import torch
import os
import random 
import json
import logzero 
from logzero import logger
import math 

if not os.environ.get('ENABLE_LOGGING'):
    logzero.logfile(None)
    logzero.loglevel(logzero.CRITICAL)

from video_qa.base import BaseVQA, work
from analysis.compare_pairwise_sims import interpolate_1d_sims
from tqdm import tqdm 

from analysis.similarity_logger import SimilarityLogger, patch_rekv_for_logging, patch_rekv_after_encoding, restore_rekv_methods 

class ReKVOfflineVQA(BaseVQA):
    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None, **ext_kwargs):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }

        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens, 
            retrieved_indices=retrieved_indices, **ext_kwargs)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None, **ext_kwargs):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, 
            retrieved_indices=retrieved_indices, **ext_kwargs) 
        if len(pred_answer.strip()) == 0:
            logger.warning(f"empty prediction for question: {question!r}")
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample, **kwargs):

        """ Process a video and answer all questions related to the video
            Args: 
                - video_sample: dictionary corresponding to the video sample 
                - use_temporal_windows: if true, then crop the video to corresponding windows for each question  
                = set_random_temporal_windows: if true, then make it a random offset 
        """

        logger.info(f"use_temporal_windows: {kwargs.get('use_temporal_windows')}")

        use_temporal_windows = kwargs.get("use_temporal_windows")
        set_random_temporal_windows = kwargs.get("set_random_temporal_windows")
        retrieve_from_temporal_windows = kwargs.get("retrieve_from_temporal_windows")
        temporal_windows = None 

        retrieved_indices_folder = kwargs.get("retrieved_indices_folder", None)
        retrieved_sims_folder = kwargs.get("retrieved_sims_folder", None)
        assert not (retrieved_indices_folder and retrieved_sims_folder) 

        # load and preprocess video frames for QA
        video_path = video_sample['video_path']
        video = self.load_video(video_path) # t, h, w, 3 
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video 

        # load retrieved indices from folder 
        if retrieved_indices_folder: 
            retrieved_indices_json_path = os.path.join(retrieved_indices_folder, video_sample['video_id'], "similarity_analysis.json")
            with open(retrieved_indices_json_path, "r") as f:
                sims_data = json.load(f)
                all_retrieved_indices = sims_data.get("logged_data", {}).get("layer_retrieved_indices", {}) # list of dicts {layer_idx: [indices]}

        # load retrieved sims from folder 
        if retrieved_sims_folder:
            retrieved_sims_json_path = os.path.join(retrieved_sims_folder, video_sample['video_id'], "similarity_analysis.json")
            with open(retrieved_sims_json_path, "r") as f:
                sims_data = json.load(f)
                all_retrieved_sims = sims_data.get("logged_data", {}).get("layer_similarities", {}) # list of dicts {layer_idx: [similarities]}

        # load temporal windows if they exist 
        try: 
            if retrieve_from_temporal_windows or use_temporal_windows or set_random_temporal_windows:   
                temporal_windows = [conv['temporal_windows'] for conv in video_sample['conversations']]
        except: 
            logger.warning("No temporal windows found!")
            temporal_windows = None 

        if temporal_windows is None or retrieve_from_temporal_windows:             
            self.qa_model.clear_cache()
            self.qa_model.encode_init_prompt()
            self.qa_model.encode_video(video_tensor)
        
        if self.ext_model is not None:
            self.ext_model.clear_cache()
            self.ext_model.encode_video(video_tensor)
            
        # Initialize similarity logging if enabled
        similarity_logger = None
        original_methods = None
        if hasattr(self, 'enable_similarity') and self.enable_similarity:
            os.makedirs(os.path.join(self.save_dir, "similarity_analysis", video_sample['video_id']), exist_ok=True)
            similarity_logger = SimilarityLogger(os.path.join(self.save_dir, "similarity_analysis", video_sample['video_id']))
            original_methods = patch_rekv_after_encoding(self.qa_model, similarity_logger)

        for idx, sample in enumerate(video_sample['conversations']):
            question = sample['question']
            answer = sample['answer']

            # Skip if this QA already completed in temp results
            if self.resume:
                completed_qs = self.completed_qas_by_video.get(video_sample['video_id'], set())
                if question in completed_qs:
                    continue

            # Capture similarities if enabled
            if similarity_logger is not None:
                for layer_kv in self.qa_model.kv_cache:
                    layer_kv.set_retrieval()
                similarity_logger.log_question(question, video_sample['video_id'])

            # encode video with cropped frames passed in 
            if temporal_windows is not None and not retrieve_from_temporal_windows: 
                # convert temporal windows to frame indices based on sample_fps
                windows = [[int(window[0] * self.sample_fps), int((window[1] + 1) * self.sample_fps)] for window in temporal_windows[idx]]
                if set_random_temporal_windows:
                    offsets = [random.randint(0, window[0]) for window in windows]
                    windows = [[window[0] - offset, window[1] - offset]  for offset, window in zip(offsets, windows)]

               
                n_frames = video_tensor.shape[0]
                windows = [[max(0, w[0]), min(n_frames, w[1])] for w in windows]
                temporal_factor = getattr(self.qa_model, 'temporal_factor', 1)
                if temporal_factor > 1:
                    def _n_clue_frames(ws):
                        return sum(max(0, window[1] - window[0]) for window in ws)

                    # check if video has enough frames to form a valid clue crop
                    if n_frames < temporal_factor:
                        raise RuntimeError(
                            f"{video_sample['video_id']}: video has {n_frames} frames, fewer than "
                            f"temporal_factor={temporal_factor}; cannot form a valid clue crop"
                        )
                  
                    # check if clue windows select no frames from the video
                    if _n_clue_frames(windows) == 0:
                        raise RuntimeError(
                            f"{video_sample['video_id']}: clue windows {windows} select no frames from "
                            f"a {n_frames}-frame video; nothing to repeat"
                        )

                video_tensor_cropped = torch.cat([video_tensor[window[0] : window[1]] for window in windows], dim=0)

                # pad the clue crop to be at least temporal_factor frames
                if temporal_factor > 1 and 0 < video_tensor_cropped.shape[0] < temporal_factor:
                    n_repeat = temporal_factor - video_tensor_cropped.shape[0]
                    last = video_tensor_cropped[-1:]
                    video_tensor_cropped = torch.cat(
                        [video_tensor_cropped] + [last] * n_repeat, dim=0
                    )

                # check prior to encoding that the cropped video has enough frames 
                assert video_tensor_cropped.shape[0] >= temporal_factor, (
                    f"cropped video has {video_tensor_cropped.shape[0]} frames, need at least "
                    f"temporal_factor={temporal_factor} - {video_tensor.shape} - {windows}"
                )

                self.qa_model.clear_cache()
                self.qa_model.encode_init_prompt()
                self.qa_model.encode_video(video_tensor_cropped)

            # Handle external retrieval 
            ext_model_outs = {}
            if self.ext_model is not None:
                question_features, video_features = self.ext_model.calc_topk(question)
                ext_model_outs = {
                    'ext_question_features': question_features,
                    'ext_video_features': video_features,
                    'ext_retrieved_block_indices': self.ext_model.retrieved_block_indices,
                    'ext_similarity': self.ext_model.similarity,
                    'ext_merge_method': self.ext_merge_method,
                    'ext_weights': self.ext_weights,
                    'ext_rrf_k': self.ext_rrf_k,
                }
            # handle cross-layer rank passing 
            elif self.ext_weights is not None: 
                ext_model_outs = {
                    'ext_weights': self.ext_weights,
                    'ext_rrf_k': self.ext_rrf_k,
                }
            
            
            # retrieve from temporal windows if enabled
            if retrieve_from_temporal_windows: 
                n_frames = video_tensor.shape[0] // self.tokens_per_second 
                # ensure we get at least 2 frames - with padding on top and bottom 
                windows = [[max(0, int(window[0] * self.sample_fps / self.tokens_per_second) - 1), 
                            min(n_frames, int(window[1] * self.sample_fps / self.tokens_per_second) + 1)] for window in temporal_windows[idx]]
                # combine into one range of frames 
                retrieved_indices = [frame for window in windows for frame in range(window[0], window[1])] # [[1, 2], [8, 10]] -> 1, 2, 8, 9, 10
                if len(retrieved_indices) > self.retrieve_size:
                    sample_size = len(retrieved_indices) // self.retrieve_size + 1
                    retrieved_indices = retrieved_indices[::sample_size]
                assert len(retrieved_indices) <= self.retrieve_size
                retrieved_indices = [retrieved_indices]
            # if provided retrieved indices from folder, use them 
            elif retrieved_indices_folder: 
                retrieved_indices = all_retrieved_indices[idx] 
            elif retrieved_sims_folder:
                n_frames = math.ceil(video_tensor.shape[0] / self.tokens_per_second)
                retrieved_sims = all_retrieved_sims[idx]
                for k, v in retrieved_sims.items():
                    if n_frames > self.retrieve_size:
                        retrieved_sims[k] = interpolate_1d_sims(v, n_frames)
                        self.qa_model.kv_cache[int(k)].similarity = torch.tensor(retrieved_sims[k]).tolist() 
                        retrieved_sims[k] = torch.tensor(retrieved_sims[k]).topk(self.retrieve_size).indices.tolist()
                    else: 
                        retrieved_sims[k] = [list(range(n_frames))]
                retrieved_indices = retrieved_sims 
            # otherwise, no retrieved indices 
            else: 
                retrieved_indices = None  

            # QA
            if 'choices' in sample:  # CloseQA
                choices = sample['choices']
                if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                    answer = choices[0]
                # manually handling an issue in VideoMME 
                if "Mendel and" in choices and "Watson and" in choices: 
                    choices = ["Pasteur and Coch", "Watson and Crick", "Mendel and Darwin", "E. Joslin and F. Benedict"]
                    answer =  "E. Joslin and F. Benedict"
                correct_choice = self.choice_letters[choices.index(answer)]
                qa_results = self.video_close_qa(question, choices, correct_choice, retrieved_indices=retrieved_indices, **ext_model_outs)
                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'choices': choices,
                    'answer': answer,
                    'correct_choice': correct_choice,
                    'pred_answer': qa_results['pred_answer'],
                    'pred_choice': qa_results['pred_choice'],
                    'qa_acc': qa_results['acc'] * 100,
                })
            else:  # OpenQA
                qa_results = self.video_open_qa(question, retrieved_indices=retrieved_indices, **ext_model_outs)
                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'answer': answer,
                    'pred_answer': qa_results['pred_answer'],
                })

            if 'question_type' in sample:
                self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']
            
            # Reset retrieval mode after QA
            if similarity_logger is not None:
                for layer_kv in self.qa_model.kv_cache:
                    layer_kv.reset_retrieval()
        
        # Cleanup similarity logging
        if similarity_logger is not None:
            restore_rekv_methods(self.qa_model, original_methods)
            similarity_logger.save_results()


if __name__ == "__main__":
    work(ReKVOfflineVQA)
