import torch
import numpy as np
import os
import random
import json
import logzero
from logzero import logger

if not os.environ.get('ENABLE_LOGGING'):
    logzero.logfile(None)
    logzero.loglevel(logzero.CRITICAL)

from decord import VideoReader, cpu

from video_qa.base import BaseVQA, work
from analysis.similarity_logger import SimilarityLogger, patch_rekv_for_logging, patch_rekv_after_encoding, restore_rekv_methods

class ReKVStreamVQA(BaseVQA):
    def load_video(self, video_path):
        if video_path.endswith('.npy'):  # FPS=1
            video = np.load(video_path)
            assert self.sample_fps <= 1
            num_frames = len(video)
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
            video = video[frame_idx]
        else:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
            video = vr.get_batch(frame_idx).asnumpy()
        return video

    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None, **ext_kwargs):
        if ext_kwargs is not None and len(ext_kwargs) > 0:
            logger.info("Using external retrieval kwargs!")

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
        
        if ext_kwargs is not None and len(ext_kwargs) > 0:
            logger.info(f"Using external retrieval kwargs!: {ext_kwargs}")

        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, 
            retrieved_indices=retrieved_indices, **ext_kwargs) 
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample, **kwargs):
        # Streaming VQA doesn't use temporal_windows kwargs (uses conversation data directly)
        video_path = video_sample['video_path']
        video_start_idx = video_end_idx = 0

        logger.info(f"Starting to load video: {video_path}")

        video = self.load_video(video_path)
        video_tensor = torch.from_numpy(video)

        logger.info(f"Loaded video: {video_path} with shape: {video_tensor.shape}")

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        if self.ext_model is not None: 
            self.ext_model.clear_cache() 

        # Track question count for incremental saving
        question_count = 0
        save_interval = 10  # Save after every 10 questions

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']

            temporal_windows = torch.tensor([sample['start_time'], sample['end_time']]) * self.sample_fps
            temporal_windows = temporal_windows.tolist()

            # encode video until receiving QA (always encode, even if skipping question)
            if temporal_windows[-1] > video_end_idx:
                old_video_start_idx = video_start_idx  # Save before updating
                video_end_idx = temporal_windows[-1]
                # Pass streaming=True for incremental encoding (important for Qwen models)
                self.qa_model.encode_video(video_tensor[int(old_video_start_idx):int(video_end_idx)], streaming=True)
                video_start_idx = video_end_idx

                if self.ext_model is not None: 
                    # Use old_video_start_idx to encode the same chunk as qa_model
                    self.ext_model.encode_video(video_tensor[int(old_video_start_idx):int(video_end_idx)], streaming=True)
            
            # Create unique key for this question (question + start_time + end_time)
            # This handles duplicate questions at different timesteps
            question_key = (question, sample['start_time'], sample['end_time'])
            
            # Skip if this question was already completed (resume mode)
            # Check AFTER encoding to ensure cache is up to date
            if self.resume:
                vid = video_sample.get('video_id')
                completed_qs = self.completed_qas_by_video.get(vid, set())
                if question_key in completed_qs:
                    logger.debug(f"Skip question (video_id={vid}) question='{question}' at [{sample['start_time']:.1f}-{sample['end_time']:.1f}s] (already in temp)")
                    continue
        
            # OpenQA
            qa_results = self.video_open_qa(question, max_new_tokens=256)
            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question': question,
                'start_time': sample['start_time'],
                'end_time': sample['end_time'],
                'answer': answer,
                'pred_answer': qa_results['pred_answer'],
            })
            
            # Update completed_qas_by_video to track this question with temporal info
            vid = video_sample.get('video_id')
            self.completed_qas_by_video.setdefault(vid, set()).add(question_key)
            
            # Incremental save after every N questions
            question_count += 1
            if question_count % save_interval == 0:
                self._save_temp_results()
                logger.info(f"Saved incremental results after {question_count} questions")
        
        # Save at end of video to catch any remaining questions
        if question_count > 0 and question_count % save_interval != 0:
            self._save_temp_results()
            logger.info(f"Saved results at end of video ({question_count} total questions)")


if __name__ == "__main__":
    work(ReKVStreamVQA)
