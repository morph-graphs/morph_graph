import warnings
import random
import json
import os
import math
import argparse

import numpy as np
from PIL import Image
import pandas as pd
import torch
import traceback
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import (
    logging,
)
import logzero
from logzero import logger

if not os.environ.get('ENABLE_LOGGING'):
    logzero.logfile(None)
    logzero.loglevel(logzero.CRITICAL)

from model import pecore_rekv, clip_rekv

# Model registration below is wrapped in try/except so a missing optional dependency does not
# break the whole registry. Record why each one failed: otherwise a requested-but-unregistered
# model surfaces only as `KeyError: '<name>'`, with the real cause (wrong conda env, missing
# weights) discarded -- and logger.error is a no-op unless ENABLE_LOGGING is set.
MODEL_IMPORT_ERRORS = []


EXT_MODELS = {
    # External ReKV models 
    "clip_vit_large_patch14": {
        'load_func': clip_rekv.load_model,
        'model_config': 'openai/clip-vit-large-patch14',
    },
    'siglip_so400m_patch14_384': {
        'load_func': clip_rekv.load_model,
        'model_config': 'google/siglip-so400m-patch14-384',
    },
    'siglip2_so400m_patch14_384': {
        'load_func': clip_rekv.load_model,
        'model_config': 'google/siglip2-so400m-patch14-384',
    },
    'pecore_l14_336': {
        'load_func': pecore_rekv.load_model,
        'model_config': 'PE-Core-L14-336',
    },
}

""" LLaVA-OneVision models """
try: 
    from transformers import (
        logging,
        LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor,
        VideoLlavaForConditionalGeneration, VideoLlavaProcessor
    )

    from model import llava_onevision_rekv, video_llava_rekv
    MODELS = {
        'llava_ov_7b': {
            'load_func': llava_onevision_rekv.load_model,
            'model_class': LlavaOnevisionForConditionalGeneration,
            'processor_class': LlavaOnevisionProcessor,
            'model_path': 'model_zoo/llava-onevision-qwen2-7b-ov-hf',
        },
        'llava_ov_72b': {
            'load_func': llava_onevision_rekv.load_model,
            'model_class': LlavaOnevisionForConditionalGeneration,
            'processor_class': LlavaOnevisionProcessor,
            'model_path': 'model_zoo/llava-onevision-qwen2-72b-ov-hf',
        },
        'video_llava_7b': {
            'load_func': video_llava_rekv.load_model,
            'model_class': VideoLlavaForConditionalGeneration,
            'processor_class': VideoLlavaProcessor,
            'model_path': 'model_zoo/Video-LLaVA-7B-hf',
        },
    }
except Exception as e:
    MODEL_IMPORT_ERRORS.append(traceback.format_exc())
    logger.error(f"Error loading older ReKV Models: {e}")

""" Qwen-VL models """
try:
    from model import qwen2_5_vl_rekv 
    from model import qwen2_vl_rekv    
    # Qwen2 VL models
    MODELS["qwen2_vl_7b"] = { 
        "load_func": qwen2_vl_rekv.load_model, 
        "model_path": 'model_zoo/Qwen2-VL-7B-Instruct'
    }
    # Qwen2.5 VL models
    MODELS["qwen2_5_vl_3b"] = { 
        "load_func": qwen2_5_vl_rekv.load_model, 
        "model_path": 'model_zoo/Qwen2.5-VL-3B-Instruct'
    }
    MODELS["qwen2_5_vl_7b"] = { 
        "load_func": qwen2_5_vl_rekv.load_model, 
        "model_path": 'model_zoo/Qwen2.5-VL-7B-Instruct'
    }
    MODELS["qwen2_5_vl_32b"] = { 
        "load_func": qwen2_5_vl_rekv.load_model, 
        "model_path": 'model_zoo/Qwen2.5-VL-32B-Instruct'
    }
except Exception as e:
    MODEL_IMPORT_ERRORS.append(traceback.format_exc())
    logger.error(f"Error loading Qwen-VL: {e}")

""" Qwen3-VL models """
try: 
    from model import qwen3_vl_rekv
    # Qwen-VL models 
    MODELS["qwen3_vl_8b"] = { 
        "load_func": qwen3_vl_rekv.load_model, 
        "model_path": 'model_zoo/Qwen3-VL-8B-Instruct'
    }
    # 4B shares the 8B's KV geometry (36 layers, 8 kv-heads, head_dim 128) and the
    # same architecture template; smaller weights + vision tower unlock A5000-class GPUs.
    MODELS["qwen3_vl_4b"] = {
        "load_func": qwen3_vl_rekv.load_model,
        "model_path": 'model_zoo/Qwen3-VL-4B-Instruct'
    }
except Exception as e:
    MODEL_IMPORT_ERRORS.append(traceback.format_exc())
    logger.error(f"Error loading Qwen-VL: {e}")

""" InternVL3 and InternVL3.5 models """
try:
    from model import internvl3_rekv
    MODELS["internvl3_8b"] = { 
        "load_func": internvl3_rekv.load_model,
        "model_path": "model_zoo/InternVL3-8B-HF"
    }
    MODELS["internvl3_5_8b"] = {
        "load_func": internvl3_rekv.load_model,
        "model_path": "model_zoo/InternVL3_5-8B-HF"
    }
except Exception as e:
    MODEL_IMPORT_ERRORS.append(traceback.format_exc())
    logger.error(f"Error loading InternVL3 and InternVL3.5: {e}")

class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None,
                 retrieve_size=64, chunk_size=1, 
                 resume=False, enable_similarity=False,
                 tokens_per_second=1.0,
                 **ext_model_kwargs) -> None:
        
        self.sample_fps = sample_fps
        self.tokens_per_second = tokens_per_second

        self.qa_model = qa_model
        self.qa_processor = qa_processor

        # External ReKV model 
        self.ext_model = ext_model_kwargs.get('ext_model', None)
        self.ext_processor = ext_model_kwargs.get('ext_processor', None)
        self.ext_merge_method = ext_model_kwargs.get('ext_merge_method', None)
        self.ext_weights = ext_model_kwargs.get('ext_weights', None)
        self.ext_rrf_k = ext_model_kwargs.get('ext_rrf_k', 60)
        
        # Retrieval Hyperparams
        assert chunk_size <= retrieve_size, f'chunk_size: {chunk_size}, retrieve_size: {retrieve_size}'
        self.retrieve_size = retrieve_size
        self.chunk_size = chunk_size

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno
        self.eval_grounding = 'temporal_windows' in anno[0]['conversations'][0]

        self.save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = {(self.retrieve_size, self.chunk_size): []}
        self.enable_similarity = enable_similarity
        
        # Checkpoint functionality
        self.resume = resume
        self.temp_file = f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv.tmp'
        self.final_file = f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv'
        
        # Check if this is streaming VQA (RVS-Ego/RVS-Movie)
        is_streaming_vqa = 'Stream' in self.__class__.__name__
        
        # Load existing results if resuming
        if self.resume:
            # For streaming VQA, also check final file if temp file doesn't exist
            resume_file = None
            if os.path.exists(self.temp_file):
                resume_file = self.temp_file
            elif os.path.exists(self.final_file):
                resume_file = self.final_file
                logger.info(f"Temp file not found, using final file for resume: {self.final_file}")
            
            if resume_file:
                try:
                    existing_df = pd.read_csv(resume_file)
                    # Convert existing results back to record format
                    for _, row in existing_df.iterrows():
                        result_dict = row.to_dict()
                        self.record[(self.retrieve_size, self.chunk_size)].append(result_dict)
                    logger.info(f"Resumed from {resume_file}: {len(self.record[(self.retrieve_size, self.chunk_size)])} existing results")
                    # Build completed QA index for skipping - only mark as complete if entry is valid
                    self.completed_qas_by_video = {}
                    if 'video_id' in existing_df.columns and 'question' in existing_df.columns:
                        for _, row in existing_df.iterrows():
                            vid = row['video_id'] if 'video_id' in row.index else None
                            q = row['question'] if 'question' in row.index else None
                            # Only mark as completed if both video_id and question exist AND entry is complete
                            if pd.notna(vid) and pd.notna(q) and self._is_entry_complete(row):
                                # For streaming VQA, use composite key (question, start_time, end_time)
                                # For non-streaming VQA, use just question text
                                if is_streaming_vqa and 'start_time' in existing_df.columns and 'end_time' in existing_df.columns:
                                    start_time = row['start_time'] if 'start_time' in row.index else None
                                    end_time = row['end_time'] if 'end_time' in row.index else None
                                    if pd.notna(start_time) and pd.notna(end_time):
                                        question_key = (q, float(start_time), float(end_time))
                                    else:
                                        # Fallback to just question if temporal info missing
                                        question_key = q
                                else:
                                    question_key = q
                                self.completed_qas_by_video.setdefault(vid, set()).add(question_key)
                    else:
                        self.completed_qas_by_video = {}
                except Exception as e:
                    logger.warning(f"Failed to load existing results: {e}")
                    self.record = {(self.retrieve_size, self.chunk_size): []}
                    self.completed_qas_by_video = {}
            else:
                self.completed_qas_by_video = {}
        elif not self.resume and os.path.exists(self.temp_file):
            # Remove temp file if not resuming
            os.remove(self.temp_file)
            logger.info("Removed existing temp file (not resuming)")
            self.completed_qas_by_video = {}
        else:
            self.completed_qas_by_video = {}

    def _is_entry_complete(self, row):
        """
        Check if a CSV row represents a complete QA entry.
        
        Args:
            row: pandas Series or dict representing a CSV row
            
        Returns:
            bool: True if entry is complete, False otherwise
        """
        # Handle both pandas Series and dict - use dictionary-style access
        def get_value(key):
            try:
                if isinstance(row, dict):
                    return row.get(key)
                else:  # pandas Series
                    return row[key] if key in row.index else None
            except (KeyError, IndexError):
                return None
        
        # Must have video_id and question
        video_id = get_value('video_id')
        question = get_value('question')
        if pd.isna(video_id) or pd.isna(question):
            return False
        
        # Must have pred_answer (required for all QA types)
        pred_answer = get_value('pred_answer')
        if pred_answer is None or pd.isna(pred_answer) or (isinstance(pred_answer, str) and len(pred_answer.strip()) == 0):
            return False
        
        # For close QA (MCQA), must also have pred_choice if the column exists
        pred_choice = get_value('pred_choice')
        if pred_choice is not None:  # Column exists
            if pd.isna(pred_choice) or (isinstance(pred_choice, str) and len(str(pred_choice).strip()) == 0):
                return False
        
        return True

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def load_video(self, video_path): 
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
        video = vr.get_batch(frame_idx).asnumpy()
        logger.debug(f'video shape: {video.shape}')
        timestamps = np.array([vr.get_frame_timestamp(idx) for idx in frame_idx])
        return video

    def get_framewise_entropy(self, video): 

        def compute_entropy_per_image(image: Image.Image) -> float:
            image_np = np.array(image)
            entropy_sum = 0.0
            for c in range(3):
                channel = image_np[..., c]
                histogram, _ = np.histogram(channel, bins=256, range=(0, 255), density=True)
                histogram += 1e-9
                entropy = -np.sum(histogram * np.log2(histogram))
                entropy_sum += entropy
            return entropy_sum / 3.0

        entropy_lst = [] 
        for frame in video: 
            entropy_lst.append(compute_entropy_per_image(Image.fromarray(frame)))
        return entropy_lst
    
    def calc_recall_precision(self, gt_temporal_windows, retrieved_mask):
        total_intersection_length = 0.0
    
        for (start_sec, end_sec) in gt_temporal_windows:
            start = math.floor(start_sec)
            end = math.ceil(end_sec)
            for i in range(start, end):
                if i < len(retrieved_mask) and retrieved_mask[i]:
                    intersection_start = max(start_sec, i)
                    intersection_end = min(end_sec, i + 1)
                    total_intersection_length += intersection_end - intersection_start

        gt_len = sum([end_sec - start_sec for start_sec, end_sec in gt_temporal_windows])
        retrieved_len = sum(retrieved_mask).item()

        recall = total_intersection_length / gt_len if gt_len > 0 else 0
        precision = total_intersection_length / retrieved_len if retrieved_len > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return recall, precision, f1
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s):
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            return s[0]

    def video_open_qa(self, question, max_new_tokens=1024):
        pass

    def video_close_qa(self, question, candidates, correct_choice):
        pass

    @torch.inference_mode()
    def analyze_a_video(self, video_sample, **kwargs): 
        pass

    def analyze(self, debug=False, **kwargs):
        video_annos = self.anno[:1] if debug else self.anno

        # Check if this is streaming VQA (questions processed sequentially at different timesteps)
        is_streaming_vqa = 'Stream' in self.__class__.__name__
        
        # If resuming, handle differently for streaming vs non-streaming VQA
        if self.resume:
            if is_streaming_vqa:
                # For streaming VQA: Don't skip entire videos because video encoding is incremental
                # Questions must be processed sequentially, and per-question skipping is handled in analyze_a_video
                total_videos = len(video_annos)
                # Count how many questions are already completed across all videos
                total_completed_qs = sum(len(qs) for qs in self.completed_qas_by_video.values())
                logger.info(f"Resume enabled (streaming): chunk {self.chunk_idx+1}/{self.num_chunks} — "
                          f"total videos: {total_videos}, completed questions: {total_completed_qs}")
                logger.info("Streaming VQA: Processing all videos (per-question skipping handled in analyze_a_video)")
            else:
                # For non-streaming VQA: Can skip entire videos if all questions are completed
                def _is_video_completed(sample):
                    vid = sample.get('video_id')
                    completed_qs = self.completed_qas_by_video.get(vid, set())
                    total_qs = {c.get('question') for c in sample.get('conversations', []) if 'question' in c}
                    # Only consider video complete if all questions are completed AND we have at least one question
                    is_complete = len(total_qs) > 0 and completed_qs.issuperset(total_qs)
                    if is_complete:
                        logger.debug(f"Video {vid} is complete: {len(completed_qs)}/{len(total_qs)} questions completed")
                    return is_complete

                total_videos = len(video_annos)
                remaining_annos = [s for s in video_annos if not _is_video_completed(s)]
                completed_videos = total_videos - len(remaining_annos)
                logger.info(f"Resume enabled: chunk {self.chunk_idx+1}/{self.num_chunks} — total videos: {total_videos}, completed: {completed_videos}, remaining: {len(remaining_annos)}")
                if completed_videos > 0:
                    logger.info(f"Skipping {completed_videos} videos with all questions already completed")
                video_annos = remaining_annos

        tqdm_desc = f"Chunk {self.chunk_idx+1}/{self.num_chunks}" if self.num_chunks is not None and self.chunk_idx is not None else None
        for video_sample in tqdm(video_annos, desc=tqdm_desc):
            logger.debug(f'video_id: {video_sample["video_id"]}')
            # Expose the current video to layer-level diagnostics
            # (kv_cache_manager reads CURRENT_VIDEO_ID when writing
            # per-video attention CSVs).
            os.environ['CURRENT_VIDEO_ID'] = str(video_sample.get('video_id', 'unknown_video'))
            # Skip video if all questions already completed (only for non-streaming VQA)
            if self.resume and not is_streaming_vqa:
                vid = video_sample.get('video_id')
                completed_qs = self.completed_qas_by_video.get(vid, set())
                total_qs = {c.get('question') for c in video_sample.get('conversations', []) if 'question' in c}
                if len(total_qs) > 0 and completed_qs.issuperset(total_qs):
                    logger.info(f"Skip video {vid} (all {len(total_qs)} questions already completed)")
                    continue
            self.analyze_a_video(video_sample, **kwargs)
            
            # Save incremental results to temp file
            self._save_temp_results()

        # Save final results and clean up temp file
        self._save_final_results()

    def _save_temp_results(self):
        """Save current results to temporary file"""
        try:
            dfs = []
            for (retrieve_size, chunk_size), dict_list in self.record.items():
                if dict_list:  # Only save if there are results
                    df = pd.DataFrame(dict_list)
                    df['retrieve_size'] = retrieve_size
                    df['chunk_size'] = chunk_size
                    dfs.append(df)
            
            if dfs:
                temp_df = pd.concat(dfs, ignore_index=True)
                temp_df.to_csv(self.temp_file, index=False)
        except Exception as e:
            logger.warning(f"Failed to save temp results: {e}")

    def _save_final_results(self):
        """Save final results and clean up temp file"""
        try:
            dfs = []
            for (retrieve_size, chunk_size), dict_list in self.record.items():
                if dict_list:  # Only save if there are results
                    df = pd.DataFrame(dict_list)
                    df['retrieve_size'] = retrieve_size
                    df['chunk_size'] = chunk_size
                    dfs.append(df)
            
            if dfs:
                final_df = pd.concat(dfs, ignore_index=True)
                final_df.to_csv(self.final_file, index=False)
                logger.info(f"Results saved to {self.final_file}")
            
            # Clean up temp file
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
                logger.info("Temp file cleaned up")
                
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def work(QA_CLASS):
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava_ov_7b")
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--encode_chunk_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing results.csv.tmp file")
    
    # Target Frames 
    parser.add_argument("--target_frames", type=int, default=None)
    parser.add_argument("--min_tokens", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false",
                        help="Do not page-lock offloaded KV blocks. Needed above ~115 GiB of "
                             "KV per video; transfer-speed only, results are unchanged.")

    # Temporal windows 
    parser.add_argument("--use_temporal_windows", action="store_true", default=False)
    parser.add_argument("--set_random_temporal_windows", action="store_true", default=False)
    parser.add_argument("--retrieve_from_temporal_windows", action="store_true", default=False)
    
    # Similarity analysis 
    parser.add_argument("--enable_similarity", action="store_true", default=False)

    # Extenal ReKV models 
    parser.add_argument("--ext_model", type=str, default=None, choices=["pecore_l14_336", "clip_vit_large_patch14", 
                                                                        "siglip_so400m_patch14_384", "siglip2_so400m_patch14_384"])
    parser.add_argument("--ext_merge_method", type=str, default="rrf", choices=['rrf', 'l2_concat'])
    parser.add_argument("--ext_weights", type=float, nargs="+", default=[0.5, 0.5], help='Weights for each external model')
    parser.add_argument("--ext_collect_attn", action="store_true", default=False) 
    parser.add_argument("--ext_rank_passing", type=int, default=None, help="share ranks across layers")
    parser.add_argument("--ext_rrf_k", type=int, default=60, help="RRF constant k for reciprocal rank fusion")

    # load retrieved indices from folder 
    parser.add_argument("--retrieved_indices_folder", type=str, default=None)
    parser.add_argument("--retrieved_sims_folder", type=str, default=None)

    # prefill retrieval methods 
    parser.add_argument("--prefill_method", type=str, default="swa")
    parser.add_argument("--retrieval_method", type=str, default="rekv")

    args = parser.parse_args()

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # VideoQA model
    if args.model not in MODELS:
        raise SystemExit(
            f"model {args.model!r} is not registered. Available: {sorted(MODELS)}\n"
            f"This usually means its import failed -- most often the wrong conda env "
            f"(Qwen3-VL needs transformers>=4.57, i.e. memstream_qwen3_cu126).\n"
            + ("\n".join(MODEL_IMPORT_ERRORS) if MODEL_IMPORT_ERRORS
               else "(no import errors were recorded)"))
    model_path = MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    # Per-frame token budget: only the Qwen2.5-VL / Qwen3-VL loaders accept it.
    tpf_kwargs = {}
    if args.max_tokens is not None:
        if not ("qwen2_5_vl" in args.model or "qwen3_vl" in args.model):
            raise ValueError(f"--min_tokens/--max_tokens unsupported for model {args.model}")
        tpf_kwargs = dict(min_tokens=args.min_tokens, max_tokens=args.max_tokens)

    if not args.pin_memory:
        # Restricted to qwen3_vl only: its load_model() is the only loader with a pin_memory
        # parameter -- the others hardcode 'pin_memory': True and would TypeError on the kwarg,
        # yet share the MemoryUnit code that hits the vm.max_map_count ceiling on long videos
        # (see kv_cache_manager.py MemoryUnit.__init__). Until pin_memory is threaded through
        # those loaders, use REKV_DISABLE_PIN=1 -- honored at the MemoryUnit site for every
        # backbone, equivalent to this flag.
        if "qwen3_vl" not in args.model:
            raise ValueError(
                f"--no_pin_memory unsupported for model {args.model}; "
                "set REKV_DISABLE_PIN=1 instead (same effect, works for all backbones)"
            )
        tpf_kwargs["pin_memory"] = False

    videoqa_model, videoqa_processor = load_func(
        **tpf_kwargs,
        model_path=model_path,
        n_local=args.n_local,
        topk=args.retrieve_size,
        encode_chunk_size=args.encode_chunk_size,
        chunk_size=args.retrieve_chunk_size,
        sample_fps = args.sample_fps, 
        target_frames=args.target_frames, 
        prefill_method=args.prefill_method,
        retrieval_method=args.retrieval_method, 
        ext_rank_passing=args.ext_rank_passing,
    )
    
    # Handle variable temporal patch size
    if "qwen" in args.model: 
        assert "temporal_patch_size" in videoqa_model.config.vision_config, f"Temporal patch size not found in {args.model}"
        temporal_patch_size = videoqa_model.config.vision_config.temporal_patch_size
    elif "llava" in args.model or "intern" in args.model:
        # LlavaOneVision and InternVL process frames individually — no temporal compression
        temporal_patch_size = 1
    else: 
        raise ValueError(f"Model {args.model} not supported")

    # Load External ReKV model
    ext_model_kwargs = {}
    if args.ext_model is not None:
        ext_model_config = EXT_MODELS[args.ext_model]['model_config']
        ext_model_load_func = EXT_MODELS[args.ext_model]['load_func']
        logger.info(f"Loading External ReKV model: {ext_model_config}")
        ext_model, ext_processor = ext_model_load_func(
            model_config=ext_model_config,
            topk=args.retrieve_size,
            chunk_size=args.retrieve_chunk_size,
            encode_chunk_size=args.encode_chunk_size,
            temporal_patch_size=temporal_patch_size,
        )
        ext_model_kwargs = {
            'ext_model': ext_model,
            'ext_processor': ext_processor,
            'ext_merge_method': args.ext_merge_method,
            'ext_weights': args.ext_weights,
        }
        logger.info(f"External Merge Method: {args.ext_merge_method}")
        logger.info(f"Fusion Weights: {args.ext_weights}")

    ext_model_kwargs['ext_collect_attn'] = args.ext_collect_attn 
    ext_model_kwargs['ext_rank_passing'] = args.ext_rank_passing
    ext_model_kwargs['ext_weights'] = args.ext_weights
    ext_model_kwargs['ext_rrf_k'] = args.ext_rrf_k

    # Load ground truth file
    anno = json.load(open(args.anno_path))

    retrieve_analyzer = QA_CLASS(
        anno=anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        retrieve_size=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        save_dir=args.save_dir,
        encode_chunk_size=args.encode_chunk_size, 
        resume=args.resume,
        enable_similarity=args.enable_similarity,
        tokens_per_second=temporal_patch_size,
        **ext_model_kwargs,
    )
    
    retrieve_analyzer.analyze(debug=args.debug, 
                              **{
                                  "use_temporal_windows": args.use_temporal_windows,
                                  "set_random_temporal_windows": args.set_random_temporal_windows,
                                  "retrieve_from_temporal_windows": args.retrieve_from_temporal_windows,
                                  "retrieved_indices_folder": args.retrieved_indices_folder, 
                                  "retrieved_sims_folder": args.retrieved_sims_folder
                              })
