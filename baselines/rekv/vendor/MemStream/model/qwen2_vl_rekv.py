import torch 
import os
import logzero
from logzero import logger 

import torch.nn.functional as F 

# Check if logger output should be enabled
if not os.environ.get('ENABLE_LOGGING'): 
    logzero.logfile(None)
    logzero.loglevel(logzero.CRITICAL)

from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration 

from model.patch import patch_hf 
from model.abstract_rekv import Abstract_ReKV

from tqdm import tqdm 
from collections import defaultdict


def _scalar_fps(fps):
    """transformers 5 rejects a list for the processor's `fps`; older versions accepted [fps]."""
    if isinstance(fps, (list, tuple)):
        assert len(fps) == 1, f"expected a single fps value, got {fps}"
        return fps[0]
    return fps


class Qwen2_VL_ReKV(Qwen2VLForConditionalGeneration, Abstract_ReKV): 

    """ 
    Relevant Links: 
    https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L962
    https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1384
    """
    
    def __init__(self, config, n_frame_tokens, init_prompt_ids, n_local, topk, 
                 encode_chunk_size, chunk_size,  
                 prefill_method, sample_fps, target_frames, 
                 **kwargs): 
        
        Qwen2VLForConditionalGeneration.__init__(self, config)

        # max tokens 
        min_tokens = 200
        max_tokens = 256

        # Get model path from kwargs or config, fallback to default
        model_path = kwargs.get('model_path', getattr(config, '_name_or_path', "model_zoo/Qwen2-VL-7B-Instruct"))

        # hard code the processor here 
        min_pixels = min_tokens*28*28
        max_pixels = max_tokens*28*28 
        processor = Qwen2VLProcessor.from_pretrained(
            model_path,
            use_fast=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        
        # adaptive qwen parameters 
        self.adaptive_n_frame_tokens = n_frame_tokens
        self.sample_fps = sample_fps 
        self.max_frames = n_local // max_tokens + 4

        # prefill method
        self.prefill_method = prefill_method
        self.encode_chunk_size = encode_chunk_size  # NOTE: set N frames, becomes N // 2 after temporal pooling (i.e., 64 --> 32)

        # set target frames to extract 
        self.target_frames = target_frames 

        # extract from kwargs 
        enable_attn_hidden_collect = kwargs.get('enable_attn_hidden_collect', False)
        ext_rank_passing = kwargs.get('ext_rank_passing', None)
        store_adjacent_similarities = kwargs.get('store_adjacent_similarities', False)

        # enable attention hidden collect
        self._enable_attn_hidden_collect = enable_attn_hidden_collect
        if enable_attn_hidden_collect:
            setattr(self.model, "_enable_attn_hidden_collect", enable_attn_hidden_collect)
            setattr(self.model, "collected_attn_hidden", defaultdict(list))
            setattr(self.model, "collected_question_hidden", defaultdict(list))
            setattr(self.model, "retrieval_scores", None)

        # enable rank passing 
        # ext_rank_passing: int - Number of previous layers to use (1, 2, 3, or -1 for all)
        self._enable_rank_passing = ext_rank_passing is not None and ext_rank_passing != 0
        if self._enable_rank_passing:
            setattr(self.model, "__enable_rank_passing", True)
            setattr(self.model, "__num_prev_ranks_to_use", ext_rank_passing)

        # self._store_adjacent_similarities = store_adjacent_similarities
        self._store_adjacent_similarities = (self.prefill_method.startswith("swa_event_chunked"))
        if self._store_adjacent_similarities:
            self._init_similarity_tracking()

        # log all configurations 
        logger.info(f"Qwen2-VL Config - max_frames: {self.max_frames} - min_pixels: {min_pixels} - max_pixels: {max_pixels} - sample_fps: {sample_fps} - target_frames: {target_frames}")
        logger.info(f"Enable Attention Hidden Collect: {enable_attn_hidden_collect}")
        logger.info(f"Enable Cross-Layer Rank Passing: {self._enable_rank_passing} (num_prev_ranks: {ext_rank_passing if self._enable_rank_passing else 'N/A'})")
        
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, encode_chunk_size,chunk_size)
    
    def clear_cache(self): 
        super().clear_cache()
        # Reset streaming video cache initialization flag
        self._cache_initialized_for_video = False
        if self._store_adjacent_similarities:
            self._reset_similarity_tracking()
            self.event_boundaries = []

    def reset_retrieval_scores(self): 
        if hasattr(self.model, "retrieval_scores"):
            setattr(self.model, "retrieval_scores", None)
            setattr(self.model, "collected_attn_hidden", defaultdict(list))
            setattr(self.model, "collected_question_hidden", defaultdict(list))
            logger.info(f"Reset retrieval scores")

    @property
    def visual(self):
        """Vision tower, wherever this transformers version keeps it.

        <= 4.5x registered it on the ForConditionalGeneration; 5.x moved it onto `self.model`.
        """
        registered = self.__dict__.get("_modules", {}).get("visual")
        if registered is not None:
            return registered
        return self.model.visual

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _calculate_n_frame_tokens(self, video_grid_thw):
        """
        Calculate the actual number of tokens per frame based on video_grid_thw.        
        Args:
            video_grid_thw: Tensor of shape (3,) containing [T, H, W] where:
                T = temporal patches (time dimension)
                H = height patches (spatial dimension)  
                W = width patches (spatial dimension)        
        Returns:
            int: Number of tokens per frame after spatial merging
        """
        # Calculate tokens per frame: (H * W) // spatial_merge_size^2
        # where T=time, H=height, W=width patches
        tokens_per_frame = video_grid_thw[1:].prod(-1) // self.visual.spatial_merge_size**2
        return tokens_per_frame.item() if tokens_per_frame.numel() == 1 else tokens_per_frame.tolist()
    
    def get_current_n_frame_tokens(self):
        return getattr(self, 'adaptive_n_frame_tokens', self.n_frame_tokens)

    def _initialize_cache_for_video(self, new_n_frame_tokens, cache_position_ids):
        """
            - Update block_size and CUDA cache unit size to match new adaptive n_frame_tokens.
            - Create M-RoPE position ids in the cache
            - Store video dims in the cache  
        """
        if hasattr(self, 'kv_cache') and self.kv_cache is not None:
            for layer_kv in self.kv_cache:
                layer_kv.block_size = new_n_frame_tokens
                layer_kv.exc_block_size = new_n_frame_tokens
                layer_kv.initialize_cache_for_video(new_n_frame_tokens, cache_position_ids, self.max_frames) 
            logger.info(f"Updated KV cache block sizes to {new_n_frame_tokens}")

    def _get_video_features(self, pixel_values_videos, video_grid_thw): # (N, 3, H, W) 
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype) 
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        # transformers 5 wraps the vision tower output; `pooler_output` is the post-merger
        # embedding that older versions returned directly.
        if not isinstance(video_embeds, torch.Tensor):
            video_embeds = video_embeds.pooler_output
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = video_embeds.split(split_sizes, dim=0) # list([Nt, 3584]) with len=N 
        video_embeds = torch.cat(video_embeds, dim=-1).unsqueeze(0) # (1, Nt*N, 3584)
        if self._store_adjacent_similarities: 
            self._update_similarity_tracking(video_embeds)
        return video_embeds 

    def get_tokens_per_frame(self, video_slice, fps=None): 
        """ return the tokens per frame as well as the thw video grid """
        fps = self.sample_fps if fps is None else fps 
        fps = _scalar_fps(fps)
        out = self.processor(videos=video_slice, fps=fps, text="", return_tensors="pt")
        video_grid_thw = out['video_grid_thw']  # [1, 3] => [T, H, W]
        new_n_frame_tokens = self._calculate_n_frame_tokens(video_grid_thw[0])
        return video_grid_thw, new_n_frame_tokens

    def initialize_cache_for_video(self, new_n_frame_tokens, cache_position_ids, force_reinit=False): 
        """ set the tokens per frame in the kv cache 
        
        Args:
            new_n_frame_tokens: number of tokens per frame
            cache_position_ids: position ids for M-RoPE
            force_reinit: if True, reinitialize even if already initialized (for non-streaming mode)
        """
        if hasattr(self, 'n_frame_tokens'):
            self.n_frame_tokens = new_n_frame_tokens
            self.adaptive_n_frame_tokens = new_n_frame_tokens
            # Only initialize if not already initialized (for streaming) or if forced
            if force_reinit or not getattr(self, '_cache_initialized_for_video', False):
                self._initialize_cache_for_video(new_n_frame_tokens, cache_position_ids)
                self._cache_initialized_for_video = True
        print(f"ADAPTIVE N_FRAME_TOKENS: {new_n_frame_tokens}")

    def get_rope_index(self, video_grid_thw, device, subsample_factor=1): 

        llm_pos_ids_list: list = []
        mrope_position_deltas: list = [] 

        video_index = 0 
        t, h, w = (
            self.max_frames, # video_grid_thw[video_index][0],
            video_grid_thw[video_index][1] // subsample_factor,
            video_grid_thw[video_index][2] // subsample_factor,
        )
           
        llm_grid_t, llm_grid_h, llm_grid_w = (
            t, #  t.item(),
            h.item() // self.visual.spatial_merge_size,
            w.item() // self.visual.spatial_merge_size,
        )

        # Qwen2-VL uses simple integer temporal indices (no time scaling)
        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        
        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index])) 
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

        position_ids = llm_positions.to(device).unsqueeze(1) # 3, 1, seq_len 
        mrope_position_deltas.append(llm_positions.max() + 1 - 0) # len(total_input_ids) = 0 bc there is no text 
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=device).unsqueeze(1) 

        return position_ids, mrope_position_deltas

    def _encode_video_chunk_vit(self, video_chunk, fps): 

        print("VIDEO CHUNK SHAPE: ", video_chunk.shape)
        fps = _scalar_fps(fps)
        out = self.processor(videos=video_chunk, text="", return_tensors="pt", fps=fps)
        pixel_values_videos = out['pixel_values_videos'].to(self.device, self.dtype)
        video_grid_thw = out['video_grid_thw']              # [1, 3] => [T, H, W]
        print("VIDEO GRID THW: ", video_grid_thw[0])

        # Store video grid thw on the model for use in forward pass
        setattr(self.model, "_block_size", self.adaptive_n_frame_tokens)

        # Project to LLM embed space
        video_features = self._get_video_features(pixel_values_videos, video_grid_thw)  # (1, N_vid, hidden)
        print("VIDEO FEATURES SHAPE @ _encode_video_chunk: ", video_features.shape)
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'

        return video_features 

    def _encode_video_chunk_llm(self, video_features):
        output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values 

    def _encode_video_chunk(self, video_chunk, fps):
        video_features = self._encode_video_chunk_vit(video_chunk, fps)
        self._encode_video_chunk_llm(video_features)

    @torch.inference_mode()
    def encode_video_chunkwise_events(self, video, target_frames=None): 

        """ encode video via dynamic chunking based on events """
        
        def reset_cache_for_new_chunk(new_n_frame_tokens, cache_position_ids):
            """Save, clear, and restore cache state for subsequent chunks."""
            saved_state = self._save_cache_state()
            super(Qwen2_VL_ReKV, self).clear_cache() 
            self.encode_init_prompt() 
            self.initialize_cache_for_video(new_n_frame_tokens, cache_position_ids)
            self._restore_cache_state(saved_state)
            del saved_state
            torch.cuda.empty_cache()
        
        def flush_event_chunk(global_start_idx, curr_idx, num_chunks):
            """Flush accumulated frames in visual buffer to LLM."""
            # Save and restore cache state before encoding (for subsequent chunks)
            if num_chunks > 0:
                reset_cache_for_new_chunk(new_n_frame_tokens, cache_position_ids)
            
            video_features = torch.cat(self.visual_buffer[global_start_idx:curr_idx], dim=1)
            logger.info(f"Video Features Shape: {video_features.shape} - Moving visual buffer to CPU - global_start_idx: {global_start_idx} - curr_idx: {curr_idx}")
            for i in range(global_start_idx, curr_idx): 
                self.visual_buffer[i] = self.visual_buffer[i].cpu()
            self._encode_video_chunk_llm(video_features)
            self.event_boundaries.append(curr_idx)
            return curr_idx, 0, num_chunks + 1  # new global_start_idx, reset n_frames_in_curr_event, increment num_chunks
        
        def process_visual_buffer_with_events(curr_idx, n_frames_in_curr_event, global_start_idx, num_chunks):
            """Process visual buffer with event-based chunking logic."""
            while curr_idx < len(self.visual_buffer): 
                # flush current chunk if it exceeds max frames per chunk 
                if n_frames_in_curr_event == max_frames_per_chunk: 
                    logger.info(f"Max Frames Per Chunk Reached - Flushing - global_start_idx: {global_start_idx} - curr_idx: {curr_idx} - n_frames_in_curr_event: {n_frames_in_curr_event}")
                    global_start_idx, n_frames_in_curr_event, num_chunks = flush_event_chunk(
                        global_start_idx, curr_idx, num_chunks)

                # flush current chunk if it contains an event and has at least min frames per chunk 
                elif curr_idx > 1 + offset and self.adjacent_vis_similarities[curr_idx - offset-1] < self.running_threshold[curr_idx-offset-2] \
                    and n_frames_in_curr_event >= min_frames_per_chunk:
                    logger.info(f"Event Detected - Flushing - global_start_idx: {global_start_idx} - curr_idx: {curr_idx} - n_frames_in_curr_event: {n_frames_in_curr_event}")
                    global_start_idx, n_frames_in_curr_event, num_chunks = flush_event_chunk(
                        global_start_idx, curr_idx, num_chunks)

                n_frames_in_curr_event += 1
                curr_idx += 1
            
            return curr_idx, n_frames_in_curr_event, global_start_idx, num_chunks 
        
        num_frames = video.shape[0] 
        encode_chunk_size = self.encode_chunk_size
        target_frames = self.target_frames

        # Sample frames if target_frames is specified
        if target_frames is not None and target_frames < num_frames:
            # Uniformly sample frames
            indices = torch.linspace(0, num_frames - 1, target_frames).long()
            video = video[indices]
            # Adjust FPS proportionally
            original_fps = self.sample_fps 
            new_fps = original_fps * target_frames / num_frames
            num_frames = target_frames
        else:
            new_fps = self.sample_fps 
            print(f"Using all {num_frames} frames")  

        # each event must have at least 64 frames and at most 128 frames -- (32 - 64 after temporal patch pooling)
        prefill_method = self.prefill_method # swa_event_chunked_{min_frames_per_chunk}_{max_frames_per_chunk}
        min_frames_per_chunk = int(prefill_method.split("_")[1]) // self.config.vision_config.temporal_patch_size
        max_frames_per_chunk = int(prefill_method.split("_")[2]) // self.config.vision_config.temporal_patch_size
        n_frames_in_curr_event = 0  # tracks the number of frames in the current event 
        global_start_idx = 0        # tracks the globalstart index of the current event 
        curr_idx = 0                # tracks the current index in the adjacent_vis_similarities
        offset = 0                  # flush to LLM after these many frames after event detection
        
        encode_chunk_size = self.encode_chunk_size
        num_fixed_chunks = num_frames // encode_chunk_size 
        num_chunks = 0 

        assert self._store_adjacent_similarities, "Enable store adjacent similarities!"

        # compute the new tokens per frame for each video 
        video_grid_thw, new_n_frame_tokens = self.get_tokens_per_frame(video[0], fps=new_fps if target_frames else None)
        cache_position_ids, mrope_position_deltas = self.get_rope_index(video_grid_thw, device=self.device)
        self.initialize_cache_for_video(new_n_frame_tokens, cache_position_ids)

        # encode video features via fixed chunking 
        for chunk_idx in tqdm(range(num_fixed_chunks), desc="Encoding video..."):
            start_idx = chunk_idx * encode_chunk_size 
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]

            # encode video chunk vit features 
            self._encode_video_chunk_vit(chunk_video, fps=[new_fps])
            
            # selectively chunk video frames for llm encoding 
            logger.info(f"Encoding chunk {chunk_idx} of {num_fixed_chunks} - {start_idx} to {end_idx} - global_start_idx: {global_start_idx} - n_frames_in_curr_event: {n_frames_in_curr_event} - visual_buffer_len: {len(self.visual_buffer)}")
            curr_idx, n_frames_in_curr_event, global_start_idx, num_chunks = process_visual_buffer_with_events(
                curr_idx, n_frames_in_curr_event, global_start_idx, num_chunks)
            
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_fixed_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            
            # encode video chunk vit features
            self._encode_video_chunk_vit(remaining_video, fps=[new_fps])
            
            # encode remaining frames in llm 
            logger.info(f"Encoding remaining {remaining_frames} frames - {start_idx} to {end_idx} - global_start_idx: {global_start_idx} - n_frames_in_curr_event: {n_frames_in_curr_event} - visual_buffer_len: {len(self.visual_buffer)}")
            curr_idx, n_frames_in_curr_event, global_start_idx, num_chunks = process_visual_buffer_with_events(
                curr_idx, n_frames_in_curr_event, global_start_idx, num_chunks)
            
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # flush the final chunk 
        if global_start_idx < len(self.visual_buffer):
            if num_chunks > 0:
                reset_cache_for_new_chunk(new_n_frame_tokens, cache_position_ids)
            
            video_features = torch.cat(self.visual_buffer[global_start_idx:], dim=1)
            for i in range(global_start_idx, len(self.visual_buffer)): 
                self.visual_buffer[i] = self.visual_buffer[i].cpu()
            self._encode_video_chunk_llm(video_features)
            self.event_boundaries.append(global_start_idx)
            
        assert self.kv_cache[0].num_global_block == len(self.visual_buffer), "num blocks saved != num blocks encoded"

        # convert list of adjacencies to tensor
        self.adjacent_vis_similarities = torch.tensor(self.adjacent_vis_similarities)
    
        video_kwargs = {
            "sample_fps": new_fps, 
            'target_frames': target_frames
        }

        return video_kwargs 

    @torch.inference_mode()
    def encode_video_chunkwise(self, video, target_frames=None): 
        # encode chunk by chunk with separate KV cache per chunk 
        
        num_frames = video.shape[0] 
        encode_chunk_size = self.encode_chunk_size
        target_frames = self.target_frames

        # Sample frames if target_frames is specified
        if target_frames is not None and target_frames < num_frames:
            # Uniformly sample frames
            indices = torch.linspace(0, num_frames - 1, target_frames).long()
            video = video[indices]
            # Adjust FPS proportionally
            original_fps = self.sample_fps 
            new_fps = original_fps * target_frames / num_frames
            num_frames = target_frames
        else:
            new_fps = self.sample_fps 
            print(f"Using all {num_frames} frames")        

        num_chunks = num_frames // encode_chunk_size

        # compute the new tokens per frame for each video 
        video_grid_thw, new_n_frame_tokens = self.get_tokens_per_frame(video[0], fps=new_fps if target_frames else None)
        cache_position_ids, mrope_position_deltas = self.get_rope_index(video_grid_thw, device=self.device)
        self.initialize_cache_for_video(new_n_frame_tokens, cache_position_ids)

        for chunk_idx in tqdm(range(num_chunks), desc="Encoding video..."):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]

            # for each subsequent chunk, reset the cache and encode the initial prompt (attention sinks)  
            if chunk_idx > 0: 

                # Save critical state before clearing
                saved_state = self._save_cache_state()
                
                super().clear_cache() 
                self.encode_init_prompt() 
                self.initialize_cache_for_video(new_n_frame_tokens, cache_position_ids)
                
                # Restore accumulated state from previous chunks
                self._restore_cache_state(saved_state)
                
                # Clean up saved_state to free memory
                del saved_state
                torch.cuda.empty_cache()

            self._encode_video_chunk(chunk_video, fps=[new_fps])

            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            if num_chunks > 0: 
                # Save critical state before clearing
                saved_state = self._save_cache_state()
                
                super().clear_cache() 
                self.encode_init_prompt()
                self.initialize_cache_for_video(new_n_frame_tokens, cache_position_ids)
                
                # Restore accumulated state from previous chunks
                self._restore_cache_state(saved_state)
                
                # Clean up saved_state to free memory
                del saved_state
                torch.cuda.empty_cache()
            self._encode_video_chunk(remaining_video, fps=[new_fps])

        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
    
        # calculate the adjacenct similarity 
        if self._store_adjacent_similarities: 
            self._finalize_similarity_tracking()

        video_kwargs = {
            "sample_fps": new_fps, 
            'target_frames': target_frames
        }

        return video_kwargs 

    @torch.inference_mode()
    def encode_video(self, video, timestamps=None,target_frames=None, streaming=False):  # video: (Nv, H, W, 3)
        
        # use non-overlapping chunk-based processing if specified 
        if self.prefill_method.startswith("swa_event_chunked"): 
            return self.encode_video_chunkwise_events(video, target_frames) 
        elif self.prefill_method.startswith("swa_chunked"): 
            return self.encode_video_chunkwise(video, target_frames) 

        # encode chunk by chunk
        num_frames = video.shape[0]
        encode_chunk_size = self.encode_chunk_size
        target_frames = self.target_frames

        # Sample frames if target_frames is specified
        if target_frames is not None and target_frames < num_frames:
            # Uniformly sample frames
            indices = torch.linspace(0, num_frames - 1, target_frames).long()
            video = video[indices]
            
            # Adjust FPS proportionally
            original_fps = self.sample_fps 
            new_fps = original_fps * target_frames / num_frames
            
            print(f"Sampled video input: {video.shape} (from {num_frames} to {target_frames} frames)")
            print(f"Adjusted FPS: {new_fps:.2f} (from {original_fps})")
            num_frames = target_frames
        else:
            new_fps = self.sample_fps 
            print(f"Using all {num_frames} frames")

        num_chunks = num_frames // encode_chunk_size

        # compute the new tokens per frame for each video 
        video_grid_thw, new_n_frame_tokens = self.get_tokens_per_frame(video[0], fps=new_fps if target_frames else None)
        cache_position_ids, mrope_position_deltas = self.get_rope_index(video_grid_thw, device=self.device)
        # For streaming mode, only initialize cache on first call (when not already initialized)
        # For non-streaming mode, always reinitialize (force_reinit=True)
        self.initialize_cache_for_video(new_n_frame_tokens, cache_position_ids, force_reinit=not streaming)

        for chunk_idx in tqdm(range(num_chunks), desc="Encoding video..."):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video, fps=[new_fps])
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            self._encode_video_chunk(remaining_video, fps=[new_fps])
        
        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # calculate the adjacenct similarity 
        if self._store_adjacent_similarities: 
            self._finalize_similarity_tracking()

        video_kwargs = {
            "sample_fps": new_fps, 
            'target_frames': target_frames
        }

        return video_kwargs 

    @torch.inference_mode()
    def video_question_answering(self, input_text, video_chunk, max_new_tokens=128, use_cache=False):
        messages = [{"role": "user", "content": [
            {"type": "video", "video": "<video-0>"},
            {"type": "text", "text": input_text["question"]}
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[text], videos=[video_chunk], padding=True, return_tensors="pt")
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        print("Video Shape: ", video_chunk.shape, batch['pixel_values_videos'].shape, batch['video_grid_thw'])
        print("Input Ids: ", batch['input_ids'].shape) 
        out = self.generate(**batch, max_new_tokens=max_new_tokens, use_cache=use_cache)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None, use_cache=True, 
                        **ext_kwargs):
        device = self.device 
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        
        output_ids = []
        stopped = False 
        
        # NOTE: Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)

        if use_cache:
            for layer_kv in self.kv_cache:  # activate retrieval mode
                layer_kv.set_retrieval()

        # Internal retrieval
        if retrieved_indices is None:  
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache, **ext_kwargs) if use_cache else \
                  self.language_model(input_ids=input_ids, use_cache=False)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        
        # External retrieval
        else:      
            # get current tokens per frame 
            current_n_frame_tokens = self.get_current_n_frame_tokens()

            # loop through all layers of kv cache 
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == current_n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {current_n_frame_tokens}'
                # Skip setting external retrieval for layers using uniform sampling
                if layer_kv.use_uniform_sampling:
                    logger.info(f"Skipping external retrieval for layer {layer_kv.layer_idx} (using uniform sampling)")
                    continue
                if isinstance(retrieved_indices, dict):
                    layer_retrieved_indices = retrieved_indices[str(layer_kv.layer_idx)]
                elif isinstance(retrieved_indices, list):
                    layer_retrieved_indices = retrieved_indices 
                else: 
                    raise ValueError(f"Invalid retrieved indices type: {type(retrieved_indices)}")
                logger.info(f"got retrieved indices for layer {layer_kv.layer_idx}: {layer_retrieved_indices}")
                layer_kv.set_retrieved_block_indices(layer_retrieved_indices)

            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache) if use_cache else \
                  self.language_model(input_ids=input_ids, use_cache=False)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        if use_cache:
            for layer_kv in self.kv_cache:  # reset to default
                layer_kv.reset_retrieval()

        logger.info(f"Beginning generation of tokens!")

        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = self.lm_head(out['last_hidden_state'])
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = self.lm_head(out['last_hidden_state'])
                past_key_values = out.past_key_values

            last_token_logits = logits[0, -1, :]
            
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:
                break

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output
    
    def _save_cache_state(self):
        """Save critical cache state from current cache for independent chunk encoding.
        
        Saves init_k/v (attention sinks), global_blocks, and block_k which accumulate across chunks.
        global_remainder is NOT saved - each chunk should offload all blocks immediately.
        """
        if self.kv_cache is None:
            return None
        
        saved_state = []
        for layer_kv in self.kv_cache:
            if hasattr(layer_kv, "global_blocks") and layer_kv.global_blocks is not None:
                for u in range(len(layer_kv.global_blocks)):
                    for block in layer_kv.global_blocks[u]:
                        if getattr(block, "gpu_data", None) is not None:
                            logger.info(f"Offloading block to CPU: {block.gpu_data.shape}")
                            block.offload()
        
            layer_state = {
                # Init KV cache (attention sinks) - direct references, no clone needed
                'init_k': layer_kv.init_k if hasattr(layer_kv, 'init_k') else None,
                'init_v': layer_kv.init_v if hasattr(layer_kv, 'init_v') else None,
                'init_exc': layer_kv.init_exc if hasattr(layer_kv, 'init_exc') else False,
                
                # Global blocks (offloaded to CPU) - these accumulate across chunks
                'global_blocks': layer_kv.global_blocks if hasattr(layer_kv, 'global_blocks') else None,
                'cached_blocks': layer_kv.cached_blocks.copy() if hasattr(layer_kv, 'cached_blocks') else None,
                'num_global_block': layer_kv.num_global_block if hasattr(layer_kv, 'num_global_block') else 0,
                'num_blocks_seen': layer_kv.num_blocks_seen if hasattr(layer_kv, 'num_blocks_seen') else 0,
                
                # Representative keys for retrieval - these accumulate across chunks
                'block_k': layer_kv.block_k if hasattr(layer_kv, 'block_k') else None,
                
            }
            if layer_kv.store_full: 
                layer_state['store_full'] = layer_kv.store_full
                # Direct references, no clone needed - saves memory
                layer_state['full_gpu_k'] = layer_kv.full_gpu_k if hasattr(layer_kv, 'full_gpu_k') else None
                layer_state['full_gpu_v'] = layer_kv.full_gpu_v if hasattr(layer_kv, 'full_gpu_v') else None
            if layer_kv.store_adjacent_similarities: 
                layer_state['store_adjacent_similarities'] = layer_kv.store_adjacent_similarities 
                layer_state['adjacent_key_similarities'] = layer_kv.adjacent_key_similarities 
            
            saved_state.append(layer_state)
        
        return saved_state

    def _restore_cache_state(self, saved_state):
        """Restore critical cache state for independent chunk encoding.
        
        Restores accumulated init_k/v, global_blocks, and block_k from previous chunks.
        Each chunk encodes independently and should offload all blocks immediately.
        global_remainder is NOT preserved - chunks should be sized to avoid leftover tokens.
        """
        if saved_state is None or self.kv_cache is None:
            return
        
        for layer_idx, layer_kv in enumerate(self.kv_cache):
            layer_state = saved_state[layer_idx]

            # Restore init KV cache - delete old tensors first to free memory
            if layer_state['init_k'] is not None:
                layer_kv.init_k = layer_state['init_k']
            if layer_state['init_v'] is not None:
                layer_kv.init_v = layer_state['init_v']
            if 'init_exc' in layer_state:
                layer_kv.init_exc = layer_state['init_exc']
            
            # Restore global blocks (these accumulate across chunks)
            if layer_state['global_blocks'] is not None:
                layer_kv.global_blocks = layer_state['global_blocks']
                # Re-bind MemoryUnit.cache to the *new* CUDA cache for this layer and explicitly drop any stale GPU-side references
                if hasattr(layer_kv, "global_blocks") and layer_kv.global_blocks is not None:
                    for u in range(len(layer_kv.global_blocks)):
                        for block in layer_kv.global_blocks[u]:
                            # reset the block cache to the new cuda cache 
                            if hasattr(layer_kv, "cuda_cache"):
                                block.cache = layer_kv.cuda_cache
                            block.gpu_data = None
                            block.gpu_data_id = None
                            block.event = None
            #     logger.info(
            #         f"Restored global_blocks for layer {layer_idx}: "
            #         f"{len(layer_kv.global_blocks[0]) if layer_kv.global_blocks else 0} blocks"
            #     )
            if layer_state['cached_blocks'] is not None:
                layer_kv.cached_blocks = layer_state['cached_blocks']
            if 'num_global_block' in layer_state:
                layer_kv.num_global_block = layer_state['num_global_block']
            if 'num_blocks_seen' in layer_state:
                layer_kv.num_blocks_seen = layer_state['num_blocks_seen']

            if "store_full" in layer_state:
                layer_kv.store_full = layer_state['store_full']
            if 'full_gpu_k' in layer_state:
                layer_kv.full_gpu_k = layer_state['full_gpu_k']
            if 'full_gpu_v' in layer_state:
                layer_kv.full_gpu_v = layer_state['full_gpu_v']

            if "store_adjacent_similarities" in layer_state: 
                layer_kv.store_adjacent_similarities = layer_state['store_adjacent_similarities']
                layer_kv.adjacent_key_similarities = layer_state['adjacent_key_similarities'] 
                                    
            # Restore representative keys (these accumulate across chunks)
            if layer_state['block_k'] is not None:
                layer_kv.block_k = layer_state['block_k']


def load_model(model_path='model_zoo/Qwen2-VL-7B-Instruct',
               n_init=None, n_local=8000, topk=32, 
               encode_chunk_size=64,
               chunk_size=1, 
               sample_fps=0.5, target_frames=None, 
               prefill_method='swa', 
               retrieval_method="rekv", head_idx=-1, 
               enable_attn_hidden_collect=False,
               ext_rank_passing=None):
    
    n_frame_tokens = 196  # Default estimate: 14*14 = 196 (patch_size^2) 
    
    processor = Qwen2VLProcessor.from_pretrained(model_path, use_fast=True)
    tokenizer = processor.tokenizer

    init_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    init_prompt_ids = tokenizer(init_prompt).input_ids
    
    inf_llm_config = {
        'n_init': len(init_prompt_ids) if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'encode_chunk_size': encode_chunk_size // 2, # account for temporal pooling  
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
        'prefill_method': prefill_method, 
        'retrieval_method': retrieval_method, 
        'head_idx': head_idx
    }
    model = Qwen2_VL_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.bfloat16, # changed from float16 to bfloat16
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        encode_chunk_size=encode_chunk_size,
        chunk_size=chunk_size,
        prefill_method=prefill_method,
        sample_fps=sample_fps,
        target_frames=target_frames,
        enable_attn_hidden_collect=enable_attn_hidden_collect, 
        ext_rank_passing=ext_rank_passing,
        model_path=model_path,
    )
    model = patch_hf(model, **inf_llm_config)
    # transformers 5 nests the decoder layers under model.model.language_model; older versions put
    # them directly on model.model.
    model.language_model = getattr(model.model, "language_model", model.model)

    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()
    return model, processor