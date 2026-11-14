import os 
import math 
import torch 
import numpy as np 
import logzero
from logzero import logger 
import torch.nn.functional as F

# Check if logger output should be enabled
if not os.environ.get('ENABLE_LOGGING'): 
    logzero.logfile(None)
    logzero.loglevel(logzero.CRITICAL)

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration 
from transformers.image_utils import SizeDict
from qwen_vl_utils import process_vision_info 

from model.patch import patch_hf 
from model.abstract_rekv import Abstract_ReKV
from model.qwen3_utils import get_qwen3_vl_rope_index

from tqdm import tqdm 
from collections import defaultdict

""" CONSTANTS """ 
TIMESTAMP_START_TOKEN_ID = 27  # "<" - corresponds to 27 - indicates the start of a new timestamp



class Qwen3_VL_ReKV(Qwen3VLForConditionalGeneration, Abstract_ReKV):

    """ 
    Relevant Links: 
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py    
    """
    
    def __init__(self, config, n_frame_tokens, init_prompt_ids, n_local, topk, 
                 encode_chunk_size, chunk_size,  
                 prefill_method, sample_fps, target_frames, 
                 **kwargs):

        Qwen3VLForConditionalGeneration.__init__(self, config)

        # Per-frame token budget. Defaults are the Qwen3-VL setting (448/512); override via
        # load_model(min_tokens=..., max_tokens=...) threaded from the CLI for tokens-per-frame
        # scaling studies. Qwen3-VL uses patch 16 with a 2x2 spatial merge, so one token is
        # 32x32 px (Qwen2.5-VL's is 28x28).
        min_tokens = kwargs.get('min_tokens', 448)
        max_tokens = kwargs.get('max_tokens', 512)

        # Get model path from kwargs or config, fallback to default
        model_path = kwargs.get("model_path", getattr(config, "_name_or_path", "model_zoo/Qwen3-VL-8B-Instruct"))

        # hard code the processor here - min/max pixels per frame 
        self.min_pixels = min_pixels = min_tokens * 32 * 32  
        self.max_pixels = max_pixels = max_tokens * 32 * 32  
        self.temporal_factor = 2 

        processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
        )
        
        # adaptive qwen parameters 
        self.adaptive_n_frame_tokens = n_frame_tokens
        self.sample_fps = sample_fps
        # Must be > topk: the per-video cache init runs the processor on a dummy window of
        # max_frames features to build cache_position_ids / init_block_sizes, and QA can pack up to
        # `topk` retrieved frames. Sizing this from n_local alone (34 frames at n_local=17000) while
        # retrieving topk=64 produces degenerate generations. Mirrors the Qwen2.5-VL wrapper.
        self.max_frames = max(n_local // max_tokens + 1, topk + 1)

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
        logger.info(f"Qwen3-VL Config - max_frames: {self.max_frames} - min_pixels: {min_pixels} - max_pixels: {max_pixels} - sample_fps: {sample_fps} - target_frames: {target_frames}")
        logger.info(f"Enable Attention Hidden Collect: {enable_attn_hidden_collect}")
        logger.info(f"Enable Cross-Layer Rank Passing: {self._enable_rank_passing} (num_prev_ranks: {ext_rank_passing if self._enable_rank_passing else 'N/A'})")
        
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, encode_chunk_size, chunk_size)

    def process_video(self, video_np, fps, curr_frame_indices):
        """ process numpy video and return processor output
            out: dict 
            - input_ids
            - attention_mask: 
            - pixel_values_videos: torch.Tensor - shape [h*w*t, patch_size**2 * 2 * 3]
            - video_grid_thw: [t, h, w] - h * w / 4 ~ max_tokens 
        """
        assert video_np.ndim == 4 and video_np.shape[-1] == 3  # (T, H, W, 3)

        T, H, W, _ = video_np.shape
        fps = fps if fps is not None else self.sample_fps 
        video_inputs = [video_np]
        video_metadatas = [{
            "fps": fps, 
            "total_num_frames": T, "height": H, "width": W, 
            "frames_indices": curr_frame_indices 
        }]
        logger.info(f"Processing video: {video_np.shape}")

        # default 
        shortest_edge = 32 * 32 * 4
        longest_edge = 32 * 32 * 24576 

        # min/max pixels scaled across all frames of the video 
        temporal_factor = self.temporal_factor 
        shortest_edge = self.min_pixels * math.ceil(T / temporal_factor) * temporal_factor
        longest_edge = self.max_pixels * math.ceil(T / temporal_factor) * temporal_factor
        size = {"shortest_edge": shortest_edge, "longest_edge": longest_edge}

        # Include video token so processor calculates timestamps
        # Use processor's actual token attributes to ensure compatibility
        text = [f"{self.processor.vision_start_token}{self.processor.video_token}{self.processor.vision_end_token}"]
        
        logger.info(f"videos: {video_inputs[0].shape}")
        out = self.processor(text=text, videos=video_inputs, video_metadata=video_metadatas, size=size, 
            do_sample_frames=False, do_resize=True, return_tensors="pt")         
        return out

    def _calculate_n_frame_tokens(self, video_grid_thw):
        """ Calculate the actual number of tokens per frame based on video_grid_thw. """
        # Calculate tokens per frame: (H * W) // spatial_merge_size^2 where T=time, H=height, W=width patches
        tokens_per_frame = video_grid_thw[1:].prod(-1) // self.config.vision_config.spatial_merge_size**2
        return tokens_per_frame.item() if tokens_per_frame.numel() == 1 else tokens_per_frame.tolist()
    
    def clear_cache(self): 
        super().clear_cache()
        # Reset streaming video cache initialization flag
        self._cache_initialized_for_video = False
        if self._store_adjacent_similarities:
            self._reset_similarity_tracking()

    def reset_retrieval_scores(self): 
        if hasattr(self.model, "retrieval_scores"):
            setattr(self.model, "retrieval_scores", None)
            setattr(self.model, "collected_attn_hidden", defaultdict(list))
            setattr(self.model, "collected_question_hidden", defaultdict(list))

    def get_prompt(self, query, mc=False): 
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt
    
    def get_current_n_frame_tokens(self):
        return getattr(self, 'adaptive_n_frame_tokens', self.n_frame_tokens)

    def _initialize_cache_for_video(self, new_n_frame_tokens, cache_position_ids, video_grid_thw, 
        n_block_prefix_tokens=0, init_block_sizes=None):
        """
            This function initializes the KV cache for a new video. 
            - Update block_size and CUDA cache unit size to match new adaptive n_frame_tokens.
            - Create M-RoPE position ids in the cache
            - Store video dims in the cache  

            Args: 
                - new_n_frame_tokens: number of tokens per frame (vision patches only)
                - cache_position_ids: position ids for M-RoPE
                - video_grid_thw: [T, H, W] - video grid (timestamp, height, width)
                - n_block_prefix_tokens: number of non-vision prefix tokens per block (e.g. timestamp + vision_start)
                - init_block_sizes: list[int] - per-frame token counts for frames in this window (including timestamp and <st> and <ed> tokens)
        """

        # apply spatial merge ratio (clone first: never mutate the caller's tensor, otherwise a
        # second call would re-divide grid_hw and corrupt it)
        video_grid_thw = video_grid_thw.clone()
        video_grid_thw[:, 1:] = video_grid_thw[:, 1:]  // self.config.vision_config.spatial_merge_size

        if hasattr(self, 'kv_cache') and self.kv_cache is not None:
            for layer_kv in self.kv_cache:
                layer_kv.block_size = new_n_frame_tokens
                layer_kv.exc_block_size = new_n_frame_tokens
                layer_kv.n_block_prefix_tokens = n_block_prefix_tokens
                layer_kv.initialize_cache_for_video(new_n_frame_tokens, cache_position_ids, video_grid_thw, self.max_frames, init_block_sizes) 
            logger.info(f"Updated KV cache block sizes to {new_n_frame_tokens}, n_block_prefix_tokens={n_block_prefix_tokens}")

    def initialize_cache_for_video(self, new_n_frame_tokens, cache_position_ids, video_grid_thw, 
                                    n_block_prefix_tokens=0, init_block_sizes=None, force_reinit=False): 
        """ set the tokens per frame in the kv cache 
        
        Args:
            new_n_frame_tokens: number of tokens per frame (vision patches only)
            cache_position_ids: position ids for M-RoPE
            n_block_prefix_tokens: number of non-vision prefix tokens per block (e.g. timestamp + vision_start)
            force_reinit: if True, reinitialize even if already initialized (for non-streaming mode)
        """
        if hasattr(self, 'n_frame_tokens'):
            self.n_frame_tokens = new_n_frame_tokens
            self.adaptive_n_frame_tokens = new_n_frame_tokens
            # Only initialize if not already initialized (for streaming) or if forced
            if force_reinit or not getattr(self, '_cache_initialized_for_video', False):
                self._initialize_cache_for_video(new_n_frame_tokens, cache_position_ids, video_grid_thw, n_block_prefix_tokens, init_block_sizes)
                self._cache_initialized_for_video = True
        logger.info(f"ADAPTIVE N_FRAME_TOKENS: {new_n_frame_tokens}, n_block_prefix_tokens: {n_block_prefix_tokens}")

    def get_rope_index(self, out): 

        video_grid_thw = out['video_grid_thw']
        input_ids = out['input_ids']
        attention_mask = out['attention_mask']

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        return get_qwen3_vl_rope_index(
            input_ids,
            video_grid_thw,
            attention_mask,
            spatial_merge_size,
            video_token_id,
            vision_start_token_id,
        )
       
    def _encode_video_chunk(self, processed_dict):

        """
       _encode_video_chunk - processes a video chunk and updates the KV-cache 
        
        :param video_chunk: np.array - [n_frames, h, w, c]
        :param processed_dict: dict - contains video_grid_thw and pixel_values_videos 
        """

        # get the input embeddings for current chunk 
        video_grid_thw = processed_dict['video_grid_thw']
        pixel_values_videos = processed_dict['pixel_values_videos'].to(self.device, self.dtype) 
        input_ids = processed_dict['input_ids'].to(self.device)

        inputs_embeds = self.get_input_embeddings()(input_ids)

        # Store video grid thw on the model for use in forward pass
        setattr(self.model, "_block_size", self.adaptive_n_frame_tokens)
        
        # get video features both for input to the MLP and the deepstack features.
        # MEMSTREAM_VIT_CACHE + MEMSTREAM_VIT_MODE={record,replay} decouple the ViT from
        # the LLM chunking: record saves per-frame-pair features; replay feeds byte-identical
        # features into any encode_chunk_size, isolating LLM-side ecs effects from ViT-side.
        _vit_dir = os.environ.get("MEMSTREAM_VIT_CACHE")
        _vit_mode = os.environ.get("MEMSTREAM_VIT_MODE")
        _vid_key = os.environ.get("CURRENT_VIDEO_ID", "unknown_video")
        if _vit_dir and _vit_mode == "replay":
            _s, _e = self._vit_frame_range
            _tf = self.temporal_factor
            _pools, _deeps = [], None
            for _pair in range(_s // _tf, _e // _tf):
                _f = os.path.join(_vit_dir, f"{_vid_key}_pair{_pair:05d}.pt")
                if not os.path.exists(_f):
                    raise FileNotFoundError(f"ViT replay cache missing: {_f}")
                _d = torch.load(_f)
                _pools.append(_d["pool"])
                if _deeps is None:
                    _deeps = [[] for _ in _d["deep"]]
                for _li, _t in enumerate(_d["deep"]):
                    _deeps[_li].append(_t)
            video_embeds = (torch.cat(_pools, dim=0).to(self.device),)
            deepstack_video_embeds = [torch.cat(x, dim=0).to(self.device) for x in _deeps]
        else:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw) 
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features
            if _vit_dir and _vit_mode == "record":
                os.makedirs(_vit_dir, exist_ok=True)
                _s, _e = self._vit_frame_range
                _tf = self.temporal_factor
                _npairs = (_e - _s) // _tf
                _pool_all = torch.cat(video_embeds, dim=0)
                _pp = _pool_all.shape[0] // _npairs
                assert _pool_all.shape[0] % _npairs == 0, (_pool_all.shape, _npairs)
                for _j in range(_npairs):
                    _pair = _s // _tf + _j
                    torch.save({
                        "pool": _pool_all[_j*_pp:(_j+1)*_pp].cpu(),
                        "deep": [t[_j*_pp:(_j+1)*_pp].cpu() for t in deepstack_video_embeds],
                    }, os.path.join(_vit_dir, f"{_vid_key}_pair{_pair:05d}.pt"))
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # update variables - video_mask and visual_pos_masks 
        video_mask = video_mask[..., 0] # [1, n_tokens, embedding_dim] --> [1, n_tokens]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

        # get position ids 
        position_ids, rope_deltas = self.get_rope_index(processed_dict)
        # split into frame-level inputs based on timestamp delimeter 
        splits = torch.where(input_ids[0] == TIMESTAMP_START_TOKEN_ID)[0]
        framewise_inputs_ids_lst = []
        for i, idx in enumerate(splits):
            start = idx
            end = splits[i + 1] if i + 1 < len(splits) else len(input_ids[0])
            framewise_inputs_ids_lst.append(input_ids[...,start:end])
        block_sizes = [len(x[0]) for x in framewise_inputs_ids_lst]

        # The retrieval buffer is sized as topk*(block_size + n_block_prefix_tokens) + n_init, so a
        # chunk whose timestamp text is longer than the window measured at cache-init time would
        # silently overflow it. Fail here instead, where the offending sizes are still visible.
        n_prefix = getattr(self, "_n_block_prefix_tokens", None)
        if n_prefix is not None:
            max_prefix = max(block_sizes) - self.adaptive_n_frame_tokens
            assert max_prefix <= n_prefix, (
                f"per-frame prefix {max_prefix} exceeds n_block_prefix_tokens={n_prefix} "
                f"(block_sizes: min={min(block_sizes)} max={max(block_sizes)}, "
                f"n_frame_tokens={self.adaptive_n_frame_tokens}); the retrieval buffer would overflow"
            )
        assert sum(block_sizes) == input_ids.size(-1), (
            f"block sizes must tile the chunk exactly: sum={sum(block_sizes)} vs "
            f"len(input_ids)={input_ids.size(-1)}"
        )

        # feed visual features to the the LLM
        output = self.language_model(
            input_ids=None, 
            inputs_embeds=inputs_embeds,
            block_sizes=block_sizes, 
            past_key_values=self.kv_cache,
            visual_pos_masks=visual_pos_masks, 
            deepstack_visual_embeds=deepstack_visual_embeds, 
            position_ids=position_ids, 
            use_cache=True, 
            return_dict=True, 
        )

        self.kv_cache = output.past_key_values 

    def resample_video(self, video, target_frames, num_frames): 
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
        return video, num_frames, new_fps

    @torch.inference_mode()
    def encode_video(self, video, target_frames=None, streaming=False):  # video: (Nv, H, W, 3)

        # encode chunk by chunk
        num_frames = video.shape[0]
        encode_chunk_size = self.encode_chunk_size
        target_frames = self.target_frames

        # handle target_frames
        video, num_frames, new_fps = self.resample_video(video, target_frames, num_frames)

        # Every chunk must contain a whole number of temporal groups. process_video sizes the pixel
        # budget as min_pixels * ceil(T/temporal_factor)*temporal_factor, so a non-multiple frame
        # count gets an inflated budget, resizes to a different grid, and silently violates the
        # cache's fixed block_size with a different tokens-per-frame (e.g. 33 frames -> 493 vs 464).
        # encode_chunk_size is already a multiple of temporal_factor, so only the tail can offend.
        remainder = num_frames % self.temporal_factor
        if remainder:
            logger.info(
                f"dropping {remainder} trailing frame(s) so the frame count is a multiple of "
                f"temporal_factor={self.temporal_factor} ({num_frames} -> {num_frames - remainder})"
            )
            video = video[: num_frames - remainder]
            num_frames -= remainder
        assert encode_chunk_size % self.temporal_factor == 0, (
            f"encode_chunk_size={encode_chunk_size} must be a multiple of "
            f"temporal_factor={self.temporal_factor}"
        )

        # Initialize the cache from a dummy window sized to the largest sliding window we may hold.
        # Use the *last* frames of the video: the per-frame timestamp text grows with the time value
        # (more digits -> more tokens), so the tail of the video yields the longest blocks and hence
        # a safe upper bound for n_block_prefix_tokens / the retrieval buffer.
        n_init_frames = self.max_frames * self.temporal_factor
        init_start = max(num_frames - n_init_frames, 0)
        init_frame_indices = list(range(init_start, init_start + n_init_frames))
        video_for_init = np.zeros((n_init_frames, *video.shape[1:]))
        out = self.process_video(video_for_init, fps=new_fps, curr_frame_indices=init_frame_indices)

        # get the init block sizes 
        input_ids = out['input_ids']
        splits = torch.where(input_ids[0] == TIMESTAMP_START_TOKEN_ID)[0]
        framewise_inputs_ids_lst = []
        for i, idx in enumerate(splits):
            start = idx
            end = splits[i + 1] if i + 1 < len(splits) else len(input_ids[0])
            framewise_inputs_ids_lst.append(input_ids[...,start:end])
        init_block_sizes = [len(x[0]) for x in framewise_inputs_ids_lst]

        # compute the new tokens per frame for each video 
        video_grid_thw = out['video_grid_thw']  # [1, 3] => [T, H, W]
        new_n_frame_tokens = self._calculate_n_frame_tokens(video_grid_thw[0])

        # Non-visual prefix tokens per block (timestamp text + vision_start + vision_end), measured
        # from the worst-case (latest-timestamp) window rather than hardcoded: this sizes the
        # retrieval global_buffer, so undersizing it overflows on long videos.
        n_block_prefix_tokens = max(init_block_sizes) - new_n_frame_tokens
        assert n_block_prefix_tokens > 0, (
            f"non-positive block prefix: max(init_block_sizes)={max(init_block_sizes)} vs "
            f"n_frame_tokens={new_n_frame_tokens}"
        )
        self._n_block_prefix_tokens = n_block_prefix_tokens

        # generate positional ids 
        cache_position_ids, mrope_position_deltas = self.get_rope_index(out)
        assert streaming is False, "we don't support streaming right now!"
        self.initialize_cache_for_video(new_n_frame_tokens, cache_position_ids, video_grid_thw,
            n_block_prefix_tokens=n_block_prefix_tokens, init_block_sizes=init_block_sizes,
            force_reinit=not streaming)

        num_chunks = num_frames // encode_chunk_size
        for chunk_idx in tqdm(range(num_chunks), desc="Encoding video..."):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            video_chunk = video[start_idx:end_idx]

            # convert to numpy array and send to the processor (expects numpy array)
            processed_dict = self.process_video(video_chunk.cpu().numpy(), new_fps, list(range(start_idx, end_idx))) 
            self._vit_frame_range = (start_idx, end_idx)
            self._encode_video_chunk(processed_dict)

            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]

            processed_dict = self.process_video(remaining_video.cpu().numpy(), new_fps, list(range(start_idx, end_idx)))
            self._vit_frame_range = (start_idx, end_idx)
            self._encode_video_chunk(processed_dict)
        
        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Encoding invariant: one appended block per (temporally-merged) frame feature. Count
        # appended blocks, not offloaded ones -- when the whole video fits inside n_local the
        # window never slides, so nothing is packaged into MemoryUnits and num_global_block
        # legitimately stays 0 (this happens at low tokens-per-frame budgets).
        n_frame_features = math.ceil(num_frames / self.temporal_factor)
        layer0 = self.kv_cache[0]
        n_appended = len(layer0.adaptive_block_sizes)
        n_offloaded = layer0.num_global_block
        assert n_appended == n_frame_features, (
            f"appended {n_appended} KV blocks but the video has {n_frame_features} frame features "
            f"({num_frames} frames / temporal_factor {self.temporal_factor})"
        )
        assert n_offloaded <= n_frame_features, (
            f"offloaded {n_offloaded} blocks > {n_frame_features} frame features"
        )
        encoded_tokens = sum(layer0.adaptive_block_sizes)
        logger.info(
            f"encode_video done: {n_appended} blocks ({n_offloaded} offloaded), "
            f"{encoded_tokens} tokens ({encoded_tokens / max(n_appended, 1):.1f} tokens/block)"
        )

        # calculate the adjacenct similarity
        if self._store_adjacent_similarities: 
            self._finalize_similarity_tracking()

        # Save video RoPE cache if in streaming mode (for later restoration)
        if streaming:
            for layer_kv in self.kv_cache: 
                layer_kv.position_embedding.save_video_rope_cache()

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
        # The question is deliberately *not* padded: the patched attention forward ignores
        # `attention_mask` entirely, so padding to a fixed length would fold pad tokens into the
        # mean-pooled retrieval query. This matches the Qwen2.5-VL reference implementation.
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

            lm_kwargs = dict(ext_kwargs)
            if use_cache:
                out = self.language_model(
                    input_ids=input_ids,
                    use_cache=True,
                    past_key_values=self.kv_cache,
                    **lm_kwargs,
                )
            else:
                out = self.language_model(input_ids=input_ids, use_cache=False, **lm_kwargs)
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

def load_model(model_path='model_zoo/Qwen3-VL-8B-Instruct',
               n_init=None, n_local=8000, topk=32, 
               encode_chunk_size=64,
               chunk_size=1, 
               sample_fps=0.5, target_frames=None, 
               prefill_method='swa', 
               retrieval_method="rekv", head_idx=-1, 
               enable_attn_hidden_collect=False,
               ext_rank_passing=None,
               min_tokens=448, max_tokens=512,
               pin_memory=True,
               torch_dtype=torch.bfloat16, fattn=True):

    n_frame_tokens = 196  # Default estimate: 14*14 = 196 (patch_size^2) 
    
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    tokenizer = processor.tokenizer

    init_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    init_prompt_ids = tokenizer(init_prompt).input_ids
    
    inf_llm_config = {
        'n_init': len(init_prompt_ids) if n_init is None else n_init,
        'n_local': n_local,
        'fattn': fattn,
        'block_size': n_frame_tokens,
        'topk': topk,
        'encode_chunk_size': encode_chunk_size // 2, # account for temporal pooling  
        'chunk_size': chunk_size,
        # Must exceed the number of blocks ONE question retrieves. CudaCache.alloc() asserts
        # on an empty idle set and _remove_lru_blocks refuses to evict a block that is about
        # to be read, so an undersized cache is a hard crash, not graceful thrashing.
        # Compressed retrieval (MEMSTREAM_CRETR=r) holds up to ~r^2 times more blocks at once.
        'max_cached_block': max(128, int(topk * (int(os.environ.get("MEMSTREAM_CRETR", "0") or 0) ** 2) * 1.15) + 8),
        'exc_block_size': n_frame_tokens,
        # Page-locks every offloaded KV block. The driver refuses past ~115 GiB pinned, which
        # surfaces as `RuntimeError: CUDA error: out of memory` from the .to("cpu") in
        # MemoryUnit.__init__ -- a host failure, so no GPU or --mem change helps. Pass
        # pin_memory=False for videos whose KV exceeds that (>=116 min at 480 tpf);
        # transfer-speed only, values are bit-identical either way.
        'pin_memory': pin_memory,
        'prefill_method': prefill_method, 
        'retrieval_method': retrieval_method, 
        'head_idx': head_idx
    }
    model = Qwen3_VL_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch_dtype, # default bfloat16; fp32 for the ecs numerics probe
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
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    model = patch_hf(model, **inf_llm_config)
    model.language_model = model.model.language_model

    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()
    return model, processor