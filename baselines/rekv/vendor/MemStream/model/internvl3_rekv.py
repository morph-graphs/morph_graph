import os 
import math 
import torch 
import numpy as np 
import logzero
from logzero import logger 
from einops import rearrange
import torch.nn.functional as F

# Check if logger output should be enabled
if not os.environ.get('ENABLE_LOGGING'): 
    logzero.logfile(None)
    logzero.loglevel(logzero.CRITICAL)

from transformers import AutoProcessor, InternVLForConditionalGeneration 

from model.patch import patch_hf 
from model.abstract_rekv import Abstract_ReKV

from tqdm import tqdm 
from collections import defaultdict

""" Documentation:

InternVL3.5 - appears to use a fixed sequence length of 256 tokens per image tile, with a variable number of tiles per frame. We can do this in two ways: 
  1. processor(images=video_chunk) - will return [T * n_tiles_per_frame, 3, H, W] - this resizes the video to 448 x 448 and uses variable number of tiles per frame - determined by processor.image_processor.max_patches  
  2. processor(videos=video_chunk) - will return [T, 3, H, W] - this resizes the video to 384 x 384 and uses 1 tile 

"""

class Intern3_5_VL_ReKV(InternVLForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, n_frame_tokens, init_prompt_ids, n_local, topk, 
                 encode_chunk_size, chunk_size,  
                 prefill_method, sample_fps, target_frames, 
                 **kwargs):
        InternVLForConditionalGeneration.__init__(self, config)

        # get model path from kwargs or config, fallback to default 
        model_path = kwargs.get('model_path', getattr(config, '_name_or_path', "model_zoo/InternVL3_5-8B-HF"))

        # processor default processes image at 448 x 448 tiles and uses preset aspect ratios 
        processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
        ) 
        processor.image_processor.max_patches = 1

        # adaptive internvl parameters 
        self.adaptive_n_frame_tokens = n_frame_tokens
        self.sample_fps = sample_fps 
        self.max_frames = n_local // n_frame_tokens + 4 

        # prefill method 
        self.prefill_method = prefill_method
        self.encode_chunk_size = encode_chunk_size 

        # target frames to extract 
        self.target_frames = target_frames 

        # extract from kwargs 
        enable_attn_hidden_collect = kwargs.get('enable_attn_hidden_collect', False)
        ext_rank_passing = kwargs.get('ext_rank_passing', None)
        store_adjacent_similarities = kwargs.get('store_adjacent_similarities', False)

        # enable attention hidden collect (must live on Qwen3Model — that is what runs model_forward_intern3_5_vl)
        self._enable_attn_hidden_collect = enable_attn_hidden_collect
        lm = self.model.language_model
        if enable_attn_hidden_collect:
            setattr(lm, "_enable_attn_hidden_collect", enable_attn_hidden_collect)
            setattr(lm, "collected_attn_hidden", defaultdict(list))
            setattr(lm, "collected_question_hidden", defaultdict(list))
            setattr(lm, "retrieval_scores", None)

        # enable rank passing 
        # ext_rank_passing: int - Number of previous layers to use (1, 2, 3, or -1 for all)
        self._enable_rank_passing = ext_rank_passing is not None and ext_rank_passing != 0
        if self._enable_rank_passing:
            setattr(lm, "__enable_rank_passing", True)
            setattr(lm, "__num_prev_ranks_to_use", ext_rank_passing)

        # self._store_adjacent_similarities = store_adjacent_similarities
        self._store_adjacent_similarities = (self.prefill_method.startswith("swa_event_chunked"))
        if self._store_adjacent_similarities:
            self._init_similarity_tracking()

        # log all configurations 
        logger.info(f"InternVL-3.5-VL Config - max_frames: {self.max_frames} - sample_fps: {sample_fps} - target_frames: {target_frames}")
        logger.info(f"Enable Attention Hidden Collect: {enable_attn_hidden_collect}")
        logger.info(f"Enable Cross-Layer Rank Passing: {self._enable_rank_passing} (num_prev_ranks: {ext_rank_passing if self._enable_rank_passing else 'N/A'})")
        
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, encode_chunk_size, chunk_size) 

    def clear_cache(self):    
        super().clear_cache()
        # Reset streaming video cache initialization flag
        self._cache_initialized_for_video = False
        if self._store_adjacent_similarities:
            self._reset_similarity_tracking()
            self.event_boundaries = []

    def get_prompt(self, query, mc=False):
        prompt = f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def get_tokens_per_frame(self, video_slice):
        return self.adaptive_n_frame_tokens

    def process_video(self, video_np): 
        """ process numpy video and return procesor output """ 
        logger.info(f"Processing video: {video_np.shape}")
        # text = [f"{self.processor.start_image_token_id}{self.processor.video_token}{self.processor.end_image_token_id}"]
        text = [(self.processor.image_token + "\n") * video_np.shape[0]]
        out = self.processor(images=video_np, text=text, return_tensors="pt")

        print("1. PIXEL_VALUES: ", out['pixel_values'].shape, out['pixel_values'].mean(), out['pixel_values'].std())
        return out 

    def resample_video(self, video, target_frames, num_frames): 
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
        return video, num_frames, new_fps 

    def _encode_video_chunk(self, processed_dict):
        """ 
        _encode_video_chunk - processes a video chunk and updates the KV-cache
        :param processed_dict: dict - contains pixel_values_videos 
        """ 

        # get the input ids and input embeddings for current chunk 
        input_ids = processed_dict['input_ids'].to(self.device)
        pixel_values_videos = processed_dict['pixel_values'].to(self.device, self.dtype)

        inputs_embeds = self.get_input_embeddings()(input_ids)

        # block_size for optional attn hidden pooling lives on Qwen3Model (see patch_intern3_5_vl.model_forward_intern3_5_vl)
        setattr(self.model.language_model, "_block_size", self.adaptive_n_frame_tokens)

        # get video features 
        video_features = self.model.get_image_features(pixel_values_videos)
        video_embeds = video_features.pooler_output # (n_frames, 256, 4096)
        batch_size = input_ids.shape[0]

        if self.processor.image_processor.max_patches > 1:
            # Rearrange tiled features into a single spatial grid per frame and
            # drop the downsampled overview tile (last 16 columns).
            n_segments = self.processor.image_processor.max_patches
            h = w = int(math.sqrt(video_embeds.shape[1]))
            video_embeds = rearrange(
                video_embeds,
                "(n_frames n_segments) (h w) d -> n_frames h (n_segments w) d",
                h=h, w=w, n_segments=n_segments,
            )
            video_embeds = video_embeds[:, :, :-w, :]  # drop overview tile (last w columns)
            video_embeds = rearrange(video_embeds, "n_frames h w d -> n_frames (h w) d")

            # Update grid_hw on every KV-cache layer to match the new spatial layout.
            # After tile rearrangement: h rows × (n_segments-1)*w cols = h × 32 for max_patches=3.
            grid_h, grid_w = h, (n_segments - 1) * w
            if self.kv_cache is not None:
                for layer_kv in self.kv_cache:
                    layer_kv.grid_hw = (grid_h, grid_w)

            # token count per frame has changed — bypass get_placeholder_mask (which
            # uses input_ids built for n_segments*256 slots) and build inputs_embeds
            # directly from the rearranged visual features.
            inputs_embeds = video_embeds.reshape(batch_size, -1, video_embeds.shape[-1])
        else:
            # Only visual patch embeddings reach the LLM — BOI/EOI excluded so the
            # KV cache only ever contains spatial visual tokens (256 per frame).
            inputs_embeds = video_embeds.reshape(batch_size, -1, video_embeds.shape[-1])

        # Position ids are sequential over the visual-only sequence.
        position_ids = torch.arange(inputs_embeds.shape[1], device=input_ids.device)

        # feed visual features to the the LLM 
        output = self.language_model(input_ids=None, inputs_embeds=inputs_embeds,
            past_key_values=self.kv_cache, position_ids=position_ids, use_cache=True, return_dict=True, 
            visual_pos_masks=None,
        )

        self.kv_cache = output.past_key_values 

    @torch.inference_mode() 
    def encode_video(self, video, target_frames=None, streaming=False): 
        
        # encode chunk by chunk 
        num_frames = video.shape[0]
        encode_chunk_size = self.encode_chunk_size 
        target_frames = self.target_frames 

        video, num_frames, new_fps = self.resample_video(video, target_frames, num_frames)

        num_chunks = num_frames // encode_chunk_size
        for chunk_idx in tqdm(range(num_chunks), desc="Encoding video..."):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            logger.info(f"Encoding chunk {chunk_idx} of {num_chunks} - {chunk_video.shape}")
            processed_dict = self.process_video(chunk_video.cpu().numpy())
            self._encode_video_chunk(processed_dict)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            processed_dict = self.process_video(remaining_video.cpu().numpy())
            self._encode_video_chunk(processed_dict)
        
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
    def question_answering(self, input_text, max_new_tokens=10, retrieved_indices=None, use_cache=True,
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

        # # NOTE: Only input the question to perform retrieval.
        # tokenizer = self.processor.tokenizer
        # # Ensure we can pad to a fixed length
        # if tokenizer.pad_token_id is None:
        #     tokenizer.pad_token = tokenizer.eos_token
        # encoded_q = tokenizer(
        #     input_text["question"],
        #     padding="max_length",
        #     truncation=True,
        #     max_length=64,
        #     return_tensors="pt",
        # )
        # input_ids = encoded_q["input_ids"].to(device)
        # attention_mask = encoded_q["attention_mask"].to(device)

        # if use_cache:
        #     for layer_kv in self.kv_cache:  # activate retrieval mode
        #         layer_kv.set_retrieval()

        # # Internal retrieval
        # if retrieved_indices is None:  
        #     lm_kwargs = dict(ext_kwargs)
        #     lm_kwargs["attention_mask"] = attention_mask
        #     if use_cache:
        #         out = self.language_model(
        #             input_ids=input_ids,
        #             use_cache=True,
        #             past_key_values=self.kv_cache,
        #             **lm_kwargs,
        #         )
        #     else:
        #         out = self.language_model(input_ids=input_ids, use_cache=False, **lm_kwargs)
        #     past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        
        # External retrieval
        else:      
            # get current tokens per frame 
            current_n_frame_tokens = self.adaptive_n_frame_tokens

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
            lm_kwargs["attention_mask"] = attention_mask
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

def load_model(model_path="model_zoo/InternVL3_5-8B-HF", 
    n_init=None, n_local=8000, topk=32, 
    encode_chunk_size=64,
    chunk_size=1, 
    sample_fps=0.5, target_frames=None, 
    prefill_method='swa', 
    retrieval_method="rekv", head_idx=-1, 
    enable_attn_hidden_collect=False,
    ext_rank_passing=None):


    n_frame_tokens = 256  # 16×16 spatial patches per frame (no BOI/EOI in KV cache)

    # n_local must be a multiple of block_size so the compression rearrange
    # (b nh (m h w) d) always sees complete frames.
    n_local = (n_local // n_frame_tokens) * n_frame_tokens

    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True,
    )
    tokenizer = processor.tokenizer

    init_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    init_prompt_ids = tokenizer(init_prompt).input_ids

    inf_llm_config = {
        'n_init': len(init_prompt_ids) if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'encode_chunk_size': encode_chunk_size,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
        'prefill_method': prefill_method,
        'retrieval_method': retrieval_method,
        'head_idx': head_idx,
    }

    model = Intern3_5_VL_ReKV.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
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
    # patch_hf detects InternVLForConditionalGeneration and patches model.language_model.model
    model = patch_hf(model, **inf_llm_config)
    model.language_model = model.model.language_model

    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()
    return model, processor
