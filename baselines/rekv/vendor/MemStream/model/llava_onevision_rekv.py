import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV


class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk,
                 encode_chunk_size, chunk_size, 
                 prefill_method, sample_fps, target_frames, 
                 **kwargs):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, encode_chunk_size, chunk_size)


        ext_rank_passing = kwargs.get('ext_rank_passing', None)

        # enable rank passing 
        # ext_rank_passing: int - Number of previous layers to use (1, 2, 3, or -1 for all)
        self._enable_rank_passing = ext_rank_passing is not None and ext_rank_passing != 0
        if self._enable_rank_passing:
            setattr(self.model, "__enable_rank_passing", True)
            setattr(self.model, "__num_prev_ranks_to_use", ext_rank_passing)

        logger.info(f"Enable Cross-Layer Rank Passing: {self._enable_rank_passing} (num_prev_ranks: {ext_rank_passing if self._enable_rank_passing else 'N/A'})")

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]

        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*196, D)
        return video_features

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None, **ext_kwargs):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        for layer_kv in self.kv_cache:  # activate retrieval mode
            layer_kv.set_retrieval()

        # Internal retrieval
        if retrieved_indices is None: 
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache, **ext_kwargs)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        # External retrieval
        else: 
            # loop through all layers of kv cache 
            for layer_kv in self.kv_cache:
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

            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache) 
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        for layer_kv in self.kv_cache:  # reset to default
            layer_kv.reset_retrieval()

        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = out.logits
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
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


def load_model(model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
               n_init=None, n_local=None, topk=64,
               encode_chunk_size=64,
               chunk_size=1,
               sample_fps=0.5, target_frames=None,
               prefill_method='swa',
               retrieval_method='rekv',
               head_idx=-1, 
               enable_attn_hidden_collect=False,
               ext_rank_passing=None): 
    device = 'cuda'
    n_frame_tokens = 196

    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    
    init_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    
    # Round n_local to nearest multiple of n_frame_tokens
    if n_local is not None:
        n_local = (n_local // n_frame_tokens) * n_frame_tokens
        logger.info(f'Rounded n_local to multiple of n_frame_tokens: {n_local}')
    
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
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
    model = LlavaOneVision_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        encode_chunk_size=encode_chunk_size,
        chunk_size=chunk_size,
        prefill_method=prefill_method,
        sample_fps=sample_fps,
        target_frames=target_frames,
        ext_rank_passing=ext_rank_passing,
    )
    model.language_model = patch_hf(model.language_model, **inf_llm_config)
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor
