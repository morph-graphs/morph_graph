import copy
import torch
from typing import Optional, List
import os

from ..kv_cache_manager import ContextManager
from ..dot_production_attention import get_multi_stage_dot_production_attention

from ezcolorlog import root_logger as logger

if not os.environ.get('ENABLE_LOGGING'):
    logger.disabled = True 

def rekv_attention_forward(
    n_local, n_init, topk, encode_chunk_size, chunk_size,
    block_size, max_cached_block,
    exc_block_size, fattn,
    async_global_stream=True,
    pin_memory=False,
    prefill_method="swa", 
    retrieval_method="rekv", 
    head_idx=-1,
    *args, **kwargs
):
    Attn, _ = get_multi_stage_dot_production_attention(fattn)
    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out, 
                    dim_head, num_heads, num_heads_kv,
                    curr_position_ids: Optional[torch.Tensor] = None, 
                    block_sizes: Optional[List] = None, 
                    q_norm=None, k_norm=None,
                    visual_pos_masks: Optional[torch.Tensor] = None,
    ):

        assert self.layer_idx is not None, "layer_idx must be set on the ContextManager"
        
        logger.info(f"\n-------------------- Start of rekv_attention_forward - Layer {self.layer_idx} --------------------------")
        
        ext_kwargs = getattr(self, "_rekv_ext_kwargs", None)

        """ 1. Project QKV """
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)

        # Apply RMSNorm for Q and K - for Qwen3-VL
        if q_norm is not None: 
            h_q = q_norm(h_q)
        if k_norm is not None:
            h_k = k_norm(h_k)

        if position_bias._cos_cached is not None and position_bias._cos_cached.device != h_q.device:
            position_bias = copy.deepcopy(position_bias)
            if position_bias.inv_freq.device != h_q.device:
                position_bias.inv_freq = position_bias.inv_freq.to(h_q.device)
            if position_bias._cos_cached is not None:
                position_bias._cos_cached = position_bias._cos_cached.to(h_q.device)
            if position_bias._sin_cached is not None:
                position_bias._sin_cached = position_bias._sin_cached.to(h_q.device)

        if past_key_value is None:
            past_key_value = ContextManager(
                self.layer_idx,
                position_bias,
                n_init, n_local,
                block_size, max_cached_block, topk, 
                encode_chunk_size, chunk_size, exc_block_size,
                fattn,
                async_global_stream,
                pin_memory,
                prefill_method, 
                retrieval_method,
                head_idx,
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        n_local_for_retr = n_local 
        update_global_buffer_size = False 
        is_context_manager = type(past_key_value) is ContextManager

        # NOTE: Question-answering, fall back to sliding-window attention (infinite_lm)
        if type(past_key_value) is not ContextManager or past_key_value.to_retrieve:
            retrieved_position_ids = None
            if is_context_manager:  # retrieval
                if past_key_value.retrieved_block_indices is None:  # retrieve based on global_q (question's query)
                    retrieved_kv = past_key_value.get_retrieved_kv(global_q, ext_kwargs, query_v=global_v)
                else:  # retrieve based on pre-computed retrieved_block_indices
                    retrieved_kv = past_key_value.get_retrieved_kv()
                    update_global_buffer_size = True 
                past_k, past_v = retrieved_kv[:2]
                retrieved_position_ids = retrieved_kv[2] if len(retrieved_kv) > 2 else None

                if past_key_value.retrieved_block_indices is not None:
                    logger.info(f"1.1 - PAST K: {past_k.shape} - PAST V: {past_v.shape}")
                else:
                    logger.info(f"1.2 - PAST K: {past_k.shape} - PAST V: {past_v.shape}")
                
                update_kv_cache = False  # We do not update KV cache with the input KV (h_k, h_v) because we only use it for retrieval
                n_local_for_retr = past_key_value.n_local # update n_local based on the value from ContextManager
                logger.info(f"n_local_for_retr: {n_local_for_retr}")
            else:  # sliding-window attention
                past_k = past_key_value[0]
                past_v = past_key_value[1]
                update_kv_cache = True
                logger.info(f"1.3 - PAST K: {past_k.shape} - PAST V: {past_v.shape}")

            """ 2. Update KV w/ past KV cache 
                h_q: query_text (1D rep) 
                h_k: cache init (1D rep) + retrieved video blocks (3D rep) + query_text (1D rep)
            """
            h_k = torch.cat([past_k, h_k], dim=-2)
            h_v = torch.cat([past_v, h_v], dim=-2)
            len_k += past_k.shape[2]

            # h_k_init = past_k[:, :, :14, :]
            # h_k_vid = past_k[:, :, 14:, :]
            # h_v_init = past_v[:, :, :14, :]
            # h_v_vid = past_v[:, :, 14:, :]
            # h_k = torch.cat([h_k_init, h_k, h_k_vid], dim=-2)
            # h_v = torch.cat([h_v_init, h_v, h_v_vid], dim=-2)
            # len_k += past_k.shape[2] 

            """ 3. Update KV cache """
            if update_kv_cache:
                if len_k <= n_local + n_init:
                    h_k_cache = h_k
                    h_v_cache = h_v
                else:
                    h_k_cache = torch.cat([h_k[:, :, :n_init, :], h_k[:, :, max(0, h_k.size(-2) - n_local_for_retr):, :]], dim=2)
                    h_v_cache = torch.cat([h_v[:, :, :n_init, :], h_v[:, :, max(0, h_k.size(-2) - n_local_for_retr):, :]], dim=2)
                current_key_value = (h_k_cache, h_v_cache)
            else:
                current_key_value = (past_k, past_v)

            """ 4. Get local QKV and apply RoPE to local QK """
            h_q_, h_k_, h_v_ = h_q, h_k, h_v
            if len_q + n_local < h_k_.size(-2):
                h_k_ = h_k_[:, :, h_k_.size(-2) - len_q - n_local_for_retr:, :]
                h_v_ = h_v_[:, :, h_v_.size(-2) - len_q - n_local_for_retr:, :]

            # set the global buffer size based on retrieved size 
            logger.info(f"2 - PAST K: {past_k.shape} - PAST V: {past_v.shape} - LEN Q: {len_q}")
            position_bias_kwargs = {}
            
            # set global_buffer_size based on retrieved_position_ids or depending on retrieval vs generation 
            if retrieved_position_ids is not None:
                position_bias_kwargs["retrieved_position_ids"] = retrieved_position_ids
                global_buffer_size = None 
            else: 
                global_buffer_size = past_k.size(-2) if is_context_manager else None
            
            local_h_q, local_h_k = position_bias(h_q_, h_k_, global_buffer_size=global_buffer_size, right=None,
                update_global_buffer_size=update_global_buffer_size, **position_bias_kwargs)
            local_h_v = h_v_

            """ 5. Get init QKV and apply RoPE to init Q (Infinite-LM assigns the same position_ids to initial tokens) """
            if len_k > n_local:
                logger.info(f"2.1 - INIT Q: {h_q.shape} - INIT K: {h_k.shape} - INIT V: {h_v.shape}")
                init_h_q = position_bias.apply_rotary_pos_emb_one_angle(
                    h_q, n_local_for_retr, L=1 
                )
                init_h_k = h_k
                init_h_v = h_v
                init_h_k = init_h_k[:, :, :n_init, :].contiguous()
                init_h_v = init_h_v[:, :, :n_init, :].contiguous()

            else:
                init_h_q = h_q
                init_h_k = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_k.device,
                    dtype=h_k.dtype
                )
                init_h_v = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_v.device,
                    dtype=h_v.dtype
                )

            """ 6. Sliding Window Attention """
            attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
            attn.append(local_h_q, local_h_k, local_h_v, sliding_window=n_local_for_retr)
            attn.append(init_h_q, init_h_k, init_h_v, end=True, sliding_window=(len_k - len_q, n_local_for_retr), complement_sliding_window=True)
            score, _ = attn.get_result()

            score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3) # (batch, len_q, num_heads, dim_head)
            score = score.reshape(batch_size, len_q, num_heads * dim_head) # (batch, len_q, num_heads * dim_head)
            score = attention_out(score)

            return score, current_key_value

        # NOTE: Encode video, managed by the KVCacheManager
        else:
            o = past_key_value.append(
                local_q, local_k, local_v,
                global_q, global_k, global_v,
                curr_position_ids, block_sizes,
                visual_pos_masks,
            )
            o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            o = o.reshape(batch_size, len_q, dim_head * num_heads)
            o = attention_out(o)

            return o, past_key_value

    return forward