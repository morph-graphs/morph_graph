import torch
import torch.nn
import torch.nn.functional as F
from transformers import Cache
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast
from einops import rearrange
from ezcolorlog import root_logger as logger
import os

if not os.environ.get('ENABLE_LOGGING'):
    logger.disabled = True

# Qwen2DecoderLayer or Qwen3DecoderLayer Forward (L309) 
def qwen_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    device = hidden_states.device

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, attn_hidden, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states.to(device)

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states.to(device)

    # get cache outputs 
    outputs = (hidden_states,)
    if kwargs.get("output_attentions", False):
        outputs += (attn_hidden,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        assert not output_attentions
        ret = forward(
            self, hidden_states, hidden_states,
            position_ids, use_cache, past_key_value,
            self.q_proj, self.k_proj, self.v_proj, self.o_proj, 
            self.head_dim, self.config.num_attention_heads, self.config.num_key_value_heads, 
            q_norm=getattr(self, "q_norm", None),
            k_norm=getattr(self, "k_norm", None),
            visual_pos_masks=kwargs.get('visual_pos_masks', None),
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None

        # (hidden_states, attn_hidden, present_key_value)
        return o, None, pkv

    return hf_forward

def cos_sim(a, b, eps: float = 1e-8, reduce: str | None = None):
    """
    a: [..., Q, D]
    b: [..., T, D]
    returns: [..., Q, T] (or reduced)
    reduce: 'mean_q' -> average over Q, 'mean_t' -> average over T, None -> no reduction
    """
    a = F.normalize(a.float(), dim=-1, eps=eps)
    b = F.normalize(b.float(), dim=-1, eps=eps)
    sim = a @ b.transpose(-1, -2)
    if reduce == 'mean_q': return sim.mean(dim=-2)
    if reduce == 'mean_t': return sim.mean(dim=-1)
    return sim


# Qwen3Model inside Qwen3ForCausalLM (InternVL3.5's language_model).
# Standard 1D RoPE — no deepstack, no M-RoPE block_sizes.
def model_forward_intern3_5_vl(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs
):
    output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
    output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = kwargs.get("return_dict", self.config.use_return_dict)

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    pkv = tuple() if use_cache else None

    # extract external model inputs
    ext_outs = {
        "ext_question_features": kwargs.get('ext_question_features', None),
        "ext_video_features": kwargs.get('ext_video_features', None),
        "ext_retrieved_block_indices": kwargs.get('ext_retrieved_block_indices', None),
        "ext_similarity": kwargs.get('ext_similarity', None),
        "ext_merge_method": kwargs.get('ext_merge_method', None),
        "ext_weights": kwargs.get('ext_weights', None),
        "ext_rrf_k": kwargs.get('ext_rrf_k', None),
    }

    hidden_states = inputs_embeds

    block_size = getattr(self, "_block_size", None)
    compute_retrieval_score = kwargs.get('compute_retrieval_score', False)

    # handle cross-layer rank passing
    enable_rank_passing = getattr(self, "__enable_rank_passing", False)
    num_prev_ranks_to_use = getattr(self, "__num_prev_ranks_to_use", 1)
    prev_layer_rankings_list = []

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[layer_idx] if past_key_values is not None and len(past_key_values) > layer_idx else None

        if enable_rank_passing and len(prev_layer_rankings_list) > 0:
            ext_outs_with_rankings = ext_outs.copy()
            rankings_to_pass = prev_layer_rankings_list if num_prev_ranks_to_use == -1 else prev_layer_rankings_list[-num_prev_ranks_to_use:]
            ext_outs_with_rankings["ext_prev_layer_full_rankings"] = rankings_to_pass
            ext_outs_with_rankings["num_prev_ranks"] = len(rankings_to_pass)
            setattr(decoder_layer.self_attn, "_rekv_ext_kwargs", ext_outs_with_rankings)
        else:
            setattr(decoder_layer.self_attn, "_rekv_ext_kwargs", ext_outs)

        # indices where visual tokens are located 
        visual_pos_masks = kwargs.get('visual_pos_masks', None)

        # Standard Qwen3DecoderLayer — no position_embeddings (1D RoPE).
        # self.position_bias (RotaryEmbeddingESM) is passed as position_ids and
        # intercepted by the patched huggingface_forward in patch.py.
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=self.position_bias,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=None,
            visual_pos_masks=visual_pos_masks,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            _cache = layer_outputs[2 if output_attentions else 1]
            pkv = pkv + (_cache,)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # extract rankings from current layer to pass to next layer
        if enable_rank_passing and use_cache:
            if hasattr(past_key_value, 'sorted_block_indices'):
                current_rankings = past_key_value.sorted_block_indices[0].copy()
                prev_layer_rankings_list.append(current_rankings)
                logger.info(f"Layer {layer_idx} - Extracted {len(current_rankings)} sorted indices, total collected: {len(prev_layer_rankings_list)}")

        attn_hidden = layer_outputs[-1]

        if hasattr(self, "collected_attn_hidden") and hasattr(self, "collected_question_hidden"):
            if block_size is not None and not compute_retrieval_score:
                pooled_attn_hidden = rearrange(attn_hidden, "b (m n) d -> b m n d", n=block_size).mean(dim=2)
                self.collected_attn_hidden[layer_idx].append(pooled_attn_hidden)
            elif compute_retrieval_score:
                self.collected_question_hidden[layer_idx].append(attn_hidden)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if hasattr(self, "collected_attn_hidden") and hasattr(self, "collected_question_hidden"):
        if compute_retrieval_score:
            collected_attn_hidden = {k: torch.cat(v, dim=1) for k, v in self.collected_attn_hidden.items()}
            collected_question_hidden = {k: torch.cat(v, dim=1) for k, v in self.collected_question_hidden.items()}
            retrieval_scores = {k: cos_sim(collected_question_hidden[k], v, reduce='mean_q').cpu().numpy().tolist() for k, v in collected_attn_hidden.items()}
            self.retrieval_scores = retrieval_scores

    if not return_dict:
        return tuple(v for v in [hidden_states, pkv, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=pkv,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
