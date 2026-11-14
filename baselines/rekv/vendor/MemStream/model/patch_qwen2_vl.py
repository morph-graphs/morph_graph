import torch
import torch.nn.functional as F
from transformers import Cache
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast
from einops import rearrange
from ezcolorlog import root_logger as logger
import os

if not os.environ.get('ENABLE_LOGGING'):
    logger.disabled = True


def qwen2_vl_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Cache | None = None,
    output_attentions: bool | None = False,
    use_cache: bool | None = False,
    **kwargs,
):
    """Qwen2/2.5-VL decoder layer that keeps the pre-transformers-5 calling convention.

    The patched ReKV attention returns `(hidden_states, attn_hidden, present_key_value)`, which is
    what the transformers <= 4.5x decoder layer expected. The transformers 5 stock layer unpacks
    only two values (and names the cache `past_key_values`), so it has to be replaced for the ReKV
    attention to be callable at all.
    """
    device = hidden_states.device
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, attn_hidden, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = residual + hidden_states.to(device)

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states.to(device)

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_hidden,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


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


def qwen_vl_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position = None,
    position_embeddings = None,
    **kwargs,
):
    """
    Patched forward for Qwen2_5_VLDecoderLayer to return attn_hidden.
    Mirrors HF semantics while appending the attention output residual clone.
    """
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

    attn_hidden = hidden_states.detach().clone()

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    outputs += (attn_hidden,)

    return outputs

def model_forward_qwen_vl(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    cache_position = None,
    *args,
    **kwargs
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
        if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
            inputs_embeds = inputs_embeds * self.config.scale_emb

    if use_cache:
        pkv = tuple()
    else:
        pkv = None

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

    # collect attention hidden states if enabled
    block_size = getattr(self, "_block_size", None)
    compute_retrieval_score = kwargs.get('compute_retrieval_score', False)

    # handle cross-layer rank passing 
    enable_rank_passing = getattr(self, "__enable_rank_passing", False)
    num_prev_ranks_to_use = getattr(self, "__num_prev_ranks_to_use", 1)
    prev_layer_rankings_list = []  # Collect rankings from previous layers

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[i] if past_key_values is not None and len(past_key_values) > i else None

        # provide external kwargs to attention without changing decoder layer signature
        # Include previous layer rankings if rank passing is enabled
        if enable_rank_passing and len(prev_layer_rankings_list) > 0:
            ext_outs_with_rankings = ext_outs.copy()
            if num_prev_ranks_to_use == -1:
                rankings_to_pass = prev_layer_rankings_list
            else:
                rankings_to_pass = prev_layer_rankings_list[-num_prev_ranks_to_use:]
            ext_outs_with_rankings["ext_prev_layer_full_rankings"] = rankings_to_pass
            ext_outs_with_rankings["num_prev_ranks"] = len(rankings_to_pass)
            setattr(decoder_layer.self_attn, "_rekv_ext_kwargs", ext_outs_with_rankings)
        else:
            setattr(decoder_layer.self_attn, "_rekv_ext_kwargs", ext_outs)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,  # attention_mask is None - flash-attention handles it internally
            position_ids=self.position_bias,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            _cache = layer_outputs[2 if output_attentions else 1]
            pkv = pkv + (_cache,)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        
        # Extract rankings from current layer to pass to next layer
        if enable_rank_passing and use_cache:
            if hasattr(past_key_value, 'sorted_block_indices'):
                # Append current layer's rankings to the list
                current_rankings = past_key_value.sorted_block_indices[0].copy()  # batch_size=1
                prev_layer_rankings_list.append(current_rankings)
                logger.info(f"Layer {i} - Extracted {len(current_rankings)} sorted indices, total collected: {len(prev_layer_rankings_list)}")

        attn_hidden = layer_outputs[-1] 

        if hasattr(self, "collected_attn_hidden") and hasattr(self, "collected_question_hidden"):
            if block_size is not None and not compute_retrieval_score: 
                pooled_attn_hidden = rearrange(attn_hidden, "b (m n) d -> b m n d", n=block_size).mean(dim=2) 
                self.collected_attn_hidden[i].append(pooled_attn_hidden) 
            elif compute_retrieval_score:
                self.collected_question_hidden[i].append(attn_hidden)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
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
