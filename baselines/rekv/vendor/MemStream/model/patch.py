import torch
import torch.nn.functional as F 
import transformers 
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding 
from collections import defaultdict
import os

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen2VLForConditionalGeneration
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import InternVLForConditionalGeneration
except ImportError:
    # These imports may not be available in older transformers versions
    pass

from model.attention import RotaryEmbeddingESM, rekv_attention_forward
from model.attention import RotaryEmbeddingQwenVL, rekv_attention_forward_qwenvl
from model.attention import RotaryEmbeddingQwen3VL

from model.patch_qwen2_vl import model_forward_qwen_vl
from model.patch_qwen3_vl import model_forward_qwen3_vl
from model.patch_qwen3_vl import huggingface_forward as huggingface_forward_qwen3_vl

from model.patch_internvl3 import model_forward_intern3_5_vl
from model.patch_internvl3 import huggingface_forward as huggingface_forward_intern3_5_vl

from ezcolorlog import root_logger as logger
from einops import rearrange

if not os.environ.get('ENABLE_LOGGING'):
    logger.disabled = True

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
            self.head_dim, self.num_heads, self.num_key_value_heads, 
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None

        return o, None, pkv

    return hf_forward

def patch_hf(
    model,
    attn_kwargs: dict = {},
    base = None, 
    distance_scale = None,
    **kwargs
):

    is_qwen_vl = False
    is_qwen3_vl = False
    is_internvl3_5 = False

    attn_kwargs.update(kwargs)
    
    # Import required model classes - note: model detection logic could be refactored
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Qwen2Model
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel, BaseModelOutputWithPast

    def model_forward(
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
        *args,
        **kwargs
    ):
        # Get compute_retrieval_score from either the model attribute (set by wrapper) or kwargs
        compute_retrieval_score = getattr(self, '_compute_retrieval_score', kwargs.pop('compute_retrieval_score', False))
        
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
        ext_kwargs = getattr(self, "ext_kwargs", {})
        ext_outs = { 
            "ext_question_features": ext_kwargs.get('ext_question_features', None),
            "ext_video_features": ext_kwargs.get('ext_video_features', None),
            "ext_retrieved_block_indices": ext_kwargs.get('ext_retrieved_block_indices', None),
            "ext_similarity": ext_kwargs.get('ext_similarity', None),
            "ext_merge_method": ext_kwargs.get('ext_merge_method', None),
            "ext_weights": ext_kwargs.get('ext_weights', None),
            "ext_rrf_k": ext_kwargs.get('ext_rrf_k', None),
        }

        hidden_states = inputs_embeds

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
                attention_mask=attention_mask,
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

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, pkv, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Wrapper for CausalLM forward to handle custom kwargs
    def causal_lm_forward_wrapper(original_forward, inner_model):
        def wrapper(*args, **kwargs):
            # Extract custom kwargs that the base CausalLM doesn't accept
            # but pass them to the inner model.model.forward()
            compute_retrieval_score = kwargs.pop('compute_retrieval_score', None)
            
            # Store in the inner model so model_forward() can access it
            if compute_retrieval_score is not None:
                inner_model._compute_retrieval_score = compute_retrieval_score
            
            result = original_forward(*args, **kwargs)
            
            # Clean up
            if hasattr(inner_model, '_compute_retrieval_score'):
                delattr(inner_model, '_compute_retrieval_score')
            
            return result
        return wrapper

    def qwen_vl_text_model(m):
        """Return the decoder-layer-carrying text model for a Qwen2/2.5-VL checkpoint.

        transformers <= 4.5x exposed the decoder layers directly on `model.model`; transformers 5
        moved them under `model.model.language_model` (the layout Qwen3-VL already used). Support
        both so the Qwen2/2.5-VL path keeps working across versions.
        """
        inner = m.model
        return inner.language_model if hasattr(inner, "language_model") else inner

    if isinstance(model, LlamaForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif isinstance(model, MistralForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif isinstance(model, Qwen2ForCausalLM) or isinstance(model, Qwen2Model):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif model.__class__.__name__ == "MiniCPMForCausalLM":
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    else:
        try:
            # Check for Qwen2.5 VL first
            if isinstance(model, Qwen2_5_VLForConditionalGeneration):
                text_model = qwen_vl_text_model(model)
                Attention = text_model.layers[0].self_attn.__class__
                Model = text_model.__class__
                logger.info(f"Attention class: {Attention}")
                logger.info(f"Model class: {Model}")
                logger.info(f"Rotary Embedding Class: {text_model.rotary_emb.__class__}")
                logger.info(f"RoPE Type: {text_model.rotary_emb.rope_type}")
                is_qwen_vl = True
            # Check for Qwen2 VL
            elif isinstance(model, Qwen2VLForConditionalGeneration):
                text_model = qwen_vl_text_model(model)
                Attention = text_model.layers[0].self_attn.__class__
                Model = text_model.__class__
                logger.info(f"Attention class: {Attention}")
                logger.info(f"Model class: {Model}")
                logger.info(f"Rotary Embedding Class: {text_model.rotary_emb.__class__}")
                if hasattr(text_model.rotary_emb, 'rope_type'):
                    logger.info(f"RoPE Type: {text_model.rotary_emb.rope_type}")
                is_qwen_vl = True  # Use same patching logic as Qwen2.5 VL

            elif isinstance(model, Qwen3VLForConditionalGeneration):
                Attention = model.model.language_model.layers[0].self_attn.__class__
                Model = model.model.__class__
                logger.info(f"Attention class: {Attention}")
                logger.info(f"Model class: {Model}")
                logger.info(f"Rotary Embedding Class: {model.model.language_model.rotary_emb.__class__}")
                logger.info(f"RoPE Type: {model.model.language_model.rotary_emb.rope_type}")
                is_qwen_vl = True
                is_qwen3_vl = True

            elif isinstance(model, InternVLForConditionalGeneration):
                # model.model is the vision encoder; LLM is model.language_model (Qwen3ForCausalLM)
                Attention = model.model.language_model.layers[0].self_attn.__class__
                Model = model.model.language_model.__class__
                logger.info(f"InternVL3.5: Attention class: {Attention}")
                logger.info(f"InternVL3.5: Model class: {Model}")
                logger.info(f"InternVL3.5: RoPE class: {model.model.language_model.rotary_emb.__class__}")
                is_internvl3_5 = True

        except Exception as e:
            logger.error(f"Error patching model: {e}")
            raise ValueError(f"Error patching model: {e}")
    
    # Transformers >= 4.45.0: Get RoPE from either model.layers or language_model
    if hasattr(model.model, "layers"):
        hf_rope = model.model.layers[0].self_attn.rotary_emb
    elif hasattr(model.model, "language_model"):
        # quen3-vl or internvl3-5
        hf_rope = model.model.language_model.rotary_emb 

    # MEMSTREAM_COMPRESS_LAYERS=<i>,<j>,...: every named layer must exist -- an out-of-range
    # index would otherwise never match and silently run that layer dense.
    from model.attention.kv_cache_manager import _COMPRESS_LAYER_SET
    if _COMPRESS_LAYER_SET is not None:
        _layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
        _bad = [i for i in _COMPRESS_LAYER_SET if i >= len(_layers)]
        if _bad:
            raise ValueError(f"MEMSTREAM_COMPRESS_LAYERS: layer(s) {_bad} out of range "
                             f"[0, {len(_layers)}) for {model.__class__.__name__}")

    # Extract RoPE parameters based on transformers version and model type
    if hasattr(hf_rope.config, 'rope_parameters'):
        # Older transformers versions with rope_parameters in config
        base = hf_rope.config.rope_parameters['rope_theta']
        distance_scale = 1.0
        partial_rotary_factor = 1.0
        # Prefer the explicit head_dim field: models like Qwen3-VL-4B decouple it from
        # hidden_size // num_attention_heads (2560//32 = 80, but head_dim is 128).
        _hd = getattr(hf_rope.config, 'head_dim', None) or (hf_rope.config.hidden_size // hf_rope.config.num_attention_heads)
        dim = int(_hd * partial_rotary_factor)
    elif isinstance(hf_rope, Qwen2RotaryEmbedding):
        # Transformers 4.45.0 and above with Qwen2
        base = hf_rope.base
        distance_scale = 1.0  # equivalent to attention_factor in Qwen2_5_VL/Qwen2
        dim = hf_rope.dim
    else:
        # Default RoPE configuration
        # Reference: https://github.com/huggingface/transformers/blob/4da03d7f57707bd6800518ce4f56e834f153377b/src/transformers/modeling_rope_utils.py#L92
        base = hf_rope.config.rope_theta
        distance_scale = distance_scale if distance_scale is not None else 1.0
        partial_rotary_factor = getattr(hf_rope.config, "partial_rotary_factor", 1.0) or 1.0
        _hd = getattr(hf_rope.config, 'head_dim', None) or (hf_rope.config.hidden_size // hf_rope.config.num_attention_heads)
        dim = int(_hd * partial_rotary_factor)

    # Patch model with appropriate RoPE and attention implementations
    if is_qwen3_vl:
        logger.info("Patching Qwen3_VL huggingface_forward")
        _rs = getattr(hf_rope.config, 'rope_scaling', None) or {}
        _trope = list(_rs.get('mrope_section', [24, 20, 20]))
        rope = RotaryEmbeddingQwen3VL(dim, base, distance_scale, trope_section=_trope)
        forward = huggingface_forward_qwen3_vl(rekv_attention_forward_qwenvl(**attn_kwargs))
    elif is_qwen_vl:
        logger.info("Patching RotaryEmbeddingQwen (Qwen2_VL/Qwen2.5_VL)")
        rope = RotaryEmbeddingQwenVL(dim, base, distance_scale)
        forward = huggingface_forward(rekv_attention_forward_qwenvl(**attn_kwargs))
    elif is_internvl3_5:
        logger.info("Patching InternVL3.5 (Qwen3ForCausalLM backbone, 1D RoPE)")
        rope = RotaryEmbeddingESM(dim, base, distance_scale)
        forward = huggingface_forward_intern3_5_vl(rekv_attention_forward(**attn_kwargs))
    else:
        logger.info("Patching RotaryEmbeddingESM")
        rope = RotaryEmbeddingESM(dim, base, distance_scale)
        forward = huggingface_forward(rekv_attention_forward(**attn_kwargs))

    # Set position_bias on the appropriate model component
    if is_qwen3_vl:
        model.model.language_model.position_bias = rope
    elif is_internvl3_5:
        model.model.language_model.position_bias = rope
    elif is_qwen_vl:
        # transformers 5 nests the text model one level deeper (see qwen_vl_text_model)
        qwen_vl_text_model(model).position_bias = rope
    else:
        model.model.position_bias = rope

    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)

    model.apply(set_forward)

    if is_qwen3_vl:
        logger.info("Patching Qwen3_VL model_forward")
        model.model.language_model._old_forward = model.model.language_model.forward
        model.model.language_model.forward = model_forward_qwen3_vl.__get__(model.model.language_model, Model)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
        from model.patch_qwen3_vl import qwen3_vl_decoder_layer_forward

        def _set_decoder_layer_forward(m):
            if isinstance(m, Qwen3VLTextDecoderLayer):
                m._old_forward = m.forward
                m.forward = qwen3_vl_decoder_layer_forward.__get__(m, Qwen3VLTextDecoderLayer)

        model.apply(_set_decoder_layer_forward)

    elif is_qwen_vl:
        logger.info("Patching Qwen2_VL/Qwen2.5_VL model_forward")
        text_model = qwen_vl_text_model(model)
        text_model._old_forward = text_model.forward
        text_model.forward = model_forward_qwen_vl.__get__(text_model, Model)

        # transformers 5 changed the stock decoder layer to unpack only 2 values from self_attn;
        # the ReKV attention returns 3, so restore the older convention with our own layer forward.
        from model.patch_qwen2_vl import qwen2_vl_decoder_layer_forward

        DecoderLayer = text_model.layers[0].__class__

        def _set_qwen_decoder_layer_forward(m):
            if isinstance(m, DecoderLayer):
                m._old_forward = m.forward
                m.forward = qwen2_vl_decoder_layer_forward.__get__(m, DecoderLayer)

        model.apply(_set_qwen_decoder_layer_forward)

    elif is_internvl3_5:
        logger.info("Patching InternVL3.5 model_forward (Qwen3ForCausalLM backbone)")
        model.model.language_model._old_forward = model.model.language_model.forward
        model.model.language_model.forward = model_forward_intern3_5_vl.__get__(model.model.language_model, Model)

        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model 
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

        # InternVL3.5 has a Qwen3ForCausalLM backbone
        if isinstance(model.model.language_model, Qwen3Model):
            from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
            from model.patch_intern3_5_vl import qwen_decoder_layer_forward

            def _set_decoder_layer_forward(m):
                if isinstance(m, Qwen3DecoderLayer):
                    m._old_forward = m.forward
                    m.forward = qwen_decoder_layer_forward.__get__(m, Qwen3DecoderLayer)

            model.apply(_set_decoder_layer_forward)
        
        # InternVL3 has a Qwen2ForCausalLM backbone
        elif isinstance(model.model.language_model, Qwen2Model):
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            from model.patch_internvl3 import qwen_decoder_layer_forward

            def _set_decoder_layer_forward(m):
                if isinstance(m, Qwen2DecoderLayer):
                    m._old_forward = m.forward
                    m.forward = qwen_decoder_layer_forward.__get__(m, Qwen2DecoderLayer)
            model.apply(_set_decoder_layer_forward)

        else:
            raise ValueError(f"Unsupported model: {model.model.language_model.__class__}")


    else:
        logger.info("Patching model_forward")
        model.model._old_forward = model.model.forward
        model.model.forward = model_forward.__get__(model.model, Model)

    # Patch the top-level CausalLM forward to handle custom kwargs
    if isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM)):
        logger.info("Patching top-level CausalLM forward to handle custom kwargs")
        model._old_forward = model.forward
        model.forward = causal_lm_forward_wrapper(model._old_forward, model.model)

    return model