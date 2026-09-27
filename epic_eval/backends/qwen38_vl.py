"""Qwen3.8 backend (local HF transformers; one model shared by all worker threads).

A *context* is one group's system turn + video, run through the processor once:
the video is decoded and patchified in open_context, and every question of the
group reuses those tensors, appending only its own text. Each question is still
prefilled from scratch (no KV reuse), so an answer does not depend on which
questions came before it. Generation is serialized by a lock, since all threads
share one model.

Model keys read from the config:
    path                 weights directory (e.g. ~/scratch/model_hub/Qwen3.8-27B)
    attn_implementation  default "flash_attention_2"
    mode                 "thinking" or "instruct": chat-template thinking switch + Qwen's sampling preset (MODES)
    max_new_tokens       output cap, thinking included (default 8192)
    seed                 reseeded before every question (default none)
    fps                  required; frames sampled per second of video
    max_frames           required; frame cap per clip (clips longer than max_frames / fps
                         get max_frames frames spread over their length)
    Both are set per launch: orchestrator --fps / --max_frames.
    processor_kwargs     other video settings for apply_chat_template (default: the model's own
                         video_preprocessor_config, i.e. ~12k video tokens per clip)
    generation_kwargs    extra model.generate kwargs, applied over the mode's preset
"""

from __future__ import annotations

import threading
import time

import torch

from ..common import parse_letter

VISION_END = "<|vision_end|>"
THINK_END = "</think>"

# Qwen's recommended sampling per mode. presence_penalty is not a transformers generate()
# argument; ask() applies it with PresencePenalty below.
MODES = {
    "thinking": {"enable_thinking": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                 "presence_penalty": 0.0, "repetition_penalty": 1.0},
    "instruct": {"enable_thinking": False, "temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0,
                 "presence_penalty": 1.5, "repetition_penalty": 1.0},
}


def _mode(mcfg: dict) -> dict:
    mode = mcfg.get("mode", "thinking")
    if mode not in MODES:
        raise ValueError(f"model.mode must be one of {sorted(MODES)}, got {mode!r}")
    return MODES[mode]


class PresencePenalty:
    """vLLM/OpenAI presence penalty: subtract a flat penalty from every token already generated.

    Only generated tokens count, not the prompt (which here holds the video).
    """

    def __init__(self, penalty: float, prompt_len: int):
        self.penalty, self.prompt_len = penalty, prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_len:]
        if generated.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, generated, True)
        return scores - self.penalty * seen.to(scores.dtype)


def make_client(mcfg: dict) -> dict:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(mcfg["path"])
    model = AutoModelForMultimodalLM.from_pretrained(
        mcfg["path"], dtype=torch.bfloat16, device_map="auto",
        attn_implementation=mcfg.get("attn_implementation", "flash_attention_2"),
    ).eval()
    print(f"  loaded {mcfg['path']} in {time.time() - t0:.0f}s (attn={model.config._attn_implementation})")
    return {"model": model, "processor": processor, "lock": threading.Lock()}


def _messages(system_instruction: str, video_path, prompt: str) -> list[dict]:
    content = [{"type": "video", "video": str(video_path)}] if video_path is not None else []
    return [{"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": content + [{"type": "text", "text": prompt}]}]


def _template_kwargs(mcfg: dict) -> dict:
    return {"enable_thinking": _mode(mcfg)["enable_thinking"]}


def _video_kwargs(mcfg: dict) -> dict:
    # The processor samples min(fps * duration, max_frames, frames in the file).
    if mcfg.get("fps") is None or mcfg.get("max_frames") is None:
        raise ValueError("model.fps and model.max_frames are required (orchestrator --fps / --max_frames)")
    return {"fps": float(mcfg["fps"]), "max_frames": int(mcfg["max_frames"])}


def open_context(client: dict, mcfg: dict, video_path, system_instruction: str, n_questions: int) -> dict:
    ctx = {"system_instruction": system_instruction, "video_path": video_path, "prefix": None,
           "seq_keys": (), "cache": None, "cache_tokens": None, "cache_error": None, "video_tokens": None,
           "video_frames": None}
    if video_path is None:
        return ctx
    proc = client["processor"]
    inputs = proc.apply_chat_template(
        _messages(system_instruction, video_path, ""), tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=False,
        **_video_kwargs(mcfg), **_template_kwargs(mcfg), **mcfg.get("processor_kwargs", {}))
    ids = inputs["input_ids"][0]
    hits = (ids == proc.tokenizer.convert_tokens_to_ids(VISION_END)).nonzero().flatten()
    if len(hits) != 1:
        raise RuntimeError(f"expected one {VISION_END} in the prompt, found {len(hits)}")
    n, seq = int(hits[0]) + 1, ids.shape[0]
    # Per-token tensors (input_ids, attention_mask, mm_token_type_ids, ...) are cut after the
    # video; everything else (pixel_values_videos, video_grid_thw, ...) is kept whole.
    seq_keys = tuple(k for k, v in inputs.items() if torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, seq))
    ctx["prefix"] = {k: (v[:, :n] if k in seq_keys else v) for k, v in inputs.items()}
    ctx["seq_keys"] = seq_keys
    ctx["video_tokens"] = n
    grid = inputs.get("video_grid_thw")  # (videos, 3): temporal patches, h, w
    if grid is not None:
        ctx["video_frames"] = int(grid[0, 0]) * getattr(proc.video_processor, "temporal_patch_size", 2)
    print(f"    {video_path.name}: fps={mcfg['fps']} -> {ctx['video_frames']} frames, "
          f"{n} prompt tokens up to the end of the video")
    return ctx


def _inputs(client: dict, mcfg: dict, ctx: dict, prompt: str) -> dict:
    proc = client["processor"]
    messages = _messages(ctx["system_instruction"], ctx["video_path"], prompt)
    if ctx["prefix"] is None:  # text-only
        return dict(proc.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                             add_generation_prompt=True, **_template_kwargs(mcfg)))
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **_template_kwargs(mcfg))
    _, sep, suffix = text.partition(VISION_END)
    if not sep:
        raise RuntimeError(f"{VISION_END} missing from the rendered prompt")
    suffix_ids = proc.tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
    out = {}
    for k, v in ctx["prefix"].items():
        if k not in ctx["seq_keys"]:
            out[k] = v
        elif k == "input_ids":
            out[k] = torch.cat([v, suffix_ids.to(v.dtype)], dim=1)
        else:  # attention_mask -> 1 for the new text, token-type ids -> 0 (text)
            fill = torch.ones if k == "attention_mask" else torch.zeros
            out[k] = torch.cat([v, fill(1, suffix_ids.shape[1], dtype=v.dtype)], dim=1)
    return out


def ask(client: dict, mcfg: dict, ctx: dict, prompt: str, n_options: int) -> dict:
    model, proc = client["model"], client["processor"]
    inputs = _inputs(client, mcfg, ctx, prompt)
    n_in = inputs["input_ids"].shape[1]
    from transformers import LogitsProcessorList

    preset = {k: v for k, v in _mode(mcfg).items() if k != "enable_thinking"}
    gen = {"do_sample": True, "max_new_tokens": int(mcfg.get("max_new_tokens", 8192)),
           **preset, **mcfg.get("generation_kwargs", {})}
    presence = float(gen.pop("presence_penalty", 0.0))
    if presence:
        gen["logits_processor"] = LogitsProcessorList([PresencePenalty(presence, n_in)])
    with client["lock"]:
        if mcfg.get("seed") is not None:
            from transformers import set_seed
            set_seed(int(mcfg["seed"]))
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**{k: v.to(model.device) if torch.is_tensor(v) else v
                                    for k, v in inputs.items()}, **gen)
        latency = time.time() - t0
    new = out[0, n_in:].cpu()

    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    stopped = len(new) > 0 and int(new[-1]) in eos

    # Split on the last </think> token; without one, a run that thought until the cap has no answer.
    think_id = proc.tokenizer.convert_tokens_to_ids(THINK_END)
    ends = (new == think_id).nonzero().flatten() if think_id is not None else []
    if len(ends):
        cut = int(ends[-1]) + 1
        thought_ids, answer_ids = new[:cut - 1], new[cut:]
    elif _mode(mcfg)["enable_thinking"] and not stopped:
        thought_ids, answer_ids = new, new[:0]
    else:
        thought_ids, answer_ids = new[:0], new
    text = proc.decode(answer_ids, skip_special_tokens=True).strip()
    thoughts = proc.decode(thought_ids, skip_special_tokens=True).strip()

    return {
        "pred_letter": parse_letter(text, n_options),
        "raw_text": text,
        "thoughts": thoughts or None,
        "finish_reason": "STOP" if stopped else "MAX_TOKENS",  # MAX_TOKENS is what run_eval treats as truncated
        "block_reason": None,
        "usage": {"prompt_token_count": n_in, "cached_content_token_count": None,
                  "candidates_token_count": len(answer_ids), "thoughts_token_count": len(thought_ids),
                  "total_token_count": n_in + len(new)},
        "latency_s": round(latency, 2),
        "cache_used": False,
        "fps": mcfg["fps"],
        "max_frames": mcfg["max_frames"],
        "video_frames": ctx["video_frames"],
        "video_tokens": ctx["video_tokens"],
    }


def close_context(client: dict, ctx: dict) -> None:
    ctx["prefix"] = None  # drop the decoded video; nothing on the GPU outlives a generate call
