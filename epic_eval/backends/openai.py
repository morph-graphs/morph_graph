"""OpenAI backend (Responses API): the video goes in as frames, since these models take no video file.

A *context* is the frames of one clip: the full video, or the [0, t_ask] clip prep cut for the streaming
questions asked at that t_ask. Frames are taken from that clip only, evenly spaced and ending on its last
frame (what is in view when the question is asked), so no frame can come from after t_ask; each answer row
records the times used (``frame_times_s``, ``last_frame_s``). At most ``max_frames`` frames, never more than
``max_fps`` a second, scaled to ``frame_side`` px and sent with ``detail``. The frames come first in every
request of a context, so the provider's automatic prompt cache can reuse them across its questions.
The key is read from the environment only (``api_key_env``, default OPENAI_API_KEY).
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path

from openai import OpenAI

from ..common import parse_letter

MAX_IMAGES, MAX_BYTES = 1500, 512 * 2 ** 20   # per request (images guide, 2026-11)


def make_client(mcfg: dict):
    key = os.environ.get(mcfg.get("api_key_env", "OPENAI_API_KEY"))
    if not key:
        raise RuntimeError(f"no OpenAI key: set ${mcfg.get('api_key_env', 'OPENAI_API_KEY')}")
    # the SDK retries 408/409/429/5xx and connection errors itself, with backoff
    return OpenAI(api_key=key.strip(), max_retries=int(mcfg.get("max_retries", 6)),
                  timeout=float(mcfg.get("timeout_s", 1800)))


def frame_pts(path) -> list[float]:
    """Presentation time of every video frame of the file, in order."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return sorted(float(x) for x in out.stdout.replace(",", " ").split() if x.strip() not in ("", "N/A"))


def pick_frames(pts: list[float], max_frames: int, max_fps: float) -> list[int]:
    """Indices into pts: k frames evenly spaced over the clip, the last one being the clip's last frame."""
    n = len(pts)
    k = max(1, min(int(max_frames), n, int((pts[-1] - pts[0]) * float(max_fps))))
    return [((i + 1) * n + k - 1) // k - 1 for i in range(k)]


def read_frames(path, idx: list[int], n_frames: int, side: int, quality: int) -> list[bytes]:
    """JPEGs of the frames idx of the file. Every frame is decoded once into a temporary folder (the clips are
    2 fps, so a 55 min clip is ~6,600 small files) and only the chosen ones are read."""
    scale = f"scale='if(gte(iw,ih),min({side},iw),-2)':'if(gte(iw,ih),-2,min({side},ih))'"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", scale,
                        "-fps_mode", "passthrough", "-q:v", str(quality), "-start_number", "0",
                        os.path.join(tmp, "%06d.jpg")], check=True)
        got = len(os.listdir(tmp))
        if got != n_frames:   # frame k on disk must be the frame with pts[k]
            raise RuntimeError(f"{path}: decoded {got} frames, the container lists {n_frames}")
        return [Path(tmp, f"{i:06d}.jpg").read_bytes() for i in idx]


def open_context(client, mcfg: dict, video_path, system_instruction: str, n_questions: int) -> dict:
    ctx = {"system_instruction": system_instruction, "images": [], "frame_times": [],
           "cache": None, "cache_tokens": None, "cache_error": None}
    if video_path is None:
        return ctx
    pts = frame_pts(video_path)
    idx = pick_frames(pts, mcfg.get("max_frames", 1024), mcfg.get("max_fps", 1.0))
    jpegs = read_frames(video_path, idx, len(pts), int(mcfg.get("frame_side", 512)), int(mcfg.get("jpeg_quality", 3)))
    ctx["frame_times"] = [round(pts[i], 3) for i in idx]
    ctx["images"] = ["data:image/jpeg;base64," + base64.b64encode(j).decode() for j in jpegs]
    ctx["request_bytes"] = sum(len(s) for s in ctx["images"])
    if len(idx) > MAX_IMAGES or ctx["request_bytes"] > 0.95 * MAX_BYTES:
        raise RuntimeError(f"{video_path}: {len(idx)} frames, {ctx['request_bytes'] / 2 ** 20:.0f} MB: over the "
                           f"request limits ({MAX_IMAGES} images, {MAX_BYTES // 2 ** 20} MB); lower max_frames or frame_side")
    return ctx


def _content(mcfg: dict, ctx: dict, prompt: str) -> list[dict]:
    times, parts = ctx["frame_times"], []
    if times:
        step = (times[-1] - times[0]) / max(1, len(times) - 1)
        parts.append({"type": "input_text", "text": (
            f"The video is given as {len(times)} frames in chronological order, sampled evenly from its start "
            f"to its end (about one every {step:.1f} s).")})
    for t, img in zip(times, ctx["images"]):
        if mcfg.get("frame_timestamps"):
            parts.append({"type": "input_text", "text": f"t={t:.1f}s"})
        parts.append({"type": "input_image", "image_url": img, "detail": mcfg.get("detail", "high")})
    return parts + [{"type": "input_text", "text": prompt}]


def ask(client, mcfg: dict, ctx: dict, prompt: str, n_options: int) -> dict:
    letters = [chr(ord("A") + i) for i in range(n_options)]
    kw = {"model": mcfg["name"], "instructions": ctx["system_instruction"], "store": False,
          "input": [{"role": "user", "content": _content(mcfg, ctx, prompt)}],
          "text": {"format": {"type": "json_schema", "name": "answer", "strict": True, "schema": {
              "type": "object", "properties": {"answer": {"type": "string", "enum": letters}},
              "required": ["answer"], "additionalProperties": False}}}}
    if mcfg.get("max_output_tokens"):
        kw["max_output_tokens"] = int(mcfg["max_output_tokens"])
    reasoning = {k: mcfg[f"reasoning_{k}"] for k in ("effort", "summary") if mcfg.get(f"reasoning_{k}")}
    if reasoning:
        kw["reasoning"] = reasoning
    t0 = time.time()
    resp = client.responses.create(**kw)
    latency = time.time() - t0
    text = resp.output_text or ""
    thoughts = [s.text for item in (resp.output or []) if getattr(item, "type", None) == "reasoning"
                for s in (getattr(item, "summary", None) or []) if getattr(s, "text", None)]
    u = resp.usage
    out_tok = getattr(u, "output_tokens", None)
    think = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None)
    usage = {   # under the Gemini backend's names, so score.py adds them up the same way
        "prompt_token_count": getattr(u, "input_tokens", None),
        "cached_content_token_count": getattr(getattr(u, "input_tokens_details", None), "cached_tokens", None),
        "candidates_token_count": out_tok - (think or 0) if out_tok is not None else None,
        "thoughts_token_count": think,
        "total_token_count": getattr(u, "total_tokens", None)} if u else {}
    why = getattr(getattr(resp, "incomplete_details", None), "reason", None)
    return {
        "pred_letter": parse_letter(text, n_options),
        "raw_text": text,
        "thoughts": "\n".join(thoughts) if thoughts else None,
        # run_eval treats MAX_TOKENS without a letter as truncated (final unless runtime.retry_truncated)
        "finish_reason": "STOP" if resp.status == "completed" else "MAX_TOKENS" if why == "max_output_tokens" else (why or resp.status),
        "block_reason": why if why == "content_filter" else None,
        "usage": usage,
        "latency_s": round(latency, 2),
        "cache_used": bool(usage.get("cached_content_token_count")),
        "model_returned": getattr(resp, "model", None),
        "n_frames": len(ctx["frame_times"]),
        "frame_times_s": ctx["frame_times"],
        "last_frame_s": ctx["frame_times"][-1] if ctx["frame_times"] else None,
        "frame_side": int(mcfg.get("frame_side", 512)) if ctx["frame_times"] else None,
        "detail": mcfg.get("detail", "high") if ctx["frame_times"] else None,
    }


def close_context(client, ctx: dict) -> None:
    ctx["images"] = []   # nothing is stored at the provider (store=False); just drop the frames
