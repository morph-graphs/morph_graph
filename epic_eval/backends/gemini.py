"""Gemini backend (Developer API, Files API + explicit context caching).

A *context* is one video input shared by a group of questions: the full video
for non-streaming questions, or one [0, t_ask] clip for the streaming questions
asked at that t_ask. The video is uploaded once per context and, when the group
has ``cache_min_questions`` or more questions, cached once so its tokens are
reused across the group's queries. Cached and direct calls carry the identical
system instruction and video part; only where they live differs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

from ..common import parse_letter

RETRY_CODES = {408, 429, 500, 502, 503, 504}


def api_key(mcfg: dict) -> str | None:
    key = os.environ.get(mcfg.get("api_key_env", "GEMINI_API_KEY"))
    if key:
        return key.strip()
    path = Path(os.path.expanduser(mcfg.get("api_key_file", "")))
    if mcfg.get("api_key_file") and path.is_file():
        return path.read_text().strip()
    return None


def make_client(mcfg: dict):
    key = api_key(mcfg)
    if not key:
        raise RuntimeError(f"no Gemini key: set ${mcfg.get('api_key_env')} or write {mcfg.get('api_key_file')}")
    http = types.HttpOptions(api_version=mcfg["api_version"]) if mcfg.get("api_version") else None
    return genai.Client(api_key=key, http_options=http)


def _retry(fn, mcfg: dict, what: str):
    delay = float(mcfg.get("retry_delay_s", 10))
    for attempt in range(int(mcfg.get("max_retries", 6)) + 1):
        try:
            return fn()
        except errors.APIError as e:
            if e.code not in RETRY_CODES or attempt == mcfg.get("max_retries", 6):
                raise
            print(f"    {what}: HTTP {e.code}, retry {attempt + 1} in {delay:.0f}s")
        except (ConnectionError, TimeoutError, OSError) as e:
            if attempt == mcfg.get("max_retries", 6):
                raise
            print(f"    {what}: {type(e).__name__}: {e}, retry {attempt + 1} in {delay:.0f}s")
        time.sleep(delay)
        delay = min(delay * 2, 300)


def _video_part(mcfg: dict, file) -> types.Part:
    part = types.Part(
        file_data=types.FileData(file_uri=file.uri, mime_type=file.mime_type),
        video_metadata=types.VideoMetadata(fps=float(mcfg["fps"])),
    )
    if mcfg.get("media_resolution"):
        part.media_resolution = types.PartMediaResolution(level=mcfg["media_resolution"])
    return part


def upload(client, mcfg: dict, path) -> object:
    f = _retry(lambda: client.files.upload(file=str(path), config=types.UploadFileConfig(
        mime_type="video/mp4", display_name=Path(path).name)), mcfg, "upload")
    t0 = time.time()
    while f.state and f.state.name == "PROCESSING":
        if time.time() - t0 > float(mcfg.get("upload_timeout_s", 1800)):
            raise TimeoutError(f"{path} still PROCESSING after {time.time() - t0:.0f}s")
        time.sleep(5)
        f = _retry(lambda: client.files.get(name=f.name), mcfg, "files.get")
    if f.state and f.state.name != "ACTIVE":
        raise RuntimeError(f"upload of {path} ended in state {f.state.name}: {f.error}")
    return f


def open_context(client, mcfg: dict, video_path, system_instruction: str, n_questions: int) -> dict:
    ctx = {"system_instruction": system_instruction, "file": None, "part": None,
           "cache": None, "cache_tokens": None, "cache_error": None, "video_tokens": None}
    if video_path is not None:
        ctx["file"] = upload(client, mcfg, video_path)
        ctx["part"] = _video_part(mcfg, ctx["file"])
    try:
        _maybe_cache(client, mcfg, ctx, video_path, n_questions)
    except BaseException:
        close_context(client, ctx)  # the caller never receives ctx, so clean up here
        raise
    return ctx


def _maybe_cache(client, mcfg: dict, ctx: dict, video_path, n_questions: int) -> None:
    system_instruction = ctx["system_instruction"]
    if ctx["part"] is not None and n_questions >= int(mcfg.get("cache_min_questions", 2)):
        try:
            cache = _retry(lambda: client.caches.create(model=mcfg["name"], config=types.CreateCachedContentConfig(
                contents=[types.Content(role="user", parts=[ctx["part"]])],
                system_instruction=system_instruction,
                ttl=f"{int(mcfg.get('cache_ttl_s', 7200))}s",
                display_name=Path(video_path).name[:120],
            )), mcfg, "caches.create")
            ctx["cache"] = cache
            um = getattr(cache, "usage_metadata", None)
            ctx["cache_tokens"] = getattr(um, "total_token_count", None)
        except errors.APIError as e:
            # e.g. below the model's minimum cacheable size: answer directly instead.
            ctx["cache_error"] = f"HTTP {e.code}: {e.message}"
            print(f"    cache not created ({ctx['cache_error']}); using direct calls")


def _generate_config(mcfg: dict, n_options: int, ctx: dict) -> types.GenerateContentConfig:
    letters = [chr(ord("A") + i) for i in range(n_options)]
    cfg = types.GenerateContentConfig(
        temperature=mcfg.get("temperature"),
        seed=mcfg.get("seed"),
        max_output_tokens=mcfg.get("max_output_tokens"),
        response_mime_type="application/json",
        response_schema={"type": "OBJECT", "properties": {"answer": {"type": "STRING", "enum": letters}},
                         "required": ["answer"]},
        thinking_config=types.ThinkingConfig(thinking_level=mcfg.get("thinking_level"),
                                             include_thoughts=bool(mcfg.get("include_thoughts"))),
    )
    if ctx["cache"] is not None:
        cfg.cached_content = ctx["cache"].name
    else:
        cfg.system_instruction = ctx["system_instruction"]
    return cfg


def ask(client, mcfg: dict, ctx: dict, prompt: str, n_options: int) -> dict:
    parts = ([ctx["part"]] if ctx["cache"] is None and ctx["part"] is not None else []) + [types.Part(text=prompt)]
    config = _generate_config(mcfg, n_options, ctx)
    t0 = time.time()
    resp = _retry(lambda: client.models.generate_content(
        model=mcfg["name"], contents=[types.Content(role="user", parts=parts)], config=config),
        mcfg, "generate_content")
    latency = time.time() - t0
    cand = resp.candidates[0] if resp.candidates else None
    answer_text, thoughts = [], []
    for p in (cand.content.parts if cand and cand.content and cand.content.parts else []):
        if p.text:
            (thoughts if p.thought else answer_text).append(p.text)
    text = "".join(answer_text)
    um = resp.usage_metadata
    usage = {k: getattr(um, k, None) for k in (
        "prompt_token_count", "cached_content_token_count", "candidates_token_count",
        "thoughts_token_count", "total_token_count")} if um else {}
    feedback = getattr(resp, "prompt_feedback", None)
    return {
        "pred_letter": parse_letter(text, n_options),
        "raw_text": text,
        "thoughts": "\n".join(thoughts) if thoughts else None,
        "finish_reason": getattr(getattr(cand, "finish_reason", None), "name", None),
        "block_reason": str(feedback.block_reason) if feedback and feedback.block_reason else None,
        "usage": usage,
        "latency_s": round(latency, 2),
        "cache_used": ctx["cache"] is not None,
    }


def close_context(client, ctx: dict) -> None:
    for kind, obj in (("cache", ctx.get("cache")), ("file", ctx.get("file"))):
        if obj is None:
            continue
        try:
            (client.caches if kind == "cache" else client.files).delete(name=obj.name)
        except Exception as e:  # cleanup is best-effort; caches expire by TTL, files after 48h
            print(f"    warning: could not delete {kind} {obj.name}: {e}")


def count_video_tokens(client, mcfg: dict, file) -> int:
    part = _video_part(mcfg, file)
    r = _retry(lambda: client.models.count_tokens(
        model=mcfg["name"], contents=[types.Content(role="user", parts=[part])]), mcfg, "count_tokens")
    return r.total_tokens
