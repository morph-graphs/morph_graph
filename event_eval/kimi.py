"""Kimi (Moonshot) client for oracle / query-agnostic event annotation.

Mirrors ``gemini.py``'s ``annotate()`` contract so ``annotate.py`` can swap
providers without touching the rest of the pipeline. Schema and duration probing
are shared; agnostic prompts differ because Kimi over-merges on the shared v3
text unless given an explicit segmentation-density floor.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gemini import (
    PROMPT_VERSION_ORACLE,
    build_agnostic_prompt,
    build_prompt,
    video_duration_sec,
)
from .schema import (
    OracleAnnotation,
    OracleEvent,
    gemini_response_schema,
    predicted_answer_from_response,
)

DEFAULT_MODEL = "kimi-k2.6"
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
# v3 merge guidance is over-applied by Kimi (few mega-events); v4 adds density floor.
PROMPT_VERSION_AGNOSTIC = "eval_agnostic_v4_kimi"


def build_agnostic_prompt_kimi(duration_sec: float) -> str:
    """Agnostic prompt with extra anti-merge guidance for Kimi's tendency to under-segment."""
    base = build_agnostic_prompt(duration_sec)
    soft_max = min(60, int(8 * (duration_sec / 60) ** 0.5) + 8)
    soft_min = max(5, soft_max // 3)
    max_span = min(120.0, duration_sec * 0.20)
    return (
        base
        + f"""

## Segmentation density (critical)
Do NOT collapse an entire build session, walk, or craft project into one or two
long events. Under-segmentation is the main failure mode to avoid.

- Target **{soft_min}–{soft_max} events** for this {duration_sec:.0f}s video.
  Fewer than {soft_min} events almost always means distinct phases were merged.
- **Split** when any of these change: sub-goal or build step, materials/tools,
  room or location, who/what is being interacted with, or activity type (e.g.
  sorting pieces → reading instructions → assembling section A → section B).
- **No single event longer than ~{max_span:.0f}s** unless the full interval is
  one uniform repetitive cycle with absolutely no phase change (rare).
- Long walks: split at trail junctions, terrain changes, or when attention
  shifts to a new landmark — not one forest-walk event for the whole video.
- Long builds/crafts: one event per instruction section or visibly distinct
  assembly phase, not one event for the entire build.
"""
    )

# Moonshot rejects uploads over 100MB. Most egolongqa source videos (15fps,
# up to ~1GB) blow past that, so anything over the cap gets re-encoded before
# upload, always at UPLOAD_FPS (never sampled coarser, so temporal coverage
# stays comparable to Gemini's fps=1.0). Rather than a fixed-quality CRF
# (which leaves most of the 100MB budget unused on simpler videos), the
# video bitrate is solved from TARGET_UPLOAD_BYTES and the clip's duration
# and encoded two-pass, so every video uses as much of the cap as it can at
# native resolution. Resolution is only reduced if bitrate-targeting still
# overshoots (rare — two-pass rate control is usually accurate to a few %).
UPLOAD_FPS = 1.0
MAX_UPLOAD_BYTES = 100_000_000
TARGET_UPLOAD_BYTES = 97_000_000  # small margin under the hard cap for muxing/rate-control variance
AUDIO_BITRATE_KBPS = 64
MIN_VIDEO_BITRATE_KBPS = 150
MIN_UPLOAD_WIDTH = 320
MAX_ENCODE_ATTEMPTS = 4


def _even(value: int) -> int:
    value = int(value)
    return value if value % 2 == 0 else value - 1


def _probe_width(video_path: str | Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _video_bitrate_kbps(duration_sec: float, target_bytes: int) -> float:
    total_kbps = (target_bytes * 8) / duration_sec / 1000
    return max(MIN_VIDEO_BITRATE_KBPS, total_kbps - AUDIO_BITRATE_KBPS)


def _encode_two_pass(
    src: Path, dst: Path, *, bitrate_kbps: float, width: int | None, passlog: Path
) -> None:
    video_filter = f"fps={UPLOAD_FPS}"
    if width is not None:
        video_filter += f",scale={width}:-2"
    bitrate_arg = f"{bitrate_kbps:.0f}k"
    common = [
        "-i",
        str(src),
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-b:v",
        bitrate_arg,
        "-passlogfile",
        str(passlog),
    ]
    subprocess.run(
        ["ffmpeg", "-y", *common, "-pass", "1", "-an", "-f", "mp4", os.devnull],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", *common, "-pass", "2", "-c:a", "aac", "-b:a", f"{AUDIO_BITRATE_KBPS}k", str(dst)],
        check=True,
        capture_output=True,
        text=True,
    )


def prepare_upload_path(path: Path, duration_sec: float) -> tuple[Path, Path | None]:
    """Return (path_to_upload, temp_dir_to_clean_up_or_None).

    Videos under the cap are uploaded as-is. Larger ones are re-encoded
    two-pass at UPLOAD_FPS with a bitrate solved to land near
    TARGET_UPLOAD_BYTES. If rate control still overshoots the hard cap,
    the bitrate/resolution are corrected and it re-encodes, up to
    MAX_ENCODE_ATTEMPTS.
    """
    if path.stat().st_size <= MAX_UPLOAD_BYTES:
        return path, None

    tmp_dir = Path(tempfile.mkdtemp(prefix="kimi_upload_"))
    candidate = tmp_dir / f"{path.stem}_upload.mp4"
    passlog = tmp_dir / "ffmpeg2pass"
    try:
        width: int | None = _probe_width(path)
        bitrate_kbps = _video_bitrate_kbps(duration_sec, TARGET_UPLOAD_BYTES)
        attempt = 0
        while True:
            _encode_two_pass(path, candidate, bitrate_kbps=bitrate_kbps, width=width, passlog=passlog)
            actual = candidate.stat().st_size
            attempt += 1
            if actual <= MAX_UPLOAD_BYTES or attempt >= MAX_ENCODE_ATTEMPTS:
                break
            if bitrate_kbps > MIN_VIDEO_BITRATE_KBPS * 1.2:
                # Rate control overshot at a workable bitrate: tighten the target.
                bitrate_kbps = max(MIN_VIDEO_BITRATE_KBPS, bitrate_kbps * (TARGET_UPLOAD_BYTES / actual) * 0.9)
            else:
                # Already near the bitrate floor: downscale instead of degrading further.
                base_width = width or _probe_width(path)
                scale = (TARGET_UPLOAD_BYTES / actual) ** 0.5
                width = max(MIN_UPLOAD_WIDTH, _even(int(base_width * scale)))
        if candidate.stat().st_size > MAX_UPLOAD_BYTES:
            raise RuntimeError(
                f"Could not compress {path} under {MAX_UPLOAD_BYTES} bytes "
                f"after {attempt} encode attempts "
                f"(final size={candidate.stat().st_size}, bitrate={bitrate_kbps:.0f}kbps, width={width})"
            )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return candidate, tmp_dir


class KimiAnnotationError(RuntimeError):
    """Raised after retries are exhausted; carries the last model payload."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        parsed_events: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.parsed_events = parsed_events


class KimiOracleAnnotator:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        max_output_tokens: int = 32768,
        max_retries: int = 5,
        retry_delay_sec: float = 30.0,
    ) -> None:
        api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("Set MOONSHOT_API_KEY or pass api_key")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install Kimi support with: pip install openai") from exc

        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = int(max_retries)
        self.retry_delay_sec = float(retry_delay_sec)
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _upload_video(self, path: Path) -> str:
        # No documented "processing" status to poll for Kimi video uploads
        # (unlike Gemini's files.get().state == ACTIVE). If large videos turn
        # out to need a readiness check, add it here once observed.
        uploaded = self._client.files.create(file=path, purpose="video")
        return uploaded.id

    def _response_format(self, annotation_mode: str) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "event_annotation",
                "schema": gemini_response_schema(annotation_mode),
                "strict": True,
            },
        }

    def annotate(
        self,
        *,
        sample_id: str,
        video_path: str | Path,
        question: str,
        options: str | None = None,
        annotation_mode: str = "query_conditioned",
    ) -> OracleAnnotation:
        if annotation_mode == "query_conditioned":
            if not question.strip():
                raise ValueError("query-conditioned annotation requires a question")
            prompt_builder = lambda duration: build_prompt(question, duration, options=options)
            prompt_version = PROMPT_VERSION_ORACLE + ("_mcq" if options else "")
            query = question
        elif annotation_mode == "query_agnostic":
            prompt_builder = build_agnostic_prompt_kimi
            prompt_version = PROMPT_VERSION_AGNOSTIC
            query = ""
        else:
            raise ValueError(f"Unknown annotation_mode={annotation_mode!r}")

        path = Path(video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        duration = video_duration_sec(path)
        upload_path, tmp_dir = prepare_upload_path(path, duration)
        try:
            file_id = self._upload_video(upload_path)
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": f"ms://{file_id}"}},
                        {"type": "text", "text": prompt_builder(duration)},
                    ],
                }
            ]
            last_error: Exception | None = None
            last_raw_text: str | None = None
            last_parsed_events: list[dict[str, Any]] | None = None
            for attempt in range(self.max_retries):
                try:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        # kimi-k2.5 rejects any temperature other than 1 (400 invalid_request_error).
                        temperature=1.0,
                        max_tokens=self.max_output_tokens,
                        response_format=self._response_format(annotation_mode),
                    )
                    raw_text = (response.choices[0].message.content or "").strip()
                    last_raw_text = raw_text
                    last_parsed_events = None
                    generated = json.loads(raw_text)
                    events = []
                    event_dicts: list[dict[str, Any]] = []
                    for value in generated.get("events", []):
                        if annotation_mode == "query_agnostic":
                            # Schema omits roles; force empty for validation.
                            value = {**value, "query_roles": []}
                        event_dicts.append(dict(value))
                        events.append(OracleEvent.from_dict(value))
                    last_parsed_events = event_dicts
                    annotation = OracleAnnotation(
                        sample_id=str(sample_id),
                        video_id=path.stem,
                        video_path=str(path),
                        video_duration_sec=duration,
                        query=query,
                        events=tuple(events),
                        model=self.model,
                        prompt_version=prompt_version,
                        # Kimi's API exposes no explicit frame-sampling control
                        # (unlike Gemini's VideoMetadata(fps=...)). Videos over
                        # the upload cap are re-encoded at UPLOAD_FPS before
                        # upload; smaller ones go through at native fps, so
                        # this is a nominal reference rate, not a guarantee.
                        video_fps=UPLOAD_FPS,
                        media_resolution=None,
                        generated_at=datetime.now(timezone.utc).isoformat(),
                        warnings=tuple(
                            str(value) for value in generated.get("warnings", [])
                        ),
                        raw_response=raw_text,
                        annotation_mode=annotation_mode,
                        predicted_answer=predicted_answer_from_response(
                            generated, annotation_mode=annotation_mode
                        ),
                    )
                    return annotation.require_valid()
                except Exception as exc:
                    last_error = exc
                    status_code = getattr(exc, "status_code", None)
                    if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                        raise
                    if attempt + 1 < self.max_retries:
                        time.sleep(self.retry_delay_sec)
            raise KimiAnnotationError(
                f"Kimi annotation failed after {self.max_retries} attempts: {last_error}",
                raw_response=last_raw_text,
                parsed_events=last_parsed_events,
            ) from last_error
        finally:
            try:
                self._client.files.delete(file_id)
            except Exception:
                # The annotation result should not be discarded because remote
                # cleanup failed; stale uploads can be removed separately.
                pass
