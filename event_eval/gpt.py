"""GPT (OpenAI) client for oracle / query-agnostic event annotation.

Mirrors ``gemini.py``'s ``annotate()`` contract so ``annotate.py`` can swap
providers without touching the rest of the pipeline.

GPT does not support native video file uploads.  Videos are decomposed into
JPEG frames via ffmpeg (see ``frames.py``) and sent as base64
``image_url`` content blocks through the OpenAI Chat Completions API.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .frames import (
    build_frame_preamble,
    extract_frames,
    frames_to_base64,
)
from .gemini import (
    PROMPT_VERSION_ORACLE,
    PROMPT_VERSION_AGNOSTIC,
    build_agnostic_prompt,
    build_prompt,
    video_duration_sec,
)
from .schema import (
    OracleAnnotation,
    OracleEvent,
    gemini_response_schema,
)

DEFAULT_MODEL = "us/openai/openai/eccn-gpt-5.6-sol"
DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"
PROMPT_VERSION_AGNOSTIC_GPT = "eval_agnostic_v3_gpt"

# Conservative default — GPT supports many images but each burns tokens.
MAX_FRAMES = 100
DEFAULT_FPS = 1.0
# "low" detail uses a fixed 85-token budget per image; "high" is much more
# expensive but preserves fine detail.  User chose "low" for now.
DEFAULT_IMAGE_DETAIL = "low"
# Max dimension for extracted frames (GPT has no hard limit but larger images
# at "high" detail cost more tokens; at "low" detail they are resized to
# 512×512 anyway, so 2000 px is a safe upper bound for either setting).
MAX_IMAGE_DIM = 2000


class GPTAnnotationError(RuntimeError):
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


class GPTOracleAnnotator:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        fps: float = DEFAULT_FPS,
        max_frames: int = MAX_FRAMES,
        image_detail: str = DEFAULT_IMAGE_DETAIL,
        max_output_tokens: int = 32768,
        max_retries: int = 5,
        retry_delay_sec: float = 30.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if image_detail not in {"low", "high", "auto"}:
            raise ValueError("image_detail must be 'low', 'high', or 'auto'")
        api_key = api_key or os.getenv("INFERENCE_API_KEY")
        if not api_key:
            raise ValueError("Set INFERENCE_API_KEY or pass api_key")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install GPT support with: pip install openai"
            ) from exc

        self.model = model
        self.fps = float(fps)
        self.max_frames = int(max_frames)
        self.image_detail = image_detail
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = int(max_retries)
        self.retry_delay_sec = float(retry_delay_sec)
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _response_format(self, annotation_mode: str) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "event_annotation",
                "schema": gemini_response_schema(annotation_mode),
                "strict": True,
            },
        }

    def _build_messages(
        self,
        frames_b64: list[tuple[str, float]],
        prompt_text: str,
    ) -> list[dict[str, Any]]:
        """Build the OpenAI chat messages payload with image_url blocks."""
        content: list[dict[str, Any]] = []

        # Add each frame as an image_url content block.
        for b64_data, _ts in frames_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_data}",
                        "detail": self.image_detail,
                    },
                }
            )

        # Add the combined prompt text.
        content.append({"type": "text", "text": prompt_text})

        return [{"role": "user", "content": content}]

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
            prompt_version = PROMPT_VERSION_ORACLE
            query = question
        elif annotation_mode == "query_agnostic":
            prompt_builder = build_agnostic_prompt
            prompt_version = PROMPT_VERSION_AGNOSTIC_GPT
            query = ""
        else:
            raise ValueError(f"Unknown annotation_mode={annotation_mode!r}")

        path = Path(video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        duration = video_duration_sec(path)

        # Extract frames and encode to base64.
        total_at_fps = max(1, int(math.floor(duration * self.fps)))
        subsampled = total_at_fps > self.max_frames

        frames, tmp_dir = extract_frames(
            path,
            fps=self.fps,
            max_frames=self.max_frames,
            max_width=MAX_IMAGE_DIM,
            max_height=MAX_IMAGE_DIM,
            duration_sec=duration,
        )
        try:
            frames_b64 = frames_to_base64(frames)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        timestamps = [ts for _, ts in frames_b64]
        preamble = build_frame_preamble(
            n_frames=len(frames_b64),
            fps=self.fps,
            duration_sec=duration,
            timestamps=timestamps,
            subsampled=subsampled,
        )

        event_prompt = prompt_builder(duration)
        full_prompt_text = preamble + event_prompt

        messages = self._build_messages(frames_b64, full_prompt_text)

        last_error: Exception | None = None
        last_raw_text: str | None = None
        last_parsed_events: list[dict[str, Any]] | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
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
                        value = {**value, "query_roles": []}
                    event_dicts.append(dict(value))
                    events.append(OracleEvent.from_dict(value))
                last_parsed_events = event_dicts
                answer = (
                    str(generated["answer"]).strip()
                    if "answer" in generated and generated["answer"] is not None
                    else None
                )
                annotation = OracleAnnotation(
                    sample_id=str(sample_id),
                    video_id=path.stem,
                    video_path=str(path),
                    video_duration_sec=duration,
                    query=query,
                    events=tuple(events),
                    model=self.model,
                    prompt_version=prompt_version,
                    video_fps=self.fps,
                    media_resolution=None,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    answer=answer,
                    warnings=tuple(
                        str(value) for value in generated.get("warnings", [])
                    ),
                    raw_response=raw_text,
                    annotation_mode=annotation_mode,
                )
                return annotation.require_valid()
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if (
                    status_code is not None
                    and 400 <= status_code < 500
                    and status_code != 429
                ):
                    raise
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_delay_sec)

        raise GPTAnnotationError(
            f"GPT annotation failed after {self.max_retries} attempts: {last_error}",
            raw_response=last_raw_text,
            parsed_events=last_parsed_events,
        ) from last_error
