"""Claude (Anthropic) client for oracle / query-agnostic event annotation.

Mirrors ``gemini.py``'s ``annotate()`` contract so ``annotate.py`` can swap
providers without touching the rest of the pipeline.

Claude does not support native video file uploads.  Videos are decomposed
into JPEG frames via ffmpeg (see ``frames.py``) and sent as base64 images
through the Anthropic Messages API.
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

DEFAULT_MODEL = "claude-opus-4-8"
# Claude has no native JSON‐schema mode.  The schema is injected into the
# prompt text and the model is instructed to return only valid JSON.
PROMPT_VERSION_AGNOSTIC_CLAUDE = "eval_agnostic_v3_claude"

# Claude limits: 100 images per request, 2000 px max dimension in multi-image.
MAX_FRAMES = 100
MAX_IMAGE_DIM = 2000
DEFAULT_FPS = 1.0


class ClaudeAnnotationError(RuntimeError):
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


def _json_schema_instruction(annotation_mode: str) -> str:
    """Return a prompt fragment that embeds the expected JSON schema."""
    schema = gemini_response_schema(annotation_mode)
    schema_json = json.dumps(schema, indent=2)
    return (
        "You MUST respond with ONLY a single valid JSON object (no markdown "
        "fences, no explanation, no chain-of-thought).  The JSON must conform "
        "exactly to this schema:\n\n"
        f"```json\n{schema_json}\n```\n"
    )


class ClaudeOracleAnnotator:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        fps: float = DEFAULT_FPS,
        max_frames: int = MAX_FRAMES,
        max_output_tokens: int = 32768,
        max_retries: int = 5,
        retry_delay_sec: float = 30.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass api_key")
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Install Claude support with: pip install anthropic"
            ) from exc

        self.model = model
        self.fps = float(fps)
        self.max_frames = int(max_frames)
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = int(max_retries)
        self.retry_delay_sec = float(retry_delay_sec)
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def _build_messages(
        self,
        frames_b64: list[tuple[str, float]],
        prompt_text: str,
        annotation_mode: str,
    ) -> list[dict[str, Any]]:
        """Build the Anthropic messages payload with interleaved images + text."""
        content: list[dict[str, Any]] = []

        # Add each frame as an image content block.
        for b64_data, _ts in frames_b64:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64_data,
                    },
                }
            )

        # Add the combined prompt text (frame preamble + event prompt + schema).
        full_prompt = (
            prompt_text
            + "\n\n"
            + _json_schema_instruction(annotation_mode)
        )
        content.append({"type": "text", "text": full_prompt})

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
            prompt_version = PROMPT_VERSION_AGNOSTIC_CLAUDE
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

        messages = self._build_messages(frames_b64, full_prompt_text, annotation_mode)

        last_error: Exception | None = None
        last_raw_text: str | None = None
        last_parsed_events: list[dict[str, Any]] | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    messages=messages,
                )
                # Extract text from the response content blocks.
                raw_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        raw_text += block.text
                raw_text = raw_text.strip()
                # Strip markdown code fences if present.
                if raw_text.startswith("```"):
                    lines = raw_text.split("\n")
                    # Remove opening fence (```json or ```) and closing ```.
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    raw_text = "\n".join(lines).strip()

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
                # Anthropic raises APIStatusError for HTTP errors.
                status_code = getattr(exc, "status_code", None)
                if (
                    status_code is not None
                    and 400 <= status_code < 500
                    and status_code != 429
                ):
                    raise
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_delay_sec)

        raise ClaudeAnnotationError(
            f"Claude annotation failed after {self.max_retries} attempts: {last_error}",
            raw_response=last_raw_text,
            parsed_events=last_parsed_events,
        ) from last_error
