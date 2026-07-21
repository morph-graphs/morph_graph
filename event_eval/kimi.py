"""Kimi (Moonshot) client for oracle / query-agnostic event annotation.

Mirrors ``gemini.py``'s ``annotate()`` contract so ``annotate.py`` can swap
providers without touching the rest of the pipeline. Prompts, schema, and
duration probing are shared with the Gemini path (same text, different
model) so the two providers stay directly comparable in ``metrics.py``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gemini import (
    PROMPT_VERSION_AGNOSTIC,
    PROMPT_VERSION_ORACLE,
    build_agnostic_prompt,
    build_prompt,
    video_duration_sec,
)
from .schema import OracleAnnotation, OracleEvent, gemini_response_schema

DEFAULT_MODEL = "kimi-k2.5"
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"


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
        annotation_mode: str = "query_conditioned",
    ) -> OracleAnnotation:
        if annotation_mode == "query_conditioned":
            if not question.strip():
                raise ValueError("query-conditioned annotation requires a question")
            prompt_builder = lambda duration: build_prompt(question, duration)
            prompt_version = PROMPT_VERSION_ORACLE
            query = question
        elif annotation_mode == "query_agnostic":
            prompt_builder = build_agnostic_prompt
            prompt_version = PROMPT_VERSION_AGNOSTIC
            query = ""
        else:
            raise ValueError(f"Unknown annotation_mode={annotation_mode!r}")

        path = Path(video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        duration = video_duration_sec(path)
        file_id = self._upload_video(path)
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
                        temperature=0.0,
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
                        # (unlike Gemini's VideoMetadata(fps=...)); this records
                        # a nominal reference rate, not a guaranteed sample rate.
                        video_fps=1.0,
                        media_resolution=None,
                        generated_at=datetime.now(timezone.utc).isoformat(),
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
