"""Gemini client and prompts for oracle / query-agnostic event annotation."""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import (
    EVENT_CATEGORIES,
    QUERY_ROLES,
    OracleAnnotation,
    OracleEvent,
    gemini_response_schema,
    predicted_answer_from_response,
)

DEFAULT_MODEL = "gemini-3.6-flash"
PROMPT_VERSION_ORACLE = "eval_oracle_v2"
PROMPT_VERSION_AGNOSTIC = "eval_agnostic_v3"
# Backward-compatible alias used by older tests/imports.
PROMPT_VERSION = PROMPT_VERSION_ORACLE


class GeminiAnnotationError(RuntimeError):
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


def build_prompt(
    question: str,
    duration_sec: float,
    *,
    options: str | None = None,
) -> str:
    """Query-conditioned oracle prompt (upper bound for one question)."""
    categories = ", ".join(sorted(EVENT_CATEGORIES))
    roles = ", ".join(sorted(QUERY_ROLES))
    options_block = f"\n\nMultiple-choice options:\n{options.strip()}" if options else ""
    mcq_rule = (
        "\nThe question has multiple-choice options: answer with the option letter\n"
        "followed by its text (e.g. \"B. A red umbrella\").\n"
        if options
        else ""
    )
    return f"""You are creating an upper-bound event oracle for video-question evaluation.

Question:
{question.strip()}{options_block}

Video duration: {duration_sec:.3f} seconds.

Return only the minimal set of meaningful video events needed to answer the
question. The question determines which events and boundaries matter. Do not
segment unrelated portions of the video.

An event is a meaningful intentional action, interaction, state change, goal
or subgoal, spatial transition, location/context transition, causal transition,
or temporal transition. While mere visibility of a random object is not an event,
if the question asks about a specific object, sign, or visual detail, the precise
temporal window where that object is clearly observed IS considered an event.
The query's focus makes the observation itself the event, even without a physical
interaction or state change.

Use fine granularity for short actions or state changes and coarse granularity
for larger activities or goals. Use both only when the hierarchy helps answer
the question. A parent-child relationship should be created only if (1) the
child's interval lies entirely within the parent's interval, and (2) the child
represents a meaningful sub-event that contributes to, realizes, or decomposes
the parent's activity or objective. Do not create parent links solely because
events overlap in time or occur consecutively.

Every returned event must have at least one query role:
- anchor: establishes a reference in the question
- evidence: is needed to connect or disambiguate the answer
- answer_bearing: directly contains the requested answer

Also return ``predicted_answer``: a concise answer to the question that must be
derivable from the events you listed alone (their descriptions, times, and
roles). Do not rely on video content outside those events. This is a sufficiency
check on the oracle: if the events are incomplete, the answer will be wrong.
If you cannot answer from the events alone, still give your best short answer
and add a warning explaining what is missing.
{mcq_rule}
Allowed categories: {categories}
Allowed query roles: {roles}

Use timestamps in seconds with 0 <= start_sec < end_sec <= {duration_sec:.3f}.
Descriptions must state observable event content concisely. Distinguish visible
or audible evidence from inferred goals. Return no chain-of-thought."""

def build_agnostic_prompt(duration_sec: float) -> str:
    """Query-agnostic event segmentation prompt (no question)."""
    categories = ", ".join(sorted(EVENT_CATEGORIES))
    # Sublinear growth so long videos aren't capped at the same budget as short ones.
    soft_max = min(60, int(8 * (duration_sec / 60) ** 0.5) + 8)
    return f"""You are segmenting a video into query-agnostic events. Do NOT
optimize for any particular question. Produce the events a careful human would
mark if asked to paraphrase what happened in the video at a natural activity
level.

Video duration: {duration_sec:.3f} seconds.

## What counts as an event
An event is a meaningful intentional action, interaction, ongoing activity or
state (including sitting, waiting, walking, or attentively watching or
listening), state change, goal or subgoal, spatial / location / context
transition, causal transition, or clear phase change in an activity. Event
boundaries fall where one such activity or state gives way to another.

A boundary is a change in what is HAPPENING, not a change in what is PRESENTED.
Changes in viewpoint, framing, camera motion, shot boundary, zoom, or which
objects are in frame are not themselves events. Neither are movements or
gestures that do not advance or alter the ongoing activity. If the underlying
activity continues across such a change, it remains one event.

Mere visibility or presence of an everyday background object is NOT an event,
and neither is its appearance or disappearance from frame. Do not create one
event per visible object. However, notable visual encounters—such as passing a
distinct object, a specific sign, or a landmark—should be recorded. For deciding
whether an object's presence matters, the test is whether it is a distinct,
notable item the wearer encounters or passes. Sustained passive engagement
(watching, listening, observing) is itself an activity and counts as an event
even when nothing else changes.

An event boundary also occurs when what happens departs from what the ongoing
situation predicts, even if no one intended it. Unexpected occurrences,
interruptions, and things encountered out of their normal context are events: a
dog running up and barking during a walk, or an animal found somewhere it would
not normally be. The test is whether the moment forces an update to what is
happening in the activity — not whether something is merely unusual to look at.

## Compact partition (critical)
- Partition the **entire** video: every time point from 0 to
  {duration_sec:.3f} must belong to at least one event (the union of event
  intervals must cover the full duration). Idle stretches count — sitting,
  waiting, or continuous walking are real events, not omitted gaps.
- Prefer merging continuous activity into one span over chopping it into many
  near-identical pieces.
- Do NOT merge adjacent spans when the goal, activity type, or
  location / context clearly changes — even if both involve walking or similar
  motion. Hallway travel, entering a trash room to dispose of items, and walking
  back are separate events. Merge only when the ongoing activity and setting are
  essentially continuous and the descriptions would otherwise be near-duplicates.
- Soft target: about **{soft_max} events or fewer** for this video. Fewer is
  fine when the video is genuinely sparse; do not under-segment by collapsing
  distinct goals or locations. Far more is almost always over-segmentation.
- When an activity consists of a repeating or alternating cycle of steps, merge
  the entire repeating phase into ONE event describing the activity as a whole.
  Do NOT emit a long alternating sequence of near-duplicate descriptions.
- Temporally adjacent or nearby events must have clearly distinct descriptions.
  If neighbouring events would read as near-identical sentences, that is
  over-segmentation — merge them. Genuinely recurring activity separated by
  other events may legitimately reuse similar descriptions.

## Granularity labels (NOT duration tags)
`fine` and `coarse` are optional scale labels. Label an event `fine` if it is a
single atomic action, `coarse` if it is a larger activity phase. Most events are
independent roots; a flat list of only one label is valid. A `coarse` event
does **not** require any children, and most coarse events should have none.

## Overlap and parent–child links
Concurrent events may overlap as independent roots. If you are walking and a
dog runs up and barks, or if you pass a notable landmark, sign, or distinct
object, keep the long continuous walk span, but add a separate, short,
overlapping event for the visual encounter or interruption (e.g., [0:00-0:30]
"Walking along the path" and [0:10-0:12] "Passing a concrete bench"). Do not force
them into one description, and do not parent one to the other unless the child
is a true sub-step of the parent.

Create a parent link only if BOTH hold:
1. Temporal containment: the child's full interval lies within the parent
   interval (not merely adjacent or successive).
2. Semantic decomposition: the child is a real sub-step of that parent activity,
   not a concurrent side event and not the next beat after it.

Do NOT parent an event to the previous one because they are consecutive,
wrap the whole video in one coarse parent, or use parent links only because
intervals overlap. When unsure, leave `parent_event_ids` empty.

## Output rules
Do not include query roles. This annotation is query-agnostic.
Allowed categories: {categories}
Categories describe what an event is about, not a license to emit events of
that kind: `state_change` means a meaningful change of state, never an object
merely being visible. Use `unexpected_or_interruption` for events that exist
because they violated the ongoing situation's expectations.
Use timestamps in seconds with 0 <= start_sec < end_sec <= {duration_sec:.3f}.
Events should be ordered by start_sec.
Descriptions must state observable content concisely. Distinguish visible,
audible, or textual evidence from inferred goals. Return no chain-of-thought."""

def video_duration_sec(video_path: str | Path) -> float:
    """Read duration with ffprobe; annotation should not require TorchCodec/CUDA."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"Could not determine video duration: {video_path}")
    return duration


class GeminiOracleAnnotator:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        fps: float = 1.0,
        media_resolution: str | None = "medium",
        max_output_tokens: int = 32768,
        max_retries: int = 5,
        retry_delay_sec: float = 30.0,
        thinking_level: str | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if media_resolution not in {None, "low", "medium"}:
            raise ValueError("video media_resolution must be low, medium, or None")
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass api_key")
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise ImportError("Install Gemini support with: pip install google-genai") from exc

        self.model = model
        self.fps = float(fps)
        self.media_resolution = media_resolution
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = int(max_retries)
        self.retry_delay_sec = float(retry_delay_sec)
        self.thinking_level = thinking_level
        self._types = types
        self._errors = errors
        self._client = genai.Client(api_key=api_key)

    def _wait_until_active(self, uploaded: Any) -> Any:
        while True:
            uploaded = self._client.files.get(name=uploaded.name)
            state = getattr(uploaded.state, "name", str(uploaded.state))
            if state == "ACTIVE":
                return uploaded
            if state == "FAILED":
                raise RuntimeError(f"Gemini upload failed: {uploaded.name}")
            time.sleep(5)

    def _config(self, annotation_mode: str, cached_content_name: str | None = None) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
            # Native JSON Schema is required here. ``response_schema`` converts
            # through the legacy Gemini Schema type and can turn keywords such
            # as additionalProperties into an invalid API payload.
            "response_json_schema": gemini_response_schema(annotation_mode),
        }
        if self.thinking_level:
            kwargs["thinking_config"] = self._types.ThinkingConfig(
                thinking_level=self.thinking_level
            )
        if cached_content_name:
            kwargs["cached_content"] = cached_content_name
        if self.media_resolution:
            kwargs["media_resolution"] = {
                "low": self._types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": self._types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
            }[self.media_resolution]
        return self._types.GenerateContentConfig(**kwargs)

    def upload_video(self, video_path: str | Path) -> Any:
        path = Path(video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return self._wait_until_active(self._client.files.upload(file=str(path)))

    def delete_file(self, name: str) -> None:
        try:
            self._client.files.delete(name=name)
        except Exception:
            pass

    def create_video_cache(self, uploaded_file: Any, ttl: str = "3600s") -> Any:
        # We assume uploaded_file is a types.File from files.upload()
        video_part = self._types.Part.from_uri(
            file_uri=uploaded_file.uri,
            mime_type=getattr(uploaded_file, "mime_type", None) or "video/mp4",
        )
        video_part.video_metadata = self._types.VideoMetadata(fps=self.fps)
        return self._client.caches.create(
            model=self.model,
            config=self._types.CreateCachedContentConfig(
                contents=[video_part],
                ttl=ttl,
            )
        )

    def delete_cache(self, name: str) -> None:
        try:
            self._client.caches.delete(name=name)
        except Exception:
            pass

    def _generate_annotation(
        self,
        *,
        sample_id: str,
        video_id: str,
        video_path: str,
        duration: float,
        query: str,
        options: str | None = None,
        prompt_builder: Any,
        prompt_version: str,
        annotation_mode: str,
        contents: list[Any],
        cached_content_name: str | None = None,
    ) -> OracleAnnotation:
        last_error: Exception | None = None
        last_raw_text: str | None = None
        last_parsed_events: list[dict[str, Any]] | None = None
        
        config_kwargs = self._config(annotation_mode, cached_content_name=cached_content_name)

        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config_kwargs,
                )
                raw_text = (response.text or "").strip()
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
                    video_id=video_id,
                    video_path=video_path,
                    video_duration_sec=duration,
                    query=query,
                    events=tuple(events),
                    model=self.model,
                    prompt_version=prompt_version,
                    video_fps=self.fps,
                    media_resolution=self.media_resolution,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    options=options,
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
                if (
                    isinstance(exc, self._errors.APIError)
                    and 400 <= getattr(exc, "code", 0) < 500
                    and getattr(exc, "code", 0) != 429
                ):
                    raise
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_delay_sec)
        raise GeminiAnnotationError(
            f"Gemini annotation failed after {self.max_retries} attempts: {last_error}",
            raw_response=last_raw_text,
            parsed_events=last_parsed_events,
        ) from last_error

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
            prompt_builder = build_agnostic_prompt
            prompt_version = PROMPT_VERSION_AGNOSTIC
            query = ""
        else:
            raise ValueError(f"Unknown annotation_mode={annotation_mode!r}")

        path = Path(video_path).resolve()
        duration = video_duration_sec(path)
        uploaded = self.upload_video(path)
        try:
            video_part = self._types.Part.from_uri(
                file_uri=uploaded.uri,
                mime_type=getattr(uploaded, "mime_type", None) or "video/mp4",
            )
            video_part.video_metadata = self._types.VideoMetadata(fps=self.fps)
            contents = [video_part, prompt_builder(duration)]
            return self._generate_annotation(
                sample_id=sample_id,
                video_id=path.stem,
                video_path=str(path),
                duration=duration,
                query=query,
                options=options,
                prompt_builder=prompt_builder,
                prompt_version=prompt_version,
                annotation_mode=annotation_mode,
                contents=contents,
            )
        finally:
            self.delete_file(uploaded.name)

    def annotate_batch(
        self,
        *,
        video_path: str | Path,
        queries: list[dict[str, str]],
        annotation_mode: str = "query_conditioned",
        max_workers: int = 5,
        ttl: str = "3600s",
    ) -> list[OracleAnnotation]:
        """
        Process multiple queries for a single video in parallel using Context Caching.
        `queries` is a list of dicts with 'sample_id', 'question', and optional 'options' keys.
        """
        path = Path(video_path).resolve()
        duration = video_duration_sec(path)
        uploaded = self.upload_video(path)
        
        try:
            cache = self.create_video_cache(uploaded, ttl=ttl)
            try:
                results = [None] * len(queries)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures_to_idx = {}
                    for idx, q_dict in enumerate(queries):
                        q_text = q_dict["question"]
                        sample_id = q_dict["sample_id"]
                        q_opts = q_dict.get("options")
                        
                        if annotation_mode == "query_conditioned":
                            prompt_builder = (lambda d, qt=q_text, opt=q_opts: build_prompt(qt, d, options=opt))
                            prompt_version = PROMPT_VERSION_ORACLE + ("_mcq" if options else "")
                            query = q_text
                        elif annotation_mode == "query_agnostic":
                            prompt_builder = build_agnostic_prompt
                            prompt_version = PROMPT_VERSION_AGNOSTIC
                            query = ""
                        else:
                            raise ValueError(f"Unknown annotation_mode={annotation_mode!r}")

                        future = executor.submit(
                            self._generate_annotation,
                            sample_id=sample_id,
                            video_id=path.stem,
                            video_path=str(path),
                            duration=duration,
                            query=query,
                            options=q_opts,
                            prompt_builder=prompt_builder,
                            prompt_version=prompt_version,
                            annotation_mode=annotation_mode,
                            contents=[prompt_builder(duration)],
                            cached_content_name=cache.name,
                        )
                        futures_to_idx[future] = idx
                        
                    for future in concurrent.futures.as_completed(futures_to_idx):
                        idx = futures_to_idx[future]
                        results[idx] = future.result()
                return results
            finally:
                self.delete_cache(cache.name)
        finally:
            self.delete_file(uploaded.name)
