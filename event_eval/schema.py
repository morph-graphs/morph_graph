"""Versioned records for query-conditioned oracle-event evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
GRANULARITIES = frozenset({"fine", "coarse"})
ANNOTATION_MODES = frozenset({"query_conditioned", "query_agnostic"})
EVENT_CATEGORIES = frozenset(
    {
        "action",
        "state_change",
        "agent_or_interaction",
        "location_or_scene",
        "goal_or_subgoal",
        "causal_transition",
        "temporal_transition",
        "unexpected_or_interruption",
    }
)
QUERY_ROLES = frozenset({"anchor", "evidence", "answer_bearing"})


@dataclass(frozen=True)
class OracleEvent:
    event_id: str
    start_sec: float
    end_sec: float
    granularity: str
    description: str
    categories: tuple[str, ...]
    query_roles: tuple[str, ...] = ()
    parent_event_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "OracleEvent":
        return cls(
            event_id=str(raw["event_id"]).strip(),
            start_sec=float(raw["start_sec"]),
            end_sec=float(raw["end_sec"]),
            granularity=str(raw["granularity"]).strip(),
            description=str(raw["description"]).strip(),
            categories=tuple(dict.fromkeys(str(value) for value in raw.get("categories", []))),
            query_roles=tuple(dict.fromkeys(str(value) for value in raw.get("query_roles", []))),
            parent_event_ids=tuple(
                dict.fromkeys(str(value) for value in raw.get("parent_event_ids", []))
            ),
        )

    def validate(
        self, duration_sec: float, *, annotation_mode: str = "query_conditioned"
    ) -> list[str]:
        errors: list[str] = []
        if not self.event_id:
            errors.append("event_id must be non-empty")
        if not self.description:
            errors.append(f"{self.event_id}: description must be non-empty")
        if self.start_sec < 0 or self.end_sec <= self.start_sec:
            errors.append(
                f"{self.event_id}: expected 0 <= start_sec < end_sec, "
                f"got [{self.start_sec}, {self.end_sec})"
            )
        if duration_sec > 0 and self.end_sec > duration_sec + 1e-3:
            errors.append(
                f"{self.event_id}: end_sec={self.end_sec} exceeds duration={duration_sec}"
            )
        if self.granularity not in GRANULARITIES:
            errors.append(
                f"{self.event_id}: unknown granularity {self.granularity!r}"
            )
        unknown_categories = set(self.categories) - EVENT_CATEGORIES
        if unknown_categories:
            errors.append(
                f"{self.event_id}: unknown categories {sorted(unknown_categories)}"
            )
        if not self.categories:
            errors.append(f"{self.event_id}: at least one category is required")
        unknown_roles = set(self.query_roles) - QUERY_ROLES
        if unknown_roles:
            errors.append(f"{self.event_id}: unknown query roles {sorted(unknown_roles)}")
        if annotation_mode == "query_conditioned" and not self.query_roles:
            errors.append(
                f"{self.event_id}: a query-conditioned oracle event needs a query role"
            )
        if annotation_mode == "query_agnostic" and self.query_roles:
            errors.append(
                f"{self.event_id}: query-agnostic events must not include query_roles"
            )
        return errors


@dataclass(frozen=True)
class OracleAnnotation:
    sample_id: str
    video_id: str
    video_path: str
    video_duration_sec: float
    query: str
    events: tuple[OracleEvent, ...]
    model: str
    prompt_version: str
    video_fps: float
    media_resolution: str | None
    generated_at: str
    options: str | None = None
    answer: str | None = None
    warnings: tuple[str, ...] = ()
    raw_response: str | None = field(default=None, repr=False)
    schema_version: str = SCHEMA_VERSION
    annotation_mode: str = "query_conditioned"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "OracleAnnotation":
        return cls(
            sample_id=str(raw["sample_id"]),
            video_id=str(raw["video_id"]),
            video_path=str(raw["video_path"]),
            video_duration_sec=float(raw["video_duration_sec"]),
            query=str(raw.get("query", "")),
            events=tuple(OracleEvent.from_dict(value) for value in raw.get("events", [])),
            model=str(raw["model"]),
            prompt_version=str(raw["prompt_version"]),
            video_fps=float(raw["video_fps"]),
            media_resolution=(
                str(raw["media_resolution"])
                if raw.get("media_resolution") is not None
                else None
            ),
            generated_at=str(raw["generated_at"]),
            options=str(raw["options"]).strip() if raw.get("options") is not None else None,
            answer=str(raw["answer"]).strip() if raw.get("answer") is not None else None,
            warnings=tuple(str(value) for value in raw.get("warnings", [])),
            raw_response=raw.get("raw_response"),
            schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
            annotation_mode=str(raw.get("annotation_mode", "query_conditioned")),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append(
                f"unsupported schema_version={self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        if self.annotation_mode not in ANNOTATION_MODES:
            errors.append(
                f"unknown annotation_mode={self.annotation_mode!r}; "
                f"expected one of {sorted(ANNOTATION_MODES)}"
            )
        if not self.sample_id or not self.video_id:
            errors.append("sample_id and video_id must be non-empty")
        if self.annotation_mode == "query_conditioned" and not self.query.strip():
            errors.append("query-conditioned annotations require a non-empty query")
        if self.annotation_mode == "query_agnostic" and self.query.strip():
            errors.append("query-agnostic annotations must leave query empty")
        if self.video_duration_sec <= 0:
            errors.append("video_duration_sec must be positive")
        if self.video_fps <= 0:
            errors.append("video_fps must be positive")
        if not self.events:
            errors.append("at least one event is required")

        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            errors.append("event_id values must be unique")
        known_ids = set(ids)
        events_by_id = {event.event_id: event for event in self.events}
        for event in self.events:
            errors.extend(
                event.validate(
                    self.video_duration_sec, annotation_mode=self.annotation_mode
                )
            )
            missing = set(event.parent_event_ids) - known_ids
            if missing:
                errors.append(
                    f"{event.event_id}: unknown parent_event_ids {sorted(missing)}"
                )
            if event.event_id in event.parent_event_ids:
                errors.append(f"{event.event_id}: event cannot parent itself")
            for parent_id in set(event.parent_event_ids) & known_ids:
                parent = events_by_id[parent_id]
                if (
                    event.start_sec < parent.start_sec - 1e-3
                    or event.end_sec > parent.end_sec + 1e-3
                ):
                    errors.append(
                        f"{event.event_id}: interval [{event.start_sec}, {event.end_sec}) "
                        f"is not contained by parent {parent_id} interval "
                        f"[{parent.start_sec}, {parent.end_sec})"
                    )
        return errors

    def require_valid(self) -> "OracleAnnotation":
        errors = self.validate()
        if errors:
            raise ValueError("Invalid oracle annotation:\n- " + "\n- ".join(errors))
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "OracleAnnotation":
        return cls.from_dict(json.loads(Path(path).read_text())).require_valid()


def gemini_response_schema(
    annotation_mode: str = "query_conditioned",
) -> dict[str, Any]:
    """JSON schema for the model-generated portion of an annotation."""
    if annotation_mode not in ANNOTATION_MODES:
        raise ValueError(f"Unknown annotation_mode={annotation_mode!r}")
    event_properties: dict[str, Any] = {
        "event_id": {"type": "string"},
        "start_sec": {"type": "number"},
        "end_sec": {"type": "number"},
        "granularity": {
            "type": "string",
            "enum": sorted(GRANULARITIES),
        },
        "description": {"type": "string"},
        "categories": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": sorted(EVENT_CATEGORIES),
            },
        },
        "parent_event_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    }
    required = [
        "event_id",
        "start_sec",
        "end_sec",
        "granularity",
        "description",
        "categories",
        "parent_event_ids",
    ]
    if annotation_mode == "query_conditioned":
        event_properties["query_roles"] = {
            "type": "array",
            "items": {
                "type": "string",
                "enum": sorted(QUERY_ROLES),
            },
        }
        required.append("query_roles")

    schema_properties: dict[str, Any] = {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": event_properties,
                "required": required,
                "additionalProperties": False,
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    }
    schema_required = ["events", "warnings"]

    if annotation_mode == "query_conditioned":
        schema_properties["answer"] = {
            "type": "string",
            "description": "Direct, concise answer to the question based on the identified events.",
        }
        schema_required.append("answer")

    return {
        "type": "object",
        "properties": schema_properties,
        "required": schema_required,
        "additionalProperties": False,
    }
