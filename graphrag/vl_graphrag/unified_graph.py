"""One public graph API for event-derived and pre-segmented video memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .event_graph import EventGraph
from .vl_base import VLQueryParam
from .vl_graphrag import VLGraphRAG

GraphSource = Literal["events", "segments"]


@dataclass
class UnifiedVideoGraph:
    """Select an indexing adapter while preserving one retrieval API.

    ``events`` consumes raw-video features plus TSM segmentation. ``segments``
    consumes the existing caption and/or vision segment records.  Both return
    the shared ``vl_query`` response whose records are normalized by
    :meth:`retrieval_trace`.
    """

    source: GraphSource
    graph: EventGraph | VLGraphRAG

    @classmethod
    def for_events(cls, **kwargs: Any) -> "UnifiedVideoGraph":
        return cls(source="events", graph=EventGraph(**kwargs))

    @classmethod
    def for_segments(cls, **kwargs: Any) -> "UnifiedVideoGraph":
        return cls(source="segments", graph=VLGraphRAG(**kwargs))

    def index_events(self, *args: Any, **kwargs: Any) -> None:
        if self.source != "events":
            raise ValueError("index_events requires a graph created with for_events()")
        self.graph.insert(*args, **kwargs)  # type: ignore[union-attr]

    def index_segments(
        self, *, caption_segments: Any | None = None, vision_segments: Any | None = None
    ) -> None:
        if self.source != "segments":
            raise ValueError("index_segments requires a graph created with for_segments()")
        if caption_segments is not None:
            self.graph.insert_text_nodes(caption_segments)  # type: ignore[union-attr]
        if vision_segments is not None:
            self.graph.insert_vision_nodes(vision_segments)  # type: ignore[union-attr]

    def query(self, question: str, param: VLQueryParam | None = None) -> dict[str, Any]:
        response = self.graph.query(question, param or VLQueryParam())  # type: ignore[union-attr]
        response["graph_source"] = self.source
        return response

    def answer(self, question: str, **kwargs: Any) -> dict[str, Any]:
        if self.source != "events":
            raise NotImplementedError(
                "Event-value QA is available only for event-backed graphs; "
                "use the video_qa runner for pre-segmented caption/video QA."
            )
        response = self.graph.answer(question, **kwargs)  # type: ignore[union-attr]
        response["graph_source"] = self.source
        return response

    @staticmethod
    def retrieval_trace(retrieval: dict[str, Any]) -> list[dict[str, Any]]:
        """Return JSON-safe, per-rank evidence for retrieval evaluation."""
        trace = []
        seeds = set(retrieval.get("seed_node_ids", []))
        for rank, segment in enumerate(retrieval.get("segments", []), start=1):
            node_id = segment.get("node_id")
            trace.append(
                {
                    "rank": rank,
                    "node_id": node_id,
                    "video_id": segment.get("video_id"),
                    "segment_idx": segment.get("segment_idx"),
                    "frame_range": segment.get("frame_range"),
                    "time_range_sec": segment.get("time_range_sec"),
                    "retrieval_score": segment.get("retrieval_score"),
                    "graph_source": retrieval.get("graph_source"),
                    "provenance": "seed" if node_id in seeds else "graph_expansion",
                }
            )
        return trace
