"""Shared types and benchmark adapter interface for VL-GraphRAG eval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class EvalSample:
    """One benchmark question with ground truth and dataset-specific raw fields."""

    sample_id: str
    question: str
    ground_truth: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexInputs:
    """Per-video data passed to VLGraphRAG indexing.

    caption: input for insert_text_nodes (caption-key VDB)
    vision: segment list for insert_vision_nodes (video-key VDB)
        - requires both `video_path` and `frame_range` for each segment
    """

    caption: Any | None = None  # insert_text_nodes
    vision: Any | None = None  # insert_vision_nodes

    def required_for(self, key_source: str) -> Any | None:
        """Return the index input needed for the active retrieval key_source."""
        return self.caption if key_source == "caption" else self.vision


@dataclass
class EvalConfig:
    """Run configuration shared across benchmarks."""

    benchmark: str
    output_dir: Path
    rag_cache_dir: Path
    retrieval_mode: str = "naive"
    key_source: Literal["caption", "video"] = "caption"
    top_k: int = 3
    max_segments_for_qa: int = 5
    max_frames_for_video_unit: int = 128
    qa_mode: Literal["caption", "video"] = "caption"
    semantic_similarity_threshold: float = 0.5
    embedding_batch_num: int = 32
    embedding_model: str | None = None  # registry key, not a checkpoint path
    retrieval_only: bool = False
    reindex: bool = False
    model_path: str | None = None
    max_new_tokens: int = 32
    max_samples: int | None = None

    @classmethod
    def from_args(cls, args: Namespace, benchmark_name: str) -> EvalConfig:
        return cls(
            benchmark=benchmark_name,
            output_dir=Path(args.output_dir),
            rag_cache_dir=Path(args.rag_cache_dir),
            retrieval_mode=args.retrieval_mode,
            key_source=args.key_source,
            top_k=args.top_k,
            max_segments_for_qa=args.max_segments_for_qa,
            max_frames_for_video_unit=args.max_frames_for_video_unit,
            qa_mode=args.qa_mode,
            semantic_similarity_threshold=args.semantic_similarity_threshold,
            embedding_batch_num=args.embedding_batch_num,
            embedding_model=args.embedding_model,
            retrieval_only=args.retrieval_only,
            reindex=args.reindex,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
        )


def parse_mcq_prediction(pred: str) -> str:
    """Extract the leading option letter from a VLM MCQ response."""
    return pred.split()[0].rstrip(".").strip().upper() if pred.strip() else ""


def base_result_row(
    sample: EvalSample,
    video_path: str,
    retrieval: dict[str, Any],
    config: EvalConfig,
) -> dict[str, Any]:
    """Common retrieval fields written to every result row."""
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "correct_answer": sample.ground_truth,
        "video_path": video_path,
        "retrieval_mode": config.retrieval_mode,
        "key_source": config.key_source,
        "top_k": config.top_k,
        "max_segments_for_qa": config.max_segments_for_qa,
        "seed_node_ids": retrieval.get("seed_node_ids", []),
        "selected_node_ids": retrieval.get("selected_node_ids", []),
        "segments": retrieval.get("segments", []),
        "caption_context": retrieval.get("caption_context", ""),
    }


def compute_accuracy_breakdown(
    results: list[dict],
    group_field: str = "question_type",
) -> dict[str, dict[str, float | int]]:
    """Per-group MCQ accuracy (default grouping: question_type)."""
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in results:
        group = row.get(group_field) or "unknown"
        by_group[group]["total"] += 1
        if row.get("correct") is True:
            by_group[group]["correct"] += 1
    return {
        group: {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
        }
        for group, counts in sorted(by_group.items())
    }


class VLGraphRAGBenchmark(ABC):
    """Dataset adapter: load samples, provide per-video index data, optional metrics.

    Subclass to add a new benchmark and register it in benchmarks/__init__.py REGISTRY.
    Override only what differs from the defaults below.
    """

    name: str

    @abstractmethod
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add benchmark-specific CLI flags."""

    @abstractmethod
    def load_samples(self, args: Namespace) -> list[EvalSample]:
        """Load questions and any auxiliary data (captions, annotations, etc.)."""

    @abstractmethod
    def get_video_path(self, sample: EvalSample, args: Namespace) -> str:
        """Resolve the on-disk video path for a sample."""

    @abstractmethod
    def get_index_inputs(self, sample: EvalSample, args: Namespace) -> IndexInputs:
        """Return caption and/or vision inputs for VLGraphRAG indexing."""

    def format_options(self, sample: EvalSample) -> list[str]:
        """Format MCQ options for the VLM prompt."""
        options = sample.raw.get("options", {})
        return [f"{k}. {v}" for k, v in sorted(options.items())]

    def build_result_row(
        self,
        sample: EvalSample,
        video_path: str,
        retrieval: dict[str, Any],
        config: EvalConfig,
    ) -> dict[str, Any]:
        return base_result_row(sample, video_path, retrieval, config)

    def parse_prediction(self, pred: str) -> str:
        return parse_mcq_prediction(pred)

    def compute_breakdown(self, results: list[dict]) -> dict[str, dict[str, float | int]]:
        return compute_accuracy_breakdown(results)

    def aggregate_retrieval_metrics(self, results: list[dict]) -> dict[str, Any]:
        """Optional benchmark-specific retrieval metrics (e.g. clue hit rate)."""
        return {}

    def results_filename(
        self,
        config: EvalConfig,
        embedding_model: str,
        model_key: str | None,
    ) -> str:
        stem = f"results_{self.name}_graph_rag_{config.retrieval_mode}_{embedding_model}"
        if config.retrieval_only:
            stem += "_retrieval"
        elif model_key:
            stem += f"_{model_key}"
        return f"{stem}.json"
