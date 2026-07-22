"""Explicit registry for supported production video-QA benchmarks."""

from __future__ import annotations

from pathlib import Path

from .base import VideoQABenchmark
from .egolongqa import EgoLongQABenchmark
from .vsi_morph import VSIMorphBenchmark

BENCHMARKS = {
    VSIMorphBenchmark.name: VSIMorphBenchmark,
    EgoLongQABenchmark.name: EgoLongQABenchmark,
}


def get_benchmark(
    name: str, *, question_path: Path | None = None, video_dir: Path | None = None
) -> VideoQABenchmark:
    try:
        cls = BENCHMARKS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}") from exc
    kwargs = {}
    if question_path is not None:
        kwargs["question_path"] = question_path
    if video_dir is not None:
        kwargs["video_dir"] = video_dir
    return cls(**kwargs)


__all__ = ["BENCHMARKS", "VideoQABenchmark", "get_benchmark"]
