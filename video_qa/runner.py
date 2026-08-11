"""Batch-capable EventGraph QA runner used by local and Slurm execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ezcolorlog import root_logger as logger

from morph_graph.events import TSMEventSegmenter
from morph_graph.video_models import get_extractor, get_video_llm
from morph_graph.video_models.video_io import decode_frames
from morph_graph.graphrag.vl_graphrag import UnifiedVideoGraph, VLQueryParam
from morph_graph.graphrag.vl_graphrag.vl_segments import _video_id_from_path

from .benchmarks.base import VideoQABenchmark, VideoQASample


@dataclass
class VideoQAConfig:
    fps: float = 1.0
    max_input_frames: int | None = None
    frame_batch_size: int | None = 128
    tokens_per_frame: int = 256
    feature_stage: str = "post"
    aligned_model: str = "pe_core_vit_l"
    kernel_width: int = 6
    peak_height: float = 1.0
    semantic_similarity_threshold: float = 0.9
    top_k: int = 3
    max_segments_for_qa: int = 5
    max_qa_frames: int = 128
    max_new_tokens: int = 128
    retrieval_only: bool = False
    retrieval_mode: str = "local"
    qafd_alpha: float = 10.0
    qafd_epsilon: float = 1e-4
    qafd_step_size: float = 0.2
    qafd_max_iterations: int = 500
    qafd_weight_scheme: str = "multiply"
    sim_mode: str = "normalized"


class VideoQARunner:
    """Process ordered samples while indexing each video exactly once per worker."""

    def __init__(self, config: VideoQAConfig, cache_dir: Path) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self._graphs: dict[str, UnifiedVideoGraph] = {}
        self._video_llm = None
        self._aligned = None

    def _load_models(self) -> None:
        if self._video_llm is None:
            self._video_llm = get_video_llm(
                "qwen3_vl_8b",
                tokens_per_frame=self.config.tokens_per_frame,
                feature_stage=self.config.feature_stage,
            )
            self._aligned = get_extractor(self.config.aligned_model)

    def _graph_for(self, sample: VideoQASample) -> UnifiedVideoGraph:
        key = str(sample.video_path)
        if key in self._graphs:
            return self._graphs[key]
        if not sample.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {sample.video_path}")
        self._load_models()
        assert self._video_llm is not None and self._aligned is not None

        frames, frame_indices = decode_frames(
            str(sample.video_path), fps=self.config.fps, max_frames=self.config.max_input_frames
        )
        features = self._video_llm.encode_frames(
            frames, frame_batch_size=self.config.frame_batch_size
        )
        segmentation = TSMEventSegmenter(
            kernel_width=self.config.kernel_width, peak_height=self.config.peak_height
        ).segment(features.pooled)
        aligned_pooled = self._aligned.encode_frames(frames).pooled
        video_id = _video_id_from_path(str(sample.video_path))
        graph = UnifiedVideoGraph.for_events(
            embedding_dim=int(aligned_pooled.shape[-1]),
            text_encoder=self._aligned.encode_text,
            working_dir=str(self.cache_dir / video_id),
            semantic_similarity_threshold=self.config.semantic_similarity_threshold,
            clear_working_dir=True,
        )
        graph.index_events(
            video_id,
            str(sample.video_path),
            features,
            segmentation,
            frame_indices,
            aligned_pooled=aligned_pooled,
            fps=self.config.fps,
        )
        logger.info(
            f"[video_qa] indexed {video_id}: {segmentation.n_events} events "
            f"from {len(frame_indices)} frames"
        )
        self._graphs[key] = graph
        return graph

    def run_sample(
        self, sample: VideoQASample, benchmark: VideoQABenchmark
    ) -> dict[str, Any]:
        graph = self._graph_for(sample)
        param = VLQueryParam(
            mode=self.config.retrieval_mode,
            top_k=self.config.top_k,
            max_segments_for_qa=self.config.max_segments_for_qa,
            max_frames_for_video_unit=self.config.max_qa_frames,
            qafd_alpha=self.config.qafd_alpha,
            qafd_epsilon=self.config.qafd_epsilon,
            qafd_step_size=self.config.qafd_step_size,
            qafd_max_iterations=self.config.qafd_max_iterations,
            qafd_weight_scheme=self.config.qafd_weight_scheme,
            sim_mode=self.config.sim_mode,
        )
        if self.config.retrieval_only:
            response = graph.query(sample.question, param)
            answer, qa_error, stitch_log = "", None, []
        else:
            assert self._video_llm is not None
            response = graph.answer(
                sample.question,
                param=param,
                qa_model=self._video_llm,
                max_new_tokens=self.config.max_new_tokens,
                qa_prompt=sample.qa_prompt,
            )
            answer = str(response.get("answer", ""))
            qa_error = response.get("qa_error")
            stitch_log = response.get("stitch_log", [])

        parsed = benchmark.parse_prediction(answer)
        correct = (
            parsed == sample.ground_truth
            if sample.ground_truth is not None and benchmark.options_are_mcq
            else None
        )
        return {
            **benchmark.result_fields(sample),
            "graph_source": "events",
            "answer": answer,
            "parsed_answer": parsed,
            "correct": correct,
            "qa_error": qa_error,
            "seed_node_ids": response.get("seed_node_ids", []),
            "selected_node_ids": response.get("selected_node_ids", []),
            "retrieval_trace": UnifiedVideoGraph.retrieval_trace(response),
            "stitch_log": stitch_log,
        }

    def run(
        self,
        benchmark: VideoQABenchmark,
        samples: list[VideoQASample],
        output_path: Path,
    ) -> list[dict[str, Any]]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict[str, Any]] = []
        completed: set[str] = set()
        if output_path.exists():
            existing = json.loads(output_path.read_text())
            completed = {str(row["sample_id"]) for row in existing if row.get("sample_id")}
        results = existing
        for sample in samples:
            if sample.sample_id in completed:
                continue
            results.append(self.run_sample(sample, benchmark))
            output_path.write_text(json.dumps(results, indent=2))
        return results
