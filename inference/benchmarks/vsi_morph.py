"""VSI-Morph MCQ benchmark adapter.

Caption index: per-video row from precomputed segment captions JSON.
Vision index: scene segments from per-video annotation JSON ({video}.json),
with optional caption merge by frame_range.
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from ezcolorlog import root_logger as logger

from morph_graph.graphrag.vl_graphrag.vl_segments import parse_caption_row
from morph_graph.inference.framework.base import (
    EvalConfig,
    EvalSample,
    IndexInputs,
    VLGraphRAGBenchmark,
    base_result_row,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
VSI_VIDEO_DIR = REPO_ROOT / "vsi_output" / "vsi_constructed"
DEFAULT_QUESTIONS = REPO_ROOT / "vsi_output" / "vsi_events_mc_poc.json"
DEFAULT_CAPTIONS = REPO_ROOT / "outputs" / "per_video_segment_captions_qwen3_vl_8b.json"
_SCENE_END = re.compile(r"^(?:end\s+of\s+|end\s+)", re.IGNORECASE)


def _load_json_list(path: str | Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    return data


def _vision_segments(video_path: str, caption_row: dict | None) -> list[dict]:
    """Build vision-index segments from scene events, not from the caption blob alone."""
    ann_path = Path(video_path).with_suffix(".json")
    if not ann_path.exists():
        logger.warning(f"Missing annotation for vision index: {ann_path}")
        return []

    ann = json.loads(ann_path.read_text())
    fps = float(ann.get("fps", 1))
    # Merge captions by frame_range when available.
    captions = {
        (int(s["frame_range"][0]), int(s["frame_range"][1])): s.get("caption", "")
        for s in parse_caption_row(caption_row or {})
        if s.get("frame_range") is not None
    }

    segments, start = [], 0
    for idx, event in enumerate(e for e in ann.get("annotations", []) if e.get("type") == "scene"):
        end = int(event["frame_number"])
        if end <= start:
            continue
        label = _SCENE_END.sub("", str(event.get("label", "")).strip()).strip() or "unknown"
        frame_range = [start, end]
        caption = captions.get((start, end), "")
        segments.append(
            {
                "segment_idx": idx,
                "scene_label": label,
                "frame_range": frame_range,
                "time_range_sec": [start / fps, end / fps],
                "video_path": video_path,
                "caption": caption,
            }
        )
        start = end
    return segments


def _clue_hit_rate(segments: list[dict], clue_frames) -> float | None:
    """Fraction of clue frames covered by at least one retrieved segment."""
    clues = [int(f) for group in (clue_frames or []) for f in (group if isinstance(group, list) else [group])]
    if not clues:
        return None
    hits = sum(
        1
        for clue in clues
        if any(
            int(s["frame_range"][0]) <= clue < int(s["frame_range"][1])
            for s in segments
            if s.get("frame_range") is not None
        )
    )
    return hits / len(clues)


class VSIMorphBenchmark(VLGraphRAGBenchmark):
    name = "vsi_morph"

    def __init__(self) -> None:
        self._captions: dict[str, dict] = {}
        self._vision: dict[str, list[dict]] = {}

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--question_path", type=str, default=str(DEFAULT_QUESTIONS))
        parser.add_argument("--captions_path", type=str, default=str(DEFAULT_CAPTIONS))
        parser.add_argument("--video_dir", type=str, default=str(VSI_VIDEO_DIR))

    def load_samples(self, args: Namespace) -> list[EvalSample]:
        required = {"qid", "video_uid", "object", "question", "options", "correct_answer"}
        samples = []
        for i, item in enumerate(_load_json_list(args.question_path)):
            missing = required - set(item)
            if missing:
                raise ValueError(f"{args.question_path}[{i}]: missing {sorted(missing)}")
            samples.append(
                EvalSample(
                    sample_id=str(item["qid"]),
                    question=item["question"],
                    ground_truth=str(item["correct_answer"]).upper(),
                    raw=item,
                )
            )

        self._captions = {
            str(Path(row["video_path"]).resolve()): row
            for row in _load_json_list(args.captions_path)
            if row.get("video_path")
        }
        self._vision = {}
        logger.info(f"Loaded {len(samples)} questions, {len(self._captions)} caption videos")
        return samples

    def get_video_path(self, sample: EvalSample, args: Namespace) -> str:
        rel = f"{sample.raw['object'].replace(' ', '_')}/{sample.raw['video_uid']}.mp4"
        return str((Path(args.video_dir) / rel).resolve())

    def get_index_inputs(self, sample: EvalSample, args: Namespace) -> IndexInputs:
        video_path = self.get_video_path(sample, args)
        if video_path not in self._vision:
            self._vision[video_path] = _vision_segments(video_path, self._captions.get(video_path))
        vision = self._vision[video_path] or None
        return IndexInputs(caption=self._captions.get(video_path), vision=vision)

    def build_result_row(
        self,
        sample: EvalSample,
        video_path: str,
        retrieval: dict[str, Any],
        config: EvalConfig,
    ) -> dict[str, Any]:
        segments = retrieval.get("segments", [])
        raw = sample.raw
        return {
            **base_result_row(sample, video_path, retrieval, config),
            "qid": raw["qid"],
            "video_uid": raw["video_uid"],
            "object": raw["object"],
            "question_type": raw.get("question_type"),
            "options": raw["options"],
            "clue_hit_rate": _clue_hit_rate(segments, raw.get("clue_frames")),
        }

    def aggregate_retrieval_metrics(self, results: list[dict]) -> dict[str, Any]:
        rates = [r["clue_hit_rate"] for r in results if r.get("clue_hit_rate") is not None]
        if not rates:
            return {}
        mean = sum(rates) / len(rates)
        logger.info(f"Mean clue frame hit rate: {mean:.3f} ({len(rates)} questions)")
        return {"mean_clue_hit_rate": mean, "num_questions_with_clues": len(rates)}
