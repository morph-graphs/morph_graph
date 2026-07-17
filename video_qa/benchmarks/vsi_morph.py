"""VSI-Morph benchmark adapter for event-backed or pre-segmented QA."""

from __future__ import annotations

import json
from pathlib import Path

from .base import VideoQABenchmark, VideoQASample

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTIONS = _REPO_ROOT / "vsi_output" / "vsi_events_mc_poc.json"
DEFAULT_VIDEO_DIR = _REPO_ROOT / "vsi_output" / "vsi_constructed"


class VSIMorphBenchmark(VideoQABenchmark):
    name = "vsi_morph"

    def __init__(
        self,
        question_path: Path = DEFAULT_QUESTIONS,
        video_dir: Path = DEFAULT_VIDEO_DIR,
    ) -> None:
        self.question_path = question_path
        self.video_dir = video_dir

    @property
    def options_are_mcq(self) -> bool:
        return True

    def load(self) -> list[VideoQASample]:
        with self.question_path.open() as handle:
            rows = json.load(handle)
        samples = []
        for row in rows:
            rel_path = f"{row['object'].replace(' ', '_')}/{row['video_uid']}.mp4"
            samples.append(
                VideoQASample(
                    sample_id=str(row["qid"]),
                    video_path=(self.video_dir / rel_path).resolve(),
                    question=str(row["question"]),
                    ground_truth=str(row["correct_answer"]).upper(),
                    options={str(k): str(v) for k, v in row.get("options", {}).items()},
                    metadata={
                        "qid": row["qid"],
                        "video_uid": row["video_uid"],
                        "object": row["object"],
                        "question_type": row.get("question_type"),
                        "clue_frames": row.get("clue_frames"),
                    },
                )
            )
        return samples
