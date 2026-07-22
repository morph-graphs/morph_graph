"""EgoLongQA adapter using the official validation JSONL."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .base import VideoQABenchmark, VideoQASample

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTIONS = (
    _REPO_ROOT
    / "data"
    / "wearable-ai"
    / "egolongqa"
    / "wearable_ai_2026_egolongqa_val_700.jsonl"
)
DEFAULT_VIDEO_DIR = _REPO_ROOT / "data" / "wearable-ai" / "egolongqa"
_OPTION = re.compile(r"(?:^|\s)([A-Z])\.\s*")


def _parse_options(raw: object) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {
            item.split(".", 1)[0].strip(): item.split(".", 1)[-1].strip()
            for item in raw
        }
    if isinstance(raw, str):
        matches = list(_OPTION.finditer(raw))
        return {
            match.group(1): raw[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None].strip()
            for index, match in enumerate(matches)
        }
    return {}


class EgoLongQABenchmark(VideoQABenchmark):
    name = "egolongqa"

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
        samples = []
        with self.question_path.open() as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_path = Path(row["video_path"])
                video_path = raw_path if raw_path.is_absolute() else self.video_dir / raw_path
                if not video_path.exists() and not raw_path.is_absolute():
                    video_path = self.video_dir / "val" / raw_path
                options = _parse_options(row.get("mcq_options", row.get("options", {})))
                samples.append(
                    VideoQASample(
                        sample_id=str(row.get("qid", row.get("id", index))),
                        video_path=video_path.resolve(),
                        question=str(row["question"]),
                        ground_truth=(
                            str(row.get("correct_answer", row.get("mcq_answer"))).upper()
                            if row.get("correct_answer", row.get("mcq_answer")) is not None
                            else None
                        ),
                        options=options,
                        metadata={
                            key: value
                            for key, value in row.items()
                            if key
                            not in {"video_path", "question", "correct_answer", "options", "mcq_options"}
                        },
                    )
                )
        return samples
