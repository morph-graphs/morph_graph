"""Dataset-neutral records for the production video-QA runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoQASample:
    sample_id: str
    video_path: Path
    question: str
    ground_truth: str | None = None
    options: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def qa_prompt(self) -> str:
        if not self.options:
            return self.question
        choices = "\n".join(f"{key}. {value}" for key, value in sorted(self.options.items()))
        return f"{self.question.strip()}\nOptions:\n{choices}\nAnswer with the option letter only."


class VideoQABenchmark(ABC):
    name: str

    @abstractmethod
    def load(self) -> list[VideoQASample]:
        """Load all question/video pairs for this benchmark."""

    def result_fields(self, sample: VideoQASample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "video_path": str(sample.video_path),
            "correct_answer": sample.ground_truth,
            "options": sample.options,
            **sample.metadata,
        }

    def parse_prediction(self, answer: str) -> str:
        return answer.split()[0].rstrip(".").upper() if self.options_are_mcq else answer.strip()

    @property
    def options_are_mcq(self) -> bool:
        return False
