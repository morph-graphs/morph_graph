"""Model-agnostic interface for extracting visual features from an MLLM.

Unit of extraction = one temporal unit (for Qwen3-VL, 2 frames fused by the
encoder's `temporal_patch_size`). Each unit yields patch-level `values` and a
mean-pooled embedding; downstream TSM/event code consumes `pooled`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class VisualFeatures:
    # Patch-level "values": (n_units, tokens_per_unit, dim). List form allowed
    # when per-unit token counts differ (variable frame resolution).
    patch_tokens: torch.Tensor | list[torch.Tensor]
    # Mean-pooled per-unit embedding fed to TSM: (n_units, dim).
    pooled: torch.Tensor
    # Merged-token spatial grid (h // merge, w // merge) per unit.
    grid_hw: tuple[int, int]
    dim: int
    model_name: str
    # Qwen3-VL's intermediate visual streams injected into early LLM layers.
    # Kept token-aligned with ``patch_tokens`` when available.
    deepstack_features: list[torch.Tensor] | None = None

    @property
    def n_units(self) -> int:
        return int(self.pooled.shape[0])


class VisualFeatureExtractor(ABC):
    """Extract per-temporal-unit visual features from a single video."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def feature_dim(self) -> int: ...

    @property
    @abstractmethod
    def temporal_patch_size(self) -> int: ...

    @property
    def supports_text_query(self) -> bool:
        """True if `pooled` lives in a text-aligned space (see AlignedFeatureExtractor)."""
        return False

    @abstractmethod
    def encode_frames(self, frames: torch.Tensor) -> VisualFeatures:
        """Encode decoded frames `(T, C, H, W)` uint8 into `VisualFeatures`."""

    def load_video(
        self,
        video_path: str,
        *,
        fps: float = 1.0,
        max_frames: int | None = None,
    ) -> torch.Tensor:
        """Decode + sample frames `(T, C, H, W)` uint8. Kept separate from
        encoding so callers can decode once and reuse."""
        from .video_io import decode_frames

        frames, _ = decode_frames(video_path, fps=fps, max_frames=max_frames)
        return frames

    def encode_video(
        self,
        video_path: str,
        *,
        fps: float = 1.0,
        max_frames: int | None = None,
    ) -> VisualFeatures:
        frames = self.load_video(video_path, fps=fps, max_frames=max_frames)
        return self.encode_frames(frames)


class VideoLLM(ABC):
    """Answer questions from retrieved model-specific event values.

    Implementations own their stored-value schema and any prompt/model-specific
    generation details. Retrieval code passes plain question text plus ordered
    event values and never depends on a particular VLM's token format.
    """

    @property
    @abstractmethod
    def visual_extractor(self) -> VisualFeatureExtractor:
        """The extractor that defines this model's persisted event-value schema."""

    @property
    def name(self) -> str:
        return self.visual_extractor.name

    def encode_frames(self, frames: torch.Tensor, **kwargs) -> VisualFeatures:
        """Extract values through the model-owned visual extractor."""
        return self.visual_extractor.encode_frames(frames, **kwargs)

    @abstractmethod
    def answer_from_event_values(
        self,
        question: str,
        event_values: list[dict],
        *,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict]:
        """Return ``(answer, qa_metadata)`` for retrieved event values."""


class AlignedFeatureExtractor(VisualFeatureExtractor):
    """A CLIP-family extractor whose `pooled` embeddings live in a joint,
    contrastively-aligned image/text space (SigLIP2, PE-Core).

    Enables text-query retrieval (step E): index events by pooled image
    embeddings and seed the search with `encode_text(query)` in the same space.
    Unlike the MLLM extractors, these are per-frame (temporal_patch_size == 1)
    and `pooled` is L2-normalized for cosine similarity.
    """

    @property
    def supports_text_query(self) -> bool:
        return True

    @property
    def temporal_patch_size(self) -> int:
        return 1

    @abstractmethod
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text queries into the joint space: `(len(texts), dim)`, L2-normalized."""


# --- registry: any MLLM plugs in under a string key ---------------------------

_EXTRACTORS: dict[str, Callable[..., VisualFeatureExtractor]] = {}
_VIDEO_LLMS: dict[str, Callable[..., VideoLLM]] = {}


def register_extractor(name: str, factory: Callable[..., VisualFeatureExtractor]) -> None:
    _EXTRACTORS[name] = factory


def get_extractor(name: str, **kwargs) -> VisualFeatureExtractor:
    if name not in _EXTRACTORS:
        raise KeyError(f"Unknown extractor {name!r}. Registered: {sorted(_EXTRACTORS)}")
    return _EXTRACTORS[name](**kwargs)


def register_video_llm(name: str, factory: Callable[..., VideoLLM]) -> None:
    _VIDEO_LLMS[name] = factory


def get_video_llm(name: str, **kwargs) -> VideoLLM:
    if name not in _VIDEO_LLMS:
        raise KeyError(f"Unknown VideoLLM {name!r}. Registered: {sorted(_VIDEO_LLMS)}")
    return _VIDEO_LLMS[name](**kwargs)
