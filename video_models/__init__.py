"""Visual feature extraction (step A): decode video, run an MLLM vision encoder,
emit per-temporal-unit patch values + pooled embeddings."""

from .base import (
    AlignedFeatureExtractor,
    VideoLLM,
    VisualFeatures,
    VisualFeatureExtractor,
    get_extractor,
    get_video_llm,
    register_extractor,
    register_video_llm,
)
from .video_io import decode_frames, sample_frame_indices

# Register bundled extractors. Each guarded so the lightweight interface still
# imports even if a given backend's heavy deps are unavailable.
for _mod in ("qwen3_vl", "siglip2", "pe_core"):
    try:  # noqa: SIM105
        __import__(f"{__name__}.{_mod}")  # side effect: register_extractor
    except Exception:  # pragma: no cover
        pass

__all__ = [
    "VisualFeatures",
    "VisualFeatureExtractor",
    "AlignedFeatureExtractor",
    "VideoLLM",
    "get_extractor",
    "get_video_llm",
    "register_extractor",
    "register_video_llm",
    "decode_frames",
    "sample_frame_indices",
]
