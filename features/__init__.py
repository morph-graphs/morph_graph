"""Visual feature extraction (step A): decode video, run an MLLM vision encoder,
emit per-temporal-unit patch values + pooled embeddings."""

from .base import (
    AlignedFeatureExtractor,
    VisualFeatures,
    VisualFeatureExtractor,
    get_extractor,
    register_extractor,
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
    "get_extractor",
    "register_extractor",
    "decode_frames",
    "sample_frame_indices",
]
