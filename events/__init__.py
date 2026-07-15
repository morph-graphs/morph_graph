"""Event segmentation over per-unit visual embeddings."""

from .base import (
    EventSegmentation,
    EventSegmenter,
    events_from_boundaries,
    get_segmenter,
    register_segmenter,
)
from .tsm import TSMEventSegmenter, checkerboard_kernel
from .viz import plot_segmentation

__all__ = [
    "EventSegmentation",
    "EventSegmenter",
    "events_from_boundaries",
    "get_segmenter",
    "register_segmenter",
    "TSMEventSegmenter",
    "checkerboard_kernel",
    "plot_segmentation",
]
