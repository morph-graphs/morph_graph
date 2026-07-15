"""Offline TSM (temporal self-similarity) event segmentation.

Foote-style checkerboard novelty on per-unit embeddings. Faithful to the online
detector in matching_tsme_online.py (same kernel + novelty), but computed in one
offline pass over all units so it's easy to validate/visualize. The streaming
version reuses the identical novelty; only buffering/peak bookkeeping differ.

Boundaries here are in *unit* indices (each unit = one Qwen3-VL temporal unit,
i.e. 2 frames), so `kernel_width`/`peak_distance` are measured in units.
"""

from __future__ import annotations

import numpy as np
from ezcolorlog import root_logger as logger
from scipy.signal import find_peaks

from .base import EventSegmentation, EventSegmenter, events_from_boundaries, register_segmenter


def checkerboard_kernel(width: int, sigma: float = 0.5) -> np.ndarray:
    """Gaussian-tapered checkerboard kernel of size `(2*width, 2*width)`."""
    cb = np.kron(np.array([[1, -1], [-1, 1]]), np.ones((width, width)))
    axis = np.linspace(-1, 1, width * 2)
    x, y = np.meshgrid(axis, axis)
    taper = np.exp(-(x**2 + y**2) / sigma)
    return cb * taper


def _to_numpy(pooled) -> np.ndarray:
    if hasattr(pooled, "detach"):
        pooled = pooled.detach().cpu().float().numpy()
    return np.asarray(pooled, dtype=np.float32)


class TSMEventSegmenter(EventSegmenter):
    def __init__(
        self,
        *,
        kernel_width: int = 6,        # half-window, in units
        peak_height: float = 1.0,     # z-score threshold if normalize else raw
        peak_distance: int | None = None,  # min units between boundaries (default kernel_width)
        normalize: bool = True,       # z-score the novelty for adaptive height
        gaussian_sigma: float = 0.5,
    ):
        self.kernel_width = int(kernel_width)
        self.peak_height = float(peak_height)
        self.peak_distance = int(peak_distance or kernel_width)
        self.normalize = bool(normalize)
        self.kernel = checkerboard_kernel(self.kernel_width, gaussian_sigma)

    @property
    def name(self) -> str:
        return "tsm_offline"

    def novelty_curve(self, pooled) -> np.ndarray:
        """Per-unit checkerboard novelty; 0 where the window doesn't fit."""
        feats = _to_numpy(pooled)
        n = feats.shape[0]
        w = self.kernel_width
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)

        novelty = np.zeros(n, dtype=np.float32)
        # Valid centers need a full 2w window: [c-w : c+w].
        for c in range(w, n - w):
            window = feats[c - w : c + w]
            S = window @ window.T          # local (2w x 2w) self-similarity
            novelty[c] = float((S * self.kernel).sum())
        return novelty

    def segment(self, pooled) -> EventSegmentation:
        feats = _to_numpy(pooled)
        n = feats.shape[0]
        w = self.kernel_width

        # Too short to score any window -> single event.
        if n < 2 * w + 1:
            logger.info(f"[tsm] n_units={n} < 2*kernel_width+1={2 * w + 1}; single event")
            return EventSegmentation([], [(0, n)], np.zeros(n, dtype=np.float32), n)

        novelty = self.novelty_curve(feats)

        score = novelty.copy()
        if self.normalize:
            valid = score[w : n - w]
            mu, sigma = float(valid.mean()), float(valid.std())
            score = (score - mu) / (sigma + 1e-6)

        peaks, _ = find_peaks(score, height=self.peak_height, distance=self.peak_distance)
        boundaries = [int(p) for p in peaks]
        events = events_from_boundaries(boundaries, n)
        logger.info(
            f"[tsm] n_units={n} kernel_width={w} -> {len(boundaries)} boundaries, {len(events)} events"
        )
        return EventSegmentation(boundaries, events, novelty, n)


register_segmenter("tsm_offline", TSMEventSegmenter)
