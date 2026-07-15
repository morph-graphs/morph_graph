"""Plotting helpers for event segmentation.

Renders (1) the full unit-level cosine self-similarity matrix with boundary
lines and (2) the novelty curve with detected peaks + shaded event spans. Useful
for eyeballing whether the TSM cuts land on real scene changes.
"""

from __future__ import annotations

import numpy as np

from .base import EventSegmentation
from .tsm import _to_numpy


def plot_segmentation(
    pooled,
    result: EventSegmentation,
    path: str,
    *,
    peak_height: float | None = None,
    normalize: bool = True,
    title: str | None = None,
) -> str:
    """Save a 2-panel figure (self-similarity + novelty) to `path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = _to_numpy(pooled)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)
    ssm = feats @ feats.T
    n = feats.shape[0]

    nov = result.novelty.astype(np.float32)
    score = nov.copy()
    if normalize:
        # z-score over the valid (non-zero-padded) region for display parity
        valid = nov[nov != 0]
        if valid.size:
            score = (nov - valid.mean()) / (valid.std() + 1e-6)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # origin="upper" -> unit 0 at the top, matrix reads top-down.
    im = ax0.imshow(ssm, origin="upper", cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02, label="cosine sim")
    for b in result.boundaries:
        ax0.axvline(b - 0.5, color="red", lw=1.2)
        ax0.axhline(b - 0.5, color="red", lw=1.2)
    ax0.set_ylabel("unit")
    ax0.set_title(title or f"TSM self-similarity ({result.n_events} events)")

    ax1.plot(score, color="steelblue", lw=1.2, label="novelty" + (" (z)" if normalize else ""))
    if result.boundaries:
        ax1.plot(result.boundaries, score[result.boundaries], "rv", ms=8, label="boundary")
    if peak_height is not None:
        ax1.axhline(peak_height, color="gray", ls="--", lw=1.0, label="threshold")
    # shade alternating event spans
    for i, (s, e) in enumerate(result.events):
        if i % 2:
            ax1.axvspan(s - 0.5, e - 0.5, color="orange", alpha=0.12)
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_xlabel("unit")
    ax1.set_ylabel("novelty")
    ax1.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
