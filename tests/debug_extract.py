"""Debug/smoke test for step A (visual feature extraction).

Run on a GPU node in the `morph_graphs_qwen3_cu126` env:

    python -m morph_graph.tests.debug_extract \
        --video morph_graph/data/wearable-ai/egolongqa/val/00cdb8ca10c069f8.mp4 \
        --tokens-per-frame 256 --fps 1.0

Logs shapes, checks invariants, and reports adjacent-unit similarity so you can
eyeball whether the pooled embeddings look sane before wiring TSM.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch.nn.functional as F
from ezcolorlog import root_logger as logger

from morph_graph.features import get_extractor

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_VIDEO = _REPO_ROOT / "data" / "wearable-ai" / "egolongqa" / "val" / "00cdb8ca10c069f8.mp4"


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test the visual feature extractor.")
    ap.add_argument("--video", type=Path, default=_DEFAULT_VIDEO)
    ap.add_argument("--model", default="qwen3_vl_8b")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--tokens-per-frame", type=int, default=256)
    args = ap.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    logger.info(f"[load] extractor={args.model} tokens_per_frame={args.tokens_per_frame}")
    ext = get_extractor(args.model, tokens_per_frame=args.tokens_per_frame)

    # 1) decode
    frames = ext.load_video(str(args.video), fps=args.fps, max_frames=args.max_frames)
    logger.info(f"[decode] frames={tuple(frames.shape)} dtype={frames.dtype}")

    # 2) encode
    feats = ext.encode_frames(frames)
    pt = feats.patch_tokens
    logger.info(f"[encode] pooled={tuple(feats.pooled.shape)} "
                f"patch_tokens={tuple(pt.shape)} grid_hw={feats.grid_hw} dim={feats.dim}")

    # 3) invariants
    n_frames = int(frames.shape[0])
    exp_units = math.ceil(n_frames / ext.temporal_patch_size)
    tokens_per_unit = feats.grid_hw[0] * feats.grid_hw[1]
    assert feats.pooled.shape == (feats.n_units, feats.dim), "pooled shape mismatch"
    assert tuple(pt.shape) == (feats.n_units, tokens_per_unit, feats.dim), "patch_tokens shape mismatch"
    assert feats.n_units == exp_units, f"n_units {feats.n_units} != ceil(frames/2) {exp_units}"
    logger.info(f"[check] n_units={feats.n_units} (expected {exp_units}), "
                f"tokens_per_unit={tokens_per_unit} (requested ~{args.tokens_per_frame})")

    # 4) adjacent-unit cosine similarity sanity check
    if feats.n_units >= 2:
        p = feats.pooled.float()
        adj = F.cosine_similarity(p[:-1], p[1:], dim=1)
        logger.info(f"[adj-sim] min={adj.min():.3f} mean={adj.mean():.3f} max={adj.max():.3f}")

    logger.info("[ok] extraction smoke test passed")


if __name__ == "__main__":
    main()
