"""Debug/smoke test for step B (offline TSM event segmentation).

Runs A -> B end-to-end on a GPU node in the `morph_graphs_qwen3_cu126` env:

    python -m morph_graph.tests.debug_events \
        --video morph_graph/data/wearable-ai/egolongqa/val/00cdb8ca10c069f8.mp4 \
        --tokens-per-frame 256 --fps 1.0 --kernel-width 6 --peak-height 1.0 \
        --plot morph_graph/tests/out/seg_00cdb8ca10c069f8.png

Extracts pooled per-unit embeddings, runs the checkerboard-novelty segmenter,
and logs boundaries + event spans so you can eyeball the cuts before wiring the
key-value store.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ezcolorlog import root_logger as logger

from morph_graph.events import TSMEventSegmenter
from morph_graph.features import get_extractor

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_VIDEO = _REPO_ROOT / "data" / "wearable-ai" / "egolongqa" / "val" / "00cdb8ca10c069f8.mp4"


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test offline TSM event segmentation.")
    ap.add_argument("--video", type=Path, default=_DEFAULT_VIDEO)
    ap.add_argument("--model", default="qwen3_vl_8b")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--tokens-per-frame", type=int, default=256)
    ap.add_argument("--feature-stage", choices=["post", "pre"], default="post",
                    help="Qwen3 vision stage for TSM: post-projection (4096) or pre-merger")
    ap.add_argument("--kernel-width", type=int, default=6)
    ap.add_argument("--peak-height", type=float, default=1.0)
    ap.add_argument("--peak-distance", type=int, default=None)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--plot", type=Path, default=None, help="save a segmentation figure to this path")
    args = ap.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    # A) extract pooled per-unit embeddings
    logger.info(f"[load] extractor={args.model} tokens_per_frame={args.tokens_per_frame} stage={args.feature_stage}")
    ext = get_extractor(args.model, tokens_per_frame=args.tokens_per_frame, feature_stage=args.feature_stage)
    frames = ext.load_video(str(args.video), fps=args.fps, max_frames=args.max_frames)
    feats = ext.encode_frames(frames)
    logger.info(f"[encode] pooled={tuple(feats.pooled.shape)} n_units={feats.n_units}")

    # B) segment
    seg = TSMEventSegmenter(
        kernel_width=args.kernel_width,
        peak_height=args.peak_height,
        peak_distance=args.peak_distance,
        normalize=not args.no_normalize,
    )
    result = seg.segment(feats.pooled)

    nov = result.novelty
    logger.info(f"[novelty] min={nov.min():.3f} mean={nov.mean():.3f} max={nov.max():.3f}")
    logger.info(f"[boundaries] units={result.boundaries}")
    logger.info(f"[events] n_events={result.n_events}")
    for i, (s, e) in enumerate(result.events):
        logger.info(f"  event {i:>2}: units [{s}, {e})  len={e - s}")

    if args.plot is not None:
        from morph_graph.events import plot_segmentation

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_segmentation(
            feats.pooled,
            result,
            str(args.plot),
            peak_height=args.peak_height,
            normalize=not args.no_normalize,
            title=f"{args.video.name} ({result.n_events} events)",
        )
        logger.info(f"[plot] saved -> {args.plot}")

    # invariants
    assert result.events[0][0] == 0 and result.events[-1][1] == feats.n_units, "events must cover all units"
    assert all(result.events[i][1] == result.events[i + 1][0] for i in range(result.n_events - 1)), \
        "events must be contiguous"
    logger.info("[ok] event segmentation smoke test passed")


if __name__ == "__main__":
    main()
