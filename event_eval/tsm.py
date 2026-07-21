"""Export TSM segmentations on an exact seconds timeline for oracle comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _duration(metadata: Any) -> float:
    duration = getattr(metadata, "duration_seconds", None)
    if (duration is None or duration <= 0) and metadata.average_fps:
        duration = metadata.num_frames / metadata.average_fps
    if duration is None or duration <= 0:
        raise ValueError("Could not determine video duration")
    return float(duration)


def _sample_position_to_sec(
    position: int,
    frame_indices: list[int],
    *,
    source_fps: float | None,
    duration_sec: float,
    requested_fps: float,
) -> float:
    if position <= 0:
        return 0.0
    if position >= len(frame_indices):
        return duration_sec
    if source_fps is not None and source_fps > 0:
        return min(duration_sec, frame_indices[position] / source_fps)
    return min(duration_sec, position / requested_fps)


def export_tsm(
    video_path: str | Path,
    *,
    model: str = "qwen3_vl_8b",
    fps: float = 1.0,
    max_frames: int | None = None,
    frame_batch_size: int | None = 128,
    tokens_per_frame: int = 256,
    feature_stage: str = "post",
    kernel_width: int = 6,
    peak_height: float = 1.0,
    peak_distance: int | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """Run the deployed TSM path and serialize boundaries using source times."""
    import torch

    from morph_graph.events import TSMEventSegmenter
    from morph_graph.video_models import get_extractor
    from morph_graph.video_models.video_io import load_decoder, sample_frame_indices

    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    decoder = load_decoder(str(path))
    metadata = decoder.metadata
    duration_sec = _duration(metadata)
    frame_indices = sample_frame_indices(decoder, fps, max_frames)
    batch = decoder.get_frames_at(frame_indices)
    frames = batch.data if hasattr(batch, "data") else batch
    if frames.dtype != torch.uint8:
        frames = frames.clamp(0, 255).to(torch.uint8)

    extractor = get_extractor(
        model,
        tokens_per_frame=tokens_per_frame,
        feature_stage=feature_stage,
    )
    features = extractor.encode_frames(frames, frame_batch_size=frame_batch_size)
    temporal_patch_size = int(extractor.temporal_patch_size)
    segmenter = TSMEventSegmenter(
        kernel_width=kernel_width,
        peak_height=peak_height,
        peak_distance=peak_distance,
        normalize=normalize,
    )
    segmentation = segmenter.segment(features.pooled)

    def unit_to_sec(unit: int) -> float:
        return _sample_position_to_sec(
            unit * temporal_patch_size,
            frame_indices,
            source_fps=metadata.average_fps,
            duration_sec=duration_sec,
            requested_fps=fps,
        )

    return {
        "video_id": path.stem,
        "video_path": str(path),
        "duration_sec": duration_sec,
        "requested_fps": fps,
        "source_fps": metadata.average_fps,
        "sampled_frame_indices": frame_indices,
        "n_sampled_frames": len(frame_indices),
        "temporal_patch_size": temporal_patch_size,
        "n_units": segmentation.n_units,
        "boundaries_units": segmentation.boundaries,
        "boundaries_sec": [round(unit_to_sec(unit), 6) for unit in segmentation.boundaries],
        "events_units": [list(span) for span in segmentation.events],
        "events_sec": [
            [round(unit_to_sec(start), 6), round(unit_to_sec(end), 6)]
            for start, end in segmentation.events
        ],
        "novelty": [float(value) for value in segmentation.novelty],
        "config": {
            "model": model,
            "tokens_per_frame": tokens_per_frame,
            "feature_stage": feature_stage,
            "frame_batch_size": frame_batch_size,
            "kernel_width": kernel_width,
            "peak_height": peak_height,
            "peak_distance": peak_distance,
            "normalize": normalize,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument(
        "--oracle-dir",
        type=Path,
        help="Run once per unique video referenced by oracle JSON files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3_vl_8b")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--tokens-per-frame", type=int, default=256)
    parser.add_argument("--feature-stage", choices=["pre", "post"], default="post")
    parser.add_argument("--kernel-width", type=int, default=6)
    parser.add_argument("--peak-height", type=float, default=1.0)
    parser.add_argument("--peak-distance", type=int)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    from .schema import OracleAnnotation

    args = build_parser().parse_args()
    if args.video:
        jobs = [(args.video, args.output)]
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        by_video: dict[str, Path] = {}
        for path in sorted(args.oracle_dir.glob("*.json")):
            if path.name.endswith(".error.json"):
                continue
            annotation = OracleAnnotation.load(path)
            by_video.setdefault(annotation.video_id, Path(annotation.video_path))
        jobs = [
            (video_path, args.output / f"{video_id}.json")
            for video_id, video_path in by_video.items()
        ]
    if not jobs:
        raise ValueError("No videos selected")

    for index, (video_path, output_path) in enumerate(jobs, start=1):
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(jobs)}] skip {output_path}")
            continue
        print(f"[{index}/{len(jobs)}] TSM {video_path}")
        result = export_tsm(
            video_path,
            model=args.model,
            fps=args.fps,
            max_frames=args.max_frames,
            frame_batch_size=args.frame_batch_size,
            tokens_per_frame=args.tokens_per_frame,
            feature_stage=args.feature_stage,
            kernel_width=args.kernel_width,
            peak_height=args.peak_height,
            peak_distance=args.peak_distance,
            normalize=not args.no_normalize,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  saved {output_path}")


if __name__ == "__main__":
    main()
