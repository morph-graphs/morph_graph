"""Run query-agnostic annotation on N random val videos via the NVIDIA endpoint.

Writes the standard per-video JSON directory format consumed by
``event_eval.viewer`` / ``metrics.py`` / ``report.py`` (one
``<video_id>.json`` per video, same as ``annotate.py``). Optionally also
emits a single shareable JSONL with *relative* video paths (two parent dirs
kept) for the lightweight ``pilot_viewer``.

Usage:
    python -m event_eval.run_pilot_gpt --n 10 --seed 0 \
        --video-dir /lustre/.../egolongqa/val \
        --output-dir outputs/event_eval/agnostic_gpt_latest \
        --jsonl jsonls/gpt_agnostic_pilot.jsonl

Resumable: videos with a valid existing JSON in --output-dir are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

from .gpt import GPTOracleAnnotator
from .schema import OracleAnnotation

DEFAULT_VIDEO_DIR = Path(
    "/lustre/fsw/portfolios/nemotron/users/pulkitk/tracking/datasets/egolongqa/egolongqa/val"
)


def _relative_video_path(video_path: str) -> str:
    """Keep two parent directories so paths port across clusters."""
    return "/".join(Path(video_path).parts[-3:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        default=None,
        help="Model id on the endpoint (default: gpt.py's DEFAULT_MODEL). "
        "Any OpenAI-compatible VLM works, e.g. us/gcp/google/eccn-gemini-3.5-flash.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Per-video JSON dir, viewer-compatible "
        "(default: outputs/event_eval/agnostic_<model-short-name>_latest).",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Also write all annotations to this JSONL with relative video "
        "paths (default: jsonls/<model-short-name>_agnostic_pilot.jsonl; "
        "pass an empty string to skip).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    short = (args.model or "gpt").rsplit("/", 1)[-1].replace(".", "_")
    if args.output_dir is None:
        args.output_dir = repo_root / "outputs" / "event_eval" / f"agnostic_{short}_latest"
    if args.jsonl is None:
        args.jsonl = repo_root / "jsonls" / f"{short}_agnostic_pilot.jsonl"
    elif str(args.jsonl) in {"", "."}:
        args.jsonl = None

    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files in {args.video_dir}")
    rng = random.Random(args.seed)
    picked = rng.sample(videos, min(args.n, len(videos)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotator = GPTOracleAnnotator(**({"model": args.model} if args.model else {}))
    annotations: list[OracleAnnotation] = []
    failed = 0
    for index, video in enumerate(picked, start=1):
        video_id = video.stem
        output_path = args.output_dir / f"{video_id}.json"
        if output_path.exists():
            try:
                annotations.append(OracleAnnotation.load(output_path))
                print(f"[{index}/{len(picked)}] {video_id} — already done, skipping")
                continue
            except (ValueError, KeyError, json.JSONDecodeError):
                print(f"[{index}/{len(picked)}] {video_id} — regenerating invalid file")
        print(f"[{index}/{len(picked)}] {video_id} — annotating...", flush=True)
        start = time.time()
        try:
            annotation = annotator.annotate(
                sample_id=video_id,
                video_path=video,
                question="",
                annotation_mode="query_agnostic",
            )
        except Exception:
            failed += 1
            print(f"[{index}/{len(picked)}] {video_id} — FAILED")
            traceback.print_exc()
            continue
        annotation.save(output_path)
        annotations.append(annotation)
        print(
            f"[{index}/{len(picked)}] {video_id} — {len(annotation.events)} events "
            f"in {time.time() - start:.0f}s"
        )

    if args.jsonl and annotations:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w") as fh:
            for annotation in annotations:
                row = annotation.to_dict()
                row["video_path"] = _relative_video_path(row["video_path"])
                fh.write(json.dumps(row) + "\n")
        print(f"JSONL (portable, relative paths): {args.jsonl}")

    print(f"Output dir (viewer format): {args.output_dir}  failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
