"""Run GPT query-agnostic annotation on N random val videos, dump to one JSONL.

Usage:
    python -m event_eval.run_pilot_gpt --n 10 --seed 0 \
        --video-dir /lustre/.../egolongqa/val \
        --output jsonls/gpt_agnostic_pilot.jsonl

Resumable: videos whose sample_id already appears in the output JSONL are
skipped, so re-running after a partial failure continues where it left off.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

from .gpt import GPTOracleAnnotator

DEFAULT_VIDEO_DIR = Path(
    "/lustre/fsw/portfolios/nemotron/users/pulkitk/tracking/datasets/egolongqa/egolongqa/val"
)


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
        "--output",
        type=Path,
        default=None,
        help="Output JSONL (default: jsonls/<model-short-name>_agnostic_pilot.jsonl).",
    )
    args = parser.parse_args()

    if args.output is None:
        short = (args.model or "gpt").rsplit("/", 1)[-1].replace(".", "_")
        args.output = (
            Path(__file__).resolve().parents[1]
            / "jsonls"
            / f"{short}_agnostic_pilot.jsonl"
        )

    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files in {args.video_dir}")
    rng = random.Random(args.seed)
    picked = rng.sample(videos, min(args.n, len(videos)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.output.exists():
        with args.output.open() as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["sample_id"])

    annotator = GPTOracleAnnotator(**({"model": args.model} if args.model else {}))
    for index, video in enumerate(picked, start=1):
        sample_id = video.stem
        if sample_id in done:
            print(f"[{index}/{len(picked)}] {sample_id} — already done, skipping")
            continue
        print(f"[{index}/{len(picked)}] {sample_id} — annotating...", flush=True)
        start = time.time()
        try:
            annotation = annotator.annotate(
                sample_id=sample_id,
                video_path=video,
                question="",
                annotation_mode="query_agnostic",
            )
        except Exception:
            print(f"[{index}/{len(picked)}] {sample_id} — FAILED")
            traceback.print_exc()
            continue
        row = annotation.to_dict()
        # Keep only two parent directories so the path is portable across
        # clusters (consumers prepend their own data root).
        row["video_path"] = "/".join(Path(row["video_path"]).parts[-3:])
        with args.output.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"[{index}/{len(picked)}] {sample_id} — {len(annotation.events)} events "
            f"in {time.time() - start:.0f}s"
        )

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
