"""Command-line entrypoint for supported event-backed video QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarks import BENCHMARKS, get_benchmark
from .runner import VideoQAConfig, VideoQARunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--question-path", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-ids-json", type=Path,
                        help="Optional JSON list of sample IDs for one Slurm shard")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-input-frames", type=int)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--tokens-per-frame", type=int, default=256)
    parser.add_argument("--feature-stage", choices=["pre", "post"], default="post")
    parser.add_argument("--aligned-model", default="pe_core_vit_l")
    parser.add_argument("--kernel-width", type=int, default=6)
    parser.add_argument("--peak-height", type=float, default=1.0)
    parser.add_argument("--sim-threshold", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-segments-for-qa", type=int, default=5)
    parser.add_argument("--max-qa-frames", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retrieval-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    benchmark = get_benchmark(
        args.benchmark, question_path=args.question_path, video_dir=args.video_dir
    )
    samples = benchmark.load()
    if args.sample_ids_json:
        selected_ids = {str(value) for value in json.loads(args.sample_ids_json.read_text())}
        samples = [sample for sample in samples if sample.sample_id in selected_ids]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    samples.sort(key=lambda sample: str(sample.video_path))
    config = VideoQAConfig(
        fps=args.fps,
        max_input_frames=args.max_input_frames,
        frame_batch_size=args.frame_batch_size,
        tokens_per_frame=args.tokens_per_frame,
        feature_stage=args.feature_stage,
        aligned_model=args.aligned_model,
        kernel_width=args.kernel_width,
        peak_height=args.peak_height,
        semantic_similarity_threshold=args.sim_threshold,
        top_k=args.top_k,
        max_segments_for_qa=args.max_segments_for_qa,
        max_qa_frames=args.max_qa_frames,
        max_new_tokens=args.max_new_tokens,
        retrieval_only=args.retrieval_only,
    )
    results = VideoQARunner(config, args.cache_dir).run(
        benchmark, samples, args.output_json
    )
    correct = sum(row.get("correct") is True for row in results)
    evaluated = sum(row.get("correct") is not None for row in results)
    if evaluated:
        print(f"Accuracy: {correct}/{evaluated} = {correct / evaluated:.3f}")
    print(f"Saved {len(results)} results to {args.output_json}")


if __name__ == "__main__":
    main()
