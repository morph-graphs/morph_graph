"""VL-GraphRAG eval: index → retrieve → optional VLM answer.

Entry point: morph_graph/eval/eval_qwen3_vl_rag.py

Examples:
  python3.11 morph_graph/eval/eval_qwen3_vl_rag.py --benchmark vsi_morph --retrieval_only
  python3.11 morph_graph/eval/eval_qwen3_vl_rag.py --key_source caption --retrieval_mode local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))

from morph_graph.inference.benchmarks import add_benchmark_argument, get_benchmark  # noqa: E402
from morph_graph.inference.framework.base import EvalConfig  # noqa: E402
from morph_graph.inference.framework.runner import run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VL-GraphRAG on a video QA benchmark")
    add_benchmark_argument(parser)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--rag_cache_dir", default=str(REPO_ROOT / "outputs" / "vl_graphrag_cache"))
    parser.add_argument("--embedding_model", default="qwen3_vl_embedding_2b", choices=["qwen3_vl_embedding_2b", "qwen3_vl_embedding_8b"])
    parser.add_argument("--retrieval_mode", default="naive", choices=["naive", "local"])
    parser.add_argument("--key_source", default="caption", choices=["caption", "video"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--semantic_similarity_threshold", type=float, default=0.5)
    parser.add_argument("--embedding_batch_num", type=int, default=32)
    parser.add_argument("--max_segments_for_qa", type=int, default=5)
    parser.add_argument("--max_frames_for_video_unit", type=int, default=128)
    parser.add_argument("--qa_mode", default="caption", choices=["caption", "video"])
    parser.add_argument("--model_path", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--retrieval_only", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)

    # Two-pass parse: generic flags first, then benchmark-specific flags.
    args, _ = parser.parse_known_args()
    benchmark = get_benchmark(args.benchmark)
    benchmark.add_arguments(parser)
    args = parser.parse_args()

    run_benchmark(benchmark, EvalConfig.from_args(args, benchmark.name), args)


if __name__ == "__main__":
    main()
