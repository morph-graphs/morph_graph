"""Retrieval debug: index one video, query GraphRAG, print retrieved segments.

No VLM question-answering — use ``python -m morph_graph.video_qa.cli`` for
benchmark QA.

Examples:
  python -m tests.debug_oracle_retrieval
  python -m tests.debug_oracle_retrieval --query "where is the kitchen"
  python -m tests.debug_oracle_retrieval --plot_graph
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from ezcolorlog import root_logger as logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))

from morph_graph.graphrag.vl_graphrag.event_viz import plot_from_graphml  # noqa: E402
from morph_graph.graphrag.vl_graphrag.vl_base import VLQueryParam  # noqa: E402
from morph_graph.graphrag.vl_graphrag.vl_segments import _video_id_from_path  # noqa: E402
from morph_graph.video_qa.oracle import (  # noqa: E402
    OracleIndexInputs,
    build_oracle_scene_segments,
    index_oracle,
    resolve_embedding_model,
)

DEFAULT_VIDEO = REPO_ROOT / "data" / "vsi_morph" / "vsi_constructed" / "Cat" / "0000.mp4"
DEFAULT_QUERY = "In what order was the Cat first seen across the Office, the Bedroom/Toy Room, and the Industrial Room?"
DEFAULT_CAPTIONS = REPO_ROOT / "outputs" / "per_video_segment_captions_qwen3_vl_8b.json"

logger.info(f"Starting debug_oracle_retrieval: QUERY: {DEFAULT_QUERY}")


def _load_caption_rows(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    return data


def _find_caption_row(rows: list[dict], video_path: Path) -> dict | None:
    resolved = str(video_path.resolve())
    for row in rows:
        row_path = row.get("video_path")
        if row_path and str(Path(row_path).resolve()) == resolved:
            return row
    # Fall back to matching by filename when paths differ across machines.
    name = video_path.name
    for row in rows:
        row_path = row.get("video_path")
        if row_path and Path(row_path).name == name:
            return row
    return None


def _print_segments(retrieval: dict) -> None:
    segments = retrieval.get("segments", [])
    print(f"\nRetrieved {len(segments)} segment(s)")
    print(f"Seed nodes:     {retrieval.get('seed_node_ids', [])}")
    print(f"Selected nodes: {retrieval.get('selected_node_ids', [])}")
    print("-" * 72)
    for i, seg in enumerate(segments, 1):
        frame_range = seg.get("frame_range")
        frames = f"frames {frame_range[0]}-{frame_range[1]}" if frame_range else "frames ?"
        label = seg.get("scene_label") or "unknown"
        score = seg.get("retrieval_score")
        score_str = f" score={score:.4f}" if isinstance(score, (int, float)) else ""
        caption = (seg.get("caption") or "").strip()
        if len(caption) > 200:
            caption = caption[:200] + "..."
        print(f"{i}. [{label}, {frames}]{score_str}")
        print(f"   {caption or '(no caption)'}")
    print("-" * 72)
    context = retrieval.get("caption_context", "")
    if context:
        logger.info("\nCaption context:\n")
        logger.info(context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug VL-GraphRAG indexing + retrieval for one video (no QA)"
    )
    parser.add_argument("--video_path", type=str, default=str(DEFAULT_VIDEO))
    parser.add_argument("--captions_path", type=str, default=str(DEFAULT_CAPTIONS))
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    parser.add_argument("--rag_cache_dir", type=str, default=str(REPO_ROOT / "outputs" / "vl_graphrag_cache_debug"))
    parser.add_argument("--embedding_model", default="qwen3_vl_embedding_2b", choices=["qwen3_vl_embedding_2b", "qwen3_vl_embedding_8b"])
    parser.add_argument("--key_source", default="caption", choices=["caption", "video"])
    parser.add_argument("--retrieval_mode", default="naive", choices=["naive", "local"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--semantic_similarity_threshold", type=float, default=0.4)
    parser.add_argument("--embedding_batch_num", type=int, default=32)
    parser.add_argument("--max_segments_for_qa", type=int, default=5)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--save", type=str, default=None, help="Optional path to write retrieval JSON")
    parser.add_argument("--plot_graph", action="store_true", help="Save segment graph PNG after query")
    parser.add_argument("--plot_graph_out", type=str, default=None, help="Output path for graph PNG")
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip indexing/query; plot existing graph_segment_relation.graphml in cache dir",
    )
    args = parser.parse_args()

    args.rag_cache_dir = f"{args.rag_cache_dir}_{args.key_source}"

    cache_root = Path(args.rag_cache_dir)
    video_path = Path(args.video_path).resolve()
    working_dir = cache_root / _video_id_from_path(str(video_path)).replace("/", "__")
    graphml_path = working_dir / "graph_segment_relation.graphml"

    if args.plot_only:
        if not graphml_path.exists():
            raise FileNotFoundError(f"No graph at {graphml_path}; run without --plot_only first")
        out = args.plot_graph_out or graphml_path.with_suffix(".png")
        plot_from_graphml(
            graphml_path,
            out,
            title=str(video_path.name),
            question=args.query.strip() or None,
        )
        logger.info(f"Wrote graph plot to {out}")
        return

    if not args.query.strip():
        parser.error("--query must be non-empty (unless --plot_only)")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    captions_path = Path(args.captions_path)
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")

    caption_row = _find_caption_row(_load_caption_rows(captions_path), video_path)
    if caption_row is None:
        raise ValueError(f"No caption row found for {video_path} in {captions_path}")

    # Patch video_path in the caption row so indexing uses the local file.
    caption_row = {**caption_row, "video_path": str(video_path)}
    vision = build_oracle_scene_segments(str(video_path), caption_row) or None
    inputs = OracleIndexInputs(caption=caption_row, vision=vision)

    if not inputs.for_source(args.key_source):
        raise ValueError(f"No {args.key_source} index data available for {video_path}")

    embedding_model = resolve_embedding_model(args.embedding_model)
    cache_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Video:           {video_path}")
    logger.info(f"Embedding model: {embedding_model}")
    logger.info(f"Key source:     {args.key_source}")
    logger.info(f"Retrieval mode: {args.retrieval_mode}")
    logger.info(f"Embedding batch: {args.embedding_batch_num}")
    logger.info(f"Query:          {args.query!r}")
    logger.info(f"Cache dir:      {cache_root}")

    rag = index_oracle(
        str(video_path),
        cache_root,
        embedding_model,
        args.key_source,
        inputs,
        reindex=args.reindex,
        cache={},
        semantic_similarity_threshold=args.semantic_similarity_threshold,
        embedding_batch_num=args.embedding_batch_num,
    )

    retrieval = rag.query(
        args.query,
        VLQueryParam(
            mode=args.retrieval_mode,
            key_source=args.key_source,
            top_k=args.top_k,
            max_segments_for_qa=args.max_segments_for_qa,
            only_need_context=False,
        ),
    )

    _print_segments(retrieval)

    if args.plot_graph:
        if not graphml_path.exists():
            raise FileNotFoundError(f"Expected graph at {graphml_path}")
        highlight = set(retrieval.get("selected_node_ids", []))
        out = args.plot_graph_out or graphml_path.with_suffix(".png")
        plot_from_graphml(
            graphml_path,
            out,
            highlight_nodes=highlight,
            title=str(video_path.name),
            question=args.query,
            retrieved_segments=retrieval.get("segments", []),
            retrieval_scores=retrieval.get("scores_by_id", {}),
        )
        print(f"Wrote graph plot to {out}")

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(retrieval, f, indent=2)
        print(f"\nWrote retrieval output to {out_path}")


if __name__ == "__main__":
    main()
