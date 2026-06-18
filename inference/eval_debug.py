"""Debug script: index one video + captions, query GraphRAG, print retrieved segments.

No VLM question-answering — just indexing and retrieval smoke test.

Examples:
  python3.11 morph_graph/inference/eval_debug.py
  python3.11 morph_graph/inference/eval_debug.py --query "where is the kitchen"
  python3.11 morph_graph/inference/eval_debug.py --video_path vsi_output/vsi_constructed/Cat/0000.mp4
  python3.11 morph_graph/inference/eval_debug.py --retrieval_mode local --top_k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from morph_graph.graphrag.vl_graphrag.vl_base import VLQueryParam  # noqa: E402
from morph_graph.inference.benchmarks.vsi_morph import (  # noqa: E402
    DEFAULT_CAPTIONS,
    _vision_segments,
)
from morph_graph.inference.framework.base import IndexInputs  # noqa: E402
from morph_graph.inference.framework.rag import index_rag, resolve_embedding_key  # noqa: E402

DEFAULT_VIDEO = REPO_ROOT / "vsi_output" / "vsi_constructed" / "Cat" / "0000.mp4"
DEFAULT_QUERY = ""


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
        caption = (seg.get("caption") or seg.get("content") or "").strip()
        if len(caption) > 200:
            caption = caption[:200] + "..."
        print(f"{i}. [{label}, {frames}]{score_str}")
        print(f"   {caption or '(no caption)'}")
    print("-" * 72)
    context = retrieval.get("caption_context", "")
    if context:
        print("\nCaption context:\n")
        print(context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug VL-GraphRAG indexing + retrieval for one video")
    parser.add_argument("--video_path", type=str, default=str(DEFAULT_VIDEO))
    parser.add_argument("--captions_path", type=str, default=str(DEFAULT_CAPTIONS))
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    parser.add_argument("--rag_cache_dir", type=str, default=str(REPO_ROOT / "outputs" / "vl_graphrag_cache_debug"))
    parser.add_argument("--embedding_key", default=None)
    parser.add_argument("--key_source", default="caption", choices=["caption", "video"])
    parser.add_argument("--retrieval_mode", default="naive", choices=["naive", "local"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_segments_for_qa", type=int, default=5)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--save", type=str, default=None, help="Optional path to write retrieval JSON")
    args = parser.parse_args()

    video_path = Path(args.video_path).resolve()
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
    vision = _vision_segments(str(video_path), caption_row) or None
    inputs = IndexInputs(caption=caption_row, vision=vision)

    if not inputs.required_for(args.key_source):
        raise ValueError(f"No {args.key_source} index data available for {video_path}")

    embedding_key = resolve_embedding_key(args.embedding_key)
    cache_root = Path(args.rag_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Video:          {video_path}")
    print(f"Embedding:      {embedding_key}")
    print(f"Key source:     {args.key_source}")
    print(f"Retrieval mode: {args.retrieval_mode}")
    print(f"Query:          {args.query!r}")
    print(f"Cache dir:      {cache_root}")

    rag = index_rag(
        str(video_path),
        cache_root,
        embedding_key,
        args.reindex,
        {},
        inputs,
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

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(retrieval, f, indent=2)
        print(f"\nWrote retrieval output to {out_path}")


if __name__ == "__main__":
    main()
