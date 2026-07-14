"""Per-video VLGraphRAG index cache and embedding model registry resolution."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ezcolorlog import root_logger as logger

from morph_graph.graphrag.vl_graphrag._vlm_embed import (
    VLEMBEDDING_REGISTRY,
    get_embedding_spec,
)
from morph_graph.graphrag.vl_graphrag.vl_graphrag import VLGraphRAG
from morph_graph.graphrag.vl_graphrag.vl_segments import _video_id_from_path
from morph_graph.inference.framework.base import IndexInputs

# One working_dir per video; each index kind has its own vector DB file.
_VDB_BY_KIND = {
    "caption": "vdb_caption_segments.json",
    "vision": "vdb_video_clip_segments.json",
}


def resolve_embedding_model(requested: str | None) -> str:
    """Resolve a VL embedding registry key (e.g. qwen3_vl_embedding_2b), not a checkpoint path."""
    if requested:
        return requested
    for key in ("qwen3_vl_embedding_8b", "qwen3_vl_embedding_2b"):
        if get_embedding_spec(key).model_path.exists():
            logger.info(f"Auto-selected embedding key {key!r}")
            return key
    known = ", ".join(sorted(VLEMBEDDING_REGISTRY))
    raise FileNotFoundError(
        f"No VL embedding checkpoint found under MORPH_GRAPH_MODEL_ROOT. "
        f"Known keys: {known}"
    )


def _working_dir(cache_root: Path, video_path: str) -> Path:
    return cache_root / _video_id_from_path(video_path).replace("/", "__")


def _needs_index(working_dir: Path, kind: str) -> bool:
    """Return True if this index kind's vector DB is missing or empty."""
    vdb_path = working_dir / _VDB_BY_KIND[kind]
    if not vdb_path.exists():
        return True
    with open(vdb_path) as f:
        return not json.load(f).get("data")


def index_rag(
    video_path: str,
    cache_root: Path,
    embedding_model: str,
    key_source: str,
    reindex: bool,
    rag_cache: dict[str, VLGraphRAG],
    inputs: IndexInputs,
    **graphrag_kwargs: Any,
) -> VLGraphRAG:
    """Return a cached VLGraphRAG for video_path, indexing caption/vision as needed.

    embedding_model: registry key from VLEMBEDDING_REGISTRY (not a checkpoint path).
    """
    working_dir = _working_dir(cache_root, video_path)
    key = str(working_dir)
    if key in rag_cache:
        return rag_cache[key]

    if reindex and working_dir.exists():
        shutil.rmtree(working_dir)

    rag = VLGraphRAG(working_dir=str(working_dir), 
        embedding_model=embedding_model,
        key_source=key_source,
        **graphrag_kwargs,
    )  

    if key_source == "caption":
        assert inputs.caption is not None, "Caption index inputs are required for caption key source"
        if _needs_index(working_dir, "caption"):
            logger.info(f"Indexing caption nodes: {video_path}")
            rag.insert_text_nodes(inputs.caption)
    elif key_source == "video":
        assert inputs.vision is not None, "Video index inputs are required for video key source"
        if _needs_index(working_dir, "vision"):
            logger.info(f"Indexing vision nodes: {video_path}")
            rag.insert_vision_nodes(inputs.vision)

    if inputs.caption is None and inputs.vision is None:
        raise ValueError(f"No index inputs for {video_path}")

    rag_cache[key] = rag
    return rag
