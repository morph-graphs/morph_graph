"""Pre-segmented oracle indexing for supplied or pseudo-GT video events."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ezcolorlog import root_logger as logger

from morph_graph.graphrag.vl_graphrag._vlm_embed import (
    VLEMBEDDING_REGISTRY,
    get_embedding_spec,
)
from morph_graph.graphrag.vl_graphrag.vl_graphrag import VLGraphRAG
from morph_graph.graphrag.vl_graphrag.vl_segments import _video_id_from_path
from morph_graph.graphrag.vl_graphrag.vl_segments import parse_caption_row

_VDB_BY_SOURCE = {
    "caption": "vdb_caption_segments.json",
    "vision": "vdb_video_clip_segments.json",
}
_SCENE_END = re.compile(r"^(?:end\s+of\s+|end\s+)", re.IGNORECASE)


@dataclass
class OracleIndexInputs:
    """Externally supplied segments used as oracle or pseudo-GT event inputs."""

    caption: Any | None = None
    vision: Any | None = None

    def for_source(self, source: str) -> Any | None:
        return self.caption if source == "caption" else self.vision


def build_oracle_scene_segments(video_path: str, caption_row: dict | None) -> list[dict]:
    """Convert VSI scene annotations into pre-segmented oracle vision records."""
    annotation_path = Path(video_path).with_suffix(".json")
    if not annotation_path.exists():
        return []
    annotation = json.loads(annotation_path.read_text())
    captions = {
        tuple(segment["frame_range"]): segment.get("caption", "")
        for segment in parse_caption_row(caption_row or {})
        if segment.get("frame_range") is not None
    }
    fps, start, segments = float(annotation.get("fps", 1)), 0, []
    for index, event in enumerate(
        event for event in annotation.get("annotations", []) if event.get("type") == "scene"
    ):
        end = int(event["frame_number"])
        if end <= start:
            continue
        label = _SCENE_END.sub("", str(event.get("label", "")).strip()).strip() or "unknown"
        frame_range = [start, end]
        segments.append(
            {
                "segment_idx": index,
                "scene_label": label,
                "frame_range": frame_range,
                "time_range_sec": [start / fps, end / fps],
                "video_path": video_path,
                "caption": captions.get(tuple(frame_range), ""),
            }
        )
        start = end
    return segments


def resolve_embedding_model(requested: str | None) -> str:
    if requested:
        return requested
    for key in ("qwen3_vl_embedding_8b", "qwen3_vl_embedding_2b"):
        if get_embedding_spec(key).model_path.exists():
            return key
    raise FileNotFoundError(
        f"No VL embedding checkpoint found. Known: {sorted(VLEMBEDDING_REGISTRY)}"
    )


def index_oracle(
    video_path: str,
    cache_root: Path,
    embedding_model: str,
    key_source: str,
    inputs: OracleIndexInputs,
    *,
    reindex: bool = False,
    cache: dict[str, VLGraphRAG] | None = None,
    **kwargs: Any,
) -> VLGraphRAG:
    """Index supplied oracle/pseudo-GT segments once per video cache directory."""
    working_dir = cache_root / _video_id_from_path(video_path).replace("/", "__")
    cache = cache if cache is not None else {}
    key = str(working_dir)
    if key in cache:
        return cache[key]
    if reindex and working_dir.exists():
        shutil.rmtree(working_dir)

    records = inputs.for_source(key_source)
    if records is None:
        raise ValueError(f"Oracle {key_source!r} segments are required for {video_path}")
    rag = VLGraphRAG(
        working_dir=str(working_dir),
        embedding_model=embedding_model,
        key_source=key_source,
        **kwargs,
    )
    vdb_path = working_dir / _VDB_BY_SOURCE[key_source]
    needs_index = not vdb_path.exists() or not json.loads(vdb_path.read_text()).get("data")
    if needs_index:
        logger.info(f"Indexing oracle {key_source} segments for {video_path}")
        if key_source == "caption":
            rag.insert_text_nodes(records)
        else:
            rag.insert_vision_nodes(records)
    cache[key] = rag
    return rag
