import re
from collections import defaultdict
from typing import Any

import numpy as np

from .._storage import NanoVectorDBStorage
from .._utils import logger
from ..base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from .vl_base import VLQueryParam
from .vl_segments import normalize_text_segments, normalize_vision_segments

_DEFAULT_SEMANTIC_TOP_K = 32

# Node ids follow the "{video_id}::seg_{idx:04d}" scheme; used to detect
# temporal adjacency so semantic edges don't clobber the temporal spine
# (an undirected nx.Graph keeps only one edge per node pair).
_SEG_ID_RE = re.compile(r"^(?P<video>.*)::seg_(?P<idx>\d+)$")


def _seg_key(node_id: str) -> tuple[str, int] | None:
    match = _SEG_ID_RE.match(node_id)
    if match is None:
        return None
    return match.group("video"), int(match.group("idx"))


def _are_temporally_adjacent(src_id: str, tgt_id: str) -> bool:
    a, b = _seg_key(src_id), _seg_key(tgt_id)
    if a is None or b is None:
        return False
    return a[0] == b[0] and abs(a[1] - b[1]) == 1


def _video_decoder(video_path: str):
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(video_path)


def segment_value(segment: dict, exclude_fields: set[str] | None = None) -> dict:
    exclude_fields = exclude_fields or set()
    return {k: v for k, v in segment.items() if k not in exclude_fields}


async def upsert_segment_values_and_graph(
    segments: list[dict],
    segments_kv: BaseKVStorage,
    segments_relation_graph: BaseGraphStorage,
    exclude_value_fields: set[str] | None = None,
):
    node_ids = [segment["node_id"] for segment in segments]
    existing_values = await segments_kv.get_by_ids(node_ids)
    segment_values = {}

    for node_id, existing_value, segment in zip(
        node_ids, existing_values, segments
    ):
        value = segment_value(segment, exclude_value_fields)
        if existing_value is not None:
            value = {**existing_value, **value}
        segment_values[node_id] = value
    await segments_kv.upsert(segment_values)

    graph_nodes = [
        (
            segment["node_id"],
            {
                k: v
                for k, v in {
                    "source_id": segment["node_id"],
                    "video_id": segment["video_id"],
                    "segment_idx": segment["segment_idx"],
                    "scene_label": segment.get("scene_label"),
                    "description": segment.get("text_key")
                    or segment.get("description")
                    or segment["node_id"],
                }.items()
                if v is not None
            },
        )
        for segment in segments
    ]
    await segments_relation_graph.upsert_nodes_batch(graph_nodes)

    segments_by_video = defaultdict(list)
    for segment in segments:
        segments_by_video[segment["video_id"]].append(segment)

    temporal_edges = []
    for video_segments in segments_by_video.values():
        ordered_segments = sorted(
            video_segments, key=lambda segment: segment["segment_idx"]
        )
        for src_segment, tgt_segment in zip(
            ordered_segments, ordered_segments[1:]
        ):
            temporal_edges.extend(
                _temporal_edge(
                    src_segment["node_id"],
                    tgt_segment["node_id"],
                    1.0,
                )
            )
    if temporal_edges:
        await segments_relation_graph.upsert_edges_batch(temporal_edges)


async def expand_node_ids(
    node_ids: list[str],
    segments_relation_graph: BaseGraphStorage,
) -> list[str]:
    seen = set(node_ids)
    expanded = list(node_ids)
    node_edges = await segments_relation_graph.get_nodes_edges_batch(node_ids)
    for edges in node_edges:
        if not edges:
            continue
        for _, neighbor_id in edges:
            if neighbor_id in seen:
                continue
            seen.add(neighbor_id)
            expanded.append(neighbor_id)
    return expanded


def rank_and_cap_node_ids(
    node_ids: list[str], scores_by_id: dict[str, float], max_count: int
) -> list[str]:
    deduped_node_ids = list(dict.fromkeys(node_ids))
    deduped_node_ids.sort(
        key=lambda node_id: (
            node_id not in scores_by_id,
            -float(scores_by_id.get(node_id, 0.0)),
        )
    )
    return deduped_node_ids[:max_count]


def format_caption_context(segments: list[dict]) -> str:
    ordered_segments = sorted(
        segments,
        key=lambda segment: (
            str(segment.get("video_id", "")),
            int(segment.get("segment_idx", 0)),
        ),
    )
    lines = []
    for segment in ordered_segments:
        caption = str(
            segment.get("caption")
            or segment.get("text_key")
            or ""
        ).strip()
        if not caption:
            continue
        label = segment.get("scene_label")
        frame_range = segment.get("frame_range")
        metadata = []
        if label is not None:
            metadata.append(str(label))
        if frame_range is not None:
            metadata.append(f"frames {frame_range[0]}-{frame_range[1]}")
        prefix = f"[{', '.join(metadata)}] " if metadata else ""
        lines.append(f"- {prefix}{caption}")
    return "\n".join(lines)


async def vl_query(
    query: str,
    param: VLQueryParam,
    vdb: BaseVectorStorage,
    segments_kv: BaseKVStorage,
    segments_relation_graph: BaseGraphStorage,
) -> str | dict[str, Any]:
    results = await vdb.query(query, top_k=param.top_k)
    if not results:
        return {"query": query, "selected_node_ids": [], "segments": [], "caption_context": ""}

    seed_node_ids = [result["id"] for result in results]
    scores_by_id = {result["id"]: result.get("distance", 0.0) for result in results}

    selected_node_ids = list(seed_node_ids)
    if param.mode == "local":
        selected_node_ids = await expand_node_ids(
            selected_node_ids, segments_relation_graph
        )

    selected_node_ids = rank_and_cap_node_ids(
        selected_node_ids, scores_by_id, param.max_segments_for_qa
    )
    segment_values = await segments_kv.get_by_ids(selected_node_ids)
    segments = [
        {**segment_value, "node_id": node_id, "retrieval_score": scores_by_id.get(node_id)}
        for node_id, segment_value in zip(selected_node_ids, segment_values)
        if segment_value is not None
    ]
    caption_context = format_caption_context(segments)
    response = {
        "query": query,
        "key_source": param.key_source,
        "mode": param.mode,
        "seed_node_ids": seed_node_ids,
        "selected_node_ids": [segment["node_id"] for segment in segments],
        "segments": segments,
        "scores_by_id": scores_by_id,
        "caption_context": caption_context,
    }
    if param.only_need_context:
        return caption_context
    return response

def _temporal_edge(
    src_id: str,
    tgt_id: str,
    weight: float,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """Single undirected temporal-adjacency edge between consecutive segments.

    NetworkXStorage is undirected, so a forward/backward pair collapses to one
    edge (last write wins); we emit a single `temporal_adjacent` edge instead.
    Ordering/direction is recoverable from `segment_idx` on the nodes.
    """
    edges = (
        (
            src_id,
            tgt_id,
            {
                "edge_type": "temporal_adjacent",
                "weight": weight,
                "description": "Temporal adjacency between consecutive video segments.",
                "source_id": src_id,
            },
        ),
    )
    return edges

def _add_bidirectional_temporal_edge(
    edges: dict[tuple[str, str], tuple[str, str, dict[str, Any]]],
    src_id: str,
    tgt_id: str,
    weight: float,
):
    # Undirected: dedupe on the unordered pair.
    key = tuple(sorted((src_id, tgt_id)))
    if key not in edges:
        (edge,) = _temporal_edge(src_id, tgt_id, weight)
        edges[key] = edge

    return edges


async def build_temporal_edges(
    segments: list[dict],
    segments_kv: BaseKVStorage,
) -> list[tuple[str, str, dict[str, Any]]]: 
    
    node_ids = [segment["node_id"] for segment in segments]
    segment_ids = [segment["segment_idx"] for segment in segments]

    edges: dict[tuple[str, str], tuple[str, str, dict[str, Any]]] = {}

    # get the segment ids from the node ids 
    video_id = segments[0]["video_id"]
    start_segment_id = min(segment_ids)
    prev_segment_id = start_segment_id - 1

    # keep track of the previous node id and start node id 
    prev_node_id = f"{video_id}::seg_{prev_segment_id:04d}" if prev_segment_id >= 0 else None
    start_node_id = f"{video_id}::seg_{start_segment_id:04d}"

    if prev_node_id is not None:
        prev_value = await segments_kv.get_by_id(prev_node_id)
        if prev_value is not None:
            _add_bidirectional_temporal_edge(edges, prev_node_id, start_node_id, 1.0)
    
    return list(edges.values())
    
def _semantic_edge(
    src_id: str,
    tgt_id: str,
    score: float,
    embedding_source: str,
    edge_type: str = "semantic_sim",
) -> tuple[str, str, dict[str, Any]]:
    return (
        src_id,
        tgt_id,
        {
            "edge_type": edge_type,
            "embedding_source": embedding_source,
            "weight": score,
            "description": "Semantic similarity between segment embeddings",
            "source_id": src_id,
        },
    )


def _add_bidirectional_semantic_edge(
    edges: dict[tuple[str, str], tuple[str, str, dict[str, Any]]],
    src_id: str,
    tgt_id: str,
    score: float,
    embedding_source: str,
):
    for edge_src, edge_tgt in ((src_id, tgt_id), (tgt_id, src_id)):
        key = (edge_src, edge_tgt)
        existing = edges.get(key)
        if existing is None or float(existing[2]["weight"]) < score:
            edges[key] = _semantic_edge(edge_src, edge_tgt, score, embedding_source)


async def build_semantic_similarity_edges(
    vdb: NanoVectorDBStorage,
    list_data: list[dict],
    embeddings: np.ndarray,
    similarity_threshold: float,
    similarity_top_k: int | None,
    embedding_source: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    if not list_data:
        return []

    edges: dict[tuple[str, str], tuple[str, str, dict[str, Any]]] = {}
    top_k = similarity_top_k if similarity_top_k is not None else _DEFAULT_SEMANTIC_TOP_K
    new_ids = {item["__id__"] for item in list_data}

    # query previous segments to build semantic similarity edges
    for item, embedding in zip(list_data, embeddings):
        node_id = item["__id__"]
        neighbors = await vdb.query_with_embedding(
            embedding,
            top_k=top_k,
            better_than_threshold=similarity_threshold,
        )
        for neighbor in neighbors:
            tgt_id = neighbor["id"]
            if tgt_id == node_id or tgt_id in new_ids:
                continue
            # temporally-adjacent pairs already share the (single) undirected edge
            # slot as a temporal_adjacent edge; skip to preserve the temporal spine.
            if _are_temporally_adjacent(node_id, tgt_id):
                continue
            score = float(neighbor.get("distance", 0.0))
            _add_bidirectional_semantic_edge(
                edges, node_id, tgt_id, score, embedding_source
            )

    # query within current batch of segments to build semantic similarity edges
    if len(list_data) > 1:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_embeddings = embeddings / norms
        similarity_matrix = normalized_embeddings @ normalized_embeddings.T
        for src_idx, src_item in enumerate(list_data):
            candidates = []
            for tgt_idx, tgt_item in enumerate(list_data):
                if src_idx == tgt_idx:
                    continue
                if _are_temporally_adjacent(src_item["__id__"], tgt_item["__id__"]):
                    continue
                score = float(similarity_matrix[src_idx, tgt_idx])
                if score >= similarity_threshold:
                    candidates.append((score, tgt_item["__id__"]))
            candidates.sort(reverse=True)
            if similarity_top_k is not None:
                candidates = candidates[:similarity_top_k]
            for score, tgt_id in candidates:
                _add_bidirectional_semantic_edge(
                    edges, src_item["__id__"], tgt_id, score, embedding_source
                )

    return list(edges.values())

async def upsert_temporal_edges(
    segments: list[dict],
    segments_kv: BaseKVStorage,
    segments_relation_graph: BaseGraphStorage,
):
    temporal_edges = await build_temporal_edges(segments, segments_kv)
    if temporal_edges:
        await segments_relation_graph.upsert_edges_batch(temporal_edges)

async def upsert_segment_embeddings_with_semantic_edges(
    vdb: NanoVectorDBStorage,
    key_data: dict[str, dict],
    segments_relation_graph: BaseGraphStorage,
    similarity_threshold: float,
    similarity_top_k: int | None,
    embedding_source: str,
):
    list_data, embeddings = await vdb.embed_data(key_data)
    if not list_data:
        return

    semantic_edges = await build_semantic_similarity_edges(
        vdb,
        list_data,
        embeddings,
        similarity_threshold,
        similarity_top_k,
        embedding_source,
    )
    await vdb.upsert_prepared(list_data)
    if semantic_edges:
        await segments_relation_graph.upsert_edges_batch(semantic_edges)


async def insert_text_nodes(
    string_or_strings,
    segments_kv: BaseKVStorage,
    segments_relation_graph: BaseGraphStorage,
    caption_segments_vdb: NanoVectorDBStorage,
    semantic_similarity_threshold: float,
    semantic_similarity_top_k: int | None,
):

    """ 
        Insert text or texts to VLGraphRAG. 
        1. Normalize text or texts to segments (TODO: validate this function)
        2. Upsert segment values to key-value database (JsonKVStorage)
        3. Upsert graph nodes and temporal edges to graph database (NetworkXStorage)
        4. Upsert segment embeddings to vector database (NanoVectorDBStorage)
        5. Upsert semantic similarity edges to graph database (NetworkXStorage) - no 
           new nodes are added to the graph. 
    """ 

    segments = normalize_text_segments(string_or_strings)
    if not segments:
        logger.warning("No text segments found for VLGraphRAG insertion")
        return

    logger.info(f"[VLGraphRAG] inserting {len(segments)} caption-key segments")

    # upsert segment values to key-value database (JsonKVStorage)
    # upsert temporal edges within batch to graph database (NetworkXStorage)
    await upsert_segment_values_and_graph(
        segments, segments_kv, segments_relation_graph
    )

    # insert temporal edges across batch to graph database (NetworkXStorage)
    await upsert_temporal_edges(segments, segments_kv, segments_relation_graph)

    logger.info("Insert caption segment embeddings")
    caption_key_data = {}
    for segment in segments:
        # "content" is the nano-graphrag VDB field passed to the embedding func.
        vdb_row = {
            "content": segment["text_key"],
            "video_id": segment["video_id"],
            "segment_idx": segment["segment_idx"],
        }
        if "scene_label" in segment:
            vdb_row["scene_label"] = segment["scene_label"]
        caption_key_data[segment["node_id"]] = vdb_row
    await upsert_segment_embeddings_with_semantic_edges(
        caption_segments_vdb,
        caption_key_data,
        segments_relation_graph,
        semantic_similarity_threshold,
        semantic_similarity_top_k,
        embedding_source="caption",
    )

def get_video_frames(decoder, frame_indices):
    """Return video frames at the given frame indices."""
    if not frame_indices:
        return []
    frames = decoder.get_frames_at(frame_indices)
    return frames.data if hasattr(frames, "data") else frames


def sample_frame_indices(frame_range, max_frames: int | None = None):
    """Get frame indices in the half-open range [start, end), optionally subsampled."""
    start, end = map(int, frame_range)
    if end <= start:
        return []
    if max_frames is None:
        return list(range(start, end))
    n = min(max_frames, end - start)
    if n <= 0:
        return []
    if n == 1:
        return [start]
    return sorted(
        {start + round(i * (end - start - 1) / (n - 1)) for i in range(n)}
    )


def _allocate_frame_budgets(segments: list[dict], max_frames: int) -> list[int]:
    """Split a global frame budget across segments proportional to span length."""
    spans = [
        max(0, int(seg["frame_range"][1]) - int(seg["frame_range"][0]))
        for seg in segments
        if seg.get("frame_range") is not None
    ]
    if not spans or max_frames <= 0:
        return [0] * len(segments)

    total_span = sum(spans)
    budgets: list[int] = []
    remaining = max_frames
    for i, span in enumerate(spans):
        if i == len(spans) - 1:
            budgets.append(max(0, remaining))
            continue
        n = max(1, round(max_frames * span / total_span)) if span > 0 else 0
        n = min(n, remaining)
        budgets.append(n)
        remaining -= n
    return budgets


def stitch_retrieved_segments_video(
    segments: list[dict],
    *,
    default_video_path: str | None = None,
    max_frames: int = 128,
) -> tuple[Any, dict]:
    """Decode retrieved segments via torchcodec and concatenate for VLM QA."""
    import torch

    if not segments:
        raise ValueError("No segments to stitch")

    ordered = sorted(
        segments,
        key=lambda segment: (
            str(segment.get("video_id", "")),
            int(segment.get("segment_idx", 0)),
        ),
    )
    segments_with_frames = [
        segment for segment in ordered if segment.get("frame_range") is not None
    ]
    if not segments_with_frames:
        raise ValueError("Retrieved segments have no frame_range metadata")

    budgets = _allocate_frame_budgets(segments_with_frames, max_frames)
    decoders: dict[str, Any] = {}
    frame_chunks = []
    total_duration_sec = 0.0

    for segment, budget in zip(segments_with_frames, budgets):
        if budget <= 0:
            continue
        video_path = str(segment.get("video_path") or default_video_path or "")
        if not video_path:
            raise ValueError(f"Segment {segment.get('node_id')} missing video_path")
        if video_path not in decoders:
            decoders[video_path] = _video_decoder(video_path)

        frame_indices = sample_frame_indices(segment["frame_range"], max_frames=budget)
        if not frame_indices:
            continue
        frames = get_video_frames(decoders[video_path], frame_indices)
        if frames is None or (hasattr(frames, "numel") and frames.numel() == 0):
            continue
        frame_chunks.append(frames)

        time_range = segment.get("time_range_sec")
        if time_range is not None and len(time_range) == 2:
            total_duration_sec += max(float(time_range[1]) - float(time_range[0]), 0.0)
        else:
            start, end = map(int, segment["frame_range"])
            total_duration_sec += max(end - start, 0.0)

    if not frame_chunks:
        raise ValueError("Failed to decode any frames from retrieved segments")

    video_tensor = (
        torch.cat(frame_chunks, dim=0) if len(frame_chunks) > 1 else frame_chunks[0]
    )
    n_frames = int(video_tensor.shape[0])
    video_metadata = {
        "fps": n_frames / max(total_duration_sec, 1e-6),
        "frames_indices": list(range(n_frames)),
        "total_num_frames": n_frames,
    }
    return video_tensor, video_metadata

async def insert_vision_nodes(
    video_clip_or_clips,
    segments_kv: BaseKVStorage,
    segments_relation_graph: BaseGraphStorage,
    video_clip_segments_vdb: NanoVectorDBStorage,
    semantic_similarity_threshold: float,
    semantic_similarity_top_k: int | None,
):

    """ 
        Insert video clip or clips to VLGraphRAG. 
        1. Normalize video clip or clips to segments (TODO: validate this function)
        2. Upsert segment values to key-value database (JsonKVStorage)
        3. Upsert segment embeddings (graph nodes) and temporal edges to graph database (NetworkXStorage)
        4. Upsert segment embeddings to vector database (NanoVectorDBStorage)
        5. Upsert semantic similarity edges to graph database (NetworkXStorage) - no 
           new nodes are added to the graph. 
    """ 
    segments = normalize_vision_segments(video_clip_or_clips)
    if not segments:
        logger.warning("No vision segments found for VLGraphRAG insertion")
        return

    logger.info(f"[VLGraphRAG] inserting {len(segments)} video-key segments")
    segments_by_video: dict[str, list[dict]] = defaultdict(list)
    for segment in segments:
        segments_by_video[str(segment["video_path"])].append(segment)

    # upsert segment values to key-value database (JsonKVStorage)
    # upsert temporal edges within batch to graph database (NetworkXStorage)
    await upsert_segment_values_and_graph(
        segments,
        segments_kv,
        segments_relation_graph,
        exclude_value_fields={"vision_key"},
    )

    # insert temporal edges across batch to graph database (NetworkXStorage)
    await upsert_temporal_edges(segments, segments_kv, segments_relation_graph)

    logger.info("Insert video segment embeddings")
    video_key_data = {}
    for video_path, video_segments in segments_by_video.items():
        logger.info(f"Decoding video frames from {video_path}")
        decoder = _video_decoder(video_path)
        for segment in video_segments:
            frame_indices = sample_frame_indices(segment["frame_range"])

            if not frame_indices:
                raise ValueError(f"No frame indices found for segment {segment.get('node_id')}")
            frames = get_video_frames(decoder, frame_indices)
            if frames is None or (hasattr(frames, "numel") and frames.numel() == 0):
                raise ValueError(f"No frames decoded for segment {segment.get('node_id')}")

            time_range = segment["time_range_sec"]
            sample_fps = len(frame_indices) / max(
                time_range[1] - time_range[0], 1e-6
            )
            video_metadata = {
                "fps": sample_fps,
                "frames_indices": list(range(len(frame_indices))),
                "total_num_frames": len(frame_indices),
            }
            embed_input = {"video": frames, "video_metadata": video_metadata}
            vdb_row = {
                "content": embed_input,
                "video_id": segment["video_id"],
                "segment_idx": segment["segment_idx"],
            }
            if "scene_label" in segment:
                vdb_row["scene_label"] = segment["scene_label"]
            video_key_data[segment["node_id"]] = vdb_row

    if not video_key_data:
        logger.warning("No video segments with decodable frames for VLGraphRAG insertion")
        return

    await upsert_segment_embeddings_with_semantic_edges(
        video_clip_segments_vdb,
        video_key_data,
        segments_relation_graph,
        semantic_similarity_threshold,
        semantic_similarity_top_k,
        embedding_source="video",
    )
