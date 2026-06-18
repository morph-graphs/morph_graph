from collections import defaultdict
from typing import Any

import numpy as np

from .._storage import NanoVectorDBStorage
from .._utils import logger
from ..base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from .vl_base import VLQueryParam
from .vl_segments import normalize_text_segments, normalize_vision_segments

_DEFAULT_SEMANTIC_TOP_K = 32


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
                    "description": segment.get("key_text")
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
            or segment.get("content")
            or segment.get("key_text")
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
        "caption_context": caption_context,
    }
    if param.only_need_context:
        return caption_context
    return response

def _temporal_edge(
    src_id: str,
    tgt_id: str,
    weight: float,
) -> tuple[tuple[str, str, dict[str, Any]]]:
    """ 
    Build bidirectional temporal edges between two video segments. 
    """  
    edges = ((
        src_id,
        tgt_id,
        {
            "edge_type": "temporal_next",
            "weight": weight,
            "description": "Forward temporal adjacency between consecutive video segments.",
            "source_id": src_id,
        },
    ), 
    (
        tgt_id,
        src_id,
        {
            "edge_type": "temporal_prev",
            "weight": weight,
            "description": "Backward temporal adjacency between consecutive video segments.",
            "source_id": tgt_id,
        },
    ))
    return edges

def _add_bidirectional_temporal_edge(
    edges: dict[tuple[str, str], tuple[str, str, dict[str, Any]]],
    src_id: str,
    tgt_id: str,
    weight: float,
):
    forward_key = (src_id, tgt_id)
    backward_key = (tgt_id, src_id)

    existing_forward_edge = edges.get(forward_key)
    existing_backward_edge = edges.get(backward_key)

    if existing_forward_edge is None or existing_backward_edge is None: 
        forward_edge, backward_edge = _temporal_edge(src_id, tgt_id, weight)
        edges[forward_key] = forward_edge
        edges[backward_key] = backward_edge
    
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
        metadata = {
            "content": segment["key_text"],
            "video_id": segment["video_id"],
            "segment_idx": segment["segment_idx"],
        }
        if "scene_label" in segment:
            metadata["scene_label"] = segment["scene_label"]
        caption_key_data[segment["node_id"]] = metadata
    await upsert_segment_embeddings_with_semantic_edges(
        caption_segments_vdb,
        caption_key_data,
        segments_relation_graph,
        semantic_similarity_threshold,
        semantic_similarity_top_k,
        embedding_source="caption",
    )


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

    # upsert segment values to key-value database (JsonKVStorage)
    # upsert temporal edges within batch to graph database (NetworkXStorage)
    await upsert_segment_values_and_graph(
        segments,
        segments_kv,
        segments_relation_graph,
        exclude_value_fields={"vision_key", "content"},
    )

    # insert temporal edges across batch to graph database (NetworkXStorage)
    await upsert_temporal_edges(segments, segments_kv, segments_relation_graph)

    logger.info("Insert video segment embeddings")
    video_key_data = {}
    for segment in segments:
        metadata = {
            "content": segment["vision_key"],
            "video_id": segment["video_id"],
            "segment_idx": segment["segment_idx"],
        }
        if "scene_label" in segment:
            metadata["scene_label"] = segment["scene_label"]
        video_key_data[segment["node_id"]] = metadata

    # upsert semantic similarity edges to graph database (NetworkXStorage)
    await upsert_segment_embeddings_with_semantic_edges(
        video_clip_segments_vdb,
        video_key_data,
        segments_relation_graph,
        semantic_similarity_threshold,
        semantic_similarity_top_k,
        embedding_source="video",
    )
