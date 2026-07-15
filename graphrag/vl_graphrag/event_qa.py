"""Steps F+G: fetch event values, stitch retrieved frames, run VLM QA.

v1 pixel path (per POC): load stored `frame_indices` from `EventValueStore`,
re-decode those source frames via torchcodec, concatenate in timeline order, and
feed the stitched clip to Qwen3-VL. Stored `pooled` / `patch_tokens` are loaded
for logging/inspection; embedding injection is deferred.
"""

from __future__ import annotations

from typing import Any

import torch
from ezcolorlog import root_logger as logger

from .vl_base import VLQueryParam
from .vl_ops import _video_decoder, get_video_frames


def _segment_sort_key(segment: dict) -> tuple[str, int]:
    return str(segment.get("video_id", "")), int(segment.get("segment_idx", 0))


def _indices_from_value(value: dict[str, Any]) -> list[int]:
    raw = value.get("frame_indices")
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        return [int(x) for x in raw.tolist()]
    return [int(x) for x in raw]


def _subsample_sorted_indices(indices: list[int], max_count: int) -> list[int]:
    if max_count <= 0 or not indices:
        return []
    if len(indices) <= max_count:
        return list(indices)
    if max_count == 1:
        return [indices[0]]
    n = len(indices)
    return sorted({indices[round(i * (n - 1) / (max_count - 1))] for i in range(max_count)})


def _allocate_index_budgets(index_counts: list[int], max_frames: int) -> list[int]:
    if not index_counts or max_frames <= 0:
        return [0] * len(index_counts)
    total = sum(index_counts)
    if total == 0:
        return [0] * len(index_counts)

    budgets: list[int] = []
    remaining = max_frames
    for i, count in enumerate(index_counts):
        if i == len(index_counts) - 1:
            budgets.append(max(0, remaining))
            continue
        n = max(1, round(max_frames * count / total)) if count > 0 else 0
        n = min(n, remaining, count)
        budgets.append(n)
        remaining -= n
    return budgets


def fetch_event_values(value_store, segments: list[dict]) -> list[dict]:
    """Attach stored tensor values to each retrieved segment (step F)."""
    enriched: list[dict] = []
    for segment in segments:
        node_id = str(segment["node_id"])
        value = value_store.load(node_id)
        enriched.append({**segment, "values": value})
        pooled = value.get("pooled")
        n_units = int(pooled.shape[0]) if pooled is not None and hasattr(pooled, "shape") else "?"
        logger.info(
            f"[qa][fetch] {node_id} seg={segment.get('segment_idx')} "
            f"stored_frames={len(_indices_from_value(value))} pooled_units={n_units}"
        )
    return enriched


def _log_retrieved_segments(question: str, retrieval: dict[str, Any]) -> None:
    seeds = retrieval.get("seed_node_ids") or []
    selected = retrieval.get("selected_node_ids") or []
    segments = retrieval.get("segments") or []
    logger.info(
        f"[qa][retrieve] query={question!r} seeds={len(seeds)} "
        f"selected={len(selected)} (cap after 1-hop expand)"
    )
    for rank, segment in enumerate(segments, 1):
        score = segment.get("retrieval_score")
        score_s = f"{float(score):.3f}" if score is not None else "n/a"
        fr = segment.get("frame_range")
        tr = segment.get("time_range_sec")
        logger.info(
            f"[qa][retrieve] #{rank} seg={segment.get('segment_idx')} "
            f"node={segment.get('node_id')} score={score_s} "
            f"units={segment.get('unit_range')} frames={fr} time_sec={tr}"
        )


def stitch_retrieved_event_values(
    segments: list[dict],
    *,
    default_video_path: str | None = None,
    max_frames: int = 128,
    require_values: bool = True,
) -> tuple[torch.Tensor, dict, list[dict]]:
    """Decode frames from stored `frame_indices` and concatenate (step G, pixels).

    `segments` may already include a `values` dict (from `fetch_event_values`);
    otherwise `value_path` / `EventValueStore` is not consulted here — call
    `fetch_event_values` first.

    Returns `(video_tensor, video_metadata, stitch_log)` where `stitch_log` lists
    per-segment decode stats.
    """
    if not segments:
        raise ValueError("No segments to stitch")

    ordered = sorted(segments, key=_segment_sort_key)
    segments_with_indices: list[tuple[dict, list[int]]] = []
    for segment in ordered:
        value = segment.get("values")
        if value is None:
            if require_values:
                raise ValueError(
                    f"Segment {segment.get('node_id')} missing values; "
                    "call fetch_event_values first"
                )
            continue
        indices = _indices_from_value(value)
        if indices:
            segments_with_indices.append((segment, indices))

    if not segments_with_indices:
        raise ValueError("Retrieved segments have no stored frame_indices")

    index_counts = [len(indices) for _, indices in segments_with_indices]
    budgets = _allocate_index_budgets(index_counts, max_frames)
    logger.info(
        f"[qa][stitch] {len(segments_with_indices)} events, "
        f"frame_budget={max_frames} per_event_budgets={budgets}"
    )

    decoders: dict[str, Any] = {}
    frame_chunks: list[torch.Tensor] = []
    total_duration_sec = 0.0
    stitch_log: list[dict] = []

    for (segment, indices), budget in zip(segments_with_indices, budgets):
        if budget <= 0:
            logger.info(
                f"[qa][stitch] seg={segment.get('segment_idx')} skipped (budget=0, "
                f"stored_frames={len(indices)})"
            )
            continue

        video_path = str(segment.get("video_path") or default_video_path or "")
        if not video_path:
            raise ValueError(f"Segment {segment.get('node_id')} missing video_path")

        picked = _subsample_sorted_indices(indices, budget)
        if not picked:
            continue

        if video_path not in decoders:
            decoders[video_path] = _video_decoder(video_path)

        frames = get_video_frames(decoders[video_path], picked)
        if frames is None or (hasattr(frames, "numel") and frames.numel() == 0):
            logger.warning(
                f"[qa][stitch] seg={segment.get('segment_idx')} decode returned empty "
                f"for indices={picked[:5]}{'...' if len(picked) > 5 else ''}"
            )
            continue

        frame_chunks.append(frames)
        chunk_shape = tuple(frames.shape)
        logger.info(
            f"[qa][stitch] seg={segment.get('segment_idx')} "
            f"decoded {len(picked)}/{len(indices)} frames "
            f"(budget={budget}) chunk_shape={chunk_shape} "
            f"src_indices=[{picked[0]}..{picked[-1]}]"
        )
        time_range = segment.get("time_range_sec")
        if time_range is not None and len(time_range) == 2:
            total_duration_sec += max(float(time_range[1]) - float(time_range[0]), 0.0)

        stitch_log.append(
            {
                "node_id": segment.get("node_id"),
                "segment_idx": segment.get("segment_idx"),
                "retrieval_score": segment.get("retrieval_score"),
                "frame_range": segment.get("frame_range"),
                "time_range_sec": segment.get("time_range_sec"),
                "n_stored_frames": len(indices),
                "frame_budget": budget,
                "n_decoded_frames": len(picked),
                "frame_indices": picked,
                "chunk_shape": chunk_shape,
            }
        )

    if not frame_chunks:
        raise ValueError("Failed to decode any frames from retrieved event values")

    video_tensor = torch.cat(frame_chunks, dim=0) if len(frame_chunks) > 1 else frame_chunks[0]
    n_frames = int(video_tensor.shape[0])
    logger.info(
        f"[qa][stitch] video_tensor shape={tuple(video_tensor.shape)} "
        f"dtype={video_tensor.dtype} device={video_tensor.device} "
        f"({n_frames} frames from {len(stitch_log)} events)"
    )
    video_metadata = {
        "fps": n_frames / max(total_duration_sec, 1e-6),
        "frames_indices": list(range(n_frames)),
        "total_num_frames": n_frames,
    }
    return video_tensor, video_metadata, stitch_log


def answer_from_retrieval(
    event_graph,
    question: str,
    *,
    param: VLQueryParam | None = None,
    default_video_path: str | None = None,
    max_frames: int = 128,
    max_new_tokens: int = 128,
    model=None,
    processor=None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """End-to-end E→F→G: query graph, load values, stitch pixels, VLM answer."""
    from morph_graph.inference.framework.vlm import (
        build_messages_open_qa,
        load_model,
        run_inference_on_video_tensor,
    )

    param = param or VLQueryParam(mode="local")
    retrieval = event_graph.query(question, param)
    segments = retrieval.get("segments") or []
    if not segments:
        logger.warning(f"[qa] no segments retrieved for query={question!r}")
        return {**retrieval, "answer": "", "stitch_log": [], "qa_error": "no segments retrieved"}

    _log_retrieved_segments(question, retrieval)

    try:
        segments_with_values = fetch_event_values(event_graph.value_store, segments)
        video_tensor, video_metadata, stitch_log = stitch_retrieved_event_values(
            segments_with_values,
            default_video_path=default_video_path,
            max_frames=max_frames,
        )
    except (ValueError, OSError, FileNotFoundError) as exc:
        logger.warning(f"[qa] stitch failed: {exc}")
        return {**retrieval, "answer": "", "stitch_log": [], "qa_error": str(exc)}

    if model is None or processor is None:
        if not model_path:
            raise ValueError("model_path required when model/processor are not provided")
        logger.info(f"[qa][llm] loading VLM from {model_path}")
        model, processor = load_model(model_path)

    messages = build_messages_open_qa(question)
    logger.info(f"[qa][llm] running inference on stitched video_tensor {tuple(video_tensor.shape)}")
    answer = run_inference_on_video_tensor(
        model,
        processor,
        video_tensor,
        messages,
        video_metadata=video_metadata,
        max_new_tokens=max_new_tokens,
    )
    logger.info(f"[qa][llm] answer ({len(answer)} chars): {answer}")
    return {
        **retrieval,
        "segments": segments_with_values,
        "answer": answer,
        "stitch_log": stitch_log,
        "video_metadata": video_metadata,
        "stitched_video_shape": tuple(video_tensor.shape),
        "qa_error": None,
    }
