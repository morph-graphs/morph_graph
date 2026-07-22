"""Convert (VisualFeatures, EventSegmentation) into event 'segment' records.

Bridges step A/B output to the nano-graphrag segment schema so the existing
`vl_ops` builders (KV upsert, temporal + semantic edges) apply unchanged. We keep
the `{video_id}::seg_{idx:04d}` node-id scheme (not `evt_`) so the existing
`build_temporal_edges` cross-batch logic works verbatim — here "segment" == event.

Each record carries JSON-safe metadata plus two private fields the orchestrator
strips before KV upsert:
  - `_key`   : np.float32 event key vector (mean of unit `pooled`, L2-normalized).
  - `_value` : dict of value tensors persisted to the EventValueStore.
"""

from __future__ import annotations

from typing import Any

import torch


def _node_id(video_id: str, idx: int) -> str:
    return f"{video_id}::seg_{idx:04d}"


def build_event_segments(
    video_id: str,
    video_path: str,
    features,
    segmentation,
    frame_indices: list[int],
    *,
    aligned_pooled=None,
    fps: float = 1.0,
    temporal_patch_size: int = 2,
    store_patch_tokens: bool = True,
) -> list[dict[str, Any]]:
    """One record per event span in `segmentation.events`.

    `frame_indices` are the source frame indices returned by `decode_frames`
    (len T == n sampled frames). Unit u covers sampled frames
    `[u*temporal_patch_size, (u+1)*temporal_patch_size)`.

    Event *key* (retrieval): if `aligned_pooled` `(T, dim)` (per-frame CLIP-family
    embeddings over the same sampled frames) is given, the key is the mean of the
    aligned rows over the event's sampled-frame slice (text-aligned space, step E).
    Otherwise it falls back to the mean of the Qwen3 unit `pooled` (vision space).
    """
    pooled = features.pooled  # (n_units, dim)
    patch_tokens = features.patch_tokens  # (n_units, tokens, dim) or list
    deepstack_features = getattr(features, "deepstack_features", None)
    n_sampled = len(frame_indices)

    segments: list[dict[str, Any]] = []
    for idx, (u0, u1) in enumerate(segmentation.events):
        # map units -> sampled-frame slice
        fs0 = min(u0 * temporal_patch_size, n_sampled)
        fs1 = min(u1 * temporal_patch_size, n_sampled)

        # event key, L2-normalized for cosine.
        if aligned_pooled is not None:
            key = aligned_pooled[fs0:fs1].float().mean(dim=0)
        else:
            key = pooled[u0:u1].float().mean(dim=0)
        key = key / (key.norm() + 1e-6)

        # sampled-frame slice -> source frame indices
        member_frames = [int(frame_indices[i]) for i in range(fs0, fs1)]
        if member_frames:
            frame_range = [member_frames[0], member_frames[-1] + 1]
        else:
            frame_range = [0, 0]

        # approximate seconds from the sampling fps (uniform sampling assumption).
        time_range_sec = [round(fs0 / fps, 3), round(fs1 / fps, 3)]

        value: dict[str, torch.Tensor] = {
            "unit_range": torch.tensor([u0, u1]),
            "frame_indices": torch.tensor(member_frames, dtype=torch.long),
            "grid_hw": torch.tensor(features.grid_hw, dtype=torch.long),
        }
        if store_patch_tokens and not isinstance(patch_tokens, list):
            value["patch_tokens"] = patch_tokens[u0:u1].to(torch.float16).cpu()
            if deepstack_features is not None:
                value["deepstack_features"] = [
                    stream[u0:u1].to(torch.float16).cpu()
                    for stream in deepstack_features
                ]

        node_id = _node_id(video_id, idx)
        segments.append(
            {
                "node_id": node_id,
                "source_id": node_id,
                "video_id": video_id,
                "video_path": video_path,
                "segment_idx": idx,
                # human-readable label + frame span (the plotter parses "frames a-b").
                "scene_label": f"event {idx}",
                "description": f"event {idx} frames {frame_range[0]}-{frame_range[1]}",
                "unit_range": [int(u0), int(u1)],
                "n_units": int(u1 - u0),
                "frame_range": frame_range,
                "frame_indices": member_frames,
                "time_range_sec": time_range_sec,
                "_key": key.detach().cpu().numpy().astype("float32"),
                "_value": value,
            }
        )
    return segments
