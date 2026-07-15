"""Event value store + precomputed embedding func.

The event *keys* are already vectors (mean-pooled Qwen3-VL vision features), so we
feed them through the existing NanoVectorDB / `vl_ops` edge builders via an
identity ("precomputed") embedding func instead of re-embedding.

The event *values* are tensors (per-unit `pooled` / `patch_tokens`), which don't
fit the JSON KV store. `EventValueStore` persists them to disk (fp16) keyed by
node_id; the JSON KV record only holds a `value_path` pointer + metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .._utils import EmbeddingFunc, logger, wrap_embedding_func_with_attrs


TextEncoder = Callable[[list[str]], Any]


def aligned_embedding_func(
    embedding_dim: int, text_encoder: TextEncoder | None = None
) -> EmbeddingFunc:
    """EmbeddingFunc for the event key VDB.

    Two content types share one index:
      - indexing: `content` is a precomputed event key vector (image side) ->
        passed through unchanged.
      - querying: `content` is a text string -> encoded via `text_encoder` into
        the same aligned space (step E). If no `text_encoder` was provided (keys
        in a non-text-aligned space), text queries fail loudly.

    All vectors are expected L2-normalized already (cosine == dot).
    """

    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=0)
    async def embedding_func(contents: list[Any]) -> np.ndarray:
        if not len(contents):
            return np.array([])
        if all(isinstance(c, str) for c in contents):
            if text_encoder is None:
                raise NotImplementedError(
                    "No text_encoder configured; event keys are not in a "
                    "text-aligned space, so text-query retrieval is unavailable."
                )
            vecs = text_encoder(list(contents))
            if hasattr(vecs, "detach"):
                vecs = vecs.detach().cpu().float().numpy()
            return np.asarray(vecs, dtype=np.float32)
        vecs = [np.asarray(c, dtype=np.float32).reshape(-1) for c in contents]
        return np.stack(vecs, axis=0)

    return embedding_func


def precomputed_embedding_func(embedding_dim: int) -> EmbeddingFunc:
    """Back-compat alias: index-only embedding func (no text queries)."""
    return aligned_embedding_func(embedding_dim, text_encoder=None)


@dataclass
class EventValueStore:
    """On-disk store for per-event value tensors (the 'V' in the k/V store)."""

    working_dir: str
    namespace: str = "event_values"

    def __post_init__(self):
        self.dir = os.path.join(self.working_dir, self.namespace)
        os.makedirs(self.dir, exist_ok=True)

    def path_for(self, node_id: str) -> str:
        safe = node_id.replace("/", "__").replace("::", "--")
        return os.path.join(self.dir, f"{safe}.pt")

    def save(self, node_id: str, value: dict[str, torch.Tensor]) -> str:
        path = self.path_for(node_id)
        torch.save(value, path)
        return path

    def load(self, node_id: str) -> dict[str, torch.Tensor]:
        path = self.path_for(node_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No stored value for {node_id!r} at {path}")
        return torch.load(path, map_location="cpu")

    def load_many(self, node_ids: list[str]) -> dict[str, dict[str, torch.Tensor]]:
        return {node_id: self.load(node_id) for node_id in node_ids}
