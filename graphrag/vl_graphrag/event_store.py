"""CPU-resident event value store + precomputed embedding func.

The event *keys* are already vectors (mean-pooled Qwen3-VL vision features), so we
feed them through the existing NanoVectorDB / `vl_ops` edge builders via an
identity ("precomputed") embedding func instead of re-embedding.

The event *values* are cached `patch_tokens` and Qwen DeepStack streams, which
don't fit the JSON KV store. `EventValueStore` keeps them in CPU RAM keyed by
node ID for the lifetime of one EventGraph worker. Pooled TSM features are
transient and not stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """CPU-memory store for per-event value tensors (the 'V' in the k/V store)."""

    _values: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def _to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach()
            return value.to(dtype=torch.float16).cpu() if value.is_floating_point() else value.cpu()
        if isinstance(value, list):
            return [EventValueStore._to_cpu(item) for item in value]
        return value

    def save(self, node_id: str, value: dict[str, Any]) -> None:
        self._values[node_id] = {
            key: self._to_cpu(item) for key, item in value.items()
        }

    def load(self, node_id: str) -> dict[str, Any]:
        try:
            return self._values[node_id]
        except KeyError as exc:
            raise KeyError(
                f"No in-memory value for {node_id!r}; re-index in this worker before QA"
            ) from exc

    def load_many(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {node_id: self.load(node_id) for node_id in node_ids}
