import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import torch
import numpy as np

from .._utils import EmbeddingFunc, logger, wrap_embedding_func_with_attrs

_MODEL_ROOT = Path(
    os.environ.get(
        "MORPH_GRAPH_MODEL_ROOT",
        Path(__file__).resolve().parents[3] / "models",
    )
)
_EMBEDDING_SRC = (
    Path(__file__).resolve().parents[2]
    / "embeddings_zoo"
    / "qwen3_vl_embedding"
    / "src"
)

_EMBEDDERS: dict[str, Any] = {}
_EMBEDDING_FUNCS: dict[str, EmbeddingFunc] = {}


@dataclass(frozen=True)
class VideoHyperparams:
    """Video preprocessing and model-load defaults from vsi_poc/eval_qwen3vl.py."""

    min_pixels: int = 64 * 32 * 32
    max_pixels: int = 1800 * 32 * 32
    total_pixels: int = 20480 * 32 * 32
    fps: float = 1.0
    max_frames: int = 2048
    torch_dtype: str = "auto"
    attn_implementation: str = "flash_attention_2"


def _resolve_torch_dtype(name: str):
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, name)


@dataclass(frozen=True)
class VLEmbeddingSpec:
    model_path: Path
    embedding_dim: int
    max_token_size: int = 8192
    default_instruction: str = "Represent the user's input."
    video_hyperparams: VideoHyperparams = field(default_factory=VideoHyperparams)


VLEMBEDDING_REGISTRY: dict[str, VLEmbeddingSpec] = {
    "qwen3_vl_embedding_2b": VLEmbeddingSpec(
        model_path=_MODEL_ROOT / "Qwen3-VL-Embedding-2B",
        embedding_dim=2048,
    ),
    "qwen3_vl_embedding_8b": VLEmbeddingSpec(
        model_path=_MODEL_ROOT / "Qwen3-VL-Embedding-8B",
        embedding_dim=4096,
    ),
}


def list_embedding_keys() -> list[str]:
    return sorted(VLEMBEDDING_REGISTRY)


def get_embedding_spec(key: str) -> VLEmbeddingSpec:
    if key not in VLEMBEDDING_REGISTRY:
        known = ", ".join(list_embedding_keys())
        raise KeyError(f"Unknown VL embedding key {key!r}. Known keys: {known}")
    return VLEMBEDDING_REGISTRY[key]


def get_embedder(key: str):
    """Lazy-loaded singleton Qwen3VLEmbedder per registry key."""
    if key in _EMBEDDERS:
        return _EMBEDDERS[key]

    spec = get_embedding_spec(key)
    if not spec.model_path.exists():
        raise FileNotFoundError(
            f"Embedding model for {key!r} not found at {spec.model_path}. "
            f"Download the checkpoint or set MORPH_GRAPH_MODEL_ROOT."
        )

    src = str(_EMBEDDING_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from models.qwen3_vl_embedding import Qwen3VLEmbedder

    hp = spec.video_hyperparams
    logger.info(f"Loading VL embedder {key} from {spec.model_path}")
    _EMBEDDERS[key] = Qwen3VLEmbedder(
        str(spec.model_path),
        default_instruction=spec.default_instruction,
        min_pixels=hp.min_pixels,
        max_pixels=hp.max_pixels,
        total_pixels=hp.total_pixels,
        fps=hp.fps,
        max_frames=hp.max_frames,
        torch_dtype=_resolve_torch_dtype(hp.torch_dtype),
        attn_implementation=hp.attn_implementation,
    )
    return _EMBEDDERS[key]


def _to_embedder_input(
    content: Any,
    video_hp: VideoHyperparams | None = None,
) -> dict[str, Any]:
    if isinstance(content, str):
        return {"text": content}
    if not isinstance(content, dict):
        raise TypeError(
            f"Unsupported embedding content type {type(content)!r}; "
            "expected str or dict."
        )

    item: dict[str, Any] = {}
    text = content.get("text") or content.get("caption") or content.get("key_text")
    video = content.get("video") or content.get("clip_path") or content.get("video_path")
    if text:
        item["text"] = text
    if video:
        item["video"] = video
        if video_hp is not None:
            item.setdefault("fps", video_hp.fps)
            item.setdefault("max_frames", video_hp.max_frames)
    for key in ("instruction", "fps", "max_frames"):
        if key in content:
            item[key] = content[key]

    if not item:
        raise ValueError(
            "Embedding dicts need text (`text`/`caption`/`key_text`) "
            "and/or video (`video`/`clip_path`/`video_path`)."
        )
    return item


def _embed_sync(key: str, contents: list[Any]) -> np.ndarray:
    video_hp = get_embedding_spec(key).video_hyperparams
    inputs = [_to_embedder_input(content, video_hp) for content in contents]
    embeddings = get_embedder(key).process(inputs, normalize=True)
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    return np.asarray(embeddings, dtype=np.float32)


def get_embedding_func(key: str) -> EmbeddingFunc:
    """Cached EmbeddingFunc wrapper per registry key."""
    if key in _EMBEDDING_FUNCS:
        return _EMBEDDING_FUNCS[key]

    spec = get_embedding_spec(key)

    @wrap_embedding_func_with_attrs(
        embedding_dim=spec.embedding_dim,
        max_token_size=spec.max_token_size,
    )
    async def embedding_func(contents: list[Any]) -> np.ndarray:
        if not contents:
            return np.array([])
        return await asyncio.to_thread(_embed_sync, key, contents)

    _EMBEDDING_FUNCS[key] = embedding_func
    return embedding_func


def resolve_embedding_func(
    embedding_key: str | None,
    embedding_func: EmbeddingFunc | None,
) -> EmbeddingFunc:
    if embedding_func is not None:
        return embedding_func
    if embedding_key is None:
        raise ValueError("Provide embedding_func or embedding_key.")
    return get_embedding_func(embedding_key)
