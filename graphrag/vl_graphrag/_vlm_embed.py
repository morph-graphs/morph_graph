import asyncio
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal
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

EmbedMode = Literal["text", "video"]

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
    device_map: str | None = "auto"


def _resolve_torch_dtype(name: str):
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, name)


@dataclass(frozen=True)
class VLEmbeddingSpec:
    model_path: Path
    embedding_dim: int
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


def get_embedding_spec(model_key: str) -> VLEmbeddingSpec:
    if model_key not in VLEMBEDDING_REGISTRY:
        known = ", ".join(list_embedding_keys())
        raise KeyError(f"Unknown VL embedding key {model_key!r}. Known keys: {known}")
    return VLEMBEDDING_REGISTRY[model_key]


def get_embedder(model_key: str):
    """Lazy-loaded singleton Qwen3VLEmbedder per registry key."""
    if model_key in _EMBEDDERS:
        return _EMBEDDERS[model_key]

    spec = get_embedding_spec(model_key)
    if not spec.model_path.exists():
        raise FileNotFoundError(
            f"Embedding model for {model_key!r} not found at {spec.model_path}. "
            f"Download the checkpoint or set MORPH_GRAPH_MODEL_ROOT."
        )

    src = str(_EMBEDDING_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from models.qwen3_vl_embedding import Qwen3VLEmbedder

    hp = spec.video_hyperparams
    logger.info(f"Loading VL embedder {model_key} from {spec.model_path}")
    _EMBEDDERS[model_key] = Qwen3VLEmbedder(
        str(spec.model_path),
        max_length=None, # ensure no max length truncation 
        default_instruction=spec.default_instruction,
        min_pixels=hp.min_pixels,
        max_pixels=hp.max_pixels,
        total_pixels=hp.total_pixels,
        fps=hp.fps,
        max_frames=hp.max_frames,
        torch_dtype=_resolve_torch_dtype(hp.torch_dtype),
        attn_implementation=hp.attn_implementation,
        device_map=hp.device_map if torch.cuda.is_available() else None,
    )
    return _EMBEDDERS[model_key]


def _to_embedder_input(
    content: Any,
    embed_mode: EmbedMode,
    video_hp: VideoHyperparams | None = None,
) -> dict[str, Any]:
    if isinstance(content, str):
        return {"text": content}
    if not isinstance(content, dict):
        raise TypeError(
            f"Unsupported embedding content type {type(content)!r}; "
            "expected str or dict."
        )

    if embed_mode == "text":
        text = content.get("text_key") or content.get("caption")
        if not text:
            raise ValueError(
                "Text embedding dicts require `text_key` or `caption`."
            )
        return {"text": text}

    video = content.get("video")
    video_path = content.get("video_path")
    if video is None:
        if not video_path:
            raise ValueError("Video embedding dicts require `video` (decoded frames) or `video_path`")
        video = video_path

    item: dict[str, Any] = {"video": video}
    if "video_metadata" in content:
        item["video_metadata"] = content["video_metadata"]
    if isinstance(video, str):
        if video_hp is not None:
            item.setdefault("fps", video_hp.fps)
            item.setdefault("max_frames", video_hp.max_frames)
    for key in ("instruction", "fps", "max_frames"):
        if key in content:
            item[key] = content[key]
    return item


def _embed_video_tensors(embedder, items: list[dict[str, Any]]):
    """Embed pre-decoded segment tensors via processor (vsi_poc caption_video_tensor path)."""
    import torch.nn.functional as F

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": embedder.default_instruction}],
        },
        {"role": "user", "content": [{"type": "video"}]},
    ]
    text = embedder.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    processor_kwargs: dict[str, Any] = {
        "text": [text] * len(items),
        "videos": [item["video"] for item in items],
        "video_metadata": [item["video_metadata"] for item in items],
        "padding": True,
        "do_resize": False,
        "return_tensors": "pt",
    }
    if embedder.max_length is None:
        processor_kwargs["truncation"] = False
    else:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = embedder.max_length

    inputs = embedder.processor(**processor_kwargs)
    inputs = {k: v.to(embedder.model.device) for k, v in inputs.items()}
    outputs = embedder.forward(inputs)
    embeddings = embedder._pooling_last(
        outputs["last_hidden_state"], outputs["attention_mask"]
    )
    return F.normalize(embeddings, p=2, dim=-1)


def _embed_sync(model_key: str, contents: list[Any], embed_mode: EmbedMode) -> np.ndarray:
    video_hp = get_embedding_spec(model_key).video_hyperparams
    embedder = get_embedder(model_key)
    items = [
        _to_embedder_input(content, embed_mode, video_hp) for content in contents
    ]
    if embed_mode == "video" and items and "video_metadata" in items[0]:
        embeddings = _embed_video_tensors(embedder, items)
    else:
        embeddings = embedder.process(items, normalize=True)
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().float().numpy()
    return np.asarray(embeddings, dtype=np.float32)


def get_embedding_func(model_key: str) -> EmbeddingFunc:
    """Cached EmbeddingFunc wrapper per registry model key."""
    if model_key in _EMBEDDING_FUNCS:
        return _EMBEDDING_FUNCS[model_key]

    spec = get_embedding_spec(model_key)

    @wrap_embedding_func_with_attrs(
        embedding_dim=spec.embedding_dim,
        max_token_size=0,
    )
    async def embedding_func(
        contents: list[Any], embed_mode: EmbedMode = "text"
    ) -> np.ndarray:
        if not contents:
            return np.array([])
        return await asyncio.to_thread(_embed_sync, model_key, contents, embed_mode)

    _EMBEDDING_FUNCS[model_key] = embedding_func
    return embedding_func


def bind_embed_mode(embedding_func: EmbeddingFunc, embed_mode: EmbedMode):
    """Bind embed_mode for VDB calls; preserve attrs needed by vector storage."""
    bound = partial(embedding_func, embed_mode=embed_mode)
    bound.embedding_dim = embedding_func.embedding_dim
    return bound


def resolve_embedding_func(
    embedding_model_key: str | None,
    embedding_func: EmbeddingFunc | None,
) -> EmbeddingFunc:
    if embedding_func is not None:
        return embedding_func
    if embedding_model_key is None:
        raise ValueError("Provide embedding_func or embedding_model_key.")
    return get_embedding_func(embedding_model_key)
