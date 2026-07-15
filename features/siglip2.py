"""SigLIP2-ViT-L aligned feature extractor (per-frame CLIP-family).

Emits one L2-normalized joint-space image embedding per frame as `pooled`
(the event *retrieval* key), and `encode_text` for text-query seeding. The
joint space is contrastively aligned, so cosine(text, frame) is meaningful --
unlike the Qwen3-VL vision-pooled space.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from ezcolorlog import root_logger as logger

from ._clip_utils import batched, frames_to_pil
from .base import AlignedFeatureExtractor, VisualFeatures, register_extractor

_HF_ID = "google/siglip2-large-patch16-512"
# Prefer a local model_zoo copy if present, else fall back to the HF hub id.
_LOCAL = Path(
    os.environ.get("MORPH_GRAPH_MODEL_ROOT", Path(__file__).resolve().parents[1] / "model_zoo")
) / "siglip2-large-patch16-512"
_DEFAULT_MODEL_PATH = str(_LOCAL) if _LOCAL.exists() else _HF_ID


class SigLIP2FeatureExtractor(AlignedFeatureExtractor):
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        *,
        batch_size: int = 64,
        dtype: torch.dtype | None = None,
    ):
        from transformers import AutoModel, AutoProcessor

        self.model_path = model_path
        self.batch_size = int(batch_size)
        self._dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        logger.info(f"Loading {self.name} from {model_path} (dtype={self._dtype})")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self._dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self._dim = int(self.model.config.vision_config.hidden_size)
        logger.info(f"Loaded {self.name}: dim={self._dim}")

    @property
    def name(self) -> str:
        return "siglip2_vit_l"

    @property
    def feature_dim(self) -> int:
        return self._dim

    @property
    def _device(self):
        return self.model.device

    @torch.inference_mode()
    def encode_frames(self, frames: torch.Tensor) -> VisualFeatures:
        images = frames_to_pil(frames)
        chunks = []
        for batch in batched(images, self.batch_size):
            inputs = self.processor(images=batch, return_tensors="pt").to(self._device)
            feats = self.model.get_image_features(**inputs)
            chunks.append(feats.float())
        pooled = F.normalize(torch.cat(chunks, dim=0), dim=-1)  # (T, dim)
        self._dim = int(pooled.shape[-1])
        logger.info(f"Encoded {pooled.shape[0]} frames -> pooled {tuple(pooled.shape)}")

        # CLIP-family: no meaningful multi-token grid in the joint space, so the
        # per-frame pooled embedding is exposed as a single "token".
        patch_tokens = pooled.unsqueeze(1).contiguous()  # (T, 1, dim)
        return VisualFeatures(
            patch_tokens=patch_tokens,
            pooled=pooled.contiguous(),
            grid_hw=(1, 1),
            dim=self._dim,
            model_name=self.name,
        )

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        # Processor lowercases + pads to max_length=64 (SigLIP2 training regime).
        inputs = self.processor(
            text=texts, padding="max_length", return_tensors="pt"
        ).to(self._device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats.float(), dim=-1)


register_extractor("siglip2_vit_l", SigLIP2FeatureExtractor)
