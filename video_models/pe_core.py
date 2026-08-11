"""PE-Core-ViT-L aligned feature extractor (Meta Perception Encoder).

Same role as the SigLIP2 extractor (per-frame joint-space keys + text queries),
but backed by Meta's `perception_models` package. That package isn't a standard
pip dep, so it's imported lazily; install from
https://github.com/facebookresearch/perception_models to use this extractor.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from ezcolorlog import root_logger as logger

from ._clip_utils import batched, frames_to_pil
from .base import AlignedFeatureExtractor, VisualFeatures, register_extractor

_DEFAULT_CONFIG = "PE-Core-L14-336"

# Meta's `perception_models` isn't pip-installable as `core`; it's used from a
# source checkout. Point MORPH_GRAPH_PERCEPTION_MODELS at that repo root, else we
# fall back to the known local checkout.
_PE_REPO = os.environ.get(
    "MORPH_GRAPH_PERCEPTION_MODELS",
    "/fs/cfar-projects/mem_open_vocab/perception_models",
)


def _ensure_pe_on_path() -> None:
    pe_repo = os.environ.get(
        "MORPH_GRAPH_PERCEPTION_MODELS",
        "/fs/cfar-projects/mem_open_vocab/perception_models",
    )
    if Path(pe_repo, "core", "vision_encoder", "pe.py").exists():
        if pe_repo in sys.path:
            sys.path.remove(pe_repo)
        sys.path.insert(0, pe_repo)



class PECoreFeatureExtractor(AlignedFeatureExtractor):
    def __init__(
        self,
        config_name: str = _DEFAULT_CONFIG,
        *,
        pretrained: bool | str = True,
        batch_size: int = 64,
    ):
        _ensure_pe_on_path()
        try:
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as pe_transforms
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PE-Core requires Meta's `perception_models` package "
                "(https://github.com/facebookresearch/perception_models). "
                f"Looked in {_PE_REPO}; set MORPH_GRAPH_PERCEPTION_MODELS to override."
            ) from e

        self.config_name = config_name
        self.batch_size = int(batch_size)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading {self.name} config={config_name} pretrained={pretrained}")
        self.model = pe.CLIP.from_config(config_name, pretrained=pretrained).to(self._device).eval()
        self.preprocess = pe_transforms.get_image_transform(self.model.image_size)
        self.tokenizer = pe_transforms.get_text_tokenizer(self.model.context_length)
        # CLIP dim (L/14 -> 1024); refined from the first encode output.
        self._dim = int(getattr(self.model, "output_dim", 1024))
        logger.info(f"Loaded {self.name}: image_size={self.model.image_size} dim~{self._dim}")

    @property
    def name(self) -> str:
        return "pe_core_vit_l"

    @property
    def feature_dim(self) -> int:
        return self._dim

    def _autocast(self):
        return torch.autocast(self._device) if self._device == "cuda" else torch.autocast("cpu", enabled=False)

    @torch.inference_mode()
    def encode_frames(self, frames: torch.Tensor) -> VisualFeatures:
        images = frames_to_pil(frames)
        chunks = []
        for batch in batched(images, self.batch_size):
            pixels = torch.stack([self.preprocess(im) for im in batch]).to(self._device)
            with self._autocast():
                feats = self.model.encode_image(pixels)
            chunks.append(feats.float())
        pooled = F.normalize(torch.cat(chunks, dim=0), dim=-1)  # (T, dim)
        self._dim = int(pooled.shape[-1])
        logger.info(f"Encoded {pooled.shape[0]} frames -> pooled {tuple(pooled.shape)}")

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
        tokens = self.tokenizer(texts).to(self._device)
        with self._autocast():
            feats = self.model.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)


register_extractor("pe_core_vit_l", PECoreFeatureExtractor)
