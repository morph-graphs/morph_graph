"""Qwen3-VL visual feature extractor.

Runs the vision tower in isolation from the LLM (FluxMem-style separation):
`self.visual(pixel_values_videos, grid_thw=...)` -> merged tokens (dim 4096),
which we split into temporal units and mean-pool. Uses the native video path,
so 2 frames are fused per unit by `temporal_patch_size=2`.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from ezcolorlog import root_logger as logger
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import VisualFeatureExtractor, VisualFeatures, register_extractor

# model_zoo symlink lives at morph_graph/model_zoo/
_DEFAULT_MODEL_PATH = str(
    Path(os.environ.get("MORPH_GRAPH_MODEL_ROOT", Path(__file__).resolve().parents[1] / "model_zoo"))
    / "Qwen3-VL-8B-Instruct"
)


class Qwen3VLFeatureExtractor(VisualFeatureExtractor):
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        *,
        # Preferred knob: target tokens per frame (== tokens per temporal unit).
        # One merged token covers a `factor x factor` px region where
        # factor = patch_size * spatial_merge_size (=32). We pin the per-frame
        # area to `tokens_per_frame * factor**2` at encode time.
        tokens_per_frame: int | None = 256,
        # Manual override (used only when tokens_per_frame is None). These map to
        # the video processor's `size` and are TOTAL-across-frames budgets:
        # shortest_edge = min, longest_edge = max (smart_resize scales to fit).
        min_pixels: int = 256 * 32 * 32,
        total_pixels: int = 20480 * 32 * 32,
        # Which vision-tower stage to emit (lets us A/B TSM on either):
        #   "post" -> merged tokens after the spatial merger/projector (dim 4096,
        #             LLM input space; these are the reusable values).
        #   "pre"  -> hidden states before the merger (vision hidden_size, 4x more
        #             tokens/unit; not in LLM space -- analysis only).
        feature_stage: str = "post",
        dtype: torch.dtype | None = None,
    ):
        if feature_stage not in ("post", "pre"):
            raise ValueError(f"feature_stage must be 'post' or 'pre', got {feature_stage!r}")
        self.model_path = model_path
        self.tokens_per_frame = tokens_per_frame
        self.min_pixels = min_pixels
        self.total_pixels = total_pixels
        self.feature_stage = feature_stage
        self._dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        logger.info(f"Loading {self.name} vision extractor from {model_path} (dtype={self._dtype})")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=self._dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Vision tower, decoupled from the LLM.
        self.visual = self.model.model.visual
        self._merge = int(self.visual.spatial_merge_size)
        vcfg = self.model.config.vision_config
        self._dim = int(vcfg.out_hidden_size)       # post-projection (merged)
        self._vhid = int(vcfg.hidden_size)          # pre-projection (pre-merger)
        self._temporal_patch_size = int(vcfg.temporal_patch_size)
        self._patch = int(vcfg.patch_size)
        # px per merged token along one side (16 * 2 = 32)
        self._factor = self._patch * self._merge
        logger.info(
            f"Loaded {self.name}: stage={self.feature_stage} "
            f"dim={self.feature_dim} (post={self._dim}, pre={self._vhid}), merge={self._merge}, "
            f"temporal_patch_size={self._temporal_patch_size}, tokens_per_frame={self.tokens_per_frame}"
        )

    def _size_for(self, n_frames: int) -> dict[str, int]:
        """Video-processor `size` budget (total across frames). When
        `tokens_per_frame` is set, pin per-frame area to `tpf * factor**2`."""
        if self.tokens_per_frame is None:
            return {"shortest_edge": self.min_pixels, "longest_edge": self.total_pixels}
        per_frame = self.tokens_per_frame * self._factor**2
        # smart_resize rounds frames up to an even count (temporal_patch_size=2).
        t_bar = -(-n_frames // self._temporal_patch_size) * self._temporal_patch_size
        budget = per_frame * t_bar
        return {"shortest_edge": budget, "longest_edge": budget}  # equal => pinned

    @property
    def name(self) -> str:
        return "qwen3_vl_8b"

    @property
    def feature_dim(self) -> int:
        return self._vhid if self.feature_stage == "pre" else self._dim

    @property
    def temporal_patch_size(self) -> int:
        return self._temporal_patch_size

    @torch.inference_mode()
    def encode_frames(self, frames: torch.Tensor) -> VisualFeatures:
        # Preprocess our torchcodec frames directly; do_sample_frames=False so
        # the processor keeps exactly the frames we sampled.
        proc = self.processor.video_processor.preprocess(
            videos=[frames],
            return_tensors="pt",
            do_sample_frames=False,
            size=self._size_for(int(frames.shape[0])),
        )
        grid_thw = proc["video_grid_thw"]  # (1, 3) -> [t, h, w] in patch units
        vis_device = next(self.visual.parameters()).device
        pixel_values = proc["pixel_values_videos"].to(vis_device, self.visual.dtype)
        grid_thw = grid_thw.to(vis_device)

        # `pooler_output` = merged (post-projection) tokens; `last_hidden_state`
        # = pre-merger tokens. Deepstack features dropped (only needed for LLM
        # re-injection). See feature_stage.
        out = self.visual(pixel_values, grid_thw=grid_thw, return_dict=True)
        t, h, w = (int(x) for x in grid_thw[0].tolist())
        n_units = t

        if self.feature_stage == "pre":
            tokens = out.last_hidden_state       # (t * h * w, vhid)
            gh, gw = h, w                        # pre-merge patch grid
            dim = self._vhid
        else:
            tokens = out.pooler_output           # (t * h/merge * w/merge, 4096)
            gh, gw = h // self._merge, w // self._merge
            dim = self._dim

        tokens_per_unit = gh * gw
        # (n_units, tokens_per_unit, dim); mean over spatial tokens -> pooled.
        patch_tokens = tokens.reshape(n_units, tokens_per_unit, dim)
        pooled = patch_tokens.mean(dim=1)
        logger.info(
            f"Encoded {n_units} units x {tokens_per_unit} tokens "
            f"(stage={self.feature_stage}, grid {gh}x{gw}, dim={dim})"
        )

        return VisualFeatures(
            patch_tokens=patch_tokens.contiguous(),
            pooled=pooled.contiguous(),
            grid_hw=(gh, gw),
            dim=dim,
            model_name=self.name,
        )


register_extractor("qwen3_vl_8b", Qwen3VLFeatureExtractor)
