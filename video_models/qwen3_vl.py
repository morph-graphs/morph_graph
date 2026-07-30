"""Qwen3-VL visual feature extractor.

Runs the vision tower in isolation from the LLM (FluxMem-style separation):
`self.visual(pixel_values_videos, grid_thw=...)` -> merged tokens (dim 4096),
which we split into temporal units and mean-pool. Uses the native video path,
so 2 frames are fused per unit by `temporal_patch_size=2`.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
from ezcolorlog import root_logger as logger
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils import is_flash_attn_2_available

from .base import (
    VideoLLM,
    VisualFeatureExtractor,
    VisualFeatures,
    register_extractor,
    register_video_llm,
)

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
            attn_implementation=(
                "flash_attention_2"
                if (torch.cuda.is_available() and is_flash_attn_2_available())
                else "sdpa"
            ),
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
    def encode_frames(
        self, frames: torch.Tensor, frame_batch_size: int | None = None
    ) -> VisualFeatures:
        """Encode frames, optionally in fixed-size batches to bound VRAM.

        Qwen's vision tower groups frames by ``temporal_patch_size``.  Batches
        must therefore preserve that grouping: pad the final batch to an even
        number of frames, then discard only the padded temporal unit(s).
        """
        if frame_batch_size is not None:
            if frame_batch_size < self._temporal_patch_size:
                raise ValueError(
                    f"frame_batch_size must be >= {self._temporal_patch_size}"
                )
            if frames.shape[0] > frame_batch_size:
                pooled_chunks = []
                token_chunks = []
                deepstack_chunks = None
                metadata = None
                for start in range(0, frames.shape[0], frame_batch_size):
                    chunk = frames[start : start + frame_batch_size]
                    n_real_frames = chunk.shape[0]

                    if n_real_frames < frame_batch_size:
                        pad_len = frame_batch_size - n_real_frames
                        pad_frames = chunk[-1:].repeat(pad_len, 1, 1, 1)
                        chunk = torch.cat([chunk, pad_frames], dim=0)

                    encoded_chunk = self.encode_frames(
                        chunk, frame_batch_size=frame_batch_size
                    )
                    n_real_units = (
                        n_real_frames + self._temporal_patch_size - 1
                    ) // self._temporal_patch_size

                    pooled_chunks.append(encoded_chunk.pooled[:n_real_units])
                    if encoded_chunk.patch_tokens is not None:
                        token_chunks.append(encoded_chunk.patch_tokens[:n_real_units])
                    if encoded_chunk.deepstack_features is not None:
                        if deepstack_chunks is None:
                            deepstack_chunks = [
                                [] for _ in encoded_chunk.deepstack_features
                            ]
                        for target, features in zip(
                            deepstack_chunks, encoded_chunk.deepstack_features
                        ):
                            target.append(features[:n_real_units])
                    if metadata is None:
                        metadata = encoded_chunk

                assert metadata is not None
                return VisualFeatures(
                    patch_tokens=(
                        torch.cat(token_chunks, dim=0) if token_chunks else None
                    ),
                    pooled=torch.cat(pooled_chunks, dim=0),
                    grid_hw=metadata.grid_hw,
                    dim=metadata.dim,
                    model_name=metadata.model_name,
                    deepstack_features=(
                        [torch.cat(features, dim=0) for features in deepstack_chunks]
                        if deepstack_chunks is not None
                        else None
                    ),
                )

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
        # = pre-merger tokens. DeepStack streams are needed to replay cached
        # visual tokens through Qwen's language path without re-encoding pixels.
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
        deepstack_features = None
        if self.feature_stage == "post" and out.deepstack_features is not None:
            deepstack_features = [
                feature.reshape(n_units, tokens_per_unit, dim).contiguous()
                for feature in out.deepstack_features
            ]
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
            deepstack_features=deepstack_features,
        )

register_extractor("qwen3_vl_8b", Qwen3VLFeatureExtractor)


def answer_from_cached_tokens(
    model,
    processor,
    patch_tokens: torch.Tensor,
    deepstack_features: list[torch.Tensor],
    grid_hw: tuple[int, int],
    question: str,
    *,
    max_new_tokens: int = 32,
) -> str:
    """Generate from cached Qwen visual outputs without another vision pass."""
    if patch_tokens.ndim != 3 or not deepstack_features:
        raise ValueError("Cached patch tokens and DeepStack streams are required")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": question.strip()}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], padding=True, return_tensors="pt")
    input_ids = encoded.input_ids
    video_token_id = int(model.config.video_token_id)
    placeholders = (input_ids == video_token_id).nonzero(as_tuple=False)
    if placeholders.shape[0] != 1:
        raise ValueError(f"Expected one video placeholder, found {placeholders.shape[0]}")

    units, tokens_per_unit, _ = patch_tokens.shape
    count, start = units * tokens_per_unit, int(placeholders[0, 1])
    video_ids = torch.full((1, count), video_token_id, dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids[:, :start], video_ids, input_ids[:, start + 1 :]], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids)
    merge = int(model.config.vision_config.spatial_merge_size)
    video_grid_thw = torch.tensor([[units, grid_hw[0] * merge, grid_hw[1] * merge]], device=device)
    cached_output = SimpleNamespace(
        pooler_output=(patch_tokens.reshape(count, -1).to(device, model.dtype),),
        deepstack_features=[
            stream.reshape(count, -1).to(device, model.dtype) for stream in deepstack_features
        ],
    )
    original = model.model.get_video_features
    model.model.get_video_features = lambda *args, **kwargs: cached_output
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                video_grid_thw=video_grid_thw,
                pixel_values_videos=torch.empty(0, device=device),
                max_new_tokens=max_new_tokens,
            )
    finally:
        model.model.get_video_features = original
    return processor.batch_decode(
        [out[len(inp):] for inp, out in zip(input_ids, generated)],
        skip_special_tokens=True,
    )[0].strip()


class Qwen3VideoLLM(VideoLLM):
    """Qwen3-VL model wrapper owning extraction and cached-token QA."""

    def __init__(self, **extractor_kwargs):
        self._visual_extractor = Qwen3VLFeatureExtractor(**extractor_kwargs)

    @property
    def visual_extractor(self) -> Qwen3VLFeatureExtractor:
        return self._visual_extractor

    def answer_from_event_values(
        self,
        question: str,
        event_values: list[dict],
        *,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict]:
        patch_chunks = []
        deepstack_chunks = None
        grid_hw = None
        for value in event_values:
            patch_tokens = value.get("patch_tokens")
            deepstack = value.get("deepstack_features")
            raw_grid = value.get("grid_hw")
            if patch_tokens is None or deepstack is None or raw_grid is None:
                raise ValueError(
                    "Qwen cached-token QA requires patch_tokens, DeepStack streams, "
                    "and grid_hw; re-index with store_patch_tokens=True"
                )
            current_grid = tuple(int(x) for x in raw_grid.tolist())
            if grid_hw is None:
                grid_hw = current_grid
            elif grid_hw != current_grid:
                raise ValueError("Retrieved event values have incompatible Qwen grids")
            if deepstack_chunks is None:
                deepstack_chunks = [[] for _ in deepstack]
            if len(deepstack) != len(deepstack_chunks):
                raise ValueError("Inconsistent number of Qwen DeepStack streams")
            patch_chunks.append(patch_tokens)
            for chunks, stream in zip(deepstack_chunks, deepstack):
                chunks.append(stream)

        if not patch_chunks or deepstack_chunks is None or grid_hw is None:
            raise ValueError("No cached Qwen event values supplied")
        patch_tokens = torch.cat(patch_chunks, dim=0)
        deepstack_features = [torch.cat(chunks, dim=0) for chunks in deepstack_chunks]
        answer = answer_from_cached_tokens(
            self.visual_extractor.model,
            self.visual_extractor.processor,
            patch_tokens,
            deepstack_features,
            grid_hw,
            question,
            max_new_tokens=max_new_tokens,
        )
        return answer, {
            "cached_token_shape": tuple(patch_tokens.shape),
            "cached_grid_hw": grid_hw,
        }


register_video_llm("qwen3_vl_8b", Qwen3VideoLLM)
