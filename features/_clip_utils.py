"""Shared helpers for CLIP-family (per-frame) feature extractors."""

from __future__ import annotations

from typing import Iterator

import torch


def frames_to_pil(frames: torch.Tensor) -> list:
    """`(T, C, H, W)` uint8 -> list of PIL.Image (RGB)."""
    from PIL import Image

    arr = frames.detach().cpu()
    if arr.dtype != torch.uint8:
        arr = arr.clamp(0, 255).to(torch.uint8)
    hwc = arr.permute(0, 2, 3, 1).numpy()  # (T, H, W, C)
    return [Image.fromarray(f, mode="RGB") for f in hwc]


def batched(seq: list, batch_size: int) -> Iterator[list]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]
