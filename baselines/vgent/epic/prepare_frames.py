#!/usr/bin/env python3
"""Decode one HD-EPIC video into the frames Vgent works on, kept on disk instead of in memory.

Upstream (models/qwenvl.py load_video) reads the whole video at 1 fps with decord at full resolution and
resizes it in memory: ~21 GB of raw frames plus ~16 GB of floats per hour of 1408x1408 video. This writes the
same frames (upstream's frame indices, upstream's target size, bicubic with antialias) to a uint8 .npy that
the later stages memory-map:

  <out>/<video>.npy    (n, 3, H, W) uint8
  <out>/<video>.json   n, H, W, fps (upstream's sample fps), times (each frame's time in the video, seconds)

Two differences from upstream, both from not holding the raw video: the resized floats are rounded to uint8,
and the answer step downsizes these frames again instead of the full-resolution ones.

    python prepare_frames.py --video <mp4> --out <dir> --name <video id>
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torchvision.transforms import InterpolationMode, functional as TF

sys.path.insert(0, os.environ.get("VGENT_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Vgent"))
from models.utils import VIDEO_MAX_PIXELS, smart_nframes, smart_resize  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--fps", type=float, default=1.0)            # upstream defaults
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--total_pixels", type=int, default=16384)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    npy, meta = os.path.join(a.out, a.name + ".npy"), os.path.join(a.out, a.name + ".json")
    if os.path.exists(npy) and os.path.exists(meta):
        print(f"frames exist: {npy}")
        return
    os.makedirs(a.out, exist_ok=True)
    import decord
    vr = decord.VideoReader(a.video, num_threads=a.threads)
    total, vfps = len(vr), vr.get_avg_fps()
    # models/utils.py _read_video_decord
    n = smart_nframes({"fps": a.fps}, total_frames=total, video_fps=vfps)
    idx = torch.linspace(0, total - 1, n).round().long().tolist()
    h, w = vr[0].shape[:2]
    # models/qwenvl.py load_video + models/utils.py resize_video: the per-frame size for the whole video
    total_px = a.total_pixels * max(1, math.ceil(n / a.chunk_size)) * 28 * 28
    min_px = 16 * 28 * 28
    assert n <= int(total_px / (min_px * 1.05) * 2), "resize_video would subsample the frames"
    max_px = min(total_px, max(min(VIDEO_MAX_PIXELS, total_px / n * 2.0), int(min_px * 1.05)))
    H, W = smart_resize(h, w, factor=28, min_pixels=min_px, max_pixels=max_px)
    print(f"{a.video}: {total} frames at {vfps:.2f} fps, {h}x{w} -> {n} frames at {H}x{W}", flush=True)
    out = np.lib.format.open_memmap(npy + ".tmp", mode="w+", dtype=np.uint8, shape=(n, 3, H, W))
    for s in range(0, n, 32):
        x = torch.from_numpy(vr.get_batch(idx[s:s + 32]).asnumpy()).permute(0, 3, 1, 2)
        x = TF.resize(x, [H, W], interpolation=InterpolationMode.BICUBIC, antialias=True)
        out[s:s + len(x)] = x.numpy()
        if s % 320 == 0:
            print(f"  {s}/{n}", flush=True)
    out.flush()
    del out
    os.replace(npy + ".tmp", npy)
    json.dump({"video": a.name, "n": n, "H": H, "W": W, "source": [h, w], "video_fps": vfps,
               "fps": round(n / max(total, 1e-6) * vfps, 2), "times": [round(i / vfps, 3) for i in idx]},
              open(meta, "w"))
    print(f"wrote {npy}: {n} frames, {os.path.getsize(npy) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
