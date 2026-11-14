#!/usr/bin/env python3
"""EGAgent's visual memory of one video: SigLIP 2 embeddings of its 1 fps frames (GPU; open weights, no API).

    embed_frames.py --frames_dir <work>/frames/<V> --out <results>/<V>_frames.npy

Upstream's own embedding code (retrieval_model.embed_frames_batch, siglip2-giant-opt-patch16-384, the vision
tower's pooled output). Row i is the frame of second i + 1. run_qa.py puts the rows a question may see into
upstream's `frames` SQLite table.
"""
import argparse
import os

import numpy as np
from PIL import Image

import eg_paths

os.environ.setdefault("EGAGENT_MODEL_ROOT", os.path.dirname(eg_paths.EMBEDDER))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=256, help="frames held in RAM at once")
    a = ap.parse_args()
    files = eg_paths.frame_files(a.frames_dir)
    assert files, f"no frames in {a.frames_dir}"
    from retrieval_model import device, embed_frames_batch   # loads the model
    embs = []
    for i in range(0, len(files), a.chunk):
        images = [Image.open(os.path.join(a.frames_dir, f)).convert("RGB") for f in files[i:i + a.chunk]]
        embs.append(embed_frames_batch(images, device, batch_size=a.batch_size).astype("float32"))
    embs = np.vstack(embs)
    assert embs.shape[0] == len(files) and np.isfinite(embs).all(), embs.shape
    np.save(a.out + ".tmp.npy", embs)
    os.replace(a.out + ".tmp.npy", a.out)
    print(f"{embs.shape[0]} frames x {embs.shape[1]} dims -> {a.out}")


if __name__ == "__main__":
    main()
