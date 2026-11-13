"""Import first in every epic/ script that uses EGAgent: puts the upstream code on sys.path and holds the few
things the scripts share (memory file names, time formats, the paper's models). Imports nothing heavy.

Upstream is not installed; its modules are the clone's root (paths, utils, retrieval_model), egagent/ and
prepare_datasources/. Its settings come from the environment (our patch, ../epic_morph.patch):
EGAGENT_DATASET, EGAGENT_BACKBONE, EGAGENT_DB_ROOT, EGAGENT_MODEL_ROOT, OPENAI_API_KEY, GOOGLE_API_KEY.
"""
import os
import sys

R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "egagent"))
for p in ("prepare_datasources", "egagent", ""):
    sys.path.insert(0, os.path.join(R, p) if p else R)

# the paper's configuration: arm -> the agent's backbone; the memory is the same for every arm
BACKBONES = {"gpt41": "gpt-4.1", "gemini25pro": "gemini-2.5-pro"}
MEMORY_LLM = "gpt-4.1"                               # captions, graph extraction, relationship time stamps
EMBEDDER = "google/siglip2-giant-opt-patch16-384"    # frame / text-query embeddings (open weights, local)
CLIP_S = 30                                          # caption window (paper: 30 s clips at 1 fps)


def memory_files(video):
    """The memory of one video, as flat files of its results folder (what `upload` shares)."""
    return [f"{video}_frames.npy", f"{video}_captions.json", f"{video}_graph.json"]


def hhmmss(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}{sec % 3600 // 60:02d}{sec % 60:02d}"


def clock(sec, ms=False):
    s = hhmmss(sec)
    return f"{s[:2]}:{s[2:4]}:{s[4:]}" + (",000" if ms else "")


def frame_files(frames_dir):
    """The 1 fps frames in time order: frame i (0-based) is named for second i + 1, the time it is complete."""
    return sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
