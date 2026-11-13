"""Import first in every epic/ script that uses WorldMM: puts the WorldMM submodule's code on sys.path.

WorldMM is not installed; its packages are the submodule's src/ (worldmm, the vendored HippoRAG) and its
script folders (preprocess/, eval/). worldmm.memory is imported here, before anything imports hipporag:
the worldmm <-> hipporag import cycle otherwise leaves worldmm.memory.episodic.memory holding the
hipporag.HippoRAG module instead of the class.
"""
import os
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WorldMM")
for p in ("eval", "preprocess/episodic_memory", "preprocess", "src/HippoRAG/src", "src"):
    sys.path.insert(0, os.path.normpath(os.path.join(R, p)))

import worldmm.memory  # noqa: E402,F401

GRANULARITIES = ["10sec", "30sec", "3min", "10min"]   # WorldMM's per-video (Video-MME) granularities
MODELS = {"m8": "qwen3vl-8b", "m27": "qwen38-27b"}    # arm -> WorldMM model name (one backbone for every role)


def memory_files(video):
    """The memory of one video and arm, as flat files of its results folder (what `upload` shares):
    captions per granularity, the semantic snapshots, the visual embeddings, HippoRAG's OpenIE per granularity."""
    return ([f"{video}_{g}.json" for g in GRANULARITIES] + [f"{video}_semantic.json", f"{video}_visual.pkl"]
            + [f"{video}_openie_{g}.json" for g in GRANULARITIES])
