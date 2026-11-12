#!/usr/bin/env python3
"""Caption every 30 s clip with a local Qwen3-VL (MERIT's own pipeline uses GPT-5-mini).

Reads <subject>_30sec_skeleton.json from prepare_epic.py, writes <subject>_30sec.json with a
"text" field per clip. Resumes: clips that already have text are skipped.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MERIT", "src"))
from merit.llm import LLMModel  # noqa: E402

# The LVBench caption prompt from MERIT, switched to first person because HD-EPIC is
# egocentric and the questions are asked as "I" (matching the EgoLife captions' style).
PROMPT = """You are an expert video understanding assistant.

You receive frames, one per second, from a ~30-second clip of an egocentric (head-mounted camera) video recorded in a kitchen.
Write a single, information-rich paragraph that can replace the video in a long document.

Rules:
- Write in the first person ("I"), as the camera wearer.
- Describe only what is visible in the frames.
- Name every object, ingredient, container, appliance and location I interact with, and what I do with it (take, put, open, close, pour, add, stir, wash, weigh, cut ...), in order.
- Note where things come from and where they are put (fridge, cupboard, drawer, hob, counter, sink, which bowl or pan).
- Do NOT mention clip length, timestamps, or phrases like "this clip".
- Stay factual. No speculation.

Now write one coherent paragraph (roughly 120-220 words) that describes what I do in these frames.

Caption:"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skeleton", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="qwen3vl-8b")
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--max_clips", type=int, default=None, help="caption only the first N uncaptioned clips (smoke test)")
    args = p.parse_args()

    clips = json.load(open(args.skeleton))
    if os.path.exists(args.output):
        done = {c["start_time"]: c.get("text") for c in json.load(open(args.output))}
        for c in clips:
            if done.get(c["start_time"]):
                c["text"] = done[c["start_time"]]
    todo = [c for c in clips if not c.get("text")][: args.max_clips]
    print(f"{len(clips)} clips, {len(todo)} to caption", flush=True)
    if not todo:
        return

    llm = LLMModel(model_name=args.model, max_new_tokens=512)
    for n, c in enumerate(todo, 1):
        frames = sorted(glob.glob(os.path.join(c["frame_dir"], "*.jpg")))
        step = max(1, len(frames) // args.max_frames)
        frames = frames[::step][: args.max_frames]
        msg = [{"role": "user", "content": [{"type": "image", "image": f} for f in frames]
                + [{"type": "text", "text": PROMPT}]}]
        t0 = time.time()
        c["text"] = llm.generate(msg).strip()
        print(f"[{n}/{len(todo)}] {c['start_time']} {len(frames)} frames {time.time() - t0:.1f}s: "
              f"{c['text'][:120]}", flush=True)
        json.dump(clips, open(args.output, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
