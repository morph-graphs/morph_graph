#!/usr/bin/env python3
"""Caption every 30 s clip with the arm's memory model: a local Qwen (m8, m27) or, as in MERIT's own
pipeline, GPT-5-mini through the OpenAI API (g5m; calls counted by openai_usage.py).

Reads <subject>_30sec_skeleton.json from prepare_epic.py, writes <subject>_30sec.json with a
"text" field per clip. Resumes: clips that already have text are skipped. With an API model a clip whose
request failed is left without text and the script exits 1: rerun to retry only those.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    p.add_argument("--concurrency", type=int, default=1, help="clips captioned at once (an API model; 1 for a local one)")
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

    api = args.model.startswith("gpt")
    if api:   # no output cap: a GPT-5 model's reasoning tokens count against it, and a capped reply can come back empty
        import openai_usage  # noqa: F401  (counts the calls)
        llm = LLMModel(model_name=args.model)
    else:
        llm = LLMModel(model_name=args.model, max_new_tokens=512)

    def caption(c):
        frames = sorted(glob.glob(os.path.join(c["frame_dir"], "*.jpg")))
        step = max(1, len(frames) // args.max_frames)
        frames = frames[::step][: args.max_frames]
        msg = [{"role": "user", "content": [{"type": "image", "image": f} for f in frames]
                + [{"type": "text", "text": PROMPT}]}]
        t0 = time.time()
        return (llm.generate(msg) or "").strip(), len(frames), time.time() - t0

    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency if api else 1) as pool:
        jobs = {pool.submit(caption, c): c for c in todo}
        for n, job in enumerate(as_completed(jobs), 1):
            c = jobs[job]
            try:
                text, n_frames, dt = job.result()
                if api and not text:
                    raise ValueError("empty reply")
            except Exception as e:
                if not api:
                    raise
                failed += 1   # the API's error text, with anything shaped like a key removed
                print(f"[{n}/{len(todo)}] {c['start_time']} failed: {type(e).__name__}: "
                      f"{re.sub(r'sk-[A-Za-z0-9_*-]+', 'sk-...', str(e))[:300]}", flush=True)
                continue
            c["text"] = text
            print(f"[{n}/{len(todo)}] {c['start_time']} {n_frames} frames {dt:.1f}s: {text[:120]}", flush=True)
            json.dump(clips, open(args.output, "w"), indent=2, ensure_ascii=False)
    if failed:
        sys.exit(f"{failed} of {len(todo)} clips have no caption yet: rerun to retry them")


if __name__ == "__main__":
    main()
