#!/usr/bin/env python3
"""WorldMM fine captions (10 s units) for one HD-EPIC video, with WorldMM's own captioning code.

Uses preprocess/episodic_memory/generate_fine_caption.py unchanged (its system prompt, 1 fps frames at whole
seconds, segment prompt, caption entry format); the only difference from its Video-MME run is that HD-EPIC has
no transcript, so every segment gets "No transcript lines overlap this segment." Resumes from a partial output:
segments already captioned are skipped. The backbone is the LLM server (WORLDMM_SERVER).

    python caption_epic.py --skeleton <skeleton_root>/<V>/10sec_skeleton.json --output <captions>/<V>/10sec.json --model qwen3vl-8b
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import wmm_paths  # noqa: F401  (first)
from decord import VideoReader, cpu
from generate_fine_caption import Segment, VideoReaderContext, generate_caption
from worldmm.llm import LLMModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skeleton", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="qwen3vl-8b")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    skel = json.load(open(a.skeleton))
    done = {}
    if os.path.exists(a.output):
        done = {c["start_time"]: c["text"] for c in json.load(open(a.output)) if c.get("text")}
    todo = [i for i, s in enumerate(skel) if s["start_time"] not in done]
    print(f"{len(skel)} segments, {len(todo)} to caption", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)

    def save():
        # field order of WorldMM's build_caption_entry: start_time, end_time, text, date, video_path
        rows = [{"start_time": s["start_time"], "end_time": s["end_time"], "text": done[s["start_time"]],
                 "date": s["date"], "video_path": s["video_path"]} for s in skel if s["start_time"] in done]
        json.dump(rows, open(a.output + ".tmp", "w"), indent=2, ensure_ascii=False)
        os.replace(a.output + ".tmp", a.output)

    if todo:
        vr = VideoReader(skel[0]["video_path"], ctx=cpu(0))
        ctx = VideoReaderContext(reader=vr, average_fps=float(vr.get_avg_fps()), total_frames=len(vr), lock=Lock())
        llm = LLMModel(model_name=a.model)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(generate_caption, Segment(skel[i]["start_seconds"], skel[i]["end_seconds"], []), ctx, llm): i
                    for i in todo}
            for n, f in enumerate(as_completed(futs), 1):
                i = futs[f]
                done[skel[i]["start_time"]] = text = f.result()
                if n % 20 == 0 or n == len(todo):
                    save()
                if n % 10 == 0 or n == len(todo):
                    print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s  {skel[i]['start_time']}: {text[:100]}", flush=True)
    save()
    assert len(done) == len(skel) and all(done.values()), "empty or missing captions"
    print(f"{len(skel)} captions in {a.output}")


if __name__ == "__main__":
    main()
