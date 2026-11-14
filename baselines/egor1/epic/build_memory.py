#!/usr/bin/env python3
"""Ego-R1's hierarchical memory (H-RAG database) of one video, question-independent.

As the paper describes it (the official repo ships the finished EgoLife database, not the code that made it):
every 30 s clip is captioned, with its dialogue, by Gemini 1.5 Pro (retired: here Gemini 2.5 Pro, apis.py); GPT-4.1 merges the captions bottom-up into
10 minute, 1 hour and 1 day summaries. The files have the format of Ego-R1/h-rag_database (egolife/A1_JAKE):
a list of {"generated_text", "date": 1, "start_time": HHMMSSFF as int, "end_time"}.

    <out>/<video>_30sec.json   <out>/<video>_10min.json   <out>/<video>_1hour.json   <out>/<video>_1day.json

The caption and summary prompts are NOT upstream's (unpublished): they are written here after the paper's
description and the look of the released database (first person, "I ...").

Both steps call paid APIs and stop with exit code 5 when the keys are missing (apis.py). Resumes: captions are
saved as they come, a finished level is kept.

    python build_memory.py --video <V> --clips <clip_root>/<V> --out <dir>
    python build_memory.py --memory_files          (the memory's file name patterns, for run_egor1.sh)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import apis   # noqa: E402

MEMORY = ["{video}_30sec.json", "{video}_10min.json", "{video}_1hour.json", "{video}_1day.json"]
LEVELS = [("10min", 600), ("1hour", 3600), ("1day", 86400)]
CAPTION_PROMPT = (
    "This is a 30-second clip of an egocentric (first-person) video. Describe in detail, in the first person "
    "(\"I ...\"), what I do and what happens: my actions in order, the objects I use and where they are, the "
    "place, and other people. Transcribe any speech you hear. Write one dense paragraph.")
SUMMARY_PROMPT = (
    "You are given consecutive first-person descriptions of an egocentric video, each with its time span "
    "(HHMMSSFF). Merge them into one summary of this {span} in the first person (\"I ...\"), in chronological "
    "order, keeping the key actions, objects, places, people and anything that was said. Return only the summary.")


def secs(t):
    t = f"{int(t):08d}"
    return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def jload(p):
    return json.load(open(p))


def jsave(obj, p):
    json.dump(obj, open(p + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(p + ".tmp", p)


def summarize(name, group):
    text = "\n".join(f"[{e['start_time']:08d}-{e['end_time']:08d}] {e['generated_text']}" for e in group)
    span = {"10min": "10-minute interval", "1hour": "hour", "1day": "day"}[name]
    return apis.gpt41("summary", [{"role": "system", "content": SUMMARY_PROMPT.format(span=span)},
                                  {"role": "user", "content": text}], max_tokens=1000, temperature=0.0)


def build_levels(caps, reuse=None):
    """{"10min": [...], "1hour": [...], "1day": [...]} from 30 s captions, bottom-up. A window that spans the same
    time in `reuse` (the logs of the whole video) is taken from there: it was made from the same captions."""
    out, lower = {}, caps
    for name, span in LEVELS:
        have = {(e["start_time"], e["end_time"]): e for e in (reuse or {}).get(name, [])}
        groups = {}
        for e in lower:
            groups.setdefault(secs(e["start_time"]) // span, []).append(e)
        cur = []
        for _, g in sorted(groups.items()):
            s, t = g[0]["start_time"], g[-1]["end_time"]
            cur.append(have.get((s, t)) or {"generated_text": summarize(name, g), "date": 1, "start_time": s, "end_time": t})
        out[name], lower = cur, cur
    return out


def clip_list(clips):
    d = os.path.join(clips, "A1_JAKE", "DAY1")
    return [os.path.join(d, x) for x in open(os.path.join(d, "_clips.txt")).read().split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--memory_files", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--clips", help="<clip_root>/<video> (prepare_clips.py)")
    ap.add_argument("--out")
    ap.add_argument("--clip_sec", type=int, default=30)
    a = ap.parse_args()
    if a.memory_files:
        print(" ".join(MEMORY))
        return
    paths = {m.split("_")[-1][:-5]: os.path.join(a.out, m.format(video=a.video)) for m in MEMORY}
    if all(os.path.exists(p) for p in paths.values()):
        print(f"{a.video}: memory already in {a.out}")
        return
    apis.need("caption", "summary")
    os.makedirs(a.out, exist_ok=True)
    apis.USAGE = os.path.join(a.out, "api_usage.jsonl")
    clips = clip_list(a.clips)
    part = paths["30sec"] + ".partial"
    caps = jload(paths["30sec"]) if os.path.exists(paths["30sec"]) else jload(part) if os.path.exists(part) else []
    for i, c in enumerate(clips):
        if i < len(caps):
            continue
        start = int(os.path.basename(c).split("_")[-1][:8])               # HHMMSSFF of the clip's start
        s = secs(start) + a.clip_sec
        end = int(f"{s // 3600:02d}{s % 3600 // 60:02d}{s % 60:02d}00")
        caps.append({"generated_text": apis.gemini_video("caption", c, CAPTION_PROMPT).strip(), "date": 1,
                     "start_time": start, "end_time": end})
        jsave(caps, part)
        print(f"caption {i + 1}/{len(clips)}", flush=True)
    assert len(caps) == len(clips) and all(e["generated_text"] for e in caps), "empty or missing captions"
    jsave(caps, paths["30sec"])
    if os.path.exists(part):
        os.remove(part)
    for name, log in build_levels(caps).items():
        jsave(log, paths[name])
    print(f"{a.video}: {len(caps)} captions -> " + ", ".join(f"{len(jload(paths[n]))} x {n}" for n, _ in LEVELS))


if __name__ == "__main__":
    main()
