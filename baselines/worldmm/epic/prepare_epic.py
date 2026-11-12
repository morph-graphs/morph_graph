#!/usr/bin/env python3
"""HD-EPIC video and epic-morph bank -> WorldMM's per-video inputs (CPU only, no WorldMM import).

    prepare_epic.py video     --video <HD-EPIC mp4> --subject V --video_root DIR --skeleton_root DIR
    prepare_epic.py questions --bank <bank>/<V>_questions.json --skeleton <skeleton> --out <qa file>

`video` writes (skip-existing) <video_root>/<V>.mp4, the video at 512 px, 30 fps, a keyframe every second, no
audio, and <skeleton_root>/<V>/10sec_skeleton.json, WorldMM's 10 s segments (start/end HHMMSS00, DAY1). The
whole video is DAY1 from 00:00:00, as in WorldMM's Video-MME path and MERIT's adapter. The skeleton has no
question in it: the memory is built once for the whole video, and each question sees it only up to its t_ask.

`questions` writes the bank's questions for the video in WorldMM/EgoLifeQA row format: asked at t_ask (a
non-streaming question at the end of the video); ID is the 1-based position in the bank file and key the
bank's stable question key.
"""
import argparse
import json
import math
import os
import subprocess

LETTERS = "ABCD"


def hhmmss(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}{sec % 3600 // 60:02d}{sec % 60:02d}"


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def convert_questions(qfile, video_end):
    """Bank questions -> WorldMM rows (MERIT's mapping, but a streaming t_ask is floored, not ceiled)."""
    rows = []
    for i, q in enumerate(json.load(open(qfile))["questions"], 1):
        mcq = q["gemini_mcq"]
        opts = [str(o) for o in mcq["options"]]
        assert 2 <= len(opts) <= 4, (i, opts)
        # a streaming question is asked at t_ask, floored: memory items end on whole seconds and are seen when they
        # end <= the query time, so ceil would show an item ending up to 1 s after t_ask; else at the video's end
        streaming = q.get("streaming") and "t_ask" in q
        t = q["t_ask"] if streaming else video_end
        row = {
            "ID": str(i),
            "key": q.get("key"),
            "video_id": q["video"],
            "query_time": {"date": "DAY1", "time": hhmmss(math.floor(t) if streaming else math.ceil(t)) + "00"},
            "t_ask": t,
            "type": q["form"],
            "question": q["q"],
            "answer": LETTERS[mcq["answer_index"]],
            "meta": {k: q.get(k) for k in ("form", "hops", "streaming", "origin", "answer")},
        }
        for letter, o in zip(LETTERS, opts):
            row[f"choice_{letter.lower()}"] = o
        rows.append(row)
    return rows


def prep_video(a):
    os.makedirs(a.video_root, exist_ok=True)
    small = os.path.join(a.video_root, f"{a.subject}.mp4")
    if not os.path.exists(small):
        tmp = small + ".tmp.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-threads", str(a.threads), "-i", a.video,
                        "-vf", f"scale={a.size}:{a.size}", "-an", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-g", "30", "-keyint_min", "30", tmp], check=True)
        os.replace(tmp, small)
    dur = min(duration(a.video), duration(small))
    # WorldMM's build_segments / build_caption_entry: start floored, end ceiled to whole seconds
    skel = []
    for i in range(max(1, math.ceil(dur / a.unit_time))):
        s, e = i * a.unit_time, min((i + 1) * a.unit_time, dur)
        skel.append({"start_time": hhmmss(math.floor(s)) + "00", "end_time": hhmmss(math.ceil(e - 1e-9)) + "00",
                     "start_seconds": s, "end_seconds": e, "date": "DAY1", "video_path": small})
    out = os.path.join(a.skeleton_root, a.subject, "10sec_skeleton.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(skel, open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    print(f"{a.subject}: {dur:.1f}s -> {len(skel)} segments of {a.unit_time}s; video {small}")


def prep_questions(a):
    end = json.load(open(a.skeleton))[-1]["end_seconds"]
    rows = convert_questions(a.bank, end)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"{len(rows)} questions -> {a.out}; example: {json.dumps(rows[0])[:300]}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("video")
    v.add_argument("--video", required=True)
    v.add_argument("--subject", required=True)
    v.add_argument("--video_root", required=True)
    v.add_argument("--skeleton_root", required=True)
    v.add_argument("--unit_time", type=int, default=10)
    v.add_argument("--size", type=int, default=512)
    v.add_argument("--threads", type=int, default=8)
    q = sub.add_parser("questions")
    q.add_argument("--bank", required=True)
    q.add_argument("--skeleton", required=True)
    q.add_argument("--out", required=True)
    a = p.parse_args()
    prep_video(a) if a.cmd == "video" else prep_questions(a)


if __name__ == "__main__":
    main()
