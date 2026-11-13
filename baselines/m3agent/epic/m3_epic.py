#!/usr/bin/env python3
"""HD-EPIC videos and epic-morph banks in and out of M3-Agent's formats (run_m3agent.sh calls each step).

  clips     <video.mp4> -> <clip_dir>/<k>.mp4, clip k = [30k, 30k + 30) s with audio (upstream's cut, re-encoded
            so the cut is exact; long side scaled to --size), and <clip_dir>/clips.json
  questions bank file -> M3-Agent's control input ({video: {mem_path, qa_list}}), one row per question with
            its options and before_clip: the last clip that ended by t_ask (a question asked at t sees clips
            0 .. floor(ceil(t) / 30) - 1, as upstream's M3-Bench annotations and our MERIT runs); a question
            that is not streaming is asked at the end of the video
  collect   control output (jsonl) -> answers file, one row per bank question keyed by its `key`, with the
            chosen option, correctness and the clips each search returned (with times, for evidence scoring);
            prints a summary line, then the keys still unanswered (empty line when none)
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

CLIP_SEC = 30
LETTERS = "ABCDEFGH"
VERSION = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "VERSION")).read().strip()


def probe(path):
    import av   # memorization environment only
    with av.open(path) as c:
        dur = float(c.duration) / av.time_base
        return dur, any(s.type == "audio" for s in c.streams), [s for s in c.streams if s.type == "video"][0]


def clips(a):
    dur, has_audio, vs = probe(a.video)
    if not has_audio:
        raise SystemExit(f"{a.video} has no audio track: M3-Agent's memorization reads the clips' audio")
    os.makedirs(a.clip_dir, exist_ok=True)
    meta = os.path.join(a.clip_dir, "clips.json")
    if os.path.exists(meta):
        print(f"clips already made: {meta}")
        return
    scale = f"scale='if(gte(iw,ih),{a.size},-2)':'if(gte(iw,ih),-2,{a.size})'"
    n = math.ceil(dur / CLIP_SEC)
    if dur - (n - 1) * CLIP_SEC < 2:   # a last sliver under 2 s: too short for the memorization model's video reader
        n -= 1

    def cut(k):   # [30k, 30k + 30) as upstream, re-encoded so the cut is exact (seeking decodes from the keyframe before)
        out, tmp = os.path.join(a.clip_dir, f"{k}.mp4"), os.path.join(a.clip_dir, f".{k}.tmp.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-ss", str(k * CLIP_SEC), "-i", a.video,
             "-t", str(CLIP_SEC), "-vf", scale, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k", tmp], check=True)
        os.replace(tmp, out)
        return {"clip": k, "start": k * CLIP_SEC, "end": round(min((k + 1) * CLIP_SEC, dur), 2),
                "seconds": round(probe(out)[0], 2)}

    with ThreadPoolExecutor(max(1, a.threads // 2)) as ex:
        rows = list(ex.map(cut, range(n)))
    json.dump({"video": a.video, "duration": dur, "clip_sec": CLIP_SEC, "size": a.size, "clips": rows}, open(meta, "w"), indent=1)
    print(f"video {dur:.1f} s ({vs.codec_context.width}x{vs.codec_context.height}, audio) -> {len(rows)} clips in {a.clip_dir}")


def before_clip(q, video_end):
    t = q["t_ask"] if q.get("streaming") and "t_ask" in q else video_end
    return math.ceil(t) // CLIP_SEC - 1


def questions(a):
    bank = json.load(open(a.bank))
    end = json.load(open(a.clips))["duration"]
    qa_list = []
    for i, q in enumerate(bank["questions"], 1):
        mcq = q["gemini_mcq"]
        qa_list.append({"question_id": q["key"], "ID": str(i), "key": q["key"], "question": q["q"],
                        "options": [str(o) for o in mcq["options"]], "answer": LETTERS[mcq["answer_index"]],
                        "before_clip": before_clip(q, end), "t_ask": q.get("t_ask") if q.get("streaming") else None,
                        "meta": {k: q.get(k) for k in ("form", "hops", "streaming", "guessable", "answer")}})
    assert len({r["key"] for r in qa_list}) == len(qa_list), "question keys must be unique within a video"
    json.dump({a.video_id: {"mem_path": a.mem_path, "qa_list": qa_list}}, open(a.out, "w"), indent=1)
    print(f"{len(qa_list)} questions -> {a.out}; before_clip {min(r['before_clip'] for r in qa_list)}..{max(r['before_clip'] for r in qa_list)}")


def collect(a):
    want = json.load(open(a.questions))
    (video, v), = want.items()
    clip_rows = {c["clip"]: c for c in json.load(open(a.clips))["clips"]}
    sha = hashlib.sha256(open(a.mem_path, "rb").read()).hexdigest()[:16] if a.mem_path else None
    got = {}
    for f in a.runs:
        if os.path.exists(f):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    got[r["id"]] = r
    rows, missing = [], []
    for q in v["qa_list"]:
        r = got.get(q["question_id"])
        if r is None:
            missing.append(q["key"])
            continue
        retrieved = [{"round": i + 1, "query": s.get("query"), "mem_wise": s.get("mem_wise", False),
                      "clips": [{"clip": c, "start": clip_rows[c]["start"], "end": clip_rows[c]["end"]} for c in s.get("clips", [])]}
                     for i, s in enumerate(r.get("rounds", [])) if s["action"] == "search"]
        rows.append({"key": q["key"], "ID": q["ID"], "question": q["question"], "options": q["options"],
                     "answer": q["answer"], "chosen": r.get("chosen"), "correct": bool(r.get("correct")),
                     "parse": r.get("parse"), "answered_in_loop": r.get("answered_in_loop", r.get("finish")),
                     "response": r.get("response"), "t_ask": q["t_ask"],
                     "before_clip": q["before_clip"], "n_rounds": len(r.get("rounds", [])), "retrieved": retrieved,
                     "conversations": r.get("conversations"), "memory_sha256_16": sha, "adapter_version": VERSION,
                     "meta": q["meta"]})
    with (gzip.open(a.out, "wt") if a.out.endswith(".gz") else open(a.out, "w")) as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    right = sum(r["correct"] for r in rows)
    strict = sum(r["correct"] and bool(r["answered_in_loop"]) for r in rows)
    print(f"# {len(rows)}/{len(v['qa_list'])} answered, {right} right ({strict} answered within the rounds, upstream's protocol), "
          f"{sum(r['chosen'] is None for r in rows)} with no option, "
          f"{len(missing)} missing -> {a.out}", flush=True)
    print(" ".join(missing))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("clips")
    p.add_argument("--video", required=True)
    p.add_argument("--clip_dir", required=True)
    p.add_argument("--size", type=int, default=720, help="long side of the clips, px")
    p.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("questions")
    p.add_argument("--bank", required=True, help="epic-morph <bank>/<video>_questions.json")
    p.add_argument("--video_id", required=True)
    p.add_argument("--clips", required=True, help="clips.json of the video")
    p.add_argument("--mem_path", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--questions", required=True, help="the control input written by `questions`")
    p.add_argument("--clips", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="control output jsonl file(s)")
    p.add_argument("--mem_path", default=None, help="the memory the answers read (its hash goes in every row)")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"clips": clips, "questions": questions, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
