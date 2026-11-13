#!/usr/bin/env python3
"""Turn one HD-EPIC video (+ optionally an epic-morph question file) into MERIT's EgoLife layout.

Writes (all skip-existing):
  <clip_root>/<subject>/DAY1/DAY1_<subject>_<HHMMSS>00.mp4   30 s clips, re-encoded small
  <frame_root>/<subject>/<clip>/tNNNNN.jpg                    1 fps frames for captioning
  <data_dir>/captions/<subject>/<subject>_30sec_skeleton.json  clip list without captions
  <data_dir>/EgoLifeQA/EgoLifeQA_<subject>.json               questions in EgoLifeQA format (with --questions)

The whole video is treated as DAY1 starting at 00:00:00. A streaming question is asked
at its t_ask; every other question is asked at the end of the video.

A question's ID is its bank `key`, so answers stay attached to it when a tranche adds or reorders a video's
questions; a file without keys (e.g. M2 probes) keeps the 1-based position. match_answers() reuses an answer
only while the question's wording, options and gold letter are unchanged (qhash).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess

LETTERS = "ABCD"


def hhmmss(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}{sec % 3600 // 60:02d}{sec % 60:02d}"


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def make_clips(video, clip_dir, subject, n_clips, clip_sec, size, threads):
    """Cut 30 s clips. The container's duration (ffprobe format) can run a few seconds past the last video
    frame, the audio being longer, so ffmpeg may cut one segment fewer than ceil(duration / clip_sec): the
    clip list is what ffmpeg cut, recorded in _clips_<subject>.txt so a rerun does not cut again."""
    want = [os.path.join(clip_dir, f"DAY1_{subject}_{hhmmss(i * clip_sec)}00.mp4") for i in range(n_clips)]
    done = os.path.join(clip_dir, f"_clips_{subject}.txt")
    if os.path.exists(done):
        have = [os.path.join(clip_dir, l.strip()) for l in open(done) if l.strip()]
        if have and all(os.path.exists(p) for p in have):
            return have
    elif all(os.path.exists(p) for p in want):
        return want
    tmp = os.path.join(clip_dir, "_seg")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-threads", str(threads), "-i", video,
         "-vf", f"scale={size}:{size}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-force_key_frames", f"expr:gte(t,n_forced*{clip_sec})",
         "-f", "segment", "-segment_time", str(clip_sec), "-reset_timestamps", "1",
         os.path.join(tmp, "seg_%04d.mp4")], check=True)
    segs = sorted(glob.glob(os.path.join(tmp, "seg_*.mp4")))
    # one fewer is the video ending before the container does (no frames to cut); anything else is an error
    assert max(1, n_clips - 1) <= len(segs) <= n_clips, f"expected {n_clips} segments, got {len(segs)}"
    want = want[:len(segs)]
    for s, p in zip(segs, want):
        os.replace(s, p)
    os.rmdir(tmp)
    open(done, "w").write("".join(os.path.basename(p) + "\n" for p in want))
    return want


def make_frames(video, frame_root, clip_names, clip_sec, size, threads):
    if all(len(glob.glob(os.path.join(frame_root, c, "*.jpg"))) > 0 for c in clip_names):
        return
    tmp = os.path.join(frame_root, "_all")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    # fps=1 emits the frame nearest t = 0, 1, 2, ... s, so frame k (0-based) belongs to clip k // clip_sec.
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-threads", str(threads), "-i", video,
         "-vf", f"fps=1,scale={size}:{size}", "-q:v", "3", os.path.join(tmp, "%05d.jpg")], check=True)
    frames = sorted(glob.glob(os.path.join(tmp, "*.jpg")))
    for k, f in enumerate(frames):
        c = min(k // clip_sec, len(clip_names) - 1)
        d = os.path.join(frame_root, clip_names[c])
        os.makedirs(d, exist_ok=True)
        os.replace(f, os.path.join(d, f"t{k:05d}.jpg"))  # t = seconds into the video
    shutil.rmtree(tmp)


def qhash(question, options):
    """Short hash of a question's wording and options (as in baselines/magic): an edit under the same key changes it."""
    return hashlib.sha1(json.dumps([question, [str(o) for o in options]]).encode()).hexdigest()[:12]


def row_qhash(r):
    """qhash of a MERIT question row (choice_a..) or answer row (choices dict): both carry the text they were asked."""
    ch = r.get("choices") or {k: r[f"choice_{k.lower()}"] for k in LETTERS if f"choice_{k.lower()}" in r}
    return qhash(r["question"], [ch[k] for k in sorted(ch)])


def match_answers(qa_rows, answers):
    """Attach answer rows (MERIT's or blind_qa's, first valid one wins) to the questions of qa_rows:
    {question ID: answer row}, the row's ID set to the question's and key / qhash / pos added. An answer counts
    only for the same question text, options and gold letter. It is found by ID; an ID not in qa_rows (a
    positional ID from before keys, or a key since dropped) is found by that content instead: the one question
    asked that way, or among several the one at that position (otherwise it is answered again)."""
    sig = lambda r: (row_qhash(r), r.get("answer"))
    by_id = {str(q["ID"]): q for q in qa_rows}
    pos = {str(q["ID"]): i for i, q in enumerate(qa_rows, 1)}
    by_sig = {}
    for q in qa_rows:
        by_sig.setdefault(sig(q), []).append(q)
    got = {}
    for r in answers:
        if (r.get("response") or "").strip() in ("", "Error"):
            continue
        q = by_id.get(str(r["ID"]))
        if q:
            q = q if sig(q) == sig(r) else None
        else:   # by content: the only question asked this way, or the one at that position (never a second one,
            # which a copy of this row already matched elsewhere could otherwise also fill)
            same = by_sig.get(sig(r), [])
            q = next((q for q in same if len(same) == 1 or str(pos[str(q["ID"])]) == str(r["ID"])), None)
        if q and str(q["ID"]) not in got:
            i = str(q["ID"])
            got[i] = {**r, "ID": q["ID"], "key": q.get("key"), "qhash": row_qhash(q), "pos": pos[i]}
    return got


def convert_questions(qfile, subject, video_end):
    raw = json.load(open(qfile))
    keyed = all("key" in q for q in raw["questions"])
    if keyed:
        keys = [q["key"] for q in raw["questions"]]
        assert len(set(keys)) == len(keys), f"duplicate question keys in {qfile}"
    rows = []
    for i, q in enumerate(raw["questions"], 1):
        mcq = q["gemini_mcq"]
        opts = [str(o) for o in mcq["options"]]
        assert 2 <= len(opts) <= 4, (i, opts)
        t = video_end
        if q.get("streaming") and "t_ask" in q:
            # MERIT shows a clip once its end <= the query time: ceil(t_ask) could show a clip ending up to 1 s
            # after the question, so floor it. A question asked in the last second is asked at the end, where
            # the last clip ends at ceil(duration) (flooring would hide seconds seen before the question).
            t = video_end if video_end and q["t_ask"] > video_end - 1 else math.floor(q["t_ask"])
        row = {
            "ID": q["key"] if keyed else str(i),
            "key": q.get("key"),
            "qhash": qhash(q["q"], opts),
            "pos": i,
            "query_time": {"date": "DAY1", "time": hhmmss(t) + "00"},
            "type": q["form"],
            "question": q["q"],
            "answer": LETTERS[mcq["answer_index"]],
            # Kept for the report; MERIT ignores these.
            "meta": {k: q.get(k) for k in ("form", "hops", "streaming", "guessable", "confidence",
                                            "majority_baseline", "t_ask", "answer")},
        }
        row["meta"]["n_options"] = len(opts)
        for letter, o in zip(LETTERS, opts):
            row[f"choice_{letter.lower()}"] = o
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--questions", default=None, help="epic-morph bank file; without it only clips, frames and the skeleton are made")
    p.add_argument("--subject", required=True)
    p.add_argument("--clip_root", required=True, help="becomes MERIT --video-root")
    p.add_argument("--frame_root", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--clip_sec", type=int, default=30)
    p.add_argument("--clip_size", type=int, default=704)
    p.add_argument("--frame_size", type=int, default=448)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    dur = duration(args.video)
    n_clips = math.ceil(dur / args.clip_sec)
    clip_dir = os.path.join(args.clip_root, args.subject, "DAY1")
    os.makedirs(clip_dir, exist_ok=True)
    clips = make_clips(args.video, clip_dir, args.subject, n_clips, args.clip_sec, args.clip_size, args.threads)
    names = [os.path.splitext(os.path.basename(c))[0] for c in clips]
    frame_root = os.path.join(args.frame_root, args.subject)
    make_frames(args.video, frame_root, names, args.clip_sec, args.frame_size, args.threads)

    cap_dir = os.path.join(args.data_dir, "captions", args.subject)
    os.makedirs(cap_dir, exist_ok=True)
    skeleton = []
    for i, name in enumerate(names):
        start, end = i * args.clip_sec, min((i + 1) * args.clip_sec, math.ceil(dur))
        skeleton.append({
            "start_time": hhmmss(start) + "00",
            "end_time": hhmmss(end) + "00",
            "date": "DAY1",
            "video_path": f"{args.subject}/DAY1/{name}.mp4",
            "frame_dir": os.path.join(frame_root, name),
        })
    json.dump(skeleton, open(os.path.join(cap_dir, f"{args.subject}_30sec_skeleton.json"), "w"), indent=2)

    rows = []
    if args.questions:
        qa_dir = os.path.join(args.data_dir, "EgoLifeQA")
        os.makedirs(qa_dir, exist_ok=True)
        rows = convert_questions(args.questions, args.subject, math.ceil(dur))
        json.dump(rows, open(os.path.join(qa_dir, f"EgoLifeQA_{args.subject}.json"), "w"), indent=2)

    n_frames = [len(glob.glob(os.path.join(s["frame_dir"], "*.jpg"))) for s in skeleton]
    print(f"video {dur:.1f}s -> {len(clips)} clips, frames per clip min/max {min(n_frames)}/{max(n_frames)}")
    if rows:
        print(f"questions: {len(rows)}; streaming {sum(r['meta']['streaming'] for r in rows)}")
        print("example QA row:", json.dumps(rows[0])[:400])


if __name__ == "__main__":
    main()
