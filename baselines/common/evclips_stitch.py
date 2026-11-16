"""Evidence-clips videos: per ask_at "end" question of the EPIC-Morph banks, its evidence clips stitched into one short
mp4, for the "evidence clips only" ablation (Anubhav, 2026-11-15).

A question's spans are the union of its bank evidence.clips ranges, merged where they overlap, in time order, no
padding (epic_eval.common.evidence_spans; for an end question the cut at t_ask is the video's end). Questions of a
video with the same spans (rounded to 0.01 s) share one file, named by a hash of (video, spans). Each span is snapped
to the source's frame grid (frames round(t0 * fps) .. round(t1 * fps) - 1) and cut frame-exactly (input seek, then
re-encoded); the cut segments are concatenated at the source's constant frame rate and encoded as MERIT's prep
encodes its 448 px videos (prepare_epic.make_clips: scale=448:448, no audio, libx264 veryfast crf 23).

    python baselines/common/evclips_stitch.py plan --bank_root <epic-morph>/outputs/v3/banks --out OUT   # host python
    python baselines/common/evclips_stitch.py stitch --out OUT --video_dir <HD-EPIC>/videos --shard i --n_shards N
    python baselines/common/evclips_stitch.py check --out OUT --video_dir ... [--ids <id> ...] [--pixels]
    python baselines/common/evclips_stitch.py manifest --out OUT                 # OUT/manifest.json, all files done

stitch, check and manifest need ffmpeg/ffprobe (and numpy for check): run them inside MERIT's image
(run_evclips_stitch.sbatch). Outputs: OUT/videos/<video>/<id>.mp4 plus OUT/segments/<video>/<id>.json (what was cut),
skipped when present, so a preempted shard resumes.
"""
import argparse
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from epic_eval.common import evidence_clips, evidence_spans, make_qid  # noqa: E402

BANKS = ["batch10_agree2_end", "extra_agree2_end", "rescue_agree2_end", "t2_agree2_end", "t3_agree2_end",
         "t4_agree2_end", "t5_agree2_end"]
SIZE = 448
ENCODE = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]   # prepare_epic.make_clips


def file_id(video, spans):
    key = video + "|" + ";".join(f"{a:.2f}-{b:.2f}" for a, b in spans)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def paths(out, video, fid):
    return (os.path.join(out, "videos", video, f"{fid}.mp4"), os.path.join(out, "segments", video, f"{fid}.json"))


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} exited {r.returncode}: {r.stderr.strip()[-2000:]}\n{' '.join(cmd)}")
    return r.stdout


def plan(a):
    """OUT/plan.json: every question -> file id, every file -> (video, spans). Reads the bank files only."""
    questions, files = [], {}
    for bank in a.banks:
        for f in sorted(glob.glob(os.path.join(a.bank_root, bank, "*_questions.json"))):
            for q in json.load(open(f))["questions"]:
                if q.get("ask_at") != "end" or (a.videos and q["video"] not in a.videos):
                    continue
                t_ask = float(q["t_ask"]) if q.get("streaming") else None
                qid = make_qid(q["video"], q["q"], t_ask)
                spans = evidence_spans({"qid": qid, "t_ask": t_ask, "evidence_clips": evidence_clips(q)})
                spans = [[round(s, 2), round(e, 2)] for s, e in spans]
                fid = file_id(q["video"], spans)
                assert files.setdefault(fid, {"video": q["video"], "spans": spans})["spans"] == spans, fid
                questions.append({"bank": bank, "video": q["video"], "key": q["key"], "qid": qid, "file": fid})
    git = lambda *c: subprocess.run(["git", "-C", a.bank_root, *c], capture_output=True, text=True).stdout.strip()
    bank_files = sorted(glob.glob(os.path.join(a.bank_root, "*_end", "*_questions.json")))
    src = {"bank_root": os.path.abspath(a.bank_root), "banks": a.banks, "epic_morph_head": git("rev-parse", "HEAD"),
           "epic_morph_dirty": git("status", "--porcelain", "--", ":(top)v3/export_bank.py"), "epic_morph_commit": a.epic_morph_commit,
           "bank_sha1": {os.path.relpath(f, a.bank_root): hashlib.sha1(open(f, "rb").read()).hexdigest()
                         for f in bank_files if f.split(os.sep)[-2] in a.banks}}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"source": src, "questions": questions, "files": files}, open(os.path.join(a.out, "plan.json"), "w"),
              indent=1)
    n_vid = len({q["video"] for q in questions})
    secs = sum(e - s for f in files.values() for s, e in f["spans"])
    print(f"{len(questions)} questions, {n_vid} videos -> {len(files)} files ({len(questions) / len(files):.2f} "
          f"questions per file), {secs / 3600:.2f} h of stitched video")


def probe(src):
    s = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=r_frame_rate,avg_frame_rate,nb_frames,start_time", "-of", "json", src]))["streams"][0]
    assert s["r_frame_rate"] == s["avg_frame_rate"], f"{src}: not constant frame rate ({s})"
    assert abs(float(s.get("start_time", 0))) < 1e-6, f"{src}: start_time {s['start_time']}"
    return Fraction(s["r_frame_rate"]), int(s["nb_frames"])


def stitch_one(fid, f, out, video_dir, threads):
    mp4, meta = paths(out, f["video"], fid)
    if os.path.exists(mp4) and os.path.exists(meta):
        return None
    src = os.path.join(video_dir, f["video"].split("-")[0], f["video"] + ".mp4")
    fps, n_src = probe(src)
    t0 = time.time()
    # frame-grid spans: frames k0 .. k1 - 1 (never a frame of the previous span); the input window runs from half a
    # frame before k0 to half a frame before k1, so accurate seek keeps exactly those frames
    segs, args, chains = [], [], []
    for i, (s, e) in enumerate(f["spans"]):
        k0 = max(min(round(s * fps), n_src - 1), segs[-1][1] if segs else 0)
        k1 = max(min(round(e * fps), n_src), k0 + 1)
        ss = max(Fraction(0), (k0 - Fraction(1, 2)) / fps)
        args += ["-ss", f"{float(ss):.6f}", "-t", f"{float((k1 - Fraction(1, 2)) / fps - ss):.6f}", "-i", src]
        chains.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
        segs.append((k0, k1))
    n = sum(k1 - k0 for k0, k1 in segs)
    graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(len(segs))) + \
        f"concat=n={len(segs)}:v=1:a=0,setpts=N/({fps.numerator}/{fps.denominator})/TB,scale={SIZE}:{SIZE}[out]"
    os.makedirs(os.path.dirname(mp4), exist_ok=True)
    os.makedirs(os.path.dirname(meta), exist_ok=True)
    tag = f"{socket.gethostname()}.{os.getpid()}"   # per process: a requeued copy may overlap a dying one
    tmp = f"{mp4}.{tag}.part.mp4"
    run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-threads", str(threads), *args, "-filter_complex", graph,
         "-map", "[out]", "-r", f"{fps.numerator}/{fps.denominator}", "-threads", str(threads), *ENCODE, tmp])
    got = int(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                   "stream=nb_read_packets", "-of", "csv=p=0", tmp]).strip())
    assert got == n, f"{fid}: {got} frames written, {n} cut ({segs})"
    rows, at = [], 0
    for (s, e), (k0, k1) in zip(f["spans"], segs):
        rows.append({"src_start": round(float(k0 / fps), 6), "src_end": round(float(k1 / fps), 6),
                     "stitched_start": round(float(at / fps), 6), "stitched_end": round(float((at + k1 - k0) / fps), 6),
                     "src_frames": [k0, k1], "span": [s, e]})
        at += k1 - k0
    os.replace(tmp, mp4)
    rec = {"id": fid, "video": f["video"], "path": os.path.relpath(mp4, out), "fps": f"{fps.numerator}/{fps.denominator}",
           "frames": n, "duration": round(float(n / fps), 6), "bytes": os.path.getsize(mp4), "segments": rows,
           "encode_s": round(time.time() - t0, 2)}
    json.dump(rec, open(f"{meta}.{tag}.part", "w"), indent=1)
    os.replace(f"{meta}.{tag}.part", meta)
    return rec


def stitch(a):
    files = json.load(open(os.path.join(a.out, "plan.json")))["files"]
    ids = sorted(files)[a.shard::a.n_shards]
    if a.limit:
        ids = ids[:a.limit]
    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        done = [r for r in ex.map(lambda i: stitch_one(i, files[i], a.out, a.video_dir, a.threads), ids) if r]
    wall = time.time() - t0
    secs = sum(r["duration"] for r in done)
    print(f"shard {a.shard}/{a.n_shards}: {len(ids)} files, {len(done)} stitched ({len(ids) - len(done)} existed), "
          f"{secs:.0f} s of video, {sum(r['bytes'] for r in done) / 1e6:.1f} MB, wall {wall:.1f} s, "
          f"sum of per-file encode {sum(r['encode_s'] for r in done):.1f} s, jobs {a.jobs} x threads {a.threads}")


def frames(src, idx, scale):
    """{frame number: RGB float array} of src, decoded from the start (no seek, so the check does not share stitch's
    seeking), scaled to SIZE x SIZE the way stitch scales; stops after the last frame asked for."""
    import numpy as np
    idx = sorted(set(idx))
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-threads", "4", "-i", src, "-vf",
                          f"select='{sel}'" + (f",scale={SIZE}:{SIZE}" if scale else ""), "-vsync", "0",
                          "-frames:v", str(len(idx)), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return dict(zip(idx, np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3).astype(np.float32)))


def check_streams(path, meta):
    """Problems of a stitched file against its record: one video stream (no audio), 448 x 448, the source's frame
    rate, every frame cut, and a duration within one frame of the segments' total."""
    st = json.loads(run(["ffprobe", "-v", "error", "-count_packets", "-show_entries", "stream=codec_type,width,height,"
                         "r_frame_rate,nb_read_packets:format=duration", "-of", "json", path]))
    fps, bad = Fraction(meta["fps"]), []
    seg_s = sum(s["src_end"] - s["src_start"] for s in meta["segments"])
    [v] = st["streams"] if len(st["streams"]) == 1 else [None]
    if v is None or v["codec_type"] != "video":
        bad.append(f"streams {[x['codec_type'] for x in st['streams']]}")
    elif (v["width"], v["height"]) != (SIZE, SIZE) or Fraction(v["r_frame_rate"]) != fps \
            or int(v["nb_read_packets"]) != meta["frames"]:
        bad.append(f"stream {v}")
    if abs(float(st["format"]["duration"]) - seg_s) > 1 / fps:
        bad.append(f"duration {st['format']['duration']} vs segments {seg_s:.3f}")
    return bad, float(st["format"]["duration"]), seg_s


def check_pixels(src, path, meta):
    """Per segment, the mean absolute pixel difference of the stitched frames at its start and its end with the
    source frames k - 1, k, k + 1 there (scaled to 448): frame-exact means the minimum sits at offset 0."""
    import numpy as np
    fps = Fraction(meta["fps"])
    pairs = []   # (segment, which end, stitched frame, source frame)
    for j, s in enumerate(meta["segments"]):
        k0, k1 = s["src_frames"]
        f0 = round(s["stitched_start"] * fps)
        pairs += [(j, "start", f0, k0), (j, "end", f0 + k1 - k0 - 1, k1 - 1)]
    st = frames(path, [p[2] for p in pairs], scale=False)
    sv = frames(src, [k + d for *_, k in pairs for d in (-1, 0, 1) if k + d >= 0], scale=True)
    ok = True
    for j, name, sf, sk in pairs:
        mad = {d: float(np.abs(st[sf] - sv[sk + d]).mean()) for d in (-1, 0, 1) if sk + d in sv}
        best = min(mad, key=mad.get)
        ok &= best == 0
        print(f"  seg {j} {name}: stitched frame {sf} vs source frame {sk} (offset: MAD) "
              + ", ".join(f"{d:+d}: {m:.2f}" for d, m in mad.items()) + f"  -> best {best:+d}")
    return ok


def check(a):
    """Stream checks of the given files (all: every file written), and with --pixels the frame-exact check
    (decodes each source from its start: slow)."""
    metas = sorted(glob.glob(os.path.join(a.out, "segments", "*", "*.json")))
    if a.ids != ["all"]:
        metas = [m for m in metas if os.path.basename(m)[:-5] in a.ids]
    n_bad, worst = 0, 0.0
    for m in metas:
        meta = json.load(open(m))
        path = os.path.join(a.out, meta["path"])
        bad, dur, seg_s = check_streams(path, meta)
        worst = max(worst, abs(dur - seg_s))
        if a.pixels:
            print(f"{meta['id']} ({meta['video']}, {len(meta['segments'])} segments, {dur:.3f} s):")
            src = os.path.join(a.video_dir, meta["video"].split("-")[0], meta["video"] + ".mp4")
            if not check_pixels(src, path, meta):
                bad.append("not frame-exact")
        if bad:
            n_bad += 1
            print(f"BAD {meta['id']}: {'; '.join(bad)}")
    print(f"checked {len(metas)} files: {n_bad} bad; largest |duration - segments| {worst * 1000:.1f} ms")
    sys.exit(1 if n_bad else 0)


def manifest(a):
    p = json.load(open(os.path.join(a.out, "plan.json")))
    recs, missing = {}, []
    for fid, f in p["files"].items():
        meta = paths(a.out, f["video"], fid)[1]
        if os.path.exists(meta):
            recs[fid] = json.load(open(meta))
        else:
            missing.append(fid)
    if missing and not a.partial:
        sys.exit(f"{len(missing)} of {len(p['files'])} files not stitched yet, e.g. {missing[:3]}")
    settings = {"spans": "epic_eval.common.evidence_spans (union of evidence.clips, merged, cut at t_ask), no padding, "
                         "rounded to 0.01 s; snapped to frames round(t * fps)",
                "filter": f"per span input seek + concat, setpts=N/fps, scale={SIZE}:{SIZE}", "encode": ENCODE,
                "fps": "source (constant)", "audio": None}
    out = {"encoder": run(["ffmpeg", "-version"]).splitlines()[0], "settings": settings, "source": p["source"],
           "n_questions": len(p["questions"]), "n_files": len(recs), "missing": missing,
           "questions": [q for q in p["questions"] if q["file"] in recs], "files": recs}
    json.dump(out, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"manifest: {out['n_questions']} questions, {len(recs)} files, {len(missing)} missing, "
          f"{sum(r['bytes'] for r in recs.values()) / 1e9:.2f} GB, {sum(r['duration'] for r in recs.values()) / 3600:.2f} h")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "stitch", "check", "manifest"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        if name == "plan":
            s.add_argument("--bank_root", required=True, help="epic-morph outputs/v3/banks")
            s.add_argument("--banks", nargs="+", default=BANKS)
            s.add_argument("--videos", nargs="+", help="only these videos (a smoke test)")
            s.add_argument("--epic_morph_commit", help="the epic-morph commit whose export_bank.py wrote the banks")
        if name in ("stitch", "check"):
            s.add_argument("--video_dir", required=True, help="HD-EPIC videos/<P>/<video>.mp4")
        if name == "stitch":
            s.add_argument("--shard", type=int, default=0)
            s.add_argument("--n_shards", type=int, default=1)
            s.add_argument("--jobs", type=int, default=os.cpu_count(), help="ffmpeg processes at once")
            s.add_argument("--threads", type=int, default=1, help="threads per ffmpeg")
            s.add_argument("--limit", type=int, default=0)
        if name == "check":
            s.add_argument("--ids", nargs="+", default=["all"], help="file ids, or all")
            s.add_argument("--pixels", action="store_true", help="also the frame-exact check (decodes the sources)")
        if name == "manifest":
            s.add_argument("--partial", action="store_true", help="write it with the files done so far")
    a = p.parse_args()
    {"plan": plan, "stitch": stitch, "check": check, "manifest": manifest}[a.cmd](a)


if __name__ == "__main__":
    main()
