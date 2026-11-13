#!/usr/bin/env python3
"""Turn an epic_eval run folder into the baselines' per-video answers files.

epic_eval (the API models: one [0, t_ask] clip per question, no memory) writes answers/<video>.jsonl keyed by
its own qid (a hash of video, wording and t_ask). The baselines write <arm>/<video>/answers_<bank>.json.gz, one
row per question in bank order, keyed by the bank's `key` (rows as baselines/magic/epic/run_qa.py writes them:
key, qhash, ID, type, question, choices, answer, response, evaluate). This reads the bank files the run was
asked (the run's own questions_repo copy, never another one: the qid does not cover the options), joins each
answer to its question by qid, and writes those files plus a runs.jsonl line saying where they came from.
No model is called.

A question with no parsed letter (never answered, or cut off at the output cap) gets no row, as in the other
baselines; its key is printed and the exit code is 2. The conversion stops on an answer row that disagrees with
the bank (gold letter, or its own `correct`), on two questions sharing one answer, and on a streaming answer
that does not show its input ended by t_ask: the clip's length (video-file backends), and the last frame time
when the backend sent frames.

    python baselines/api/epic/from_epic_eval.py --run <epic_eval run folder> --out baselines/api/results \
        [--arm gemini-3.8-flash-high] [--bank batch10_agree2] [--video <video>] [--note "..."]
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from epic_eval.common import make_qid   # noqa: E402  (the qid the run wrote; standard library only)

LETTERS = "ABCD"
KEEP = ("qid", "model", "time", "raw_text", "finish_reason", "usage", "cache_used", "media_file", "media_duration_s")
FRAMES = ("model_returned", "n_frames", "last_frame_s", "frame_side", "detail", "frame_times_s")   # frame backends
# batch10_agree2 kept a question when 2+ of these 3 models answered it right, so their own scores on it are inflated
VERIFIERS = ("gemini-3.8-flash", "gpt-5.6-sol", "qwen3.8-max")
INFLATED = "inflated by selection: this model was one of the bank's three verifiers (questions kept when 2+ of 3 were right)"


def qhash(question, options):
    """As baselines/merit/epic/prepare_epic.py: a question edited under the same key gets another hash."""
    return hashlib.sha1(json.dumps([question, [str(o) for o in options]]).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="epic_eval run folder")
    ap.add_argument("--out", required=True, help="results root: files go to <out>/<arm>/<video>/")
    ap.add_argument("--arm", default=None, help="default: <model name>-<thinking level>, e.g. gemini-3.8-flash-high")
    ap.add_argument("--bank", default=None, help="bank name in the file name; default: the folder of the run's bank glob")
    ap.add_argument("--video", default=None, help="convert this video only (default: every video of the bank)")
    ap.add_argument("--note", default=None, help="a caveat for runs.jsonl; default: the verifier note for a verifier model")
    ap.add_argument("--no_thoughts", action="store_true", help="leave the model's thought summaries out of the rows")
    a = ap.parse_args()

    run = os.path.abspath(a.run)
    cfg = json.load(open(os.path.join(run, "reproducibility_config.json")))
    pattern = cfg["questions"]["glob"]
    bank = a.bank or os.path.basename(os.path.dirname(pattern))
    files = sorted(glob.glob(os.path.join(run, "questions_repo", pattern)))
    if a.video:
        files = [f for f in files if os.path.basename(f) == f"{a.video}_questions.json"]
    assert files, f"no bank files for {pattern}" + (f" and video {a.video}" if a.video else "")
    m = cfg["model"]
    arm = a.arm or "-".join(x for x in (m.get("name") or m["backend"], (m.get("thinking_level") or "").lower()) if x)
    note = a.note or (INFLATED if str(m.get("name", "")).lower().startswith(VERIFIERS) else None)

    answers = {}   # as epic_eval/score.py: only parsed answers count, the last one written wins
    for f in sorted(glob.glob(os.path.join(run, "answers", (a.video or "") + "*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue   # a line cut off when the job was killed; run_eval asks that question again
            if r.get("pred_letter") and (not a.video or r["video"] == a.video):
                answers[r["qid"]] = r
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    commit = os.environ.get("API_REPO_COMMIT") or commit

    used, total, right, missing = set(), 0, 0, []
    for f in files:
        qs = json.load(open(f))["questions"]
        if not qs:
            continue
        assert all("key" in q for q in qs), f"{f}: questions without a key (bank older than keys)"
        assert len({q["key"] for q in qs}) == len(qs), f"{f}: duplicate keys"
        video = qs[0]["video"]
        rows = []
        for pos, q in enumerate(qs, 1):
            assert q["video"] == video, f"{f}: more than one video"
            opts = [str(o) for o in q["gemini_mcq"]["options"]]
            gold = LETTERS[q["gemini_mcq"]["answer_index"]]
            t_ask = float(q["t_ask"]) if q["streaming"] else None
            qid = make_qid(video, q["q"], t_ask)
            assert qid not in used, f"{qid}: two questions of the bank share this qid (same video, wording and t_ask)"
            used.add(qid)
            r = answers.get(qid)
            if r is None:
                missing.append(q["key"])
                continue
            ok = r["pred_letter"] == gold
            assert r["gt_letter"] == gold and bool(r["correct"]) == ok, f"{qid}: the run's gold letter differs from the bank's"
            assert r["cutoff_s"] == t_ask, f"{qid}: asked with cutoff {r['cutoff_s']}, bank t_ask {t_ask}"
            if t_ask is not None:   # the streaming rule, from what the run recorded about its input
                dur, last = r.get("media_duration_s"), r.get("last_frame_s")
                assert dur is not None or last is not None, f"{qid}: the row records neither its clip's length nor its last frame time"
                # a clip's length may run one frame interval (0.5 s at the 2 fps encode) past its last frame
                assert dur is None or dur <= t_ask + 0.5, f"{qid}: clip of {dur} s for t_ask {t_ask}"
                assert last is None or last <= t_ask, f"{qid}: last frame at {last} s for t_ask {t_ask}"
                if "frame_times_s" in r:   # a frames backend: what the model saw is the frames, not the clip
                    assert last is not None and last == max(r["frame_times_s"]), f"{qid}: last_frame_s {last} is not its last frame time"
            row = {"key": q["key"], "qhash": qhash(q["q"], opts), "ID": q["key"], "pos": pos, "type": q["form"],
                   "question": q["q"], "choices": dict(zip(LETTERS, opts)), "answer": gold,
                   "response": r["pred_letter"], "evaluate": ok, "t_ask": t_ask, "cutoff": r["cutoff_s"],
                   "seconds": r.get("latency_s"), **{k: r.get(k) for k in KEEP}, **{k: r[k] for k in FRAMES if k in r}}
            if not a.no_thoughts:
                row["thoughts"] = r.get("thoughts")
            rows.append(row)
        total += len(qs)
        right += sum(r["evaluate"] for r in rows)
        if not rows:
            print(f"{video}: 0/{len(qs)} answered, nothing written", flush=True)
            continue
        d = os.path.join(a.out, arm, video)
        os.makedirs(d, exist_ok=True)
        with gzip.open(os.path.join(d, f"answers_{bank}.json.gz"), "wt") as g:
            json.dump(rows, g, indent=1)
        line = {"stage": "from_epic_eval", "video": video, "arm": arm, "bank": bank, "answered": len(rows),
                "questions": len(qs), "right": sum(r["evaluate"] for r in rows), "epic_eval_run": os.path.basename(run),
                "epic_eval_commit": cfg["code"].get("commit"), "epic_morph_commit": cfg["questions"].get("commit"),
                "model": m, "media": cfg.get("media"), "morph_graph_commit": commit, **({"note": note} if note else {})}
        log = os.path.join(d, "runs.jsonl")   # one line per (stage, bank): converting again replaces it
        old = [json.loads(x) for x in open(log)] if os.path.exists(log) else []
        old = [x for x in old if (x.get("stage"), x.get("bank")) != ("from_epic_eval", bank)]
        with open(log, "w") as g:
            g.write("".join(json.dumps(x) + "\n" for x in old + [line]))
        print(f"{video}: {len(rows)}/{len(qs)} answered, {line['right']} right -> {d}/answers_{bank}.json.gz", flush=True)

    stray = sorted(set(answers) - used)
    n = total - len(missing)
    print(f"# {arm} {bank}: {n}/{total} answered, {right} right = {100 * right / max(total, 1):.1f}% of all "
          f"({100 * right / max(n, 1):.1f}% of answered), {len(missing)} missing, "
          f"{len(stray)} answers of questions not in the bank" + (f"\n# note: {note}" if note else ""))
    if missing:
        print("missing: " + " ".join(missing))
        sys.exit(2)


if __name__ == "__main__":
    main()
