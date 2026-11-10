"""Build a scoreboard page for one question bank: every run's accuracy, broken down, question by question.

    python3 -m epic_eval.scoreboard --registry epic_eval/dashboards/v3/runs.json --out <dir>

Unlike epic_eval.dashboard (epic_eval runs only, with full replies), this also reads the memory
baselines, which write MERIT-style slices ({arm}_{video}_q{a}-{b}_final.json, rows with a 1-based
"ID" into the video's bank file and "evaluate"). A registry entry is one of
    {"id", "label", "owner", "group", "kind": "epic_eval", "run_dir": ...}
    {"id", "label", "owner", "group", "kind": "slices", "root": ..., "arm": ...}
    {"id", "label", "owner", "group", "kind": "planned"}          (listed, no results yet)
plus an optional "note". Scores come straight from each run's files, so runs in progress
show up with whatever they have answered.
"""

from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

LETTERS = "ABCDEFGH"


def qid(q):
    return f"{q['video']}__" + hashlib.sha1(f"{q['q']}|{q['t_ask']:.3f}".encode()).hexdigest()[:10]


def load_bank(bank_dir):
    qs = []
    for f in sorted(Path(bank_dir).glob("*_questions.json")):
        for pos, q in enumerate(json.load(open(f))["questions"], 1):
            mcq = q["gemini_mcq"]
            qs.append({"qid": qid(q), "video": q["video"], "pos": pos, "t": q["t_ask"], "form": q["form"],
                       "hops": q.get("hops"), "q": q["q"], "opts": mcq["options"], "gt": mcq["answer_index"]})
    return qs


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def from_epic_eval(run_dir, qs):
    """{question index: (correct, picked option index)}; epic_eval letters options in bank order."""
    by_qid = {q["qid"]: i for i, q in enumerate(qs)}
    out, mtime = {}, 0
    for f in glob.glob(f"{run_dir}/answers/*.jsonl"):
        mtime = max(mtime, Path(f).stat().st_mtime)
        for line in open(f):
            r = json.loads(line)
            i = by_qid.get(r.get("qid"))
            if i is None or not r.get("pred_letter"):
                continue
            pick = LETTERS.index(r["pred_letter"]) if r["pred_letter"] in LETTERS else None
            out[i] = (bool(r["correct"]), pick)
    return out, mtime


def _letter(r):
    if r.get("pred_letter"):
        return r["pred_letter"]
    m = re.search(r"ANSWER:\s*\(?([A-H])\b", r.get("response") or "") or re.match(r"\s*\(?([A-H])\b", r.get("response") or "")
    return m.group(1) if m else None


def from_slices(root, arm, qs):
    """{question index: (correct, picked option index)} from every finished slice of one arm under root."""
    by_pos = {(q["video"], q["pos"]): i for i, q in enumerate(qs)}
    out, mtime = {}, 0
    for f in glob.glob(f"{root}/**/{arm}_*_q*_final.json", recursive=True):
        mtime = max(mtime, Path(f).stat().st_mtime)
        video = Path(f).name[len(arm) + 1:].rsplit("_q", 1)[0]
        rows = json.load(open(f))
        for r in rows if isinstance(rows, list) else rows.values():
            if (r.get("response") or "").strip() in ("", "Error", "Unable to generate answer"):
                continue
            i = by_pos.get((video, int(r["ID"])))
            if i is None:
                continue
            pick, letter = None, _letter(r)
            choices = r.get("choices")
            if isinstance(choices, str):
                try:
                    choices = ast.literal_eval(choices)
                except (ValueError, SyntaxError):
                    choices = None
            if letter and isinstance(choices, dict) and letter in choices and choices[letter] in qs[i]["opts"]:
                pick = qs[i]["opts"].index(choices[letter])
            out[i] = (str(r["evaluate"]) == "True", pick)
    return out, mtime


def summarize(res, qs):
    def cell(idx):
        ans = [i for i in idx if i in res]
        k = sum(res[i][0] for i in ans)
        return {"n": len(idx), "ans": len(ans), "ok": k}
    groups = {}
    for key in ("form", "video", "hops"):
        by = defaultdict(list)
        for i, q in enumerate(qs):
            by[str(q[key])].append(i)
        groups[key] = {g: cell(ix) for g, ix in sorted(by.items())}
    total = cell(range(len(qs)))
    total["ci"] = wilson(total["ok"], total["ans"])
    return total, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=str(Path(__file__).with_name("scoreboard_template.html")))
    a = ap.parse_args()
    reg = json.load(open(a.registry))
    qs = load_bank(reg["bank_dir"])
    runs, results = [], {}
    for r in reg["runs"]:
        res, mtime = {}, 0
        if r["kind"] == "epic_eval":
            res, mtime = from_epic_eval(r["run_dir"], qs)
        elif r["kind"] == "slices":
            res, mtime = from_slices(r["root"], r["arm"], qs)
        total, groups = summarize(res, qs)
        runs.append({k: r.get(k) for k in ("id", "label", "owner", "group", "kind", "note")}
                    | {"total": total, "groups": groups,
                       "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else None})
        results[r["id"]] = {str(i): [int(ok), pick] for i, (ok, pick) in res.items()}
        print(f"  {r['id']:14s} {total['ans']:4d}/{len(qs)} answered, {total['ok']} correct")
    data = {"title": reg["title"], "bank": reg["bank"], "generated": time.strftime("%Y-%m-%d %H:%M %Z"),
            "questions": [{k: q[k] for k in ("video", "pos", "t", "form", "hops", "q", "opts", "gt")} for q in qs],
            "runs": runs, "results": results}
    page = Path(a.template).read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)
    (out / "data.json").write_text(json.dumps({k: v for k, v in data.items() if k != "generated"}, sort_keys=True))
    print(f"{out}/index.html: {len(runs)} runs, {len(qs)} questions, {(out / 'index.html').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
