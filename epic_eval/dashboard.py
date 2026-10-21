"""Build the run dashboard: one self-contained page comparing any set of runs.

    python -m epic_eval.dashboard --registry <dir>/runs.json --out <dir>

runs.json lists the runs to show, in column order:
    [{"id": "q768", "label": "Qwen3.8-27B · think · 1fps/768f", "run_dir": "..."}, ...]

Fills <out>/template.html into <out>/index.html with everything inline: the index (runs,
config essentials, scores, questions, every run's pick) as JSON, and each run's full
thinking + reply as gzip+base64, unpacked by the page when a question is first opened.
Nothing is fetched at view time (the artifact viewer does not serve fetch() of sibling files).
Scores are recomputed from answers/ and errors/ so runs still in progress show up too.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .common import LETTERS, load_run_config
from .explorer import load_run
from .run_eval import read_jsonl

SPLITS = [
    ("all", lambda q: True),
    ("non_binary", lambda q: not q["binary"]),
    ("binary_yes_no", lambda q: q["binary"]),
    ("streaming", lambda q: q["streaming"]),
    ("full_video", lambda q: not q["streaming"]),
    ("non_binary/streaming", lambda q: not q["binary"] and q["streaming"]),
    ("non_binary/full_video", lambda q: not q["binary"] and not q["streaming"]),
]


def tally(qs: list[dict], per: dict) -> dict:
    n = len(qs)
    ans = [q for q in qs if per.get(q["qid"], {}).get("pick") is not None]
    ok = sum(per[q["qid"]]["ok"] for q in ans)
    return {"n": n, "answered": len(ans), "correct": ok,
            "acc": ok / n if n else None, "acc_answered": ok / len(ans) if ans else None}


def answer_rows(run_dir: Path) -> list[dict]:
    return [a for p in sorted((run_dir / "answers").glob("*.jsonl")) for a in read_jsonl(p) if a.get("pred_letter")]


def essentials(cfg: dict, run_dir: Path, rows: list[dict]) -> dict:
    m, s = cfg["model"], cfg["slurm"]["eval"]
    frames = [a["video_frames"] for a in rows if a.get("video_frames")]
    tokens = [a["video_tokens"] for a in rows if a.get("video_tokens")]
    lat = [a["latency_s"] for a in rows if a.get("latency_s")]
    think = [(a.get("usage") or {}).get("thoughts_token_count") or 0 for a in rows]
    launches = list(read_jsonl(run_dir / "slurm" / "launches.jsonl")) if (run_dir / "slurm" / "launches.jsonl").exists() else []
    return {
        "model": m["name"], "backend": m["backend"], "mode": m.get("mode") or m.get("thinking_level"),
        "fps": m.get("fps"), "max_frames": m.get("max_frames"),
        "frames_used_max": max(frames) if frames else None,
        "video_tokens_median": statistics.median(tokens) if tokens else None,
        "max_new_tokens": m.get("max_new_tokens") or m.get("max_output_tokens"),
        "video": cfg["media"]["enabled"], "media": f"{cfg['media']['max_side']}px · {cfg['media']['encode_fps']} fps",
        "gpus": s.get("gres"), "partition": s.get("partition"),
        "code": cfg["code"]["commit"][:8], "questions": cfg["questions"]["commit"][:8],
        "retry_truncated": cfg.get("runtime", {}).get("retry_truncated", False),
        "latency_median_s": statistics.median(lat) if lat else None,
        "thinking_median_tok": statistics.median(think) if think else None,
        "launched": launches[0]["time"] if launches else None,
        "run_dir": str(run_dir),
    }


def build(registry: list[dict], out: Path) -> dict:
    runs, questions, picks, thoughts = [], {}, defaultdict(dict), {}
    for r in registry:
        run_dir = Path(r["run_dir"])
        _, qs, results = load_run(run_dir, r["label"], 0)
        rows = answer_rows(run_dir)
        cfg = load_run_config(run_dir)
        for qid, q in qs.items():
            questions.setdefault(qid, q)
        per = {}
        for qid, res in results.items():
            per[qid] = {k: v for k, v in res.items() if k != "thoughts"}
            picks[qid][r["id"]] = per[qid]
        ordered = list(qs.values())
        forms = sorted({q["form"] for q in ordered})
        truncated = sum(1 for v in results.values() if v.get("error") == "MAX_TOKENS")
        runs.append({
            "id": r["id"], "label": r["label"], "note": r.get("note"),
            "config": essentials(cfg, run_dir, rows),
            "progress": {"n": len(ordered), "answered": sum(v.get("pick") is not None for v in results.values()),
                         "truncated": truncated,
                         "errors": sum(1 for v in results.values() if v.get("pick") is None) - truncated},
            "splits": {name: tally([q for q in ordered if f(q)], per) for name, f in SPLITS},
            "by_form": {f: tally([q for q in ordered if q["form"] == f], per) for f in forms},
            "by_video": {v: tally([q for q in ordered if q["video"] == v], per)
                         for v in sorted({q["video"] for q in ordered})},
        })
        # full thinking, {video: {qid: ...}}, gzipped per run
        by_video = defaultdict(dict)
        for a in rows:
            by_video[a["video"]][a["qid"]] = {"thoughts": a.get("thoughts") or "", "reply": a.get("raw_text") or "",
                                              "frames": a.get("video_frames"), "latency": a.get("latency_s")}
        raw = json.dumps(by_video, separators=(",", ":")).encode()
        thoughts[r["id"]] = base64.b64encode(gzip.compress(raw, 9)).decode()
    qlist = sorted(questions.values(), key=lambda q: (q["video"], q["t_ask"] if q["t_ask"] is not None else 1e9, q["form"]))
    for q in qlist:
        q["per"] = picks.get(q["qid"], {})
    index = {"runs": runs, "questions": qlist, "letters": LETTERS}
    # "<" only occurs inside JSON strings, where \\u003c is the same character, so no "</script>"
    blob = json.dumps(index, separators=(",", ":")).replace("<", "\\u003c")
    page = (out / "template.html").read_text()
    page = page.replace("/*DATA*/", blob).replace("/*THOUGHTS*/", json.dumps(thoughts))
    (out / "index.html").write_text(page)
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", required=True, help="runs.json: [{id, label, run_dir, note?}, ...]")
    p.add_argument("--out", required=True, help="folder holding template.html; index.html is written there")
    a = p.parse_args()
    index = build(json.loads(Path(a.registry).read_text()), Path(a.out))
    for r in index["runs"]:
        print(f"{r['id']}: {r['progress']['answered']}/{r['progress']['n']} answered, "
              f"acc {r['splits']['all']['acc']:.3f}")
    print(f"index.html: {(Path(a.out) / 'index.html').stat().st_size / 1e6:.1f} MB (limit 16)")


if __name__ == "__main__":
    main()
