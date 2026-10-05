"""Build a self-contained HTML explorer over one or more eval runs.

    python -m epic_eval.explorer --runs <run_dir> [<run_dir> ...] \\
        [--labels "Gemini video,Gemini text-only"] [--out explorer.html] [--verdicts verdicts.json]

Reads each run folder (its reproducibility_config, its pinned questions_repo,
answers/ and errors/) and writes ONE html file with all of it embedded: per
question, the options, the correct answer, every run's pick and reasoning
summary, and the narration evidence with timestamps. Filters and per-run
summaries are computed in the page, so any slice can be looked at without
rerunning anything.

Verdicts ("model wrong", "bank answer wrong", ...) are kept per question.
Opened as a local file they live in that browser and can be exported / imported
as JSON; pass an exported file back with --verdicts to carry them into the next
build. Published as a claude.ai artifact with the `db` capability, they are
shared live between everyone who opens the link.

Questions are joined across runs by qid. Runs on different question commits can
be compared; where a question's options differ between runs, each run's own
options are kept and the page says so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import LETTERS, load_run_config, questions_for_run
from .run_eval import read_jsonl

TEMPLATE = Path(__file__).with_name("explorer_template.html")


def fmt_t(sec) -> str | None:
    if sec is None:
        return None
    sec = float(sec)
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def load_leaves(questions_dir: Path, video: str) -> dict:
    path = questions_dir / "data" / "leaves" / f"{video}.jsonl"
    if not path.is_file():
        return {}
    out = {}
    for row in read_jsonl(path):
        if "leaf_id" in row:
            out[row["leaf_id"]] = (row.get("start_timestamp"), row.get("text", ""))
    return out


def evidence_rows(ev_ids: list, leaves: dict) -> list:
    rows = []
    for e in ev_ids or []:
        hit = leaves.get(e) or leaves.get(str(e).split("#")[0])
        rows.append({"id": e, "t": fmt_t(hit[0]) if hit else None, "text": hit[1] if hit else ""})
    return sorted(rows, key=lambda r: (r["t"] is None, r["t"] or ""))


def load_run(run_dir: Path, label: str | None, max_thought: int) -> tuple[dict, dict, dict]:
    """(run meta, {qid: question row}, {qid: this run's result})."""
    cfg = load_run_config(run_dir)
    qdir = Path(cfg["resolved"]["questions_dir"])
    rows = questions_for_run(cfg)
    raw = {}
    for path in sorted(qdir.glob(cfg["questions"]["glob"])):
        for q in json.loads(path.read_text())["questions"]:
            raw[(q["video"], q["q"], float(q["t_ask"]) if q.get("streaming") else None)] = q
    leaves_by_video = {}
    questions = {}
    for r in rows:
        extra = raw.get((r["video"], r["question"], r["t_ask"]), {})
        if r["video"] not in leaves_by_video:
            leaves_by_video[r["video"]] = load_leaves(qdir, r["video"])
        questions[r["qid"]] = {
            "qid": r["qid"], "video": r["video"], "form": r["form"], "q": r["question"],
            "options": r["options"], "gt": r["answer_index"], "answer": r["answer"],
            "streaming": r["streaming"], "t_ask": r["t_ask"], "asked_at": fmt_t(r["t_ask"]),
            "binary": r["binary"], "hops": r["hops"], "guessable": r["guessable"],
            "payoff_form": extra.get("payoff_form"),
            "evidence": evidence_rows(extra.get("evidence"), leaves_by_video[r["video"]]),
        }
    results = {}
    for a in (x for p in sorted((run_dir / "answers").glob("*.jsonl")) for x in read_jsonl(p)):
        if a.get("pred_letter") and a["qid"] in questions:
            th = a.get("thoughts") or ""
            results[a["qid"]] = {
                "pick": LETTERS.index(a["pred_letter"]), "ok": bool(a.get("correct")),
                "think": (a.get("usage") or {}).get("thoughts_token_count"),
                "latency": a.get("latency_s"),
                "thoughts": th[:max_thought] + (" …" if len(th) > max_thought else ""),
            }
    for e in (x for p in sorted((run_dir / "errors").glob("*.jsonl")) for x in read_jsonl(p)):
        if e.get("qid") in questions and e["qid"] not in results:
            results[e["qid"]] = {"pick": None, "ok": False,
                                 "error": e.get("finish_reason") or e.get("stage") or "error"}
    m = cfg["model"]
    meta = {
        "label": label or f"{m['name']} · {'video' if cfg['media']['enabled'] else 'text-only'}",
        "run_dir": str(run_dir), "model": m["name"], "video": cfg["media"]["enabled"],
        "thinking": m.get("thinking_level"), "media_resolution": m.get("media_resolution"),
        "code": cfg["code"]["commit"][:8], "questions": cfg["questions"]["commit"][:8],
        "answered": sum(1 for v in results.values() if v.get("pick") is not None), "n": len(questions),
    }
    return meta, questions, results


def build(run_dirs: list[Path], labels: list[str | None], max_thought: int = 1500,
          verdicts: dict | None = None) -> dict:
    runs, qs = [], {}
    for i, (d, lab) in enumerate(zip(run_dirs, labels)):
        meta, questions, results = load_run(Path(d), lab, max_thought)
        runs.append(meta)
        for qid, q in questions.items():
            if qid not in qs:
                qs[qid] = {**q, "per": {}, "opts_by_run": {}}
            elif q["options"] != qs[qid]["options"] or q["gt"] != qs[qid]["gt"]:
                # same question, different option set (e.g. a later bank commit)
                qs[qid]["opts_by_run"][str(i)] = {"options": q["options"], "gt": q["gt"]}
            if qid in results:
                qs[qid]["per"][str(i)] = results[qid]
    order = sorted(qs.values(), key=lambda q: (q["video"], q["t_ask"] if q["t_ask"] is not None else 1e9, q["form"]))
    return {"runs": runs, "questions": order, "verdicts": verdicts or {}}


def load_verdicts(path) -> dict:
    doc = json.loads(Path(path).read_text())
    v = doc.get("verdicts", doc) if isinstance(doc, dict) else {r["qid"]: r for r in doc}
    return {k: r for k, r in v.items() if isinstance(r, dict) and r.get("verdict") is not None}


def render(data: dict) -> str:
    page = TEMPLATE.read_text()
    # "<" only ever occurs inside JSON strings, where \u003c is the same character; with no
    # literal "<" left, neither "</script>" nor "<!--<script" can end or derail the script block
    blob = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    return page.replace("/*DATA*/", blob)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run directories (first is run A by default)")
    p.add_argument("--labels", help="comma-separated display names, one per run")
    p.add_argument("--out", help="output html (default: <first run>/explorer.html)")
    p.add_argument("--verdicts", help="verdicts JSON exported from the page, to embed")
    p.add_argument("--max-thought-chars", type=int, default=1500)
    a = p.parse_args()
    labels = [s.strip() for s in a.labels.split(",")] if a.labels else [None] * len(a.runs)
    if len(labels) != len(a.runs):
        raise SystemExit(f"{len(labels)} labels for {len(a.runs)} runs")
    data = build([Path(r) for r in a.runs], labels, a.max_thought_chars,
                 load_verdicts(a.verdicts) if a.verdicts else None)
    out = Path(a.out) if a.out else Path(a.runs[0]) / "explorer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data))
    for r in data["runs"]:
        print(f"  {r['label']}: {r['answered']}/{r['n']} answered  ({r['run_dir']})")
    print(f"{len(data['questions'])} questions, {len(data['verdicts'])} embedded verdicts -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
