"""Score a run: accuracy over the pinned question bank, with the splits that matter.

Denominator is the question bank, not the answer files: a question with no
parsed answer counts as wrong in ``acc`` (and is listed as missing), and
``acc_answered`` is reported beside it so coverage gaps cannot hide.

Splits: streaming vs full video, binary yes/no questions (whose option layout
is a known format cue -- ``caveat: binary_answer_format_cue``) vs the rest, by
question form, and by who wrote the options.

    python -m epic_eval.score --run_dir <run>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .common import load_run_config, questions_for_run
from .run_eval import read_jsonl


def summarize(rows: list[dict], answers: dict) -> dict:
    n = len(rows)
    got = [answers[r["qid"]] for r in rows if r["qid"] in answers]
    correct = sum(bool(a["correct"]) for a in got)
    mb = [r["majority_baseline"] for r in rows if r.get("majority_baseline") is not None]
    return {
        "n": n,
        "answered": len(got),
        "missing": n - len(got),
        "correct": correct,
        "acc": correct / n if n else None,
        "acc_answered": correct / len(got) if got else None,
        "chance": sum(1 / len(r["options"]) for r in rows) / n if n else None,
        "majority_baseline": sum(mb) / len(mb) if mb else None,
    }


def score_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir)
    questions = questions_for_run(cfg)
    answers = {}
    for path in sorted((run_dir / "answers").glob("*.jsonl")):
        for a in read_jsonl(path):
            if a.get("pred_letter"):
                answers[a["qid"]] = a  # last write wins
    known = {q["qid"] for q in questions}
    stray = sorted(set(answers) - known)

    splits = {
        "all": questions,
        "streaming": [q for q in questions if q["streaming"]],
        "full_video": [q for q in questions if not q["streaming"]],
        "non_binary": [q for q in questions if not q["caveat"]],
        "binary_format_cue": [q for q in questions if q["caveat"]],
        "non_binary/streaming": [q for q in questions if not q["caveat"] and q["streaming"]],
        "non_binary/full_video": [q for q in questions if not q["caveat"] and not q["streaming"]],
    }
    by = lambda key: {k: summarize(v, answers) for k, v in sorted(_group(questions, key).items())}
    usage = defaultdict(int)
    for a in answers.values():
        for k, v in (a.get("usage") or {}).items():
            usage[k] += v or 0
    return {
        "run_dir": str(run_dir),
        "model": cfg["model"]["name"],
        "video_input": cfg["media"]["enabled"],
        "questions_commit": cfg["questions"]["commit"],
        "code_commit": cfg["code"]["commit"],
        "splits": {k: summarize(v, answers) for k, v in splits.items()},
        "by_form": by("form"),
        "by_video": by("video"),
        "by_options_model": by("options_model"),
        "missing_qids": sorted(known - set(answers)),
        "stray_answer_qids": stray,
        "cache_used_frac": (sum(bool(a.get("cache_used")) for a in answers.values()) / len(answers)) if answers else None,
        "usage_totals": dict(usage),
    }


def _group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[str(r[key])].append(r)
    return out


def fmt(x, pct=True):
    return "-" if x is None else (f"{100 * x:5.1f}" if pct else str(x))


def table(title: str, d: dict) -> str:
    lines = [f"\n{title}", f"{'':34s} {'n':>4s} {'ans':>4s} {'acc%':>6s} {'acc|ans':>7s} {'chance':>6s} {'major.':>6s}"]
    for k, s in d.items():
        lines.append(f"{k[:34]:34s} {s['n']:4d} {s['answered']:4d} {fmt(s['acc']):>6s} "
                     f"{fmt(s['acc_answered']):>7s} {fmt(s['chance']):>6s} {fmt(s['majority_baseline']):>6s}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    a = p.parse_args()
    res = score_run(a.run_dir)
    out = Path(a.run_dir) / "scores.json"
    out.write_text(json.dumps(res, indent=2))
    report = "\n".join([
        f"model={res['model']} video_input={res['video_input']} code={res['code_commit'][:10]} "
        f"questions={res['questions_commit'][:10]}",
        table("SPLITS", res["splits"]),
        table("BY FORM", res["by_form"]),
        table("BY VIDEO", res["by_video"]),
        table("BY OPTIONS MODEL", res["by_options_model"]),
        f"\nmissing: {len(res['missing_qids'])}  stray answers: {len(res['stray_answer_qids'])}  "
        f"cache_used_frac: {fmt(res['cache_used_frac'])}",
        f"usage: {res['usage_totals']}",
    ])
    (Path(a.run_dir) / "scores.txt").write_text(report + "\n")
    print(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
