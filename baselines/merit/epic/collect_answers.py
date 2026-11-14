#!/usr/bin/env python3
"""Merge one video's MERIT answer files into a single answers file, and list what is still missing.

MERIT writes a *_final.json per run and turns a per-question failure (e.g. CUDA OOM) into the answer
"Error". A video can therefore need a second run on the failed IDs (eval --question-ids). This merges
the earlier answers files (--prev: this bank's, then the video's other banks', hf_results.py earlier-answers)
and every run's answers for the video, keeps the
first valid answer per question, writes them in question-file order, and prints the IDs of the question file
that have no valid answer (empty line if none). An answer is kept only while its question's wording, options
and gold letter are unchanged (prepare_epic.match_answers): after a tranche adds or edits questions, only those
are listed. Answer files from before keys (positional IDs) are matched by that content.
An --out ending in .gz is written gzipped (the per-round retrieval history makes answers ~2 MB a video).

    python epic/collect_answers.py --qa_file <EgoLifeQA_<V>.json> --runs '<run dir>/*/*_final.json' \
        [--prev <answers.json.gz> ...] --out <answers.json[.gz]>
"""
import argparse
import glob
import gzip
import json
import os

from prepare_epic import match_answers


def load(path):
    rows = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    return rows if isinstance(rows, list) else list(rows.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qa_file", required=True)
    ap.add_argument("--runs", required=True, help="glob of MERIT *_final.json files for this video")
    ap.add_argument("--prev", nargs="*", default=[], help="earlier answers files to reuse, first wins (read before --out is written)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qa = json.load(open(a.qa_file))
    srcs = [p for p in a.prev if os.path.exists(p)] + sorted(glob.glob(a.runs))
    answers = [r for f in srcs for r in load(f)]
    got = match_answers(qa, answers)
    rows = [got[str(q["ID"])] for q in qa if str(q["ID"]) in got]
    with (gzip.open(a.out, "wt") if a.out.endswith(".gz") else open(a.out, "w")) as f:
        json.dump(rows, f, indent=1)
    right = sum(str(r.get("evaluate")) == "True" for r in rows)
    missing = [str(q["ID"]) for q in qa if str(q["ID"]) not in got]
    print(f"# {len(rows)}/{len(qa)} answered, {right} right, {len(missing)} missing -> {a.out}", flush=True)
    print(" ".join(missing))


if __name__ == "__main__":
    main()
