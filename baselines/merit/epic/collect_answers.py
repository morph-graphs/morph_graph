#!/usr/bin/env python3
"""Merge one video's MERIT answer files into a single answers file, and list what is still missing.

MERIT writes a *_final.json per run and turns a per-question failure (e.g. CUDA OOM) into the answer
"Error". A video can therefore need a second run on the failed IDs (eval --question-ids). This merges
every run's answers for the video, keeps the first valid answer per question ID, writes them in ID
order, and prints the IDs of the question file that have no valid answer (empty line if none).
An --out ending in .gz is written gzipped (the per-round retrieval history makes answers ~2 MB a video).

    python epic/collect_answers.py --qa_file <EgoLifeQA_<V>.json> --runs '<run dir>/*/*_final.json' \
        --out <answers.json[.gz]>
"""
import argparse
import glob
import gzip
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qa_file", required=True)
    ap.add_argument("--runs", required=True, help="glob of MERIT *_final.json files for this video")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = [str(r["ID"]) for r in json.load(open(a.qa_file))]
    got = {}
    for f in sorted(glob.glob(a.runs)):
        rows = json.load(open(f))
        for r in rows if isinstance(rows, list) else rows.values():
            if (r.get("response") or "").strip() not in ("", "Error"):
                got.setdefault(str(r["ID"]), r)
    rows = [got[i] for i in want if i in got]
    with (gzip.open(a.out, "wt") if a.out.endswith(".gz") else open(a.out, "w")) as f:
        json.dump(rows, f, indent=1)
    right = sum(str(r.get("evaluate")) == "True" for r in rows)
    missing = [i for i in want if i not in got]
    print(f"# {len(rows)}/{len(want)} answered, {right} right, {len(missing)} missing -> {a.out}", flush=True)
    print(" ".join(missing))


if __name__ == "__main__":
    main()
