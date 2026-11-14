"""Reuse answers a video already has, from its own bank's answers file or another bank's (hf_results.py
earlier-answers lists them), so a question answered once is not asked again: same key, and what was asked is
unchanged (each baseline's `sig`: wording, options, gold letter, query time, and its memory where the row says).
MERIT does the same in prepare_epic.match_answers."""
import gzip
import json
import os


def load(path):
    if not os.path.isfile(path):
        return []
    rows = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    return rows if isinstance(rows, list) else list(rows.values())


def reuse(want, paths, sig, failed=("", "Error")):
    """{key: row}: per key of `want`, the first row of the files (in order) with a valid response and sig(row) == want[key]."""
    got = {}
    for p in paths:
        for r in load(p):
            k = r.get("key")
            if k in want and k not in got and (r.get("response") or "").strip() not in failed and sig(r) == want[k]:
                got[k] = r
    return got
