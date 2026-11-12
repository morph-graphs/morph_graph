#!/usr/bin/env python3
"""Compare two MERIT answer files for the same questions (e.g. a rerun on new hardware against the
committed batch10 run): accuracy of each, how many questions both answered the same way, and the
paired disagreement (exact McNemar). Accepts .json or .json.gz.

    python epic/compare_answers.py <reference answers> <new answers>
"""
import gzip
import json
import math
import sys


def load(path):
    rows = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    return {str(r["ID"]): r for r in rows}


def main():
    ref, new = (load(p) for p in sys.argv[1:3])
    ids = sorted(set(ref) & set(new), key=int)
    ok = lambda rows, i: str(rows[i].get("evaluate")) == "True"
    same = sum((ref[i].get("parsed_answer") or "") == (new[i].get("parsed_answer") or "") for i in ids)
    a = sum(ok(ref, i) and not ok(new, i) for i in ids)
    b = sum(ok(new, i) and not ok(ref, i) for i in ids)
    n, k = a + b, min(a, b)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    acc = lambda rows: 100 * sum(ok(rows, i) for i in ids) / len(ids)
    print(f"{len(ids)} questions in both (reference {len(ref)}, new {len(new)})")
    print(f"accuracy: reference {acc(ref):.1f}%, new {acc(new):.1f}%")
    print(f"same option chosen: {same}/{len(ids)} ({100 * same / len(ids):.0f}%)")
    print(f"right only in reference {a}, only in new {b}: exact McNemar p = {p:.2f}")


if __name__ == "__main__":
    main()
