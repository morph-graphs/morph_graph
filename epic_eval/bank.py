"""Show the questions a named bank holds (banks.json), read straight from git at its pinned commit.

    python3 -m epic_eval.bank --list
    python3 -m epic_eval.bank v2-150155-82 --repo <epic-morph clone or mirror>
    python3 -m epic_eval.bank v2-150155-82 --repo <...> --csv v2_82.csv

--repo defaults to $EPIC_MORPH_DIR. Any clone or bare mirror that contains the bank's
commit works; the branch it has checked out does not matter. The CSV has one row per
question, exactly the rows a config with {"questions": {"bank": "<name>"}} would score.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

from .common import BANKS, LETTERS, load_json, load_questions_at


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?", help="bank name from banks.json")
    p.add_argument("--list", action="store_true", help="list the named banks")
    p.add_argument("--repo", default=os.environ.get("EPIC_MORPH_DIR"), help="epic-morph clone or mirror (default $EPIC_MORPH_DIR)")
    p.add_argument("--csv", help="write one row per question here")
    a = p.parse_args()

    banks = load_json(BANKS)
    if a.list or not a.name:
        for name, b in banks.items():
            print(f"{name:16} {b['commit'][:10]}  {b.get('questions', '?'):>4} questions  {b.get('description', '')}")
        return
    if a.name not in banks:
        sys.exit(f"unknown bank {a.name!r}; known: {', '.join(banks)}")
    if not a.repo:
        sys.exit("pass --repo <epic-morph clone or mirror> or set EPIC_MORPH_DIR")
    b = banks[a.name]
    try:
        rows = load_questions_at(a.repo, b["commit"], b["glob"])
    except Exception as e:  # git: commit missing from this repo, not a repo, ...
        sys.exit(f"could not read {b['commit'][:10]} from {a.repo}: {e}\n"
                 f"(fetch that commit into the repo, e.g. on a login node: git -C {a.repo} fetch)")

    print(f"{a.name}: {b.get('description', '')}")
    print(f"  commit {b['commit']}  glob {b['glob']}")
    want = b.get("questions")
    print(f"  {len(rows)} questions ({sum(r['streaming'] for r in rows)} streaming, "
          f"{sum(r['binary'] for r in rows)} yes/no)" + ("" if want in (None, len(rows)) else f"  -- banks.json says {want}!"))
    for v, n in sorted(Counter(r["video"] for r in rows).items()):
        print(f"    {v}: {n}")
    print("  by form: " + ", ".join(f"{f} {n}" for f, n in Counter(r["form"] for r in rows).most_common()))

    if a.csv:
        n_opt = max(len(r["options"]) for r in rows)
        cols = ["qid", "video", "form", "streaming", "t_ask", "question",
                *[f"option_{LETTERS[i]}" for i in range(n_opt)], "answer_letter", "answer", "evidence_clips", "source_file"]
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                opts = r["options"] + [""] * (n_opt - len(r["options"]))
                clips = "; ".join(f"{t0:.1f}-{t1:.1f}" for t0, t1 in r["evidence_clips"])
                w.writerow([r["qid"], r["video"], r["form"], r["streaming"], r["t_ask"], r["question"], *opts,
                            LETTERS[r["answer_index"]], r["options"][r["answer_index"]], clips, r["source_file"]])
        print(f"  wrote {a.csv}")


if __name__ == "__main__":
    main()
