#!/usr/bin/env python3
"""Everything the dashboard needs from the board, read-only, as one JSON line (run where the HF token is).

    python coord/board_read.py

{tasks: the master's open tasks (tasks.json), claims: every pool's claims with all their fields,
 pools: heartbeats, built: [[baseline, arm, video]], answered: [[baseline, arm, video, bank]], master: master.json,
 bank_questions: {bank: {video: questions}}, answered_q: [[baseline, arm, video, bank, questions answered]]}

Questions per bank come from the banks the master regenerates in its epic-morph checkout (EPIC_MORPH_DIR, default
the board master's); an answer counts when its key (or, in files from before keys, its question text) is one of the bank's questions
for that video and its response
is not empty or "Error". The answers files are small and cached (only changed ones download).
"""
import gzip
import json
import os
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobs  # noqa: E402
import master  # noqa: E402

head, tasks, pools = jobs.read()
have, files = master.built(HfApi())
answered = sorted({tuple(f.split("/")[:3]) + (f.split("/")[3][len("answers_"):-len(".json.gz")],)
                   for f in files if f.count("/") == 3 and f.split("/")[3].startswith("answers_") and f.endswith(".json.gz")})
try:
    m = json.load(open(hf_hub_download(jobs.REPO, "master.json", repo_type="dataset", revision=head)))
except Exception:
    m = {}
claims = [dict(c, pool=n, task=tid, pool_host=p.get("host", "")) for n, p in pools.items()
          for tid, c in p.get("claims", {}).items()]
heart = {n: dict(updated=p.get("updated"), running=p.get("running"), slots=p.get("slots"), host=p.get("host"))
         for n, p in pools.items()}

banks_dir = os.path.join(os.environ.get("EPIC_MORPH_DIR", "/fs/cfar-projects/mem_open_vocab/anubhav/board_master/epic-morph"),
                         "outputs", "v3", "banks")
keys = {}   # bank -> video -> question keys
for bank in sorted(os.listdir(banks_dir)) if os.path.isdir(banks_dir) else []:
    for f in sorted(os.listdir(os.path.join(banks_dir, bank))):
        if f.endswith("_questions.json"):
            qs = json.load(open(os.path.join(banks_dir, bank, f)))["questions"]
            keys.setdefault(bank, {})[f[:-len("_questions.json")]] = {q["key"]: q["q"].strip() for q in qs}
answered_q = []
mine = [a for a in answered if a[3] in keys and a[2] in keys[a[3]]]   # variants (answers_<bank>_<tag>) are not the bank
if mine:
    d = snapshot_download(master.RESULTS, repo_type="dataset",
                          allow_patterns=[f"{b}/{arm}/{v}/answers_{bank}.json.gz" for b, arm, v, bank in mine])
    for b, arm, v, bank in mine:
        try:
            rows = json.load(gzip.open(os.path.join(d, b, arm, v, f"answers_{bank}.json.gz"), "rt"))
        except Exception:
            continue
        rows = rows.values() if isinstance(rows, dict) else rows
        ok = [r for r in rows if isinstance(r, dict) and (r.get("response") or "").strip() not in ("", "Error")]
        want = keys[bank][v]
        # answers keyed by question key (since #16); older files (batch10) carry the question text instead
        got = {r["key"] for r in ok if r.get("key") in want} | \
              {k for k, q in want.items() if q in {(r.get("question") or "").strip() for r in ok if "key" not in r}}
        answered_q.append([b, arm, v, bank, len(got)])
print(json.dumps(dict(head=head, read=jobs.iso(), tasks=tasks, claims=claims, pools=heart, built=sorted(have),
                      answered=answered, master=m, answered_q=answered_q,
                      bank_questions={b: {v: len(k) for v, k in vs.items()} for b, vs in keys.items()})))
