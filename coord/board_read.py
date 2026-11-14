#!/usr/bin/env python3
"""Everything the dashboard needs from the board, read-only, as one JSON line (run where the HF token is).

    python coord/board_read.py

{tasks: the master's open tasks (tasks.json), claims: every pool's claims with all their fields,
 pools: heartbeats, built: [[baseline, arm, video]], answered: [[baseline, arm, video, bank]], master: master.json,
 bank_questions: {bank: {video: questions}}, answered_q: [[baseline, arm, video, bank, questions answered]],
 build_targets: {"<baseline>|<arm>": [videos its memory is meant for]}, held: [[baseline, arm, stage]]}

Build targets are the master's (master.desired): a build line with a bank covers that bank's videos; one without covers
every video, each bank's (master.REPORTED_BANKS, in order) then build_order.txt. `held` is master.HELD: work the
master keeps off the board (built elsewhere), whose progress the board still shows.

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
        qs = json.load(open(os.path.join(banks_dir, bank, f)))["questions"] if f.endswith("_questions.json") else []
        if qs:   # a video with no question is not in the bank (master.bank_videos)
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
        # an answer: a response (EGAgent: pred) that is not one of the master's failures
        ok = [r for r in rows if isinstance(r, dict) and str(r.get("response", r.get("pred")) or "").strip() not in master.FAILED]
        want = keys[bank][v]
        # answers keyed by question key (since #16); older files (batch10) carry the question text instead
        got = {r["key"] for r in ok if r.get("key") in want} | \
              {k for k, q in want.items() if q in {(r.get("question") or "").strip() for r in ok if "key" not in r}}
        answered_q.append([b, arm, v, bank, len(got)])
order = [v.strip() for v in open(master.BUILD_ORDER) if v.strip()]
every = list(dict.fromkeys([v for bank in master.REPORTED_BANKS for v in sorted(keys.get(bank, {}))] + order))
targets = {f"{b}|{arm}": (sorted(keys.get(bank, {})) if bank else every)
           for b, arm, stage, _, bank in master.EXPERIMENTS if stage == "build"}
print(json.dumps(dict(head=head, read=jobs.iso(), tasks=tasks, claims=claims, pools=heart, built=sorted(have),
                      build_targets=targets, held=sorted(map(list, getattr(master, "HELD", set()))),
                      answered=answered, master=m, answered_q=answered_q,
                      bank_questions={b: {v: len(k) for v, k in vs.items()} for b, vs in keys.items()})))
