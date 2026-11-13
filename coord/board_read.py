#!/usr/bin/env python3
"""Everything the dashboard needs from the board, read-only, as one JSON line (run where the HF token is).

    python coord/board_read.py

{tasks: the master's open tasks (tasks.json), claims: every pool's claims with all their fields,
 pools: heartbeats, built: [[baseline, arm, video]], answered: [[baseline, arm, video, bank]], master: master.json}
"""
import json
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

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
print(json.dumps(dict(head=head, read=jobs.iso(), tasks=tasks, claims=claims, pools=heart, built=sorted(have),
                      answered=answered, master=m)))
