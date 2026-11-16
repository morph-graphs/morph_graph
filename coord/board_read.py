#!/usr/bin/env python3
"""Everything the dashboard needs from the board, read-only, as one JSON line (run where the HF token is).

    python coord/board_read.py

{tasks: the master's open tasks (tasks.json), claims: every pool's claims with all their fields,
 pools: heartbeats, built: [[baseline, arm, video]], answered: [[baseline, arm, video, bank]] (an _end bank: also a video
 its base bank's answers cover), master: master.json,
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
import shutil
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
# an _end bank (<bank>_end, master.BANKS since 2026-11-15) is counted from its own file and its base bank's (the same
# questions), as the master does: also a video with only the base file
base_of = {b: b[:-len("_end")] for b in keys if b.endswith("_end")}
mine += sorted({(b, arm, v, e) for b, arm, v, x in answered for e, base in base_of.items()
                if x == base and v in keys[e]} - set(mine))
listed = {(t["baseline"], t["arm"], t["video"], t.get("bank")) for t in tasks}
if mine:
    # Only answer files that changed are downloaded: each is kept under its blob id, which a listing of the dataset
    # gives in a few requests. A snapshot of all of them (~1,700 files) at every new dataset commit spent the
    # account's Hugging Face API limit (1,000 requests per 5 min, shared with the workers; 429s on 2026-11-15).
    blob = {f.path: f.blob_id for f in HfApi().list_repo_tree(master.RESULTS, repo_type="dataset", recursive=True)
            if f.path.endswith(".json.gz") and "/answers_" in f.path and getattr(f, "blob_id", None)}
    cache = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "board_answers")
    os.makedirs(cache, exist_ok=True)
    def rows_of(path):
        local = os.path.join(cache, f"{blob.get(path, 'none')}.json.gz")
        try:
            if not os.path.exists(local):
                shutil.copyfile(hf_hub_download(master.RESULTS, path, repo_type="dataset"), local + ".tmp")
                os.replace(local + ".tmp", local)
            rows = json.load(gzip.open(local, "rt"))
        except Exception:
            return None
        return rows.values() if isinstance(rows, dict) else rows

    for b, arm, v, bank in mine:
        paths = [f"{b}/{arm}/{v}/answers_{x}.json.gz" for x in (bank, base_of.get(bank)) if x]
        read = [rows_of(p) for p in paths if p in files]
        if all(r is None for r in read):
            continue
        # an answer: a response (EGAgent: pred) that is not one of the master's failures
        ok = [r for rows in read for r in rows or [] if isinstance(r, dict)
              and str(r.get("response", r.get("pred")) or "").strip() not in master.FAILED]
        want = keys[bank][v]
        # answers keyed by question key (since #16); older files (batch10) carry the question text instead
        got = {r["key"] for r in ok if r.get("key") in want} | \
              {k for k, q in want.items() if q in {(r.get("question") or "").strip() for r in ok if "key" not in r}}
        answered_q.append([b, arm, v, bank, len(got)])
        if bank in base_of and len(got) == len(want) and (b, arm, v, bank) not in answered and (b, arm, v, bank) not in listed:
            # every question answered, from the base bank's file, and no task for it (the master's rule is stricter:
            # trace, wording, query time): done
            answered.append((b, arm, v, bank))
order = [v.strip() for v in open(master.BUILD_ORDER) if v.strip()]
every = list(dict.fromkeys([v for bank in master.REPORTED_BANKS for v in sorted(keys.get(bank, {}))] + order))
targets = {}   # an arm with several build lines covers all their videos
for b, arm, stage, _, bank in master.EXPERIMENTS:
    if stage == "build":
        targets.setdefault(f"{b}|{arm}", set()).update(keys.get(bank, {}) if bank else every)
targets = {k: sorted(v) for k, v in targets.items()}
print(json.dumps(dict(head=head, read=jobs.iso(), tasks=tasks, claims=claims, pools=heart, built=sorted(have),
                      build_targets=targets, held=sorted(map(list, getattr(master, "HELD", set()))),
                      answered=answered, master=m, answered_q=answered_q,
                      bank_questions={b: {v: len(k) for v, k in vs.items()} for b, vs in keys.items()})))
