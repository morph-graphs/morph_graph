#!/usr/bin/env python3
"""The job board's master: works out what should exist, subtracts what does, and lists the rest as tasks.

    python coord/master.py publish     # rewrite tasks.json in the jobs dataset (one commit)
    python coord/master.py view        # print the board's view as JSON (pools, running, built), for the dashboard

Desired state (EXPERIMENTS below) x videos, minus memories already in the results dataset, minus nothing
else: claims and leases are the workers' business. A task's `pools` says which pools may take it; the order
of tasks.json is the priority. Reassignment is automatic: a claim whose lease lapsed is free again, and a
task that failed twice is held back and shown as a failure.
"""
import argparse
import json
import os
import sys

from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobs  # noqa: E402

RESULTS = os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_ORDER = os.path.join(ROOT, "baselines", "merit", "build_order.txt")

# (baseline, arm, stage, pools that may run it). MERIT 27B builds run on arm00 and Zaratan outside the board
# for now (their loops predate it), so they are not listed here.
EXPERIMENTS = [
    ("merit", "m8", "build", ["max", "nexus", "nexus-test", "wriva"]),
]


def built(api):
    """{(baseline, arm, video)} whose build is recorded in the results dataset."""
    files = set(api.list_repo_files(RESULTS, repo_type="dataset"))
    out = set()
    for f in files:
        parts = f.split("/")
        if len(parts) == 4 and parts[3] == "runs.jsonl":
            out.add(tuple(parts[:3]))
    # a folder with runs.jsonl and its memory counts as built (MERIT: the keys file)
    return {k for k in out if f"{k[0]}/{k[1]}/{k[2]}/{k[2]}_30sec_4_key.json" in files or k[0] != "merit"}


def desired(have):
    order = [v.strip() for v in open(BUILD_ORDER) if v.strip()]
    tasks = []
    for b, arm, stage, pools in EXPERIMENTS:
        for v in order:
            if (b, arm, v) in have:
                continue
            tasks.append(dict(id=f"{b}-{arm}-{stage}-{v}", baseline=b, arm=arm, stage=stage, video=v, bank=None,
                              pools=pools, deps=[]))
    for i, t in enumerate(tasks):
        t["priority"] = i
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["publish", "view"])
    ap.add_argument("--first", nargs="*", default=[], help="task video ids to put at the top (e.g. a test video)")
    ap.add_argument("--only-pools", nargs="*", help="publish: restrict every task to these pools (for a test)")
    a = ap.parse_args()
    api = HfApi()
    have = built(api)
    head, old, pools = jobs.read()
    if a.action == "publish":
        tasks = desired(have)
        if a.first:
            tasks.sort(key=lambda t: (t["video"] not in a.first, t["priority"]))
            for i, t in enumerate(tasks):
                t["priority"] = i
        if a.only_pools is not None:
            for t in tasks:
                t["pools"] = a.only_pools
        ok = jobs.commit({"tasks.json": tasks, "master.json": {"updated": jobs.iso(), "tasks": len(tasks), "built": len(have)}},
                         head, f"master: {len(tasks)} tasks")
        print(f"{'published' if ok else 'board moved; rerun'}: {len(tasks)} tasks, {len(have)} memories already built")
    else:
        t = jobs.now()
        rows = []
        for name, p in pools.items():
            for tid, c in p.get("claims", {}).items():
                live = c["status"] == "running" and c.get("lease_until", 0) > t
                state = "running" if live else ("lease lapsed" if c["status"] == "running" else c["status"])
                rows.append(dict(pool=name, host=p.get("host", ""), task=tid, state=state, gpu=c.get("gpu", ""),
                                 started=c.get("started", ""), finished=c.get("finished", ""), seconds=c.get("seconds"),
                                 note=(c.get("note") or "")[-200:]))
        heart = {n: dict(updated=p.get("updated"), running=p.get("running"), slots=p.get("slots"), host=p.get("host"))
                 for n, p in pools.items()}
        print(json.dumps(dict(claims=rows, pools=heart, built=sorted(have), tasks=len(old))))


if __name__ == "__main__":
    main()
