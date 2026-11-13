#!/usr/bin/env python3
"""The job board's master: works out what should exist, subtracts what does, and lists the rest as tasks.

    python coord/master.py publish     # rewrite tasks.json in the jobs dataset (one commit)
    python coord/master.py publish --dry-run   # the same tasks, counted and printed; nothing is committed
    python coord/master.py view        # print the board's view as JSON (pools, running, built), for the dashboard

Desired state (EXPERIMENTS below) x videos, minus memories already in the results dataset, minus nothing
else: claims and leases are the workers' business. A task's `pools` says which pools may take it; the order
of tasks.json is the priority. Reassignment is automatic: a claim whose lease lapsed is free again, and a
task that failed twice is held back and shown as a failure.

`qa` experiments answer a named epic-morph bank: a task per video that has questions in the bank and no
answers_<bank>.json.gz in the results dataset, depending on the same video's build task while that is pending.
The bank is generated afresh each cycle (v3/export_bank.py <bank>, deterministic per commit) in the epic-morph
checkout EPIC_MORPH_DIR, the master's own (not a worker's, which may be reading its bank files). publish stops
without EPIC_MORPH_DIR: a tasks.json without the qa tasks would drop them from the board.
"""
import argparse
import functools
import json
import os
import subprocess
import sys
from collections import Counter

from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobs  # noqa: E402

RESULTS = os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_ORDER = os.path.join(ROOT, "baselines", "merit", "build_order.txt")

# (baseline, arm, stage, pools that may run it, bank). A qa task answers its bank on the build's GPU family, so it
# takes the build's pools. Every baseline's m8 needs one GPU of 40 GB or more. m27: MERIT one 80 GB GPU (H100 / H200 /
# GH200) or 2x 48 GB; MAGIC one 80 GB GPU or 3+ GPUs; WorldMM one 80 GB GPU or 2x 48 GB. arm00 (GH200, aarch64) runs
# MERIT only: the images are x86, and only MERIT has a native arm environment there. A worker takes only the
# baselines it was started with (worker.py --baselines), so listing a baseline here runs nothing until a pool opts in.
# our three Nexus workers are named after the queue their tasks go to: tron (QoS high), scavenger, and
# vulcan-ampere with QoS vulcan-sailon
M8 = ["max", "nexus-tron", "nexus-sailon", "nexus-scavenger", "wriva", "zaratan"]   # zaratan: H100 jobs
POOLS = {("merit", "m27"): ["wriva", "arm00", "zaratan", "max"], ("merit", "m8"): M8,
         ("magic", "m27"): ["wriva", "zaratan", "max"], ("magic", "m8"): M8,
         ("worldmm", "m27"): ["wriva", "zaratan", "max"], ("worldmm", "m8"): M8}
BANK = "t2_agree2"   # tranche 2 (epic-morph v3/tranche2.txt)
# Closed-source API models (baselines/api): no GPU and no memory, one qa task per video and config. They need the
# provider's key (GEMINI_API_KEY / OPENAI_API_KEY in the worker's environment), which only Max's machine has, so no
# other pool may take them. Listed first: 30 short tasks that should not wait behind the memory builds.
API_CONFIGS = ["gemini-3.1-flash-lite-high", "gemini-3.6-flash-high", "gpt-5-f1024"]
EXPERIMENTS = [("api", c, "qa", ["max"], "batch10_agree2") for c in API_CONFIGS] + [
    (b, arm, stage, POOLS[b, arm], BANK if stage == "qa" else None)
    for b in ("merit", "magic", "worldmm") for stage in ("build", "qa") for arm in ("m27", "m8")]
# the file whose presence in <baseline>/<arm>/<video>/ (next to runs.jsonl) means the memory is built: the last
# one each baseline writes ({v}: the video)
MEMORY_DONE = {"merit": "{v}_30sec_4_key.json", "magic": "graph.pkl", "worldmm": "{v}_visual.pkl"}
# a build that starts from another baseline's memory waits for that build while it is listed: MAGIC's 30 s captions
# are MERIT's (same backbone), fetched from the dataset; without the wait it would caption the video itself
BUILD_ON = {"magic": "merit"}


def bank_videos(bank):
    """Videos with at least one question in the epic-morph bank."""
    epic = os.environ.get("EPIC_MORPH_DIR")
    if not epic:
        raise SystemExit(f"set EPIC_MORPH_DIR to an epic-morph checkout with bank {bank} (qa tasks need it)")
    return exported(epic, bank)


@functools.lru_cache(maxsize=None)   # one export per bank and run, however many experiments answer it
def exported(epic, bank):
    d = os.path.join(epic, "outputs", "v3", "banks", bank)
    # banks are generated, not committed: regenerate every time, so an interrupted export or an older commit's
    # files never stand in for the checkout's bank (a few seconds)
    subprocess.run([sys.executable, "v3/export_bank.py", bank], cwd=epic, check=True, stdout=subprocess.DEVNULL)
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith("_questions.json") and json.load(open(os.path.join(d, f)))["questions"]:
            out.append(f[:-len("_questions.json")])
    return out


def built(api):
    """({(baseline, arm, video)} whose build is recorded in the results dataset, the dataset's file set)."""
    return built_from(set(api.list_repo_files(RESULTS, repo_type="dataset")))


def built_from(files):
    """A folder with runs.jsonl and its baseline's last memory file counts as built."""
    out = set()
    for f in files:
        parts = f.split("/")
        if len(parts) == 4 and parts[3] == "runs.jsonl":
            out.add(tuple(parts[:3]))
    return {k for k in out if k[0] in MEMORY_DONE and f"{k[0]}/{k[1]}/{k[2]}/{MEMORY_DONE[k[0]].format(v=k[2])}" in files}, files


def desired(have, files):
    order = [v.strip() for v in open(BUILD_ORDER) if v.strip()]
    tasks = []
    for b, arm, stage, pools, bank in EXPERIMENTS:
        if stage == "build":
            for v in order:
                if (b, arm, v) not in have:
                    base = f"{BUILD_ON.get(b)}-{arm}-build-{v}"
                    tasks.append(dict(id=f"{b}-{arm}-build-{v}", baseline=b, arm=arm, stage="build", video=v, bank=None,
                                      pools=pools, deps=[base] if any(t["id"] == base for t in tasks) else []))
            continue
        for v in bank_videos(bank):
            if f"{b}/{arm}/{v}/answers_{bank}.json.gz" in files:
                continue
            # wait for the video's build task while it is listed; a memory already built (on the board or off it)
            # needs no wait, and with neither, qa builds the memory itself first
            build = f"{b}-{arm}-build-{v}"
            deps = [build] if any(t["id"] == build for t in tasks) else []
            tasks.append(dict(id=f"{b}-{arm}-{stage}-{bank}-{v}", baseline=b, arm=arm, stage=stage, video=v, bank=bank,
                              pools=pools, deps=deps))
    for i, t in enumerate(tasks):
        t["priority"] = i
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["publish", "view", "reset"])
    ap.add_argument("--task", nargs="*", default=[], help="reset: task ids whose failed claims to clear (after fixing the cause)")
    ap.add_argument("--first", nargs="*", default=[], help="task video ids to put at the top (e.g. a test video)")
    ap.add_argument("--only-pools", nargs="*", help="publish: restrict every task to these pools (for a test)")
    ap.add_argument("--dry-run", action="store_true", help="publish: print the task counts, commit nothing")
    a = ap.parse_args()
    first = os.path.join(os.path.dirname(os.path.abspath(__file__)), "first_videos.txt")
    if not a.first and os.path.exists(first):   # videos to put at the top (e.g. a collaborator's test)
        a.first = open(first).read().split()
    api = HfApi()
    have, files = built(api)
    head, old, pools = jobs.read()
    if a.action == "reset":   # clear failed claims so the task can be claimed again
        files = {}
        for name, p in pools.items():
            gone = [t for t in a.task if p.get("claims", {}).get(t, {}).get("status") == "failed"]
            for t in gone:
                del p["claims"][t]
            if gone:
                files[f"pools/{name}.json"] = p
        ok = files and jobs.commit(files, head, f"master: reset {', '.join(a.task)}")
        print(f"reset {a.task} in {sorted(files)}" if ok else "nothing reset (or board moved; rerun)")
    elif a.action == "publish":
        tasks = desired(have, files)
        if a.first:
            # the API tasks stay on top (they are few, short and only Max's pool can take them), then the first videos
            tasks.sort(key=lambda t: (t["baseline"] != "api", t["video"] not in a.first, t["priority"]))
            for i, t in enumerate(tasks):
                t["priority"] = i
        if a.only_pools is not None:
            for t in tasks:
                t["pools"] = a.only_pools
        if a.dry_run:
            n = Counter((t["baseline"], t["arm"], t["stage"], t["bank"] or "-") for t in tasks)
            wait = Counter((t["baseline"], t["arm"], t["stage"], t["bank"] or "-") for t in tasks if t["deps"])
            for k, c in sorted(n.items()):
                print(f"{' '.join(k):30} {c:4} tasks ({wait[k]} waiting on a build)")
            print(f"dry run, nothing committed: {len(tasks)} tasks, {len(have)} memories already built; "
                  f"first: {[t['id'] for t in tasks[:5]]}")
            return
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
