#!/usr/bin/env python3
"""The job board's master: works out what should exist, subtracts what does, and lists the rest as tasks.

    python coord/master.py publish     # rewrite tasks.json in the jobs dataset (one commit)
    python coord/master.py publish --dry-run   # the same tasks, counted and printed; nothing is committed
    python coord/master.py view        # print the board's view as JSON (pools, running, built), for the dashboard

Desired state (EXPERIMENTS below) x videos, minus memories already in the results dataset, minus nothing
else: claims and leases are the workers' business. A task's `pools` says which pools may take it; the order
of tasks.json is the priority. Reassignment is automatic: a claim whose lease lapsed is free again, and a
task that failed twice is held back and shown as a failure.

`qa` experiments answer a named epic-morph bank (one experiment per bank in BANKS): a task per video that has
questions in the bank and no answers_<bank>.json.gz in the results dataset, depending on the same video's build task
while that is pending.
The bank is generated afresh each cycle (v3/export_bank.py <bank>, deterministic per commit) in the epic-morph
checkout EPIC_MORPH_DIR, the master's own (not a worker's, which may be reading its bank files). publish stops
without EPIC_MORPH_DIR: a tasks.json without the qa tasks would drop them from the board.

Arms that call a paid API (NEEDS_ENV) carry `needs_env`, the variables a worker must have set to take them, and go
to pool `max` only. The reported-configuration arms (REPORTED: a gpt-5-mini memory per video, answered by several
models) are listed right after the API tasks, and while PILOT is non-empty only the tasks it names are listed.
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
# A person's one worker over several queues (worker.py --queues, e.g. Vatsal's `vatsal` on Nexus) is listed for both
# arms: the worker itself claims only the tasks one of its queues fits, so without an 80 GB queue it never takes a 27B.
M8 = ["max", "nexus-tron", "nexus-sailon", "nexus-scavenger", "wriva", "vatsal", "zaratan"]   # zaratan: H100 jobs
M27 = ["wriva", "zaratan", "max", "vatsal"]
POOLS = {("merit", "m27"): M27 + ["arm00"], ("merit", "m8"): M8,
         ("magic", "m27"): M27, ("magic", "m8"): M8,
         ("worldmm", "m27"): M27, ("worldmm", "m8"): M8}
# the banks every baseline and arm answers, in priority order: batch10 and epic-morph tranches 2 and 3
# (v3/batch10.txt, v3/tranche2.txt, v3/tranche3.txt), disjoint by video; a video's answers under one bank are reused by
# another (hf_results.py earlier-answers), so a question is never answered twice. A bank a baseline has already
# answered (MERIT's batch10) lists nothing: its answers are in the dataset
BANKS = ["batch10_agree2", "t2_agree2", "t3_agree2"]
# Closed-source API models (baselines/api): no GPU and no memory, one qa task per video and config. They need the
# provider's key (GEMINI_API_KEY / OPENAI_API_KEY in the worker's environment), which only Max's machine has, so no
# other pool may take them. Listed first: 30 short tasks that should not wait behind the memory builds.
API_CONFIGS = ["gemini-3.1-flash-lite-high", "gemini-3.6-flash-high", "gpt-5-f1024"]
# The reported-configuration arms (baselines/<b>/README.md): g5m is the memory written by gpt-5-mini, once per video;
# the g5m-* arms answer from it with another reasoning model and keep answers plus a reference to that memory
# (<video>_memory_ref.json) in their own folders. MEMORY_ARM: the arm whose build an arm's qa waits for (default itself).
MEMORY_ARM = {("worldmm", "g5m-q8"): "g5m", ("worldmm", "g5m-g5"): "g5m",
              ("merit", "g5m-q8"): "g5m", ("merit", "g5m-q27"): "g5m", ("merit", "g5m-g5"): "g5m"}
# Arms that call a paid API: every task of one carries needs_env (a worker with any of them unset leaves the task) and
# is listed for KEY_POOLS only, whatever its EXPERIMENTS line or --only-pools says: nobody but Max has the keys.
# WorldMM's answer arms both call gpt-5-mini (its retriever); MERIT's Qwen answer arms (g5m-q8, g5m-q27) call nothing
# at answer time, so they run on the GPU pools of m8 / m27 once the memory is in the dataset.
KEY_POOLS = ["max"]
NEEDS_ENV = {("api", c): ["GEMINI_API_KEY" if c.startswith("gemini") else "OPENAI_API_KEY"] for c in API_CONFIGS} | {
    ("worldmm", "g5m"): ["OPENAI_API_KEY"], ("worldmm", "g5m-q8"): ["OPENAI_API_KEY"], ("worldmm", "g5m-g5"): ["OPENAI_API_KEY"],
    ("merit", "g5m"): ["OPENAI_API_KEY"], ("merit", "g5m-g5"): ["OPENAI_API_KEY"]}
# OpenAI shuts gpt-5 and gpt-5-mini down on 2026-12-11, and a built memory can be reused afterwards: the g5m memories
# are built for ALL videos, as every memory is (desired: build_order.txt plus the batch10 videos it lacks, batch10's
# first, then each bank's of BANKS in order, then the rest); the answer arms answer every one of these banks.
REPORTED_BANKS = BANKS
REPORTED = [("worldmm", "g5m", "build", KEY_POOLS, None), ("merit", "g5m", "build", KEY_POOLS, None)] + [
    (b, arm, "qa", pools, bank) for bank in REPORTED_BANKS for b, arm, pools in (
        ("worldmm", "g5m-q8", KEY_POOLS), ("worldmm", "g5m-g5", KEY_POOLS),
        ("merit", "g5m-q8", M8), ("merit", "g5m-q27", POOLS["merit", "m27"]), ("merit", "g5m-g5", KEY_POOLS))]
# The pilot gate: while PILOT is non-empty, these are the only tasks of the reported-configuration arms that are
# listed; all their other tasks are withheld from tasks.json (the API tasks and the local-model tasks are not
# affected). Empty it (a one-line PR) once the pilot's token usage has been read, and the rest is listed.
PILOT = ["worldmm-g5m-build-P05-20240425-201444"]
EXPERIMENTS = [("api", c, "qa", KEY_POOLS, "batch10_agree2") for c in API_CONFIGS] + REPORTED + [
    (b, arm, stage, POOLS[b, arm], bank)
    for b in ("merit", "magic", "worldmm") for stage in ("build", "qa") for arm in ("m27", "m8")
    for bank in (BANKS if stage == "qa" else [None])]
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


def reported(t):
    """A task of a reported-configuration arm: one whose memory is written through a paid API (not baselines/api)."""
    b, arm = t["baseline"], t["arm"]
    return b != "api" and (b, MEMORY_ARM.get((b, arm), arm)) in NEEDS_ENV


def desired(have, files):
    order = [v.strip() for v in open(BUILD_ORDER) if v.strip()]
    # every memory is built for all videos: each bank's (batch10's first: build_order.txt lacks them, as MERIT built
    # them before the board), then the rest of the build order; the reported arms are also ranked by that group
    videos, group = [], {}
    for g, vs in enumerate([bank_videos(bank) for bank in REPORTED_BANKS] + [order]):
        for v in vs:
            if group.setdefault(v, g) == g and v not in videos:
                videos.append(v)
    tasks, ids = [], set()
    for b, arm, stage, pools, bank in EXPERIMENTS:
        env = {"needs_env": NEEDS_ENV[b, arm]} if (b, arm) in NEEDS_ENV else {}
        pools = KEY_POOLS if env else pools
        mem = MEMORY_ARM.get((b, arm), arm)
        if mem != arm:   # for the worker: the arm whose dataset folder holds this task's memory (worker.py --prefetch)
            env["memory_arm"] = mem
        if stage == "build":
            for v in videos:
                if (b, arm, v) not in have:
                    base = f"{BUILD_ON.get(b)}-{arm}-build-{v}"
                    tasks.append(dict(id=f"{b}-{arm}-build-{v}", baseline=b, arm=arm, stage="build", video=v, bank=None,
                                      pools=pools, deps=[base] if base in ids else [], **env))
                    ids.add(tasks[-1]["id"])
            continue
        for v in bank_videos(bank):
            if f"{b}/{arm}/{v}/answers_{bank}.json.gz" in files:
                continue
            # wait for the video's build task (its memory arm's) while it is listed; a memory already built (on the
            # board or off it) needs no wait, and with neither, qa builds the memory itself first
            build = f"{b}-{mem}-build-{v}"
            tasks.append(dict(id=f"{b}-{arm}-{stage}-{bank}-{v}", baseline=b, arm=arm, stage=stage, video=v, bank=bank,
                              pools=pools, deps=[build] if build in ids else [], **env))
            ids.add(tasks[-1]["id"])
    # the reported arms by video (batch10: both memories, then its answers; then tranche 2; then the other memories),
    # so the first rows of the sheet do not wait behind 300 builds; every other task keeps its EXPERIMENTS order
    rank = {id(t): i for i, t in enumerate(tasks)}
    first = min([rank[id(t)] for t in tasks if reported(t)], default=0)
    tasks.sort(key=lambda t: (first, group[t["video"]], rank[id(t)]) if reported(t) else (rank[id(t)], 0, 0))
    for i, t in enumerate(tasks):
        t["priority"] = i
    return tasks


def gate(tasks):
    """(tasks to list, how many PILOT withholds): of the reported arms only PILOT's tasks, while it is non-empty."""
    keep = [t for t in tasks if not PILOT or not reported(t) or t["id"] in PILOT]
    return keep, len(tasks) - len(keep)


def restrict(tasks, only_pools):
    """--only-pools (a test): every task to these pools, except that a task needing a key never leaves its own."""
    for t in tasks:
        t["pools"] = [p for p in only_pools if p in t["pools"]] if t.get("needs_env") else only_pools


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
        tasks, withheld = gate(desired(have, files))
        if a.first:
            # the API tasks stay on top (they are few, short and only Max's pool can take them), then the reported
            # arms in their own order (batch10 first), then the first videos
            tasks.sort(key=lambda t: (0 if t["baseline"] == "api" else 1 if reported(t) else 2,
                                      not reported(t) and t["video"] not in a.first, t["priority"]))
        for i, t in enumerate(tasks):
            t["priority"] = i
        if a.only_pools is not None:
            restrict(tasks, a.only_pools)
        if a.dry_run:
            n = Counter((t["baseline"], t["arm"], t["stage"], t["bank"] or "-") for t in tasks)
            wait = Counter((t["baseline"], t["arm"], t["stage"], t["bank"] or "-") for t in tasks if t["deps"])
            for k, c in sorted(n.items()):
                print(f"{' '.join(k):30} {c:4} tasks ({wait[k]} waiting on a build)")
            print(f"pilot gate: {withheld} tasks of the reported-configuration arms withheld"
                  + (f"; listed of them: {[t['id'] for t in tasks if reported(t)]}" if PILOT else " (PILOT is empty)"))
            print(f"need a key (pool {', '.join(KEY_POOLS)} only): {sum(1 for t in tasks if t.get('needs_env'))} tasks")
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
