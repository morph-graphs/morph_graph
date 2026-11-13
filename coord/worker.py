#!/usr/bin/env python3
"""One pool's worker for the job board: claims tasks the master listed for this pool, runs them, reports.

Run it from a morph_graph checkout of main, on the machine that has the GPUs, with a Hugging Face token that
can write morph-graphs/epic-morph-jobs and morph-graphs/epic-morph-results (HF_TOKEN or `hf auth login`):

    python coord/worker.py --pool max --launcher docker --slots 0 1 2 3          # a GPU box: one task per GPU
    python coord/worker.py --pool nexus-test --launcher local --slots 0 --max-tasks 1
    python coord/worker.py --pool max --launcher slurm --runtime apptainer --slots 1 2 3 4 \
        --sbatch "--partition=gpu --gres=gpu:1 -c 8 --mem=64G --time=8:00:00"        # a cluster: up to 4 jobs at once

Builds need the videos; qa tasks also need EPIC_MORPH_DIR, an epic-morph checkout that has the task's bank (DESIGN.md,
"Answer (qa) tasks").

With --launcher slurm the worker runs where `sbatch` works (a login node) and each task becomes one
`sbatch --wait` job that runs the baseline's container on a GPU node (the scheduler picks the GPU); slots are
just how many jobs may be queued or running at once.

Each cycle (every --every seconds): read the board, mark finished tasks done/failed with their runtime and
GPU, renew the leases of running ones, claim the highest-priority free tasks for free slots (one commit to
this pool's own file; if anyone committed meanwhile it re-reads), then start them. A task runs the baseline's
own script (stage, then upload), so results go to the results dataset as usual. Stop with Ctrl-C; running
tasks keep going, and a restarted worker picks them up again from its state folder.
"""
import argparse
import json
import re
import shlex
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobs  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A task's fields go into a shell command, so only these shapes are ever run (anything else is skipped).
VALID = dict(baseline=re.compile(r"^(merit|worldmm|magic|m3agent)$"), arm=re.compile(r"^m(8|27|3)$"),
             stage=re.compile(r"^(prep|build|qa|blind)$"), video=re.compile(r"^P\d\d-\d{8}-\d{6}$"),
             bank=re.compile(r"^[A-Za-z0-9_]*$"), id=re.compile(r"^[A-Za-z0-9_.-]+$"))


def valid(task):
    return all(VALID[k].match(str(task.get(k) or "")) for k in VALID)


def command(task, gpus, a, log):
    """The shell command that runs one task: the stage, then the upload of its results folder."""
    b, stage, v, arm, bank = task["baseline"], task["stage"], task["video"], task["arm"], task.get("bank") or ""
    if a.launcher == "docker":
        run = f"MERIT_GPU={gpus} bash baselines/{b}/docker/run.sh"
    elif a.launcher == "slurm":
        run = f"MERIT_RUNTIME={a.runtime} bash baselines/{b}/docker/run.sh"
    else:
        run = f"CUDA_VISIBLE_DEVICES={gpus} bash baselines/{b}/run_{b}.sh"
    inner = f"{run} {stage} {v} {arm} {bank} && {run} upload {v} {arm}"
    if a.launcher == "slurm":   # one job per task; --wait returns the job's exit code when it ends
        return (f"sbatch --wait --parsable --job-name=board-{task['id'][:60]} --output={log} --open-mode=append "
                f"{a.sbatch} --wrap {shlex.quote(f'cd {REPO_ROOT} && nvidia-smi -L && {inner}')}")
    return inner


def slurm_job_alive(submit_log):
    """The Slurm job of a task whose `sbatch --wait` is gone (worker host restarted): still queued or running?"""
    try:
        jid = open(submit_log).readline().strip().split(";")[0]
        return bool(jid.isdigit() and subprocess.run(["squeue", "-h", "-j", jid], capture_output=True, text=True,
                                                    timeout=60).stdout.strip())
    except Exception:
        return False


def alive(pidfile):
    """Is the task's process (pid kept in a local file, since claims are re-read from the board) still there?"""
    try:
        os.kill(int(open(pidfile).read().strip()), 0)
        return True
    except (OSError, ValueError) as e:
        return isinstance(e, PermissionError)


def gpu_name(gpus):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout
        names = dict(l.split(", ", 1) for l in out.strip().splitlines())
        return ", ".join(names.get(g, "?") for g in gpus.split(","))
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True, help="this pool's name on the board (e.g. max, nexus, arm00)")
    ap.add_argument("--launcher", choices=["docker", "local", "slurm"], required=True)
    ap.add_argument("--runtime", choices=["apptainer", "docker"], default="apptainer", help="slurm: how the job runs the image")
    ap.add_argument("--sbatch", default="", help="slurm: resources and queue for each task's job, e.g. '--partition=gpu --gres=gpu:1 -c 8 --mem=64G --time=8:00:00'")
    ap.add_argument("--slots", nargs="+", required=True, help="GPU ids per task slot: '0 1' = two one-GPU slots, '0,1' = one two-GPU slot")
    ap.add_argument("--every", type=int, default=300, help="seconds between cycles")
    ap.add_argument("--max-tasks", type=int, default=0, help="stop claiming after this many (0 = no limit)")
    ap.add_argument("--state", default=None, help="local folder for logs and exit codes (default ~/.cache/coord/<pool>)")
    a = ap.parse_args()
    state = a.state or os.path.expanduser(f"~/.cache/coord/{a.pool}")
    os.makedirs(state, exist_ok=True)
    host = socket.gethostname()
    claimed_total = 0
    while True:
        head, tasks, pools = jobs.read()
        mine = pools.get(a.pool, {"pool": a.pool, "claims": {}})
        claims = mine.setdefault("claims", {})
        t = jobs.now()
        others = {tid: (n, c) for n, p in pools.items() if n != a.pool for tid, c in p.get("claims", {}).items()}
        # finished, still running, lost, or taken over by another pool?
        for tid, c in claims.items():
            if c["status"] != "running" or c.get("host") != host:
                continue
            ex = os.path.join(state, f"{tid}.exit")
            o = others.get(tid)
            if o and (o[1]["status"] == "done" or (o[1]["status"] == "running" and o[1].get("claimed", "") > c.get("claimed", ""))):
                # our lease lapsed and another pool took the task: stop renewing (our copy may still finish; outputs are idempotent)
                c.update(status="superseded", finished=jobs.iso(), note=f"taken over by pool {o[0]}")
                continue
            if os.path.exists(ex):
                code = int(open(ex).read().strip() or 1)
                c.update(status="done" if code == 0 else "failed", exit=code, finished=jobs.iso(),
                         seconds=int(t - c.get("started_t", t)))
                if code:   # the job's log, or (slurm, job never started) sbatch's own error
                    for f in (os.path.join(state, f"{tid}.log"), os.path.join(state, f"{tid}.log.submit")):
                        if os.path.exists(f) and os.path.getsize(f):
                            c["note"] = open(f).read()[-400:]
                            break
            elif not alive(os.path.join(state, f"{tid}.pid")) and not (
                    a.launcher == "slurm" and slurm_job_alive(os.path.join(state, f"{tid}.log.submit"))):
                c.update(status="failed", exit=-1, finished=jobs.iso(), note="the task's process ended without an exit code (killed or host restarted)")
            else:
                c["lease_until"] = t + jobs.LEASE_S
        busy = {c["slot"] for c in claims.values() if c["status"] == "running" and c.get("host") == host}
        free = [s for s in a.slots if s not in busy]
        held = jobs.taken(pools, t)
        fails = jobs.failures(pools)
        done = {tid for tid, (_, c) in held.items() if c["status"] == "done"}
        start = []
        for task in sorted(tasks, key=lambda x: x.get("priority", 0)):
            if not free or (a.max_tasks and claimed_total + len(start) >= a.max_tasks):
                break
            tid = task["id"]
            if not valid(task):
                print(f"{jobs.iso()} skipping malformed task {str(tid)[:80]!r}", flush=True)
                continue
            if tid in held or fails.get(tid, 0) >= 2 or (claims.get(tid, {}).get("status") == "failed"):
                continue
            if a.pool not in task.get("pools", []) and "*" not in task.get("pools", []):
                continue
            if any(d not in done for d in task.get("deps", [])):
                continue
            slot = free.pop(0)
            claims[tid] = dict(status="running", host=host, slot=slot,
                               gpu="(scheduler)" if a.launcher == "slurm" else gpu_name(slot), claimed=jobs.iso(),
                               started=jobs.iso(), started_t=t, lease_until=t + jobs.LEASE_S)
            start.append((task, slot))
        mine.update(pool=a.pool, host=host, slots=a.slots, launcher=a.launcher, updated=jobs.iso(),
                    running=sum(1 for c in claims.values() if c["status"] == "running"))
        if not jobs.commit({f"pools/{a.pool}.json": mine}, head, f"{a.pool}: {len(start)} claimed"):
            print(f"{jobs.iso()} board moved under us; re-reading", flush=True)
            continue
        for task, slot in start:   # start only after the claim is committed
            tid = task["id"]
            log, ex = os.path.join(state, f"{tid}.log"), os.path.join(state, f"{tid}.exit")
            for f in (ex, ex + ".tmp"):   # an exit code from an earlier run of this task must not end this one
                if os.path.exists(f):
                    os.remove(f)
            run = command(task, slot, a, log)
            out = log + ".submit" if a.launcher == "slurm" else log   # slurm: the job writes the log itself
            cmd = f"cd {REPO_ROOT} && ( {run} ) > {out} 2>&1; echo $? > {ex}.tmp && mv {ex}.tmp {ex}"
            pid = subprocess.Popen(["bash", "-c", cmd], start_new_session=True).pid
            open(os.path.join(state, f"{tid}.pid"), "w").write(str(pid))
            claimed_total += 1
            print(f"{jobs.iso()} started {tid} on GPU {slot}: {task['baseline']} {task['arm']} {task['stage']} {task['video']}", flush=True)
        running = sum(1 for c in claims.values() if c["status"] == "running" and c.get("host") == host)
        if a.max_tasks and claimed_total >= a.max_tasks and running == 0:
            print(f"{jobs.iso()} done: {claimed_total} task(s) claimed and finished", flush=True)
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
