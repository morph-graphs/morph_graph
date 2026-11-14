#!/usr/bin/env python3
"""One pool's worker for the job board: claims tasks the master listed for this pool, runs them, reports.

Run it from a morph_graph checkout of main, on the machine that has the GPUs, with a Hugging Face token that
can write morph-graphs/epic-morph-jobs and morph-graphs/epic-morph-results (HF_TOKEN or `hf auth login`):

    python coord/worker.py --pool max --launcher docker --slots 0 1 2 3 --baselines merit magic   # one task per GPU
    python coord/worker.py --pool nexus-scavenger --launcher local --slots 0 --max-tasks 1
    python coord/worker.py --pool max --launcher slurm --runtime apptainer --slots 1 2 3 4 \
        --sbatch "--partition=gpu --gres=gpu:1 -c 8 --mem=64G --time=8:00:00"        # a cluster: up to 4 jobs at once

A worker takes only the baselines named by --baselines (default merit api; api tasks go to the max pool only): name one only once this machine has its
images (docker pull, or the SIFs, below), or every task of it would fail here. With --runtime apptainer, set
MERIT_SIF_DIR to a folder of <baseline>_<arm>.sif files (apptainer pull merit_m8.sif docker://ghcr.io/morph-graphs/
merit:<tag>, the tag from baselines/<b>/docker/image_tag.sh <arm>). Every baseline's docker/run.sh also reads the
worker's MERIT_GPU, MERIT_RUNTIME, MERIT_SIF_DIR, MERIT_HF_CACHE and MERIT_WORK (one shared work folder per cluster),
so one worker and one environment run them all.

A task that calls a paid API lists the variables it needs (`needs_env`, e.g. OPENAI_API_KEY, or "A|B" for either
of two): the worker takes it only when they are set in its own environment, from where the task's container gets
the key by name. A key is never part of a command, and failure notes are scrubbed of keys before they go to the board.

Builds need the videos; qa tasks also need EPIC_MORPH_DIR, an epic-morph checkout that has the task's bank (DESIGN.md,
"Answer (qa) tasks").

With --launcher slurm the worker runs where `sbatch` works (a login node) and each task becomes one
`sbatch --wait` job that runs the baseline's container on a GPU node (the scheduler picks the GPU); slots are
just how many jobs may be queued or running at once.

With --outbox (machines whose jobs cannot reach the results dataset: Zaratan) a task's upload copies its results
folder to <checkout>/outbox/<task id>/ instead (hf_results.py, MERIT_OUTBOX). The finished task stays claimed as
`relay` (no other pool takes it, and tasks that depend on it wait) until a relay uploads the folder
(`hf_results.py relay --dir <it>`, from a machine that can) and marks it with an `.relayed` file; then it is done.
With --prefetch the worker downloads the results dataset's folders for a task's video and arm (every baseline's:
MAGIC reads MERIT's captions) into MERIT_HF_CACHE before starting it, so a job whose node reaches the Hub's API but
not its file storage still finds a memory built elsewhere: in the cache, hf_hub_download needs only the API
(Zaratan compute nodes, measured 2026-11-13: huggingface.co answers over IPv6, the xet/LFS file hosts are IPv4-only).

One worker for all of a person's queues: --queues <file> instead of --sbatch (coord/nexus_queues.json is the example,
coord/nexus_worker.sbatch the job that runs it on Nexus):

    python coord/worker.py --pool vatsal --launcher slurm --queues coord/nexus_queues.json --slots 1 2 3 4 5 6 7 8

The file names each queue's sbatch arguments and the arms its GPU fits ("m8", "m27", or "magic/m27" for one
baseline). The worker claims only tasks some queue fits, and submits each task to every fitting queue at once, under
one job name. The first copy to start runs it (coord/slurm_guard.sh: an atomic marker <task>.ran, then it cancels
the pending copies); a copy starting later exits without work. Each cycle one squeue shows which copies are still
queued: the task is done when the copy that ran wrote its exit code (<task>.rc), and failed when every copy left the
queue without one (seen on two cycles in a row, for a slow NFS). The state folder must be on a filesystem the GPU
nodes share. Slots count tasks, not copies.

Each cycle (every --every seconds): read the board, mark finished tasks done/failed with their runtime and
GPU, renew the leases of running ones, claim the highest-priority free tasks for free slots (one commit to
this pool's own file; if anyone committed meanwhile it re-reads), then start them. A task runs the baseline's
own script (stage, then upload), so results go to the results dataset as usual. Stop with Ctrl-C; running
tasks keep going, and a restarted worker picks them up again from its state folder. With --launcher slurm that
holds on any host that sees the state folder, whatever host the claims name: with --queues the jobs write their own
exit codes; with --sbatch a task whose `sbatch --wait` died with the old worker is followed through squeue, and when
its job has ended sacct decides (COMPLETED 0:0: done; anything else: failed).

The Hub allows a repository 128 commits an hour, shared by every pool: a cycle with nothing to report but renewed
leases and the heartbeat commits every 10 minutes at most, and when the Hub refuses a read or a commit (429, or the
network is down) the worker waits for the time it names, or a few minutes, and goes on. It never stops for that.
"""
import argparse
import getpass
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
# An arm is m8 / m27 / m3 for the local-model baselines, a reported-configuration arm (g5m, g5m-q8, ...), a paid row's
# arm (q25vl7b-gpt4o, egogpt_gpt4o) or a config name of baselines/api/configs/ (gpt-5-f1024): lower-case letters and
# digits in runs joined by one `.`, `-` or `_`, none of which the shell, a Slurm job name or a file pattern reads
# as anything but itself (no space, quote, `$`, `/`, `*`, and never a leading `-`, so it cannot become an option).
VALID = dict(baseline=re.compile(r"^(merit|worldmm|magic|m3agent|api|egor1|egagent|egorag|hippomm)$"),
             arm=re.compile(r"^(m(8|27|3)|[a-z0-9]+([._-][a-z0-9]+)*)$"),
             stage=re.compile(r"^(prep|build|qa|blind)$"), video=re.compile(r"^P\d\d-\d{8}-\d{6}$"),
             bank=re.compile(r"^[A-Za-z0-9_]*$"), id=re.compile(r"^[A-Za-z0-9_.-]+$"))


# What a failed task's log may carry: the value of any key or token in this worker's environment, or a key by shape
KEYLIKE = re.compile(r"\b(sk-[A-Za-z0-9_*.-]{6,}|AIza[A-Za-z0-9_-]{10,}|hf_[A-Za-z0-9]{10,})")


def scrub(text):
    """The text with keys and tokens removed, for a note that goes to the board."""
    for k, v in os.environ.items():
        if len(v) >= 8 and re.search(r"(KEY|TOKEN|SECRET)$", k):
            text = text.replace(v, f"<{k}>")
    return KEYLIKE.sub(lambda m: m.group(0)[:3] + "...", text)


def has_env(task):
    """Does this worker's environment have every variable the task needs (needs_env: an API key, by name; "A|B" is
    met by either)?"""
    return all(isinstance(e, str) and any(os.environ.get(x) for x in e.split("|")) for e in task.get("needs_env") or [])


HEARTBEAT_S = 600   # a cycle that changes nothing but leases and the heartbeat commits at most this often


def board_wait(e, what):
    """The board could not be read or written (the Hub's rate limit, 429: 128 commits an hour per repository; or the
    network): say so and wait, for the time the Hub names or a few minutes. The worker never stops for it."""
    r = getattr(e, "response", None)
    code = getattr(r, "status_code", None)
    try:
        wait = int(float(r.headers.get("Retry-After")))
    except Exception:
        wait = 0
    wait = min(3600, max(wait, 300 if code == 429 else 120))
    print(f"{jobs.iso()} board {what} failed ({type(e).__name__}{f' {code}' if code else ''}: {scrub(str(e))[:200]}); "
          f"waiting {wait} s", flush=True)
    time.sleep(wait)


def substance(pool_file):
    """A pool file without what changes every cycle (the heartbeat time, the leases): equal means nothing to report."""
    return json.dumps({**{k: v for k, v in pool_file.items() if k not in ("updated", "claims")},
                       "claims": {tid: {k: v for k, v in c.items() if k != "lease_until"}
                                  for tid, c in pool_file.get("claims", {}).items()}}, sort_keys=True)


def valid(task):
    return all(VALID[k].fullmatch(str(task.get(k) or "")) for k in VALID)   # fullmatch: `$` alone lets a final newline by


def inner(task, gpus, a):
    """The shell command that runs one task: the stage, then the upload of its results folder."""
    b, stage, v, arm, bank = task["baseline"], task["stage"], task["video"], task["arm"], task.get("bank") or ""
    if a.launcher == "docker":
        run = f"MERIT_GPU={gpus} bash baselines/{b}/docker/run.sh"
    elif a.launcher == "slurm":
        run = f"MERIT_RUNTIME={a.runtime} bash baselines/{b}/docker/run.sh"
    else:
        run = f"CUDA_VISIBLE_DEVICES={gpus} bash baselines/{b}/run_{b}.sh"
    up = f"MERIT_OUTBOX={task['id']} {run}" if getattr(a, "outbox", False) else run
    return f"{run} {stage} {v} {arm} {bank} && {up} upload {v} {arm}"


def command(task, gpus, a, log):
    if a.launcher == "slurm":   # one job per task; --wait returns the job's exit code when it ends
        return (f"sbatch --wait --parsable --job-name=board-{task['id'][:60]} --output={log} --open-mode=append "
                f"{a.sbatch} --wrap {shlex.quote(f'cd {REPO_ROOT} && nvidia-smi -L && {inner(task, gpus, a)}')}")
    return inner(task, gpus, a)


def load_queues(path):
    """--queues file: {name: {"sbatch": "<sbatch args>", "fits": ["m8", "magic/m27", ...]}}."""
    queues = json.load(open(path))
    for n, q in queues.items():
        if not (VALID["id"].match(n) and q.get("sbatch") and q.get("fits")):
            raise SystemExit(f"{path}: queue {n!r} needs a plain name, 'sbatch' and 'fits'")
    return queues


def names(task, arms):
    """Whether a queue's list of arms (m8) or baseline/arms (magic/m27) names this task."""
    return task["arm"] in arms or f"{task['baseline']}/{task['arm']}" in arms


def fitting(task, queues, idle=(), full=()):
    """The queues whose GPU can run this task: their `fits` names it, or their `fallback` names it and the queue is
    idle (no task its `fits` names is waiting: an H200 takes 8B work only when no 27B task needs it). A queue in
    `full` (as many of this worker's tasks on it as its max_jobs) takes no more."""
    return [n for n, q in queues.items() if n not in full
            and (names(task, q["fits"]) or (n in idle and names(task, q.get("fallback", []))))]


def submit_copies(task, a, queues, state, sbatch="sbatch"):
    """Submit one copy of the task's job to every queue that fits it, all under one job name. Each copy starts with
    coord/slurm_guard.sh: the first to start takes the task (marker file <tid>.ran) and cancels the pending copies; a
    copy that starts later exits 0 without work. The copy that ran writes its exit code to <tid>.rc. Returns the
    {job id: queue} it wrote to <tid>.jobs ({} and no file when no queue took a copy: sbatch's errors are in
    <tid>.log.submit)."""
    tid = task["id"]
    p = os.path.join(state, tid)
    if os.path.exists(p + ".jobs"):   # copies of an earlier run of it must not run beside this one
        subprocess.run(["scancel", *[l.split()[0] for l in open(p + ".jobs") if l.strip()]], capture_output=True)
    for ext in (".exit", ".exit.tmp", ".rc", ".rc.tmp", ".ran", ".jobs", ".log.submit"):   # from an earlier run of it
        if os.path.exists(p + ext):
            os.remove(p + ext)
    wrap = (f"bash {REPO_ROOT}/coord/slurm_guard.sh {shlex.quote(p + '.ran')} || exit 0; "
            f"cd {REPO_ROOT} && {{ nvidia-smi -L || true; }} && ( {inner(task, '', a)} ); "   # || true: a CPU queue has none
            f"echo $? > {p}.rc.tmp && mv {p}.rc.tmp {p}.rc")
    ids = {}
    for n in fitting(task, queues, idle=queues):   # the caller passes the queues chosen for it (a fallback one included)
        r = subprocess.run([sbatch, "--parsable", f"--job-name=board-{tid}", f"--output={p}.log", "--open-mode=append",
                            *shlex.split(queues[n]["sbatch"]), "--wrap", wrap], capture_output=True, text=True)
        jid = r.stdout.strip().split(";")[0]
        if r.returncode == 0 and jid.isdigit():
            ids[jid] = n
            open(p + ".jobs", "a").write(f"{jid} {n}\n")   # at once: a worker killed mid-loop still knows this copy
        else:
            open(p + ".log.submit", "a").write(f"queue {n}: sbatch exit {r.returncode}: {r.stdout}{r.stderr}\n")
    return ids


def my_jobs(squeue="squeue"):
    """Ids of this user's jobs still in the queue (pending, running, requeued), or None when squeue fails."""
    try:
        r = subprocess.run([squeue, "-h", "-u", getpass.getuser(), "-o", "%i"], capture_output=True, text=True, timeout=120)
        return set(r.stdout.split()) if r.returncode == 0 else None
    except Exception:
        return None


def copies_state(p, queued):
    """Where a task submitted with submit_copies stands (p: its path prefix in the state folder; queued: my_jobs()).
    Returns (exit code or None while it may still run, the queue that ran it or None, copy ids to cancel)."""
    jobs = dict(l.split() for l in open(p + ".jobs") if l.strip())
    ran = open(p + ".ran").read().strip() if os.path.exists(p + ".ran") else None
    cancel = [j for j in jobs if j != ran and j in queued] if ran else []   # left over: submitted after the winner started
    if any(j in queued for j in ([ran] if ran else jobs)):   # (a requeued winner is back in the queue: not over)
        return None, jobs.get(ran), cancel
    if os.path.exists(p + ".rc"):
        return int(open(p + ".rc").read().strip() or 1), jobs.get(ran), cancel
    return -1, jobs.get(ran), cancel   # every copy left the queue without an exit code


def slurm_job_state(submit_log, run=subprocess.run):
    """Where the Slurm job of a task stands, asked of Slurm itself (for when its `sbatch --wait` is gone, e.g. the
    worker was restarted, and so nothing will write the task's exit code): (state, detail) with state one of
        running   still queued or running (squeue lists it)
        done      ended COMPLETED with exit code 0:0 (sacct)
        failed    ended any other way; detail is sacct's state and exit code. Also: no job id in the submit log
        unknown   left squeue and sacct has no record of it yet (accounting lags a little): ask again next cycle
    """
    try:
        jid = open(submit_log).readline().strip().split(";")[0]
    except OSError:
        jid = ""
    if not jid.isdigit():
        return "failed", "no Slurm job was submitted"
    try:
        if run(["squeue", "-h", "-j", jid], capture_output=True, text=True, timeout=60).stdout.strip():
            return "running", jid
        rows = run(["sacct", "-j", jid, "-X", "-n", "-o", "State,ExitCode"], capture_output=True, text=True,
                   timeout=60).stdout.strip().splitlines()
    except Exception as e:   # the scheduler did not answer: no verdict from that
        return "unknown", f"job {jid}: {type(e).__name__}"
    if not rows or not rows[-1].split():
        return "unknown", f"job {jid}: not in squeue, not in sacct yet"
    *state, code = rows[-1].split()   # the last row: a requeued job has one per run; "CANCELLED by <uid>" is two words
    state = " ".join(state)
    if state in ("PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING"):
        return "running", jid   # between runs of a requeued job
    if state == "COMPLETED" and code == "0:0":
        return "done", jid
    return "failed", f"Slurm job {jid} ended {state or '?'} {code}"


def alive(pidfile):
    """Is the task's process (pid kept in a local file, since claims are re-read from the board) still there?"""
    try:
        os.kill(int(open(pidfile).read().strip()), 0)
        return True
    except (OSError, ValueError) as e:
        return isinstance(e, PermissionError)


def prefetch(task):
    """Download <baseline>/<arm>/<video>/ of every baseline from the results dataset into the jobs' model cache."""
    from huggingface_hub import snapshot_download
    mem = str(task.get("memory_arm") or "")   # an answer arm reads another arm's memory (master.MEMORY_ARM)
    arms = [task["arm"]] + ([mem] if VALID["arm"].match(mem) else [])
    cache = os.path.join(os.environ.get("MERIT_HF_CACHE") or os.path.expanduser("~/.cache/huggingface"), "hub")
    try:
        snapshot_download(os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results"), repo_type="dataset",
                          cache_dir=cache, allow_patterns=[f"*/{arm}/{task['video']}/*" for arm in arms])
    except Exception as e:   # the job then fetches (or builds) by itself
        print(f"{jobs.iso()} prefetch for {task['id']} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)


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
    ap.add_argument("--queues", help="slurm, instead of --sbatch: a JSON file of queues {name: {sbatch, fits}}; each task goes "
                                     "to every queue that fits its arm at once and runs on the first to start (coord/nexus_queues.json)")
    ap.add_argument("--slots", nargs="+", required=True, help="GPU ids per task slot: '0 1' = two one-GPU slots, '0,1' = one two-GPU slot")
    ap.add_argument("--baselines", nargs="+", default=["merit", "api"],   # api: only Max's pool gets those tasks
                    help="baselines this worker takes (it needs each one's image or SIF for every arm it may get): "
                         "e.g. merit magic worldmm")
    ap.add_argument("--every", type=int, default=300, help="seconds between cycles")
    ap.add_argument("--max-tasks", type=int, default=0, help="stop claiming after this many (0 = no limit)")
    ap.add_argument("--state", default=None, help="local folder for logs and exit codes (default ~/.cache/coord/<pool>)")
    ap.add_argument("--outbox", action="store_true", help="leave results in <checkout>/outbox/ for a relay to upload "
                    "(jobs that cannot reach the results dataset); the task is done once the relay marks it")
    ap.add_argument("--host", default=socket.gethostname(), help="this worker's name in its claims (default the hostname; "
                    "fix it when the worker itself runs as a Slurm job, so a requeued worker on another node keeps its claims)")
    ap.add_argument("--prefetch", action="store_true", help="download the task's video folders from the results dataset "
                    "into MERIT_HF_CACHE before starting it (job nodes that cannot download dataset files)")
    a = ap.parse_args()
    if a.queues and (a.launcher != "slurm" or a.sbatch):
        ap.error("--queues needs --launcher slurm and replaces --sbatch")
    queues = load_queues(a.queues) if a.queues else None
    state = os.path.abspath(a.state or os.path.expanduser(f"~/.cache/coord/{a.pool}"))   # the jobs write there too
    os.makedirs(state, exist_ok=True)
    host = a.host
    claimed_total = 0
    # Whose running claims this worker looks after. slurm: all of the pool's (a pool has one writer, the state folder
    # is on a shared filesystem, and the jobs outlive the worker), so a worker restarted on another host adopts them.
    # docker / local: only those started on this host, where their processes are.
    here = (lambda c: True) if a.launcher == "slurm" else (lambda c: c.get("host") == host)
    last_commit = 0
    while True:
        try:
            head, tasks, pools = jobs.read()
        except Exception as e:   # a 429 or a network failure must not end the worker: its tasks would be orphaned
            board_wait(e, "read")
            continue
        mine = pools.get(a.pool, {"pool": a.pool, "claims": {}})
        claims = mine.setdefault("claims", {})
        before = substance(mine) if a.pool in pools else None
        t = jobs.now()
        queued = my_jobs() if queues else None   # one squeue per cycle for every task's copies
        others = {tid: (n, c) for n, p in pools.items() if n != a.pool for tid, c in p.get("claims", {}).items()}
        # finished, still running, lost, or taken over by another pool?
        for tid, c in claims.items():
            if c["status"] == "relay" and os.path.exists(os.path.join(REPO_ROOT, "outbox", tid, ".relayed")):
                c.update(status="done", relayed=jobs.iso())
            if c["status"] != "running" or not here(c):
                continue
            ex = os.path.join(state, f"{tid}.exit")
            o = others.get(tid)
            if o and (o[1]["status"] in ("done", "relay") or (o[1]["status"] == "running" and o[1].get("claimed", "") > c.get("claimed", ""))):
                # our lease lapsed and another pool took the task: stop renewing (our copy may still finish; outputs are idempotent)
                c.update(status="superseded", finished=jobs.iso(), note=f"taken over by pool {o[0]}")
                continue
            p = os.path.join(state, tid)
            if os.path.exists(p + ".jobs"):   # submitted to several queues (--queues)
                if queued is None:   # squeue failed this cycle: assume nothing, keep the lease
                    c["lease_until"] = t + jobs.LEASE_S
                    continue
                code, q, cancel = copies_state(p, queued)
                if cancel:
                    subprocess.run(["scancel", *cancel])
                if q:
                    c["queue"] = q
                if code is None:
                    c.pop("gone_t", None)
                if code == -1 and "gone_t" not in c:   # left the queue without an exit code: look again next cycle
                    c.update(gone_t=t, lease_until=t + jobs.LEASE_S)   # (the .rc it wrote may not be visible here yet)
                    continue
                if code is not None:
                    open(ex, "w").write(str(code))   # recorded below, as for the other launchers
            code = how = None   # the exit code once the task is over, and for a failure without one, why
            if os.path.exists(ex):
                code = int(open(ex).read().strip() or 1)
            elif os.path.exists(p + ".jobs"):
                pass
            # the waiter (the process that writes the exit code): its pid means something only on the machine it runs on
            # (a claim from before `node` was recorded: slurm compares its host; docker / local trust it, as they did)
            elif not ((c.get("node") or (c.get("host") if a.launcher == "slurm" else socket.gethostname())) == socket.gethostname()
                      and alive(p + ".pid")):
                if a.launcher == "slurm":   # --sbatch: the `sbatch --wait` died with the old worker; Slurm knows the outcome
                    verdict, how = slurm_job_state(p + ".log.submit")
                    if verdict == "unknown" and c.get("unknown", 0) < 10:   # not in sacct yet: ask again next cycle
                        c["unknown"] = c.get("unknown", 0) + 1
                        verdict = "running"
                    code = {"done": 0, "running": None}.get(verdict, -1)
                    how = how if code == -1 else None
                else:
                    code, how = -1, "the task's process ended without an exit code (killed or host restarted)"
            if code is None:
                c["lease_until"] = t + jobs.LEASE_S
                continue
            if code == 0 and a.outbox and not os.path.exists(os.path.join(REPO_ROOT, "outbox", tid, ".upload.json")):
                code = 2   # nothing for the relay: the run.sh did not pass MERIT_OUTBOX on (or uploaded itself)
            c.update(status=("relay" if a.outbox else "done") if code == 0 else "failed", exit=code, finished=jobs.iso(),
                     seconds=int(t - c.get("started_t", t)))
            if code:   # what Slurm said, then the job's log, or (slurm, job never started) sbatch's own error
                note = ""
                for f in (p + ".log", p + ".log.submit"):
                    if os.path.exists(f) and os.path.getsize(f):
                        note = scrub(open(f).read()[-4000:])[-400:]   # scrub before trimming: no half key
                        break
                if how or note:
                    c["note"] = f"{how}: {note}" if how and note and a.launcher == "slurm" else how or note
        busy = {c["slot"] for c in claims.values() if c["status"] == "running" and here(c)}
        free = [s for s in a.slots if s not in busy]
        held = jobs.taken(pools, t)
        fails = jobs.failures(pools)
        done = {tid for tid, (_, c) in held.items() if c["status"] == "done"}
        def claimable(task):   # waiting on the board and this worker may take it (queues aside)
            tid = task["id"]
            return (valid(task) and tid not in held and fails.get(tid, 0) < 2 and claims.get(tid, {}).get("status") != "failed"
                    and (a.pool in task.get("pools", []) or "*" in task.get("pools", [])) and task["baseline"] in a.baselines
                    and has_env(task) and all(d in done for d in task.get("deps", [])))
        # A queue with a fallback (an H200: fits 27B, fallback 8B) takes fallback work only while it is idle: no task its
        # `fits` names is waiting, none of this worker's own is still active, and fewer than fallback_max (default 1)
        # of this worker's tasks have a copy on it.
        by_id = {x["id"]: x for x in tasks}
        active = [tid for tid, c in claims.items() if c["status"] == "running"]
        def on_queue(tid, n):   # a copy of the task on queue n still queued or running, or the one that ran it
            f, r = os.path.join(state, f"{tid}.jobs"), os.path.join(state, f"{tid}.ran")
            ran = open(r).read().strip() if os.path.exists(r) else None
            return os.path.exists(f) and any(l.split()[1:] == [n] and (l.split()[0] == ran or queued is None or l.split()[0] in queued)
                                             for l in open(f) if l.strip())
        fallback_used = {n: sum(on_queue(tid, n) for tid in active) for n, q in (queues or {}).items() if q.get("fallback")}
        idle = {n for n in fallback_used
                if not any(names(x, queues[n]["fits"]) and claimable(x) for x in tasks)
                and not any(tid in by_id and names(by_id[tid], queues[n]["fits"]) for tid in active)
                and fallback_used[n] < queues[n].get("fallback_max", 1)}
        # a queue's max_jobs (e.g. 2: QoS vulcan-high-h200 runs two jobs per user): once that many of this worker's
        # tasks have a live copy on it, it takes no more, so tasks only it fits are left for other pools
        used = {n: sum(on_queue(tid, n) for tid in active) for n, q in (queues or {}).items() if q.get("max_jobs")}
        full = {n for n in used if used[n] >= queues[n]["max_jobs"]}
        # tasks only the scarce queues fit (27B) first, so cheaper work cannot use up the free slots ahead of them
        scarce = [q["fits"] for q in (queues or {}).values() if q.get("fallback")]
        start = []
        for task in sorted(tasks, key=lambda x: (not any(names(x, f) for f in scarce), x.get("priority", 0))):
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
            if queues and not fitting(task, queues, idle, full):   # none of this worker's GPUs can run it now (e.g. 27B, no 80 GB queue)
                continue
            if task["baseline"] not in a.baselines:   # this worker has no image (or setup) for it
                continue
            if not has_env(task):   # a paid-API task and no key here: claiming it would only fail it
                continue
            if any(d not in done for d in task.get("deps", [])):
                continue
            slot = free.pop(0)
            claims[tid] = dict(status="running", host=host, node=socket.gethostname(), slot=slot,
                               gpu="(scheduler)" if a.launcher == "slurm" else gpu_name(slot), claimed=jobs.iso(),
                               started=jobs.iso(), started_t=t, lease_until=t + jobs.LEASE_S)
            qs = fitting(task, queues, idle, full) if queues else []
            for n in [n for n in qs if n in used]:
                used[n] += 1
                if used[n] >= queues[n]["max_jobs"]:
                    full.add(n)
            for n in [n for n in qs if n in idle and not names(task, queues[n]["fits"])]:   # a fallback copy used
                fallback_used[n] += 1
                if fallback_used[n] >= queues[n].get("fallback_max", 1):
                    idle.discard(n)
            for n in [n for n in list(idle) if names(task, queues[n]["fits"])]:   # its own work just got claimed
                idle.discard(n)
            start.append((task, slot, qs))
        mine.update(pool=a.pool, host=host, slots=a.slots, launcher=a.launcher, updated=jobs.iso(), queues=sorted(queues or []),
                    running=sum(1 for c in claims.values() if c["status"] == "running"))
        # nothing to report but renewed leases and the heartbeat: commit that every HEARTBEAT_S at most (leases last
        # LEASE_S, three times as long), since the Hub allows a repository 128 commits an hour, shared by every pool
        if start or substance(mine) != before or jobs.now() - last_commit >= HEARTBEAT_S:
            try:
                ok = jobs.commit({f"pools/{a.pool}.json": mine}, head, f"{a.pool}: {len(start)} claimed")
            except Exception as e:   # nothing was started: the next cycle decides again from the board
                board_wait(e, "commit")
                continue
            if not ok:
                print(f"{jobs.iso()} board moved under us; re-reading", flush=True)
                continue
            last_commit = jobs.now()
        for task, slot, qs in start:   # start only after the claim is committed
            tid = task["id"]
            if a.prefetch:   # before any copy can start
                prefetch(task)
            if queues:
                ids = submit_copies(task, a, {n: queues[n] for n in qs}, state)
                if not ids:   # no queue took it: fails next cycle with sbatch's error as its note
                    open(os.path.join(state, f"{tid}.exit"), "w").write("1")
                claimed_total += 1
                print(f"{jobs.iso()} submitted {tid} to {', '.join(f'{n} ({j})' for j, n in ids.items()) or 'no queue'}: "
                      f"{task['baseline']} {task['arm']} {task['stage']} {task['video']}", flush=True)
                continue
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
        running = sum(1 for c in claims.values() if c["status"] == "running" and here(c))
        if a.max_tasks and claimed_total >= a.max_tasks and running == 0:
            print(f"{jobs.iso()} done: {claimed_total} task(s) claimed and finished", flush=True)
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
