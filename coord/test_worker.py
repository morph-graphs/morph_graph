"""worker.py --queues: which queues a task goes to, the guard that lets one copy run, and reading the outcome back.

Slurm is faked by small scripts on PATH (sbatch, squeue, scancel, nvidia-smi) that log their calls under $FAKE.
pytest is not installed on Nexus: `python coord/test_worker.py` runs every test here.
"""
import os
import json
import subprocess
import sys
import tempfile
import types
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker as W  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = os.path.join(HERE, "slurm_guard.sh")
FAKES = {
    # every call logged; a queue whose args name partition "broken" is refused; ids count up from 101, printed "id;cluster"
    "sbatch": 'echo "$@" >> $FAKE/sbatch.log\ncase "$*" in *--partition=broken*) echo "sbatch: error: invalid partition" >&2; exit 1;; esac\n'
              'n=$(( $(cat $FAKE/next 2>/dev/null || echo 100) + 1 )); echo $n > $FAKE/next; echo "$n;nexus"',
    "squeue": 'echo "$@" >> $FAKE/squeue.log\ncat $FAKE/queue 2>/dev/null; true',   # $FAKE/queue: the ids squeue shows
    "scancel": 'echo "$@" >> $FAKE/scancel.log',
    "nvidia-smi": "echo 'GPU 0: fake'",
}


def fake_slurm(tmp):
    os.makedirs(f"{tmp}/bin", exist_ok=True)
    for name, body in FAKES.items():
        open(f"{tmp}/bin/{name}", "w").write("#!/bin/bash\n" + body + "\n")
        os.chmod(f"{tmp}/bin/{name}", 0o755)
    return dict(os.environ, FAKE=tmp, PATH=f"{tmp}/bin:{os.environ['PATH']}", USER="tester")


def lines(path):
    return open(path).read().split() if os.path.exists(path) else []


QUEUES = {"a6000": {"fits": ["m8"], "sbatch": "--partition=a --gres=gpu:rtxa6000:1"},
          "h200": {"fits": ["m27", "m8"], "sbatch": "--partition=h --gres=gpu:h200-sxm:1"},
          "pair48": {"fits": ["merit/m27"], "sbatch": "--partition=p --gres=gpu:rtxa6000:2"},
          "broken": {"fits": ["m8"], "sbatch": "--partition=broken"}}


def task(b="merit", arm="m8"):
    return dict(id=f"{b}-{arm}-build-P01-20240101-000001", baseline=b, arm=arm, stage="build", video="P01-20240101-000001")


def test_fitting():
    """m8 goes to every 48 GB+ queue; m27 only where 80 GB (or MERIT's 2x48 GB) fits; a person with no 80 GB queue gets no 27B."""
    assert W.fitting(task(arm="m8"), QUEUES) == ["a6000", "h200", "broken"]
    assert W.fitting(task(arm="m27"), QUEUES) == ["h200", "pair48"]
    assert W.fitting(task("magic", "m27"), QUEUES) == ["h200"]         # MAGIC's 27B does not fit 2x48 GB
    assert W.fitting(task(arm="m27"), {"a6000": QUEUES["a6000"]}) == []
    shipped = W.load_queues(os.path.join(HERE, "nexus_queues.json"))   # the example file for Nexus users
    assert W.fitting(task(arm="m27"), shipped) == ["h200"]
    assert len(W.fitting(task(arm="m8"), shipped)) == len(shipped) - 1 == 5   # the H200 only when idle (fallback)
    assert W.fitting(dict(task(), baseline="api", arm="gpt-5-f1024"), shipped) == []


def guard(env, jid, marker):
    return subprocess.run(["bash", GUARD, marker], env=dict(env, SLURM_JOB_ID=str(jid), SLURM_JOB_NAME="board-x",
                                                             SLURM_JOB_PARTITION="p"), capture_output=True, text=True)


def test_guard(tmp):
    """First copy to start takes the task and cancels the pending copies; a later one exits 3; a requeued winner goes on."""
    env = fake_slurm(tmp)
    marker = f"{tmp}/t.ran"
    open(f"{tmp}/queue", "w").write("101\n102\n103\n")   # squeue -t PENDING as the winner sees it (itself included)
    r = guard(env, 102, marker)
    assert r.returncode == 0 and lines(marker) == ["102"], r
    assert lines(f"{tmp}/scancel.log") == ["101", "103"]
    assert "-n board-x -t PENDING" in open(f"{tmp}/squeue.log").read()
    os.remove(f"{tmp}/scancel.log")
    r = guard(env, 101, marker)   # e.g. submitted after 102 started, or scancel lost: no work, no cancelling
    assert r.returncode == 3 and "copy 102" in r.stdout and not os.path.exists(f"{tmp}/scancel.log")
    open(f"{tmp}/t.rc", "w").write("143")   # from 102's first, preempted start
    assert guard(env, 102, marker).returncode == 0   # 102 preempted and requeued: still its task
    assert not os.path.exists(f"{tmp}/t.rc")
    assert lines(marker) == ["102"]


def test_guard_race(tmp):
    """Copies starting in the same instant: exactly one runs (the marker is created with noclobber)."""
    env = fake_slurm(tmp)
    for k in range(5):
        marker = f"{tmp}/race{k}.ran"
        with ThreadPoolExecutor(16) as ex:
            codes = list(ex.map(lambda j: guard(env, j, marker).returncode, range(200, 216)))
        assert sorted(codes) == [0] + [3] * 15, codes
        assert lines(marker) == [str(200 + codes.index(0))]


def test_copies(tmp):
    """submit_copies: one job per fitting queue, one name; copies_state reads the winner's queue and exit code back."""
    env = fake_slurm(tmp)
    os.environ.update(FAKE=env["FAKE"], PATH=env["PATH"])
    state = f"{tmp}/state"
    os.makedirs(state)
    a = types.SimpleNamespace(launcher="slurm", runtime="apptainer")
    t = task()
    p = f"{state}/{t['id']}"
    open(p + ".rc", "w").write("0")   # left from an earlier run of this task: must not end this one
    open(p + ".jobs", "w").write("90 a6000\n91 h200\n")   # copies of an earlier run: cancelled first
    ids = W.submit_copies(t, a, QUEUES, state)
    assert ids == {"101": "a6000", "102": "h200"}, ids
    assert lines(f"{tmp}/scancel.log") == ["90", "91"]           # "broken" refused; pair48 does not fit m8
    calls = open(f"{tmp}/sbatch.log").read().splitlines()
    assert len(calls) == 3 and all(f"--job-name=board-{t['id']}" in c and "slurm_guard.sh" in c for c in calls)
    assert "invalid partition" in open(p + ".log.submit").read() and not os.path.exists(p + ".rc")
    assert open(p + ".jobs").read() == "101 a6000\n102 h200\n"
    q = {"101", "102"}
    assert W.copies_state(p, q) == (None, None, [])              # both pending
    open(p + ".ran", "w").write("102\n")
    assert W.copies_state(p, q) == (None, "h200", ["101"])       # 102 runs; 101 still queued: cancel it
    assert W.copies_state(p, {"102"}) == (None, "h200", [])
    open(p + ".rc", "w").write("0\n")
    assert W.copies_state(p, {"102"}) == (None, "h200", [])      # written, but 102 is still (or again, requeued) queued
    assert W.copies_state(p, set()) == (0, "h200", [])           # done
    os.remove(p + ".rc")
    assert W.copies_state(p, {"101"}) == (-1, "h200", ["101"])   # winner gone without an exit code (time limit, OOM)
    os.remove(p + ".ran")
    assert W.copies_state(p, set()) == (-1, None, [])            # every copy gone, none ran
    assert W.my_jobs() == set() and W.my_jobs("/nonexistent/squeue") is None


def test_copies_end_to_end(tmp):
    """Run the copies' real job scripts in the order the scheduler might start them: the work happens once, with the
    exit code of that run, and the late copy does nothing."""
    env = fake_slurm(tmp)
    os.environ.update(FAKE=env["FAKE"], PATH=env["PATH"])
    state = f"{tmp}/state"
    os.makedirs(state)
    orig = W.inner
    W.inner = lambda task, gpus, a: f"echo work >> {tmp}/work.log; exit 7"
    try:
        t = task()
        ids = W.submit_copies(t, types.SimpleNamespace(launcher="slurm", runtime="apptainer"), QUEUES, state)
    finally:
        W.inner = orig
    wraps = {}
    for c in open(f"{tmp}/sbatch.log").read().splitlines():   # each call ends "--wrap <script>"
        if "--partition=broken" not in c:
            wraps[c.split("--partition=")[1].split()[0]] = c.split(" --wrap ", 1)[1]
    by_part = {"a": "101", "h": "102"}
    order = ["h", "a"]   # the H200 copy starts first
    for part in order:
        r = subprocess.run(["bash", "-c", wraps[part]], env=dict(env, SLURM_JOB_ID=by_part[part], SLURM_JOB_NAME="board-x",
                                                                SLURM_JOB_PARTITION=part), capture_output=True, text=True)
        assert r.returncode == 0, r   # the job itself ends 0 either way; the task's code is in .rc
    p = f"{state}/{t['id']}"
    assert lines(f"{tmp}/work.log") == ["work"]
    assert W.copies_state(p, set()) == (7, "h200", [])
    assert ids == {"101": "a6000", "102": "h200"}


def test_worker_loop(tmp):
    """worker.main with --queues against a fake board: claims the 8B task (not the 27B one: no queue fits it), submits a
    copy per queue, waits a cycle when the copies are gone without a visible exit code, then records done and the queue."""
    env = fake_slurm(tmp)
    os.environ.update(FAKE=env["FAKE"], PATH=env["PATH"])
    qf = f"{tmp}/queues.json"
    open(qf, "w").write(json.dumps({k: QUEUES[k] for k in ("a6000", "pair48")}))
    m8, m27 = dict(task(), pools=["p"], priority=1), dict(task("magic", "m27"), pools=["p"], priority=0)
    pools, commits, cycle = {}, [], [0]

    def commit(files, head, msg):
        commits.append(json.loads(json.dumps(files["pools/p.json"])))
        pools["p"] = commits[-1]
        return True

    def sleep(_):   # between cycles: the scheduler's side of things
        cycle[0] += 1
        p = f"{tmp}/state/{m8['id']}"
        if cycle[0] == 1:
            open(p + ".ran", "w").write("101")        # the copy started and is running
            open(f"{tmp}/queue", "w").write("101\n")
        elif cycle[0] == 2:
            open(f"{tmp}/queue", "w").write("")       # it ended, but its .rc is not visible yet
        elif cycle[0] == 3:
            open(p + ".rc", "w").write("0")
        else:
            raise AssertionError("worker did not finish")

    saved = (W.jobs.read, W.jobs.commit, W.time.sleep, sys.argv)
    W.jobs.read, W.jobs.commit, W.time.sleep = (lambda: ("h", [m27, m8], json.loads(json.dumps(pools)))), commit, sleep
    sys.argv = ["worker.py", "--pool", "p", "--launcher", "slurm", "--queues", qf, "--slots", "1", "2",
                "--max-tasks", "1", "--state", f"{tmp}/state"]
    try:
        W.main()
    finally:
        W.jobs.read, W.jobs.commit, W.time.sleep, sys.argv = saved
    claims = [c["claims"] for c in commits]
    assert list(claims[0]) == [m8["id"]] and claims[0][m8["id"]]["status"] == "running"   # 27B never claimed
    assert open(f"{tmp}/state/{m8['id']}.jobs").read() == "101 a6000\n"                   # pair48 fits MERIT 27B only
    assert claims[1][m8["id"]]["queue"] == "a6000" and claims[2][m8["id"]]["status"] == "running"   # grace cycle
    assert claims[3][m8["id"]]["status"] == "done" and claims[3][m8["id"]]["exit"] == 0
    assert commits[-1]["queues"] == ["a6000", "pair48"]


def test_fallback_queue_takes_8b_only_when_idle():
    """An H200 queue (fits m27, fallback m8) gets an 8B task's copy only while no 27B task is waiting."""
    q = {"a6000": {"sbatch": "x", "fits": ["m8"]}, "h200": {"sbatch": "y", "fits": ["m27"], "fallback": ["m8"]}}
    t8 = {"baseline": "merit", "arm": "m8"}
    t27 = {"baseline": "merit", "arm": "m27"}
    assert W.fitting(t8, q) == ["a6000"]                      # a 27B task is waiting: the H200 is not idle
    assert W.fitting(t8, q, idle={"h200"}) == ["a6000", "h200"]
    assert W.fitting(t27, q) == ["h200"] and W.fitting(t27, q, idle={"h200"}) == ["h200"]
    shipped = W.load_queues(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexus_queues.json"))
    assert "h200" not in W.fitting(t8, shipped) and "h200" in W.fitting(t8, shipped, idle={"h200"})


def run_one_cycle(tmp, queues, tasks, slots):
    """worker.main for one claim cycle against a fake board and fake slurm; returns {task id: [queues it got copies on]}."""
    env = fake_slurm(tmp)
    os.environ.update(FAKE=env["FAKE"], PATH=env["PATH"])
    qf = f"{tmp}/queues.json"
    open(qf, "w").write(json.dumps(queues))

    class Stop(Exception):
        pass

    def sleep(_):
        raise Stop
    saved = (W.jobs.read, W.jobs.commit, W.time.sleep, sys.argv)
    W.jobs.read, W.jobs.commit, W.time.sleep = (lambda: ("h", tasks, {})), (lambda *_: True), sleep
    sys.argv = ["worker.py", "--pool", "p", "--launcher", "slurm", "--queues", qf, "--slots", *slots, "--state", f"{tmp}/state"]
    try:
        W.main()
    except Stop:
        pass
    finally:
        W.jobs.read, W.jobs.commit, W.time.sleep, sys.argv = saved
    out = {}
    for t in tasks:
        f = f"{tmp}/state/{t['id']}.jobs"
        out[t["id"]] = [l.split()[1] for l in open(f) if l.strip()] if os.path.exists(f) else []
    return out


def test_fallback_in_the_loop(tmp):
    """27B tasks are claimed first even when 8B ones rank higher; with 27B work claimed the H200 gets no 8B copy; with
    none, at most fallback_max (1) 8B task per cycle gets an H200 copy."""
    q = {"a6000": {"sbatch": "--partition=a", "fits": ["m8"]},
         "h200": {"sbatch": "--partition=h", "fits": ["m27"], "fallback": ["m8"]}}
    def t(arm, n, prio):
        return dict(id=f"merit-{arm}-build-P01-20240101-00000{n}", baseline="merit", arm=arm, stage="build",
                    video=f"P01-20240101-00000{n}", pools=["p"], priority=prio)
    got = run_one_cycle(f"{tmp}/a", q, [t("m8", 1, 0), t("m8", 2, 1), t("m27", 3, 2)], ["1", "2"])
    assert got[t("m27", 3, 2)["id"]] == ["h200"]                      # claimed although it ranks last
    assert got[t("m8", 1, 0)["id"]] == ["a6000"] and got[t("m8", 2, 1)["id"]] == []   # no 8B copy beside the 27B
    got = run_one_cycle(f"{tmp}/b", q, [t("m8", 1, 0), t("m8", 2, 1)], ["1", "2"])
    assert got[t("m8", 1, 0)["id"]] == ["a6000", "h200"]               # the H200 idle: one fallback copy
    assert got[t("m8", 2, 1)["id"]] == ["a6000"]                       # fallback_max reached


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        with tempfile.TemporaryDirectory() as tmp:
            f(*([tmp] if f.__code__.co_argcount else []))
        print("ok", n)
    print(f"{len(tests)} passed")
