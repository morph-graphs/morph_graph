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


def test_has_video():
    """A stitched video's task only where run.sh would find the file ($HDEPIC_VIDEOS, else under $EPIC_MORPH_DIR);
    an HD-EPIC recording is never checked."""
    st = dict(task(), video="P05-stitched")
    with tempfile.TemporaryDirectory() as d:
        env = {k: os.environ.pop(k) for k in ("HDEPIC_VIDEOS", "EPIC_MORPH_DIR") if k in os.environ}
        try:
            assert W.has_video(task()) and not W.has_video(st)              # nothing set: a stitched one is left
            os.environ["EPIC_MORPH_DIR"] = d
            os.makedirs(f"{d}/data/HD-EPIC/videos/P05")
            assert not W.has_video(st)
            open(f"{d}/data/HD-EPIC/videos/P05/P05-stitched.mp4", "w").close()
            assert W.has_video(st)
            os.environ["HDEPIC_VIDEOS"] = f"{d}/elsewhere"                # HDEPIC_VIDEOS wins, as in run.sh
            assert not W.has_video(st) and W.has_video(task())
        finally:
            os.environ.pop("HDEPIC_VIDEOS", None), os.environ.pop("EPIC_MORPH_DIR", None)
            os.environ.update(env)


def test_fitting():
    """m8 goes to every 48 GB+ queue; m27 only where 80 GB (or MERIT's 2x48 GB) fits; a person with no 80 GB queue gets no 27B.
    A queue may also name one stage of an arm (m3agent/m3/qa)."""
    assert W.fitting(task(arm="m8"), QUEUES) == ["a6000", "h200", "broken"]
    assert W.fitting(task(arm="m27"), QUEUES) == ["h200", "pair48"]
    assert W.fitting(task("magic", "m27"), QUEUES) == ["h200"]         # MAGIC's 27B does not fit 2x48 GB
    assert W.fitting(task(arm="m27"), {"a6000": QUEUES["a6000"]}) == []
    shipped = W.load_queues(os.path.join(HERE, "nexus_queues.json"))   # the example file for Nexus users
    assert W.fitting(task(arm="m27"), shipped) == ["h200"]
    assert len(W.fitting(task(arm="m8"), shipped)) == len(shipped) - 5 == 5   # not the CPU queue, VST's two or the long one; the H200 only when idle (fallback)
    # ReKV's m8 is named by the plain m8, as every baseline's: it fits those queues' GPU and RAM
    assert W.fitting(task("rekv", "m8"), shipped) == W.fitting(task(arm="m8"), shipped)
    # VST needs 64G of RAM, which the m8 queues (40G) do not have: named as vst/vst7b, on its own 64G queues only
    vst = W.fitting(task("vst", "vst7b"), shipped, idle=shipped)
    assert vst == ["scav-a5000-64g", "tron-a5000-64g"] and all("--mem=64G" in shipped[n]["sbatch"] for n in vst)
    assert W.valid(dict(task("vst", "vst7b"), stage="qa", bank="batch10_agree2", id="vst-vst7b-qa-batch10_agree2-P01-20240101-000001"))
    # a kitchen's stitched video is a video too; nothing else of that shape. Its build goes to the long queue only,
    # and that queue takes nothing else
    st = dict(task(), video="P03-stitched", id="merit-m8-build-P03-stitched")
    assert W.valid(st)
    assert not any(W.valid(dict(task(), video=v)) for v in ("P03-stitched-x", "P03-stitched;x", "stitched", "P3-stitched"))
    assert W.fitting(st, shipped, idle=shipped) == ["sailon-a6000-long"] and "--time=3-00:00:00" in shipped["sailon-a6000-long"]["sbatch"]
    assert "sailon-a6000-long" not in W.fitting(task(arm="m8"), shipped, idle=shipped) and W.fitting(st, QUEUES) == []
    # a stage of an arm (baseline/arm/stage): M3-Agent's build fits one 48 GB GPU, its qa (a 32.8B model) the H200 only
    m3 = lambda stage: dict(task("m3agent", "m3"), stage=stage)
    assert W.fitting(m3("qa"), shipped) == ["h200"] and W.fitting(m3("build"), shipped, idle=shipped) == W.fitting(task(arm="m8"), shipped)
    assert W.fitting(m3("build"), {"q": {"fits": ["m3agent/m3/qa"]}}) == []   # another stage of the arm is not named
    # the keyed arms: MERIT's gpt-5-mini memory needs no GPU; the others one GPU, on the queues nobody preempts
    assert W.fitting(task("merit", "g5m"), shipped) == ["cpu"]
    for b, arm in (("worldmm", "g5m"), ("worldmm", "g5m-q8"), ("magic", "oss120"), ("magic", "oss120-flash")):
        assert W.fitting(task(b, arm), shipped) == ["sailon-a6000", "tron-a6000"], (b, arm)
    assert W.fitting(task("merit", "g5m-g5"), shipped) == W.fitting(task("worldmm", "g5m-g5"), shipped) == []   # Max's
    assert W.fitting(task("magic", "oss120-q27"), shipped) == ["h200"]
    assert not any("--export" in q["sbatch"] for q in shipped.values())   # the jobs get the worker's environment (the key)
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


def test_has_env():
    """needs_env: every name set in the worker's environment; "A|B" is met by either; an unset or empty one is not."""
    keep = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY")}
    try:
        either, t = "OPENAI_API_KEY|OPENROUTER_API_KEY", lambda *need: {"needs_env": list(need)}
        assert W.has_env({}) and not W.has_env(t(either)) and not W.has_env(t("OPENROUTER_API_KEY"))
        os.environ["OPENROUTER_API_KEY"] = "sk-or-test-not-a-key"
        assert W.has_env(t(either)) and W.has_env(t("OPENROUTER_API_KEY"))
        assert not W.has_env(t("OPENAI_API_KEY")) and not W.has_env(t(either, "GEMINI_API_KEY"))   # every entry must hold
        os.environ["OPENROUTER_API_KEY"] = ""
        assert not W.has_env(t(either))
        os.environ["OPENAI_API_KEY"] = "sk-test-not-a-key"
        assert W.has_env(t(either)) and W.has_env(t("OPENAI_API_KEY")) and not W.has_env(t("OPENROUTER_API_KEY"))
        assert not W.has_env({"needs_env": [["OPENAI_API_KEY"]]})   # not a name: left alone
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_key_stays_in_the_environment():
    """A keyed task's job: the key is in no sbatch argument and no file of the state folder; the job would get it from
    the worker's environment (sbatch exports it), and a failure note is scrubbed of it."""
    key = "sk-or-v1-test0not0a0key0123456789"
    with tempfile.TemporaryDirectory() as tmp:
        env = fake_slurm(tmp)
        old = {k: os.environ.get(k) for k in ("FAKE", "PATH", "OPENROUTER_API_KEY")}
        os.environ.update(FAKE=env["FAKE"], PATH=env["PATH"], OPENROUTER_API_KEY=key)
        try:
            state = f"{tmp}/state"
            os.makedirs(state)
            t = dict(task("worldmm", "g5m"), needs_env=["OPENAI_API_KEY|OPENROUTER_API_KEY"])
            q = {"gpu": {"fits": ["worldmm/g5m"], "sbatch": "--partition=a --gres=gpu:rtxa6000:1"},
                 "cpu": {"fits": ["merit/g5m"], "sbatch": "--partition=c -c 4"}}
            a = types.SimpleNamespace(launcher="slurm", runtime="apptainer")
            assert W.has_env(t) and W.fitting(t, q) == ["gpu"] and W.fitting(task("merit", "g5m"), q) == ["cpu"]
            assert W.submit_copies(t, a, q, state) == {"101": "gpu"}
            call = open(f"{tmp}/sbatch.log").read()
            assert "--export" not in call and "OPENROUTER" not in call   # sbatch's default: the worker's environment
            assert "nvidia-smi -L || true" in call                       # a CPU queue's node has no nvidia-smi
            for d, _, fs in os.walk(tmp):
                assert not any(key in open(os.path.join(d, f), errors="replace").read() for f in fs)
            assert key not in W.scrub(f"401 Incorrect API key provided: {key}")
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


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
    # a stitched video's build ahead of both, whose file this machine lacks: left, and named in the pool file
    st = dict(task(), video="P05-stitched", id="merit-m8-build-P05-stitched", pools=["p"], priority=-1)
    os.environ["HDEPIC_VIDEOS"] = f"{tmp}/videos"
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
    W.jobs.read, W.jobs.commit, W.time.sleep = (lambda: ("h", [st, m27, m8], json.loads(json.dumps(pools)))), commit, sleep
    sys.argv = ["worker.py", "--pool", "p", "--launcher", "slurm", "--queues", qf, "--slots", "1", "2",
                "--max-tasks", "1", "--state", f"{tmp}/state"]
    try:
        W.main()
    finally:
        W.jobs.read, W.jobs.commit, W.time.sleep, sys.argv = saved
        os.environ.pop("HDEPIC_VIDEOS")
    claims = [c["claims"] for c in commits]
    assert st["id"] not in claims[-1] and all(c["no_video"] == ["P05-stitched"] for c in commits)
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


def test_max_jobs_caps_a_queue(tmp):
    """With max_jobs 2 on the only queue that fits 27B, a worker with 5 free slots claims only two 27B tasks and
    gives the rest of its slots to 8B work on the other queues."""
    q = {"a6000": {"sbatch": "--partition=a", "fits": ["m8"]},
         "h200": {"sbatch": "--partition=h", "fits": ["m27"], "fallback": ["m8"], "max_jobs": 2}}
    def t(arm, n, prio):
        return dict(id=f"merit-{arm}-build-P01-20240101-00000{n}", baseline="merit", arm=arm, stage="build",
                    video=f"P01-20240101-00000{n}", pools=["p"], priority=prio)
    tasks = [t("m27", 1, 0), t("m27", 2, 1), t("m27", 3, 2), t("m27", 4, 3), t("m8", 5, 4), t("m8", 6, 5)]
    got = run_one_cycle(tmp, q, tasks, ["1", "2", "3", "4", "5"])
    on_h200 = [k for k, v in got.items() if "h200" in v]
    assert len(on_h200) == 2 and all("m27" in k for k in on_h200)             # the cap
    assert got[t("m27", 3, 2)["id"]] == [] and got[t("m27", 4, 3)["id"]] == []  # left for other pools
    assert got[t("m8", 5, 4)["id"]] == ["a6000"] and got[t("m8", 6, 5)["id"]] == ["a6000"]


def test_job_start(tmp):
    """--auto-slots reads from the cycle's one squeue whether a task's job still waits, or when it started."""
    env = fake_slurm(tmp)
    os.environ.update(FAKE=env["FAKE"], PATH=env["PATH"])
    open(f"{tmp}/queue", "w").write("101|PENDING|N/A\n102|PENDING|2026-11-15T12:00:00\n103|RUNNING|2026-11-15T08:44:16\n")
    view = W.queue_view()
    assert view == {"101": ("PENDING", "N/A"), "102": ("PENDING", "2026-11-15T12:00:00"), "103": ("RUNNING", "2026-11-15T08:44:16")}
    assert W.my_jobs() == {"101", "102", "103"}
    p = f"{tmp}/t"
    assert W.job_start(p, view) is None                             # nothing submitted
    open(p + ".log.submit", "w").write("102;zaratan\n")            # --sbatch: the one job (a pending job's %S is an estimate)
    assert W.job_ids(p) == ["102"] and W.job_start(p, view) == "pending"
    open(p + ".log.submit", "w").write("103\n")
    assert W.time.strftime("%H:%M:%S", W.time.localtime(W.job_start(p, view))) == "08:44:16"
    assert W.job_start(p, {}) == 0                                  # came and went between two cycles
    open(p + ".log.submit", "w").write("sbatch: error: invalid partition\n")
    assert W.job_start(p, view) is None
    open(p + ".jobs", "w").write("101 a\n102 b\n")                  # --queues: waiting while every copy in the queue is
    assert W.job_start(p, view) == "pending" and W.job_start(p, {"101": ("PENDING", "N/A")}) == "pending"
    assert W.job_start(p, {}) is None                               # every copy gone and none ran: not a start
    running = W.job_start(p, dict(view, **{"101": ("RUNNING", "2026-11-15T08:44:16")}))   # 101 started, its marker not yet seen:
    assert W.time.strftime("%H:%M", W.time.localtime(running)) == "08:44"   # its start, not pending 102's estimate
    open(p + ".ran", "w").write("102\n")
    assert W.job_start(p, view) == os.path.getmtime(p + ".ran")     # also once the winner is requeued and pending again
    # the measurements never cause a commit by themselves
    pool = {"claims": {"x": {"status": "running", "lease_until": 1}}}
    assert W.substance(pool) == W.substance({"auto": {"pending": 3}, "claims": {"x": {"status": "running", "lease_until": 2, "wait": 28}}})


def test_auto_slots_rule():
    """`waiting` tasks may wait while the last job to start was fast; one once it was slow or a task has waited over
    2 F, and then the others that have are released; groups are sized apart."""
    F, t = 300, 10000
    c = lambda claimed, wait=None: dict(status="running", started_t=claimed, **({} if wait is None else {"wait": wait}))
    g = lambda tid: tid[0]
    assert W.auto_slots({}, set(), g, t, F, 4) == ({}, [])                              # nothing known: the loop allows `waiting`
    fast = {"a1": c(t - 5000, 900), "a2": dict(c(t - 500, 28), status="done")}          # the last to start was fast
    assert W.auto_slots(fast, set(), g, t, F, 4) == ({"a": 4}, [])
    assert W.auto_slots(dict(fast, a3=c(t - 100), a4=c(t - 400)), {"a3", "a4"}, g, t, F, 4) == ({"a": 2}, [])   # under 2 F: kept
    # the last job to start had waited longer than F: one task may wait (the probe), whatever started before it
    slow = dict(fast, a5=c(t - 450, 400))
    assert W.auto_slots(slow, set(), g, t, F, 4) == ({"a": 1}, [])
    assert W.auto_slots(dict(slow, a6=c(t - 60)), {"a6"}, g, t, F, 4) == ({"a": 0}, [])
    assert W.auto_slots(dict(slow, a6=c(t - 60, 30)), set(), g, t, F, 4) == ({"a": 4}, [])   # it started fast: room again
    # tasks waiting longer than 2 F: all of them but the oldest are released, the young one may still start
    stuck = dict(fast, a6=c(t - 900), a7=c(t - 700), a8=c(t - 650), a9=c(t - 100), b1=c(t - 50, 10))
    room, release = W.auto_slots(stuck, {"a6", "a7", "a8", "a9"}, g, t, F, 4)
    assert release == ["a7", "a8"] and room == {"a": -1, "b": 4}
    # groups are the queues a task's kind fits
    q = {"a6000": {"sbatch": "x", "fits": ["m8"]}, "h200": {"sbatch": "y", "fits": ["m27", "m3agent/m3/qa"], "fallback": ["m8"]}}
    assert W.queue_group("merit-m8-build-P01-20240101-000001", q) == ("a6000",)
    assert W.queue_group("worldmm-g5m-q27-qa-t2_agree2-P01-20240101-000001", {"h": {"fits": ["worldmm/g5m-q27"]}}) == ("h",)
    assert W.queue_group("m3agent-m3-qa-b-P01-20240101-000001", q) == ("h200",) and W.queue_group("odd", q) == ()
    assert W.queue_group("merit-m8-build-P01-20240101-000001", None) == ()


# Zaratan, 2026-11-15 (sacct -X -o Submit,Start,End of the 25 board-* jobs the worker, --slots 25, submitted at 08:43:48-50;
# read at 16:10, when 11 of its tasks were pending with reason QOSMaxGRESPerUser, two of these 25 among them): QoS
# gpu-h100 allows a user 16 H100s and two other jobs of ours held 2, so 14 started at once and 9 more only as 9 ended.
ZARATAN = """08:43:48 08:44:16 08:51:12|08:43:48 08:44:16 09:07:47|08:43:48 08:44:16 09:07:49|08:43:48 08:44:16 -
08:43:49 08:44:16 08:51:12|08:43:49 08:44:16 08:53:51|08:43:49 08:44:16 08:56:08|08:43:49 08:44:16 08:58:30
08:43:49 08:44:16 09:00:09|08:43:49 08:44:16 09:09:08|08:43:49 08:44:16 -|08:43:49 08:44:16 -|08:43:49 08:44:16 -
08:43:49 08:44:16 -|08:43:49 08:51:22 -|08:43:49 08:51:22 -|08:43:50 08:53:54 -|08:43:50 08:56:26 -|08:43:50 08:58:58 -
08:43:50 09:00:30 -|08:43:50 09:08:06 -|08:43:50 09:08:06 -|08:43:50 09:09:33 -|08:43:50 - -|08:43:50 - -"""


def replay(policy, hours=7, every=300, ceiling=25):
    """The worker's cycles against a queue rebuilt from the trace: as many GPUs as started at once (14), a free one
    taken 28 s after a job asks (the trace's delay), the n-th job to start runs as long as the trace's n-th did (the
    ones still running then: for good), and there is always another task to claim. policy: ("slots",) today's fixed
    --slots; ("waiting", k) a fixed budget of k waiting tasks; ("auto", F minutes, k) worker.auto_slots.
    Returns the limit and one (minute, running, waiting, released) per cycle."""
    sec = lambda s: sum(int(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))
    rows = [r.split() for r in ZARATAN.replace("\n", "|").split("|")]
    first = min(sec(r[1]) for r in rows if r[1] != "-")
    cap, delay = sum(sec(r[1]) == first for r in rows if r[1] != "-"), first - sec(rows[0][0])
    lasts = [sec(r[2]) - sec(r[1]) if r[2] != "-" else None for r in rows]
    claims, queue, running, started, out, n = {}, [], {}, 0, [], 0   # queue: waiting tasks, oldest first; running: {tid: end}
    for now in range(0, hours * 3600 + 1):
        for tid in [x for x, end in running.items() if end is not None and end <= now]:
            del running[tid]
            claims[tid]["status"] = "done"
        while queue and len(running) < cap and claims[queue[0]]["started_t"] + delay <= now:
            tid = queue.pop(0)
            running[tid] = now + lasts[started] if started < len(lasts) and lasts[started] else None
            claims[tid]["wait"] = now - claims[tid]["started_t"]
            started += 1
        if now % every:
            continue
        released = 0
        want = ceiling - len(running) - len(queue)
        if policy[0] == "waiting":
            want = min(want, policy[1] - len(queue))
        if policy[0] == "auto":
            room, release = W.auto_slots(claims, set(queue), lambda tid: (), now, policy[1] * 60, policy[2])
            for tid in release:
                queue.remove(tid)
                del claims[tid]
            released, want = len(release), min(want, room.get((), policy[2]))
        for _ in range(max(0, want)):
            n += 1
            claims[f"t{n:04d}"] = dict(status="running", started_t=now)
            queue.append(f"t{n:04d}")
        out.append((now // 60, len(running), len(queue), released))
    return cap, out


def test_auto_slots_zaratan_replay():
    """The real Zaratan morning through the rules, 7 hours as on the day. Fixed 25 slots is what happened: 14 running
    and 11 tasks claimed but waiting at the limit all day. --auto-slots 5 has the same 14 GPUs busy within 30 minutes,
    gives back what waits over 10 minutes and holds one task at the limit. A fixed budget of waiting tasks without the
    release cannot do both: 2 take an hour to fill the GPUs (the jobs that end meanwhile use up the claims), 5 fill
    them and leave 5 waiting."""
    show = {}
    for name, policy in (("--slots 25", ("slots",)), ("2 waiting", ("waiting", 2)), ("5 waiting", ("waiting", 5)),
                         ("--auto-slots 5", ("auto", 5, 4))):
        cap, out = replay(policy)
        show[name] = dict(full=next(m for m, run, _, _ in out if run == cap),             # first cycle that sees every GPU busy
                          idle=sum(cap - run for _, run, _, _ in out) * 5,                # GPU-minutes, at 5 min a cycle
                          stranded=sum(w for _, run, w, _ in out if run == cap) * 5 / 60,   # task-hours waiting at the limit
                          end=out[-1][1:3], released=sum(o[3] for o in out), out=out)
        r = show[name]
        print(f"   {name:15s} {cap} GPUs busy from minute {r['full']:2d}; idle {r['idle']:3d} GPU-min; at the limit, claimed and "
              f"waiting: {r['stranded']:4.1f} task-hours in 7 h; ends {out[-1][1]} running + {out[-1][2]} waiting; released {r['released']}")
        print("                   running+waiting per 5-min cycle, first hour: " + " ".join(f"{run}+{w}" for _, run, w, _ in out[:13]))
    fixed, two, five, auto = (show[k] for k in ("--slots 25", "2 waiting", "5 waiting", "--auto-slots 5"))
    assert cap == 14 and fixed["end"] == (14, 11)                    # as squeue showed at 16:10: 14 running, 11 pending
    assert auto["end"] == (14, 1) and two["end"] == (14, 2) and five["end"] == (14, 5)
    assert [run + w for _, run, w, _ in auto["out"][:4]] == [4, 8, 11, 14] and auto["full"] <= 30 < two["full"]
    assert max(w for m, _, w, _ in auto["out"] if m >= 60) == 1      # one probe at the limit, for good: no flapping
    assert auto["released"] <= 4 and auto["stranded"] < fixed["stranded"] / 8 and auto["stranded"] < five["stranded"] / 4


def test_auto_slots_in_the_loop(tmp):
    """worker.main --auto-slots with --queues on a fake board and clock: --waiting 2 of five tasks are claimed while
    five slots are free; one starts and one more is claimed; when two have waited over 2 F the newer one is released
    (its claim dropped, then its pending job cancelled) and nothing more is claimed while the older one waits."""
    env = fake_slurm(tmp)
    os.environ.update(FAKE=env["FAKE"], PATH=env["PATH"])
    qf = f"{tmp}/queues.json"
    open(qf, "w").write(json.dumps({"a6000": QUEUES["a6000"]}))
    def t(n):
        return dict(id=f"merit-m8-build-P01-20240101-00000{n}", baseline="merit", arm="m8", stage="build",
                    video=f"P01-20240101-00000{n}", pools=["p"], priority=n)
    tasks, pools, commits, cycle, clock, refused = [t(n) for n in range(1, 6)], {}, [], [0], [1000000.0], []

    def commit(files, head, msg):
        if len(commits) == 2 and not refused:   # the commit that drops the claim is refused once (someone else committed):
            refused.append(lines(f"{tmp}/scancel.log"))   # nothing may have been cancelled for it
            return False
        commits.append(json.loads(json.dumps(files["pools/p.json"])))
        pools["p"] = commits[-1]
        return True

    class Stop(Exception):
        pass

    def sleep(_):
        cycle[0] += 1
        clock[0] += 300
        if cycle[0] == 1:   # the first job started; the second waits
            open(f"{tmp}/state/{tasks[0]['id']}.ran", "w").write("101")
            os.utime(f"{tmp}/state/{tasks[0]['id']}.ran", (clock[0] - 270, clock[0] - 270))
            open(f"{tmp}/queue", "w").write("101|RUNNING|x\n102|PENDING|N/A\n")
        elif cycle[0] == 2:   # the third waits too
            open(f"{tmp}/queue", "w").write("101|RUNNING|x\n102|PENDING|N/A\n103|PENDING|N/A\n")
        elif cycle[0] == 5:   # 103 was cancelled
            open(f"{tmp}/queue", "w").write("101|RUNNING|x\n102|PENDING|N/A\n")
        elif cycle[0] == 6:
            raise Stop
    saved = (W.jobs.read, W.jobs.commit, W.jobs.now, W.time.sleep, sys.argv)
    W.jobs.read, W.jobs.commit, W.time.sleep = (lambda: ("h", tasks, json.loads(json.dumps(pools)))), commit, sleep
    W.jobs.now = lambda: clock[0]
    sys.argv = ["worker.py", "--pool", "p", "--launcher", "slurm", "--queues", qf, "--slots", "1", "2", "3", "4", "5",
                "--auto-slots", "5", "--waiting", "2", "--state", f"{tmp}/state"]
    try:
        W.main()
    except Stop:
        pass
    finally:
        W.jobs.read, W.jobs.commit, W.jobs.now, W.time.sleep, sys.argv = saved
    ids = [x["id"] for x in tasks]
    assert list(commits[0]["claims"]) == ids[:2]                                   # two may wait, though five slots are free
    assert list(commits[1]["claims"]) == ids[:3] and commits[1]["claims"][ids[0]]["wait"] == 30   # one started (fast): one more
    assert list(commits[2]["claims"]) == ids[:2]                                   # minute 20: 103 (15 min > 2 F) goes, the older 102 stays
    assert refused == [[]] and lines(f"{tmp}/scancel.log") == ["--state=PENDING", "103"]   # cancelled once, after the commit
    assert len(commits) == 3 and commits[2]["auto"]["pending"] == 1                # then nothing claimed, nothing to commit


def test_prefetch_lists_what_the_dataset_has(tmp):
    """prefetch writes, per arm, the dataset's commit and its files of the video: a job without the Hub reads it
    (hf_results.py remote_memory) to take them at that commit and to tell an absent memory from one missing in its
    cache. A failed prefetch leaves no listing, also not an earlier one; a video with nothing in the dataset gets an
    empty one."""
    import huggingface_hub
    v, snap = "P01-20240101-000001", os.path.join(tmp, "snap")
    for f in (f"worldmm/g5m/{v}/{v}_visual.pkl", f"worldmm/g5m/{v}/runs.jsonl", f"merit/g5m-q8/{v}/runs.jsonl",
              "magic/g5m/P01-20240101-000002/graph.pkl"):
        os.makedirs(os.path.dirname(os.path.join(snap, f)), exist_ok=True)
        open(os.path.join(snap, f), "w").close()
    seen = lambda arm: os.path.join(tmp, "hub", "prefetched", f"{v}__{arm}.json")
    saved = (huggingface_hub.snapshot_download, os.environ.get("MERIT_HF_CACHE"))
    os.environ["MERIT_HF_CACHE"] = tmp
    try:
        huggingface_hub.snapshot_download = lambda *a, **k: snap
        W.prefetch(dict(id="t", video=v, arm="g5m-q8", memory_arm="g5m"))
        assert json.load(open(seen("g5m"))) == dict(commit="snap", files=[f"worldmm/g5m/{v}/{v}_visual.pkl", f"worldmm/g5m/{v}/runs.jsonl"])
        assert json.load(open(seen("g5m-q8"))) == dict(commit="snap", files=[f"merit/g5m-q8/{v}/runs.jsonl"])
        huggingface_hub.snapshot_download = lambda *a, **k: (_ for _ in ()).throw(OSError("429"))
        W.prefetch(dict(id="t", video=v, arm="g5m-q8", memory_arm="g5m"))
        assert not os.path.exists(seen("g5m")) and not os.path.exists(seen("g5m-q8"))
        huggingface_hub.snapshot_download = lambda *a, **k: os.path.join(tmp, "no-such-snapshot")
        W.prefetch(dict(id="t", video=v, arm="m8"))
        assert json.load(open(seen("m8"))) == dict(commit="no-such-snapshot", files=[])
    finally:
        huggingface_hub.snapshot_download = saved[0]
        os.environ.pop("MERIT_HF_CACHE") if saved[1] is None else os.environ.update(MERIT_HF_CACHE=saved[1])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        with tempfile.TemporaryDirectory() as tmp:
            f(*([tmp] if f.__code__.co_argcount else []))
        print("ok", n)
    print(f"{len(tests)} passed")
