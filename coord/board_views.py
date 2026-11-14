"""The dashboard's datasets, computed from what collect_live.py gathered: the board (coord/board_read.py: open
tasks, every claim, heartbeats, what the results dataset holds), our own Slurm queues and the off-board loops.

    matrix.json     one row per experiment (baseline, model, stage, bank): progress in its natural unit (videos
                    with a memory; questions with a valid answer, of the bank's questions), and as detail the
                    tasks: done / open and what the open ones are doing, rate and ETA
    tasks.json      every open task with its state now and where
    running.json    what is running or waiting right now: board tasks (state from our squeue where we can see it,
                    "claimed" on other people's clusters) and our other jobs
    pools.json      one row per site (Nexus, wriva, Max, Zaratan, arm00): who, its workers' queues, heartbeat, counts
    people.json     the same rows, with a state for the overview cards
    attention.json  current problems only: failures grouped by (site, baseline, model, stage, cause), unreachable
                    machines, stale workers, open decisions
    progress.json, tranches.json, decisions.json   fixed context (tranche 1 results, banks, decisions)

Readers see sites only: a pool's queue is detail text ("Nexus · vulcan-sailon"), never a row or a column value,
so renamed pools and several workers on one cluster read as one machine. A task state is decided once, here, and every view counts from it. History kept next to the outputs:
ledger.json (when and where each memory first appeared) and board_raw.json (the last good board read).
"""
import json
import os
import re
import time

RATE_WINDOW_H = 6
NAMES = {"merit": "MERIT", "magic": "MAGIC-Video", "worldmm": "WorldMM", "m3agent": "M3-Agent", "api": "API"}
MODELS = {"m8": "8B", "m27": "27B"}
# pool -> (site, who runs it, hardware, queue). A reader sees only the site; the queue is detail text
# ("Nexus · vulcan-sailon"), so a renamed pool (nexus -> nexus-tron, #28) reads the same as before.
POOLS = {   # the old Nexus names (nexus, nexus-test, nexus-slurm) until their last claims end (renamed in #28)
    "nexus-tron": ("Nexus", "Anubhav's agent", "A6000-class GPUs on tron (QoS high)", "tron"),
    "nexus-scavenger": ("Nexus", "Anubhav's agent", "scavenger GPUs", "scavenger"),
    "nexus-sailon": ("Nexus", "Anubhav's agent", "vulcan-ampere (QoS vulcan-sailon)", "vulcan-sailon"),
    "nexus": ("Nexus", "Anubhav's agent", "A6000-class GPUs on tron (QoS high)", "tron"),
    "nexus-test": ("Nexus", "Anubhav's agent", "scavenger GPUs", "scavenger"),
    "nexus-slurm": ("Nexus", "Anubhav's agent", "vulcan-ampere (QoS vulcan-sailon)", "vulcan-sailon"),
    "vatsal-sailon": ("Nexus", "Vatsal's agent", "vulcan-ampere (his QoS vulcan-sailon)", "vulcan-sailon, Vatsal"),
    "vatsal-h200": ("Nexus", "Vatsal's agent", "H200 (his QoS vulcan-high-h200)", "H200, Vatsal"),
    "vatsal": ("Nexus", "Vatsal's agent", "his vulcan-sailon and H200 queues (one worker, #36)", "Vatsal"),
    "wriva": ("wriva", "Archana and Aaron", "4x H100 (cml-wriva, their accounts)", "cml-wriva"),
    "max": ("Max", "Max's agent", "H100 80 GB on his Slurm cluster", ""),
    "zaratan": ("Zaratan", "Anubhav's agent", "H100 / A100 (AAC)", ""),
    "arm00": ("arm00", "Anubhav's agent", "1x GH200 96 GB (vulcan-arm00)", ""),
}
SITES = ["Nexus", "wriva", "Max", "Zaratan", "arm00"]
OURS = {p for p, v in POOLS.items() if v[0] in ("Nexus", "wriva")}   # pools on Nexus: the collector's squeue shows their jobs


def site(pool):
    """The machine a reader sees for a pool: Nexus, wriva, Max, Zaratan, arm00 (an unlisted nexus-*/vatsal-* is Nexus)."""
    if pool in POOLS:
        return POOLS[pool][0]
    return "Nexus" if pool.startswith(("nexus", "vatsal")) else pool


def place(pool):
    """Site plus queue as detail text: 'Nexus · vulcan-sailon', or the site alone."""
    q = POOLS.get(pool, ("", "", "", ""))[3]
    return f"{site(pool)} · {q}" if q else site(pool)


VID = re.compile(r"P\d\d-\d{8}-\d{6}")

TRANCHES = [
    dict(tranche="t1", bank="batch10_agree2", videos=10, questions=598, state="frozen",
         source="epic-morph tag bank/batch10_agree2 (0f30904)"),
    dict(tranche="t2", bank="t2_agree2", videos=20, questions=None, state="answers on the board",
         source="epic-morph v3/tranche2.txt; questions from Max's Stage A + B"),
    dict(tranche="later", bank="—", videos=126, questions=None, state="building questions",
         source="Max's Stage A on the remaining videos; memories are built ahead of their questions"),
]
# tranche 1, run before the board existed (2026-11-12); fixed, not read live
PROGRESS = [
    dict(experiment="MERIT 8B · answers (with frames)", tranche="t1", done=598, total=598, unit="questions", note=""),
    dict(experiment="MERIT 27B · answers (with frames)", tranche="t1", done=598, total=598, unit="questions", note=""),
    dict(experiment="MAGIC-Video 8B · build + answers", tranche="t1", done=10, total=10, unit="videos", note="Vatsal's port"),
    dict(experiment="M2 probes · 8B no context", tranche="t1", done=1513, total=1513, unit="probes", note=""),
    dict(experiment="M2 probes · 27B no context", tranche="t1", done=1513, total=1513, unit="probes", note=""),
    dict(experiment="M2 probes · MERIT 8B memory only", tranche="t1", done=1513, total=1513, unit="probes", note=""),
]
DECISIONS = [
    dict(id="D1", question="Bill Zaratan's AAC account for MERIT 27B memory builds?", status="decided",
         why="CMSC has 3.4 kSU left.", options="Decided 2026-11-12: yes, AAC approved."),
    dict(id="D2", question="Who runs wriva, and how many H100s?", status="decided",
         why="Sizing their queue.", options="Decided: Archana and Aaron, 2-4 H100s."),
    dict(id="D3", question="Make the MERIT container images public?", status="decided",
         why="Only the public PyTorch base image and public pip packages.",
         options="Done 2026-11-13: ghcr.io/morph-graphs/merit is public."),
    dict(id="D4", question="Create the private dataset morph-graphs/epic-morph-jobs?", status="decided",
         why="Job claims apart from results.", options="Done: created and reachable with our token."),
    dict(id="D5", question="Which GPUs may run each model size?", status="decided",
         why="Captions differ slightly across GPU models.",
         options="Decided: anywhere the model fits; every output records its GPU; one cross-GPU check per arm."),
    dict(id="D6", question="Qwen-MM-Plugins?", status="decided",
         why="API-only defaults, no paper, ~$150 through OpenRouter.", options="Decided 2026-11-12: deferred."),
    dict(id="D7", question="M3-Agent: speech and text embeddings through the published APIs or local models?",
         status="decided", why="The paper used Gemini (speech) and OpenAI embeddings.",
         options="Decided 2026-11-13: the APIs as published, run by Max; local models only as a fallback."),
]


def _load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _t(s):
    """ISO 'YYYY-MM-DDTHH:MM(:SS)Z' -> epoch seconds (None if empty)."""
    if not s:
        return None
    s = s.rstrip("Z")
    for f in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return time.mktime(time.strptime(s, f)) - time.timezone
        except ValueError:
            pass
    return None


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(t))


def experiment(b, arm, stage, bank):
    """The label a person reads: 'MERIT 27B · memory build', 'API gpt-5-f1024 · answers batch10_agree2'."""
    what = "memory build" if stage == "build" else f"answers {bank}"
    return f"{NAMES.get(b, b)} {MODELS.get(arm, arm)} · {what}"


def parse_task(tid):
    """baseline, arm, stage, bank, video from a task id (for claims whose task left the list)."""
    v = VID.search(tid)
    video = v.group(0) if v else ""
    head = tid[:v.start() - 1] if v else tid
    for stage in ("-build", "-qa-"):
        if stage in head:
            pre, post = head.split(stage, 1)
            b, arm = pre.split("-", 1)
            return b, arm, stage.strip("-"), (post or None), video
    return head, "", "", None, video


def cause(c):
    """The one line of a failed claim's note that says why, with videos, ids and numbers masked so equal causes group."""
    if c.get("exit") == -1:
        return "cancelled or lost: the job ended without an exit code"
    lines = [l.strip() for l in (c.get("note") or "").splitlines() if l.strip()]
    keep = [l for l in lines if not l.startswith(("srun:", "slurmstepd", "Traceback", "File ")) and "UUID: GPU-" not in l]
    keep = keep or lines or ["no log"]
    line = keep[-1]
    if len(keep) > 1 and "error" in keep[-2].lower():   # the error line before a script's own summary says why
        line = keep[-2] + " → " + line
    if line.startswith("run: git submodule"):            # baselines' run.sh, when a submodule is not checked out
        line = "submodules not checked out on the worker (" + line + ")"
    line = VID.sub("<video>", line)
    line = re.sub(r"\b[0-9a-f]{8,}\b|\b\d{4,}\b", "#", line)
    return line[:220]


def update(out, raw, live, nxq, built_src):
    """raw: board_read.py's JSON (or None); live: off-board rows from collect_live; nxq: {slurm job name: job};
    built_src: {(baseline, arm, video): where seen first} for memories outside the results dataset."""
    t = time.time()
    snap = _iso(t)
    L = lambda n, d: _load(os.path.join(out, n), d)
    ledger = L("ledger.json", {})
    first_run = not ledger
    raw = raw or {"tasks": [], "claims": [], "pools": {}, "built": [], "answered": []}

    # memories: when and where each first appeared (the board's builds are named by their pool below)
    done_by = {}
    for c in raw["claims"]:
        if c["status"] == "done":
            done_by.setdefault(c["task"], c["pool"])
    # a memory a board worker reported done is that pool's, even when an off-board scan also saw it (arm00 runs both)
    src = {k: done_by.get(f"{k[0]}-{k[1]}-build-{k[2]}", w) for k, w in built_src.items()}
    for b, arm, v in raw["built"]:
        src.setdefault((b, arm, v), done_by.get(f"{b}-{arm}-build-{v}", "results dataset"))
    for (b, arm, v), where in sorted(src.items()):
        k = f"{b}|{arm}|{v}"
        if k not in ledger:
            ledger[k] = dict(t=None if first_run else t, pool=where)
        elif ledger[k]["pool"] == "results dataset" and where != "results dataset":
            ledger[k]["pool"] = where

    tasks = raw["tasks"]
    # a memory seen by an off-board scan (arm00 loop, Zaratan arrays) that no board claim reported done
    off_board = lambda k, e: e["pool"] in ("arm00", "zaratan") and "{}-{}-build-{}".format(*k.split("|")) not in done_claims
    built = {tuple(k.split("|")) for k in ledger} | {tuple(x) for x in raw["built"]}
    done_claims = {c["task"] for c in raw["claims"] if c["status"] == "done"}
    by_task = {}
    for c in raw["claims"]:
        by_task.setdefault(c["task"], []).append(c)
    for cs in by_task.values():
        cs.sort(key=lambda c: c.get("claimed") or c.get("started") or "")

    def live_state(c):
        """running / queued / claimed for a live claim; where; since."""
        if c["status"] == "running" and (c.get("lease_until") or 0) <= t:
            return "lease lapsed", f"{place(c['pool'])}: worker stopped renewing its lease", c.get("claimed", "")
        if c.get("gpu") and c["gpu"] != "(scheduler)":   # a local launcher: the claim names its GPU
            gpu = f"GPU {c['gpu']}" if str(c["gpu"]).isdigit() else c["gpu"]
            return "running", f"{place(c['pool'])} · {gpu}", c.get("started", "")
        if c["pool"] in OURS:
            j = nxq.get(f"board-{c['task'][:60]}")
            if j:
                if j["state"] == "RUNNING":
                    return "running", f"{place(c['pool'])} · {j['where']}, {j['elapsed']} elapsed", c.get("started", "")
                why = f" ({j['where']})" if j["where"] not in ("", "None") else ""
                return "queued", f"{place(c['pool'])} · waiting{why}", c.get("claimed", "")
            return "claimed", f"{place(c['pool'])} · no Slurm job seen yet", c.get("claimed", "")
        # someone else's Slurm: the claim cannot say whether the job has started
        return "claimed", f"{place(c['pool'])} · start not visible", c.get("claimed", "")

    # one state per open task
    answered = {tuple(a) for a in raw["answered"]}
    finished = {x["id"] for x in tasks if x["id"] in done_claims or (x["baseline"], x["arm"], x["video"]) in built and x["stage"] == "build"
                or (x["baseline"], x["arm"], x["video"], x.get("bank")) in answered}
    tasks = [x for x in tasks if x["id"] not in finished]   # done, but the master has not dropped them yet
    task_rows, problems = [], {}
    for x in tasks:
        cs = by_task.get(x["id"], [])
        last = cs[-1] if cs else None
        fails = sum(1 for c in cs if c["status"] == "failed")
        state, on, where, since, note = "ready", "", "", "", ""
        if last and last["status"] == "running":
            state, where, since = live_state(last)
            on = site(last["pool"])
        elif last and last["status"] == "failed":
            cancelled = last.get("exit") == -1
            state = "held" if fails >= 2 else ("cancelled" if cancelled else "failed")
            why = next((c for c in reversed(cs) if c["status"] == "failed" and c.get("exit") != -1), last)
            on, where, since, note = site(why["pool"]), place(why["pool"]), last.get("finished", ""), cause(why)
            if not cancelled or fails >= 2:
                key = (site(why["pool"]), x["baseline"], x["arm"], x["stage"], x.get("bank"), note)
                problems.setdefault(key, []).append(x["id"])
        else:
            pending = [d for d in x.get("deps", []) if d not in done_claims
                       and (parse_task(d)[0], parse_task(d)[1], parse_task(d)[4]) not in built]
            if pending:
                state = "waiting"
                note = "waits for the " + ", ".join(experiment(*parse_task(d)[:4]).split(" · ")[0] + " memory" for d in pending)
        task_rows.append(dict(id=x["id"], experiment=experiment(x["baseline"], x["arm"], x["stage"], x.get("bank")),
                              method=NAMES.get(x["baseline"], x["baseline"]), model=MODELS.get(x["arm"], x["arm"]),
                              stage="memory build" if x["stage"] == "build" else "answers", bank=x.get("bank") or "",
                              video=x["video"], state=state, on=on, where=where, since=since,
                              can_run_on=", ".join(sorted({site(p) for p in x.get("pools", [])}, key=lambda s_: (SITES + [s_]).index(s_))), failures=fails, note=note,
                              priority=x.get("priority", 0)))

    # rates: board finishes in the last RATE_WINDOW_H hours, plus memories that appeared off the board
    since_t = t - RATE_WINDOW_H * 3600
    rate = {}
    for c in raw["claims"]:
        if c["status"] == "done" and (_t(c.get("finished")) or 0) >= since_t:
            b, arm, stage, bank, _ = parse_task(c["task"])
            k = experiment(b, arm, stage, bank)
            rate[k] = rate.get(k, 0) + 1 / RATE_WINDOW_H
    for k, e in ledger.items():
        if e["t"] and e["t"] >= since_t and off_board(k, e):   # off-board loops (ledger names them)
            b, arm, _ = k.split("|")
            ek = experiment(b, arm, "build", None)
            rate[ek] = rate.get(ek, 0) + 1 / RATE_WINDOW_H

    # matrix: one row per experiment, in the board's priority order; finished ones (no open tasks) too
    exps = {}
    for x, r in zip(tasks, task_rows):
        e = exps.setdefault(r["experiment"], dict(experiment=r["experiment"], method=r["method"], model=r["model"],
                                                  stage=r["stage"], bank=r["bank"], key=(x["baseline"], x["arm"], x["stage"], x.get("bank")),
                                                  open=0, running=0, queued=0, claimed=0, failing=0, waiting=0, ready=0,
                                                  pools=r["can_run_on"], first=r["priority"]))
        e["open"] += 1
        s = r["state"]
        e[{"running": "running", "queued": "queued", "claimed": "claimed", "lease lapsed": "claimed", "failed": "failing",
           "held": "failing", "cancelled": "ready", "waiting": "waiting"}.get(s, "ready")] += 1
    for b, arm, v in raw["built"]:
        k = experiment(b, arm, "build", None)
        exps.setdefault(k, dict(experiment=k, method=NAMES.get(b, b), model=MODELS.get(arm, arm), stage="memory build", bank="",
                                key=(b, arm, "build", None), open=0, running=0, queued=0, claimed=0, failing=0, waiting=0,
                                ready=0, pools="", first=10 ** 6))
    for b, arm, v, bank in raw["answered"]:
        k = experiment(b, arm, "qa", bank)
        exps.setdefault(k, dict(experiment=k, method=NAMES.get(b, b), model=MODELS.get(arm, arm), stage="answers", bank=bank,
                                key=(b, arm, "qa", bank), open=0, running=0, queued=0, claimed=0, failing=0, waiting=0,
                                ready=0, pools="", first=10 ** 6))
    matrix = []
    for e in sorted(exps.values(), key=lambda e: e["first"]):
        b, arm, stage, bank = e.pop("key")
        e.pop("first")
        if stage == "build":
            e["done"] = sum(1 for k in built if k[0] == b and k[1] == arm)
        else:
            e["done"] = sum(1 for a in raw["answered"] if (a[0], a[1], a[3]) == (b, arm, bank))
        e["total"] = e["done"] + e["open"]
        # the headline in natural units: videos with a memory, questions with a valid answer (task counts stay as detail)
        bq = (raw.get("bank_questions") or {}).get(bank or "", {})
        if stage == "build" or not bq:
            e["unit"], e["n"], e["of"] = "videos", e["done"], e["total"]
        else:
            e["unit"], e["of"] = "questions", sum(bq.values())
            e["n"] = sum(a[4] for a in raw.get("answered_q", []) if (a[0], a[1], a[3]) == (b, arm, bank))
        e["pct"] = round(100 * e["n"] / e["of"], 1) if e["of"] else None
        e["system"] = f"{NAMES.get(b, b)} {MODELS.get(arm, arm)}"
        rph = rate.get(e["experiment"])
        e["rate_per_h"] = round(rph, 1) if rph else None
        e["eta_h"] = round(e["open"] / rph, 1) if rph and e["open"] else None
        e["state"] = ("done" if not e["open"] else "failing" if e["failing"] else
                      "running" if e["running"] else "queued" if e["queued"] or e["claimed"] else
                      "waiting on builds" if e["waiting"] == e["open"] else "partly done" if e["done"] else "not started")
        matrix.append(e)

    # running now: board claims that hold a task, then our other jobs; one group per site
    running = []
    for c in raw["claims"]:
        if c["status"] != "running":
            continue
        state, where, since = live_state(c)
        b, arm, stage, bank, v = parse_task(c["task"])
        running.append(dict(site=site(c["pool"]), kind="board task", what=f"{experiment(b, arm, stage, bank)} · {v}",
                            state=state, where=where, since=since))
    for r in live:
        if r["state"] == "alert":
            continue                      # alerts go to attention
        running.append(dict(site=site(r["pool"]), kind=r.get("kind", "other job"), what=r["what"], state=r["state"],
                            where=r.get("where", ""), since=r.get("since", "")))
    order = {"running": 0, "queued": 1, "claimed": 2, "lease lapsed": 3}
    running.sort(key=lambda r: (SITES.index(r["site"]) if r["site"] in SITES else 9, order.get(r["state"], 5), r["what"]))

    # machines: one row per site, its workers (queues) as detail text
    hb = raw["pools"]
    done24, prob = {}, {}
    for c in raw["claims"]:
        if c["status"] == "done" and (_t(c.get("finished")) or 0) >= t - 86400:
            done24[site(c["pool"])] = done24.get(site(c["pool"]), 0) + 1
    for k, e in ledger.items():
        if e["t"] and e["t"] >= t - 86400 and off_board(k, e):
            done24[site(e["pool"])] = done24.get(site(e["pool"]), 0) + 1
    for key, ids in problems.items():
        prob[key[0]] = prob.get(key[0], 0) + len(ids)
    age = {n: (t - _t(h["updated"])) / 60 if h.get("updated") else None for n, h in hb.items()}
    up = {n for n, a in age.items() if a is not None and a <= 30}
    # a silent worker whose queue has a live one (an old pool name after its rename, #28) is not stale
    stale = {n for n, a in age.items() if a is not None and a > 30 and (hb[n].get("running") or 0) > 0
             and place(n) not in {place(u) for u in up}}
    pools = []
    for s_ in SITES + sorted({site(n) for n in hb} - set(SITES)):
        names = [n for n in list(POOLS) + sorted(set(hb) - set(POOLS)) if site(n) == s_]
        mine = [r for r in running if r["site"] == s_]
        board = [r for r in mine if r["kind"] in ("board task", "loop")]   # the arm00 loop builds board work too
        hbs = [hb[n]["updated"] for n in names if n in hb and hb[n].get("updated")]
        def qs(ns, word):   # "3 up (tron, scavenger, vulcan-sailon)": an old and a renamed pool on one queue count once
            q = list(dict.fromkeys(POOLS.get(n, ("", "", "", n))[3] or n for n in ns))
            labels = [x for x in q if x not in POOLS and x not in hb]   # never a bare pool name
            return f"{len(q)} {word}" + (f" ({', '.join(labels)})" if labels else "") if ns else ""
        workers = "; ".join(x for x in (qs([n for n in names if n in up], "up"), qs([n for n in names if n in stale], "stale")) if x) \
            or "no board worker running"
        known = [n for n in names if n in hb] or names   # describe the workers that exist, else what is planned
        pools.append(dict(site=s_, person=", ".join(dict.fromkeys(POOLS.get(n, ("", "?"))[1] for n in known)),
                          hardware="; ".join(dict.fromkeys(POOLS.get(n, ("", "", "?"))[2] for n in known)),
                          workers=workers,
                          state=("worker stale" if any(n in stale for n in names) else "worker up" if any(n in up for n in names)
                                 else "no worker"),
                          heartbeat=max(hbs) if hbs else "",
                          running=sum(r["state"] == "running" for r in board), queued=sum(r["state"] == "queued" for r in board),
                          claimed=sum(r["state"] in ("claimed", "lease lapsed") for r in board),
                          other_jobs=len(mine) - len(board), done_24h=done24.get(s_, 0), problems=prob.get(s_, 0), snapshot=snap))
    people = [dict(p, state="working" if p["running"] + p["queued"] + p["claimed"] else
                   "idle, worker up" if p["state"] == "worker up" else "nothing running") for p in pools]

    # attention: current problems only
    att = []
    for r in live:
        if r["state"] == "alert":
            att.append(dict(kind="unreachable", severity="bad", site=site(r["pool"]), what=r["what"], count=None, cause=r.get("note", ""), fix=""))
    for (s_, b, arm, stage, bank, why), ids in sorted(problems.items(), key=lambda kv: -len(kv[1])):
        held = sum(1 for i in ids if sum(1 for c in by_task[i] if c["status"] == "failed") >= 2)
        att.append(dict(kind="failing", severity="bad", site=s_, what=f"{experiment(b, arm, stage, bank)} on {s_}",
                        count=len(ids), cause=why,
                        fix=(f"{held} held after two failures: fix the cause, then coord/master.py reset --task <id>. " if held else "")
                        + "The worker that failed them will not retry them; other machines may."))
    for name in sorted(stale):
        att.append(dict(kind="stale worker", severity="warn", site=site(name), what=f"{place(name)} worker", count=None,
                        cause=f"no heartbeat for {int(age[name])} min", fix="its tasks return to the list when their leases lapse"))
    for d in DECISIONS:
        if d["status"] != "decided":
            att.append(dict(kind="decision", severity="warn", site="Anubhav", what=f"{d['id']}. {d['question']}", count=None,
                            cause=d["why"], fix=d["options"]))

    W = lambda n, o: json.dump(o, open(os.path.join(out, n), "w"), indent=0)
    W("matrix.json", matrix)
    W("tasks.json", task_rows)
    W("running.json", running)
    W("pools.json", pools)
    W("people.json", people)
    W("attention.json", att)
    W("progress.json", PROGRESS)
    W("tranches.json", TRANCHES)
    W("decisions.json", DECISIONS)
    W("ledger.json", ledger)
    return dict(tasks=len(task_rows), running=len(running), attention=len(att))
