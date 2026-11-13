"""The dashboard's datasets, computed from what collect_live.py gathered: the board (coord/board_read.py: open
tasks, every claim, heartbeats, what the results dataset holds), our own Slurm queues and the off-board loops.

    matrix.json     one row per experiment (baseline, model, stage, bank): done / open, and what the open tasks
                    are doing (running, queued, claimed, failing, waiting on a build, ready), rate and ETA
    tasks.json      every open task with its state now and where
    running.json    what is running or waiting right now: board tasks (state from our squeue where we can see it,
                    "claimed" on other people's clusters) and our other jobs
    pools.json      one row per pool: who, worker heartbeat, counts, problems
    people.json     one row per site (Nexus, wriva, Max, Zaratan, arm00): the same, summed
    attention.json  current problems only: failures grouped by (pool, baseline, model, stage, cause), unreachable
                    machines, stale workers, open decisions
    events.json     activity, grouped: claims, finishes, failures (marked resolved or cancelled), memories built
    progress.json, tranches.json, decisions.json   fixed context (tranche 1 results, banks, decisions)

A task state is decided once, here, and every view counts from it. History kept next to the outputs:
ledger.json (when and where each memory first appeared) and board_raw.json (the last good board read).
"""
import json
import os
import re
import time

RATE_WINDOW_H = 6
NAMES = {"merit": "MERIT", "magic": "MAGIC-Video", "worldmm": "WorldMM", "m3agent": "M3-Agent", "api": "API"}
MODELS = {"m8": "8B", "m27": "27B"}
# pool -> (site, who runs it, hardware). Sites are what the People view shows.
POOLS = {   # the old Nexus names (nexus, nexus-test, nexus-slurm) until their last claims end (renamed in #28)
    "nexus-tron": ("Nexus", "Anubhav's agent", "Nexus Slurm, tron (QoS high)"),
    "nexus-scavenger": ("Nexus", "Anubhav's agent", "Nexus Slurm, scavenger"),
    "nexus-sailon": ("Nexus", "Anubhav's agent", "Nexus Slurm, vulcan-ampere (QoS vulcan-sailon)"),
    "nexus": ("Nexus", "Anubhav's agent", "Nexus Slurm, tron (QoS high); old name of nexus-tron"),
    "nexus-test": ("Nexus", "Anubhav's agent", "Nexus Slurm, scavenger; old name of nexus-scavenger"),
    "nexus-slurm": ("Nexus", "Anubhav's agent", "Nexus Slurm, vulcan-sailon; old name of nexus-sailon"),
    "vatsal-sailon": ("Nexus", "Vatsal's agent", "Nexus Slurm, vulcan-ampere (his QoS vulcan-sailon)"),
    "vatsal-h200": ("Nexus", "Vatsal's agent", "Nexus Slurm, H200 (his QoS vulcan-high-h200)"),
    "wriva": ("wriva", "Archana and Aaron", "4x H100 (cml-wriva, their accounts)"),
    "max": ("Max", "Max's agent", "H100 80 GB on his Slurm cluster"),
    "zaratan": ("Zaratan", "Anubhav's agent", "H100 / A100 (AAC)"),
    "arm00": ("arm00", "Anubhav's agent", "1x GH200 96 GB, tmux loop"),
}
SITES = ["Nexus", "wriva", "Max", "Zaratan", "arm00"]
OURS = {p for p, v in POOLS.items() if v[0] in ("Nexus", "wriva")}   # pools on Nexus: the collector's squeue shows their jobs
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
    src = dict(built_src)
    for b, arm, v in raw["built"]:
        src.setdefault((b, arm, v), done_by.get(f"{b}-{arm}-build-{v}", "results dataset"))
    for (b, arm, v), where in sorted(src.items()):
        k = f"{b}|{arm}|{v}"
        if k not in ledger:
            ledger[k] = dict(t=None if first_run else t, pool=where)
        elif ledger[k]["pool"] == "results dataset" and where != "results dataset":
            ledger[k]["pool"] = where

    tasks = raw["tasks"]
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
            return "lease lapsed", c.get("pool_host", ""), c.get("claimed", "")
        if c.get("gpu") and c["gpu"] != "(scheduler)":
            return "running", f"{c.get('host', '')} {c['gpu']}".strip(), c.get("started", "")
        if c["pool"] in OURS:
            j = nxq.get(f"board-{c['task'][:60]}")
            if j:
                if j["state"] == "RUNNING":
                    return "running", f"{j['partition']} on {j['where']}, {j['elapsed']} elapsed", c.get("started", "")
                return "queued", f"{j['partition']}, waiting {j['where']}", c.get("claimed", "")
            return "claimed", "no Slurm job seen yet", c.get("claimed", "")
        # someone else's Slurm: the claim cannot say whether the job has started
        return "claimed", f"{c['pool']} Slurm (start not visible)", c.get("claimed", "")

    # one state per open task
    answered = {tuple(a) for a in raw["answered"]}
    finished = {x["id"] for x in tasks if x["id"] in done_claims or (x["baseline"], x["arm"], x["video"]) in built and x["stage"] == "build"
                or (x["baseline"], x["arm"], x["video"], x.get("bank")) in answered}
    tasks = [x for x in tasks if x["id"] not in finished]   # done, but the master has not dropped them yet
    open_ids = {x["id"] for x in tasks}
    task_rows, problems = [], {}
    for x in tasks:
        cs = by_task.get(x["id"], [])
        last = cs[-1] if cs else None
        fails = sum(1 for c in cs if c["status"] == "failed")
        state, on, where, since, note = "ready", "", "", "", ""
        if last and last["status"] == "running":
            state, where, since = live_state(last)
            on = last["pool"]
        elif last and last["status"] == "failed":
            cancelled = last.get("exit") == -1
            state = "held" if fails >= 2 else ("cancelled" if cancelled else "failed")
            why = next((c for c in reversed(cs) if c["status"] == "failed" and c.get("exit") != -1), last)
            on, since, note = why["pool"], last.get("finished", ""), cause(why)
            if not cancelled or fails >= 2:
                key = (why["pool"], x["baseline"], x["arm"], x["stage"], x.get("bank"), note)
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
                              can_run_on=", ".join(x.get("pools", [])), failures=fails, note=note,
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
        if e["t"] and e["t"] >= since_t and e["pool"] in ("arm00", "zaratan"):
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
        rph = rate.get(e["experiment"])
        e["rate_per_h"] = round(rph, 1) if rph else None
        e["eta_h"] = round(e["open"] / rph, 1) if rph and e["open"] else None
        e["state"] = ("done" if not e["open"] else "failing" if e["failing"] else
                      "running" if e["running"] else "queued" if e["queued"] or e["claimed"] else
                      "waiting on builds" if e["waiting"] == e["open"] else "partly done" if e["done"] else "not started")
        matrix.append(e)

    # running now: board claims that hold a task, then our other jobs
    running = []
    for c in raw["claims"]:
        if c["status"] != "running":
            continue
        state, where, since = live_state(c)
        b, arm, stage, bank, v = parse_task(c["task"])
        running.append(dict(site=POOLS.get(c["pool"], (c["pool"],))[0], pool=c["pool"], kind="board task",
                            what=f"{experiment(b, arm, stage, bank)} · {v}", state=state, where=where, since=since))
    for r in live:
        if r["state"] == "alert":
            continue                      # alerts go to attention
        running.append(dict(site=POOLS.get(r["pool"], (r["pool"],))[0], pool=r["pool"], kind=r.get("kind", "other job"),
                            what=r["what"], state=r["state"], where=r.get("where", ""), since=r.get("since", "")))
    order = {"running": 0, "queued": 1, "claimed": 2, "lease lapsed": 3}
    running.sort(key=lambda r: (SITES.index(r["site"]) if r["site"] in SITES else 9, order.get(r["state"], 5), r["what"]))

    # pools and sites
    hb = raw["pools"]
    done24 = {}
    for c in raw["claims"]:
        if c["status"] == "done" and (_t(c.get("finished")) or 0) >= t - 86400:
            done24[c["pool"]] = done24.get(c["pool"], 0) + 1
    for k, e in ledger.items():
        if e["t"] and e["t"] >= t - 86400 and e["pool"] in ("arm00", "zaratan"):
            done24[e["pool"]] = done24.get(e["pool"], 0) + 1
    prob_pool = {}
    for key, ids in problems.items():
        prob_pool[key[0]] = prob_pool.get(key[0], 0) + len(ids)
    pools = []
    for name in sorted(set(POOLS) | set(hb), key=lambda n: (list(POOLS).index(n) if n in POOLS else 99, n)):
        site, who, hw = POOLS.get(name, (name, "?", "?"))
        h = hb.get(name, {})
        mine = [r for r in running if r["pool"] == name and (r["kind"] == "board task" or name in ("arm00", "zaratan"))]
        other = sum(1 for r in running if r["pool"] == name and r["kind"] != "board task" and name not in ("arm00", "zaratan"))
        age = (t - _t(h["updated"])) / 60 if h.get("updated") else None
        state = ("worker stale" if age is not None and age > 30 and (h.get("running") or 0) > 0 else
                 "worker up" if age is not None and age <= 30 else
                 "loop or queue" if name in ("arm00", "zaratan") else "no worker")
        pools.append(dict(pool=name, site=site, person=who, hardware=hw, state=state, heartbeat=h.get("updated") or "",
                          slots=len(h.get("slots") or []) or None,
                          running=sum(r["state"] == "running" for r in mine), queued=sum(r["state"] == "queued" for r in mine),
                          claimed=sum(r["state"] in ("claimed", "lease lapsed") for r in mine),
                          other_jobs=other, done_24h=done24.get(name, 0), problems=prob_pool.get(name, 0), snapshot=snap))
    people = []
    for site in SITES:
        ps = [p for p in pools if p["site"] == site]
        if not ps:
            continue
        S = lambda f: sum(p[f] or 0 for p in ps)
        hbs = [p["heartbeat"] for p in ps if p["heartbeat"]]
        up = [p for p in ps if p["state"] == "worker up"]
        people.append(dict(site=site, person=ps[0]["person"], pools=", ".join(p["pool"] for p in ps),
                           hardware="; ".join(dict.fromkeys(p["hardware"] for p in ps)),
                           state=("working" if S("running") + S("queued") + S("claimed") else
                                  "idle, worker up" if up else "nothing running"),
                           running=S("running"), queued=S("queued"), claimed=S("claimed"), other_jobs=S("other_jobs"), done_24h=S("done_24h"),
                           problems=S("problems"), heartbeat=max(hbs) if hbs else ""))

    # attention: current problems only
    att = []
    for r in live:
        if r["state"] == "alert":
            att.append(dict(kind="unreachable", severity="bad", pool=r["pool"], what=r["what"], count=None, cause=r.get("note", ""), fix=""))
    for (pool, b, arm, stage, bank, why), ids in sorted(problems.items(), key=lambda kv: -len(kv[1])):
        held = sum(1 for i in ids if sum(1 for c in by_task[i] if c["status"] == "failed") >= 2)
        att.append(dict(kind="failing", severity="bad", pool=pool, what=f"{experiment(b, arm, stage, bank)} on {pool}",
                        count=len(ids), cause=why,
                        fix=(f"{held} held after two failures: fix the cause, then coord/master.py reset --task <id>. " if held else "")
                        + f"{pool} will not retry these; other pools may."))
    for name, h in hb.items():
        age = (t - _t(h["updated"])) / 60 if h.get("updated") else None
        if age is not None and age > 30 and (h.get("running") or 0) > 0:
            att.append(dict(kind="stale worker", severity="warn", pool=name, what=f"{name} worker on {h.get('host')}", count=None,
                            cause=f"no heartbeat for {int(age)} min", fix="its tasks return to the list when their leases lapse"))
    for d in DECISIONS:
        if d["status"] != "decided":
            att.append(dict(kind="decision", severity="warn", pool="Anubhav", what=f"{d['id']}. {d['question']}", count=None,
                            cause=d["why"], fix=d["options"]))

    # activity: claims still running and every claim's end, grouped by (pool, event, experiment) within 30 minutes
    ev = []
    for c in raw["claims"]:
        b, arm, stage, bank, v = parse_task(c["task"])
        exp = experiment(b, arm, stage, bank)
        if c["status"] == "running" and c.get("claimed"):   # finished claims show as their end only
            ev.append((c["claimed"], c["pool"], "claimed", exp, "still running"))
        if c["status"] in ("done", "failed", "superseded") and c.get("finished"):
            if c["status"] == "failed":
                newer = by_task[c["task"]][-1]
                now_ = ("since resolved" if c["task"] not in open_ids else
                        f"retried on {newer['pool']}" if newer["status"] == "running" else "still open")
                e = "cancelled" if c.get("exit") == -1 else "failed"
                note = cause(c) if e == "failed" else ""
                note = (note + "; " if note else "") + now_
                ev.append((c["finished"], c["pool"], e, exp, note))
            else:
                ev.append((c["finished"], c["pool"], c["status"], exp, ""))
    for k, e in ledger.items():
        if e["t"] and e["pool"] in ("arm00", "zaratan"):
            b, arm, _ = k.split("|")
            ev.append((_iso(e["t"]), e["pool"], "built", experiment(b, arm, "build", None), "off the board"))
    ev.sort()
    groups = []
    for ts, pool, e, exp, note in ev:
        hit = next((g for g in reversed(groups) if (g["pool"], g["event"], g["what"], g["note"]) == (pool, e, exp, note)
                    and (_t(ts) or 0) - (_t(g["t_last"]) or 0) <= 1800), None)
        if hit:
            hit["count"] += 1
            hit["t_last"] = ts
        else:
            groups.append(dict(t=ts, t_last=ts, pool=pool, event=e, what=exp, count=1, note=note))
    events = sorted(groups, key=lambda g: g["t_last"], reverse=True)[:150]

    W = lambda n, o: json.dump(o, open(os.path.join(out, n), "w"), indent=0)
    W("matrix.json", matrix)
    W("tasks.json", task_rows)
    W("running.json", running)
    W("pools.json", pools)
    W("people.json", people)
    W("attention.json", att)
    W("events.json", events)
    W("progress.json", PROGRESS)
    W("tranches.json", TRANCHES)
    W("decisions.json", DECISIONS)
    W("ledger.json", ledger)
    return dict(tasks=len(task_rows), running=len(running), attention=len(att), events=len(events))
