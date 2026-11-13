"""The dashboard's summary views, computed from what collect_live.py gathered (all files in --out).

    matrix.json     progress per experiment and tranche, with running/queued counts, rate (per hour, last 6 h)
                    and ETA
    people.json     per pool: who runs it, state, heartbeat, running now, done in the last 24 h, failures,
                    rate, what it picks next
    attention.json  only what needs a person: alerts, failed tasks, stale workers, idle capacity, open decisions
    events.json     the last 200 events: memory built (where), task claimed / done / failed (which pool)

History lives in ledger.json (when and where each memory first appeared) and prev_state.json (each claim's
last state), both kept next to the outputs.
"""
import json
import os
import time

RATE_WINDOW_H = 6


def _load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(t))


def update(out, built_src, view):
    t = time.time()
    L = lambda n, d: _load(os.path.join(out, n), d)
    running, tasks, pools, progress, decisions = L("running.json", []), L("tasks.json", []), L("pools.json", []), \
        L("progress.json", []), L("decisions.json", [])
    ledger, events, prev = L("ledger.json", {}), L("events_log.json", []), L("prev_state.json", {})
    first_run = not ledger

    # memories: when and where each first appeared
    for (b, arm, v), src in sorted(built_src.items()):
        k = f"{b}|{arm}|{v}"
        if k not in ledger:
            ledger[k] = dict(t=None if first_run else t, pool=src)
            if not first_run:
                events.append(dict(t=_iso(t), pool=src, what=f"{b} {arm} memory · {v}", event="built"))
        elif ledger[k]["pool"] == "results dataset" and src != "results dataset":
            ledger[k]["pool"] = src
    # claims: state changes since the last cycle
    seen = {}
    for c in (view or {}).get("claims", []):
        k = f"{c['pool']}|{c['task']}"
        seen[k] = c["state"]
        if prev.get(k) != c["state"]:
            events.append(dict(t=_iso(t), pool=c["pool"], what=c["task"],
                               event={"running": "claimed"}.get(c["state"], c["state"]),
                               note=f"{c.get('gpu', '')}" + (f", {c['seconds'] // 60} min" if c.get("seconds") else "")))
    events = events[-200:]

    # rates per (baseline, arm) memory, and per pool, over the last RATE_WINDOW_H hours
    since = t - RATE_WINDOW_H * 3600
    rate, pool_rate, pool_24h = {}, {}, {}
    for k, e in ledger.items():
        b, arm, _ = k.split("|")
        if e["t"] and e["t"] >= since:
            rate[(b, arm)] = rate.get((b, arm), 0) + 1 / RATE_WINDOW_H
            pool_rate[e["pool"]] = pool_rate.get(e["pool"], 0) + 1 / RATE_WINDOW_H
        if e["t"] and e["t"] >= t - 86400:
            pool_24h[e["pool"]] = pool_24h.get(e["pool"], 0) + 1

    # matrix: the progress rows, plus live counts and ETA
    counts = {}
    for r in tasks:
        if r.get("stage") == "build":
            key = (r["method"], r["arm"])
            c = counts.setdefault(key, {"running": 0, "queued": 0, "ready": 0, "blocked": 0})
            if r["status"] in c:
                c[r["status"]] += 1
    names = {"merit": "MERIT", "magic": "MAGIC-Video", "worldmm": "WorldMM", "m3agent": "M3-Agent"}
    matrix = []
    for p in progress:
        row = dict(p)
        row.update(running=0, queued=0, rate_per_h=None, eta_h=None)
        if p["experiment"].startswith("Memories · "):
            label = p["experiment"].split(" · ", 1)[1]            # e.g. "MERIT 27B"
            meth, size = label.rsplit(" ", 1)
            arm = {"8B": "m8", "27B": "m27"}.get(size, size)
            b = {v: k for k, v in names.items()}.get(meth, meth.lower())
            c = counts.get((meth, arm), {})
            row.update(running=c.get("running", 0), queued=c.get("queued", 0))
            rph = rate.get((b, arm))
            if rph:
                row["rate_per_h"] = round(rph, 2)
                row["eta_h"] = round((p["total"] - p["done"]) / rph, 1)
        matrix.append(row)

    # people: one card per pool, from the workers' records and the loops we can read
    hb = (view or {}).get("pools", {})
    people = []
    for p in pools:
        name = p["pool"]
        mine = [r for r in running if r["pool"] == name]
        claims = [c for c in (view or {}).get("claims", []) if c["pool"] == name]
        heartbeat = hb.get(name, {}).get("updated") or (mine and mine[0].get("since")) or None
        people.append(dict(
            pool=name, person=p["person"], hardware=p["hardware"], state=p["state"], note=p.get("note", ""),
            running=[dict(what=r["what"], job=r["job"], since=r.get("since", ""), note=r.get("note", ""))
                     for r in mine if r["state"] == "running" and r["job"] != "worker"],
            queued=sum(1 for r in mine if r["state"] == "queued"),
            done_24h=pool_24h.get(name, 0) + sum(1 for c in claims if c["state"] == "done" and c.get("finished", "") >= _iso(t - 86400)),
            failed=[c["task"] for c in claims if c["state"] == "failed"],
            rate_per_h=round(pool_rate.get(name, 0), 2) or None,
            heartbeat=heartbeat,
            next=[r["id"] + " · " + f"{r['method']} {r['arm']} {r['stage']} {r['video']}"
                  for r in sorted((r for r in tasks if r.get("pool") == name and r["status"] == "ready"),
                                  key=lambda r: (r["level"], -(r.get("est_gpu_h") or 0)))[:3]]))

    # attention: only what needs a person
    att = []
    for r in running:
        if r["state"] == "alert":
            att.append(dict(kind="alert", what=r["what"], pool=r["pool"], note=r.get("note", "")))
    for c in (view or {}).get("claims", []):
        if c["state"] == "failed":
            att.append(dict(kind="failed task", what=c["task"], pool=c["pool"],
                            note=(c.get("note") or "")[-160:] + " (after a fix: coord/master.py reset --task " + c["task"] + ")"))
    for name, h in hb.items():
        u = h.get("updated")
        if u and time.mktime(time.strptime(u, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone < t - 1800 and (h.get("running") or 0) > 0:
            att.append(dict(kind="stale worker", what=f"{name} on {h.get('host')}", pool=name,
                            note=f"no heartbeat since {u}; its tasks return to the list when their leases lapse"))
    for p in people:
        ready = sum(1 for r in tasks if r.get("pool") == p["pool"] and r["status"] == "ready")
        if p["state"] == "active" and not p["running"] and not p["queued"] and ready:
            att.append(dict(kind="idle capacity", what=f"{p['pool']} has nothing running", pool=p["pool"],
                            note=f"{ready} tasks fit it"))
    for d in decisions:
        if d.get("status") != "decided":
            att.append(dict(kind="decision", what=f"{d['id']}. {d['question']}", pool="Anubhav", note=d.get("options", "")))

    W = lambda n, o: json.dump(o, open(os.path.join(out, n), "w"), indent=0)
    W("matrix.json", matrix)
    W("people.json", people)
    W("attention.json", att)
    W("events.json", list(reversed(events)))
    W("ledger.json", ledger)
    W("prev_state.json", seen)
    json.dump(events, open(os.path.join(out, "events_log.json"), "w"))
    return dict(matrix=len(matrix), people=len(people), attention=len(att), events=len(events))
