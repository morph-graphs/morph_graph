#!/usr/bin/env python3
"""Read our live jobs from every pool we can see and rebuild the board's data files.

Runs on the Mac (it uses zr, nx and ssh): Zaratan and Nexus queues, the arm00 build loop, and the built
MERIT memories on Zaratan and arm00. Writes running.json and built.txt, then runs board_snapshot.py, and
prints one line per data file with its sha256 so the board publisher uploads only what changed.

    python3 coord/collect_live.py --durations <video -> seconds json> --out <dir>
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = "anubhav@vulcan-arm00.umiacs.umd.edu"
ARM_RUNS = "/fs/cfar-projects/mem_open_vocab/anubhav/merit_runs"
ZR_RUNS = "/scratch/zt1/project/abhinav2-prj/user/anubhav/merit_runs"
SSH = ["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=os.path.expanduser("~/Work/morph"))
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 124, ""


def now():
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())


def zaratan(rows, built, alerts):
    rc, out = sh(["zr", "squeue -u $USER -h -r -o '%i|%j|%T|%R|%M'; echo ===; "
                        f"ls {ZR_RUNS}/results/m27 2>/dev/null | while read v; do "
                        f"grep -qs '\"stage\": \"build\"' {ZR_RUNS}/results/m27/$v/runs.jsonl && echo $v; done; echo ===; "
                        "sacct -n -X -S now-1days -o JobID%20,JobName%20,State%14 | grep merit- | grep -vE 'COMPLETED|PENDING|RUNNING|CANCELLED' || true"])
    if rc != 0 or "===" not in out:
        alerts.append(dict(pool="zaratan", job="—", what="Zaratan not readable", state="alert", since=now(),
                           note="zr failed (connection down? run `zr login`)"))
        return
    q, b, failed = (out.split("===\n") + ["", ""])[:3]
    jobs = {}
    for line in q.splitlines():
        if line.count("|") < 4:
            continue
        jid, name, state, reason, el = line.split("|")[:5]
        j = jobs.setdefault((jid.split("_")[0], name), {"RUNNING": 0, "PENDING": 0, "reason": reason})
        j[state] = j.get(state, 0) + 1
    for (jid, name), j in jobs.items():
        state = "running" if j.get("RUNNING") else "queued"
        rows.append(dict(pool="zaratan", job=jid, what=name, state=state, since=now(),
                         note=f"{j.get('RUNNING', 0)} running, {j.get('PENDING', 0)} pending ({j['reason']})"))
    for v in b.split():
        built.add(("merit", "m27", v))
    for line in failed.splitlines():
        if line.strip():
            alerts.append(dict(pool="zaratan", job=line.split()[0], what=" ".join(line.split()[1:2]), state="alert",
                               since=now(), note="job ended " + line.split()[-1]))


def arm00(rows, built, alerts):
    script = (f"tmux has-session -t merit-arm 2>/dev/null && echo LOOP=up || echo LOOP=down; "
              f"nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; "
              f"for v in $(ls {ARM_RUNS}/results/m27 2>/dev/null); do grep -qs '\"stage\": \"build\"' {ARM_RUNS}/results/m27/$v/runs.jsonl && echo BUILT=$v; done; "
              f"f=$(ls -t {ARM_RUNS}/logs/arm-build-*.log 2>/dev/null | head -1); [ -n \"$f\" ] && echo CUR=$(basename $f .log | sed s/arm-build-//) && "
              f"echo LAST=$(grep '^===' $f | tail -1) && echo TAIL=$(tail -c 400 $f | tr '\\r' '\\n' | grep -v '^\\s*$' | tail -1 | cut -c1-90) && echo AGE=$(( $(date +%s) - $(stat -c %Y $f) ))")
    rc, out = sh(SSH + [ARM, script], timeout=60)
    kv = [l.split("=", 1) for l in out.splitlines() if "=" in l]
    if rc != 0 or not kv:
        alerts.append(dict(pool="arm00", job="tmux merit-arm", what="arm00 not reachable", state="alert", since=now(),
                           note="ssh failed (UMIACS VPN down?)"))
        return
    d = {}
    for k, v in kv:
        if k == "BUILT":
            built.add(("merit", "m27", v))
        else:
            d[k] = v
    gpu = [l for l in out.splitlines() if "," in l and "=" not in l]
    mem, util = (gpu[0].split(",") + ["?", "?"])[:2] if gpu else ("?", "?")
    n_built = sum(1 for b in built if b[0] == "merit")
    if d.get("LOOP") == "down":
        alerts.append(dict(pool="arm00", job="tmux merit-arm", what="arm00 build loop stopped", state="alert", since=now(),
                           note=f"last video {d.get('CUR', '?')}: {d.get('LAST', '')}"))
        return
    cur, last, age = d.get("CUR", "?"), d.get("LAST", ""), int(d.get("AGE", "0") or 0)
    done = cur and last.endswith("done")
    rows.append(dict(pool="arm00", job="tmux merit-arm", what=f"MERIT 27B build · {cur}", state="running", since=now(),
                     note=f"{(d.get('TAIL') or last.replace('=== ', ''))[:90]}; GPU {int(float(mem)) // 1024 if mem.strip().replace('.', '').isdigit() else mem} GB, {util.strip()}% busy; log updated {age // 60} min ago"))
    if not done and age > 3 * 3600:
        alerts.append(dict(pool="arm00", job="tmux merit-arm", what=f"no progress on {cur}", state="alert", since=now(),
                           note=f"log unchanged for {age // 3600} h"))


def nexus(rows, alerts):
    rc, out = sh(["nx", "squeue -u anubhav -h -o '%i|%j|%P|%T|%M|%R' | grep -v keepalive"], timeout=120)
    if rc not in (0, 1):
        alerts.append(dict(pool="nexus", job="—", what="Nexus not readable", state="alert", since=now(),
                           note="nx failed (connection down? run `nx login`)"))
        return
    for line in out.splitlines():
        if line.count("|") < 5:
            continue
        jid, name, part, state, el, reason = line.split("|")[:6]
        kind = ("M2 probes (on hold)" if name.startswith("v3-m2") else "port setup / smoke test" if any(k in name for k in ("env", "smoke", "wmm", "magic", "m3a", "qmem")) else "")
        rows.append(dict(pool="nexus", job=jid, what=f"{name}" + (f" · {kind}" if kind else ""),
                         state="running" if state == "RUNNING" else "queued", since=now(),
                         note=f"{part}, {el} elapsed" if state == "RUNNING" else f"{part}, {reason}"))


def board(rows, built, alerts):
    """Claims from every pool's worker (the jobs dataset) and every memory in the results dataset."""
    rc, out = sh(["nx", "cd /fs/cfar-projects/mem_open_vocab/morph_graphs/anubhav/pkg/wt-coord && HF_TOKEN=$(cat ~/.config/hf_write_token) "
                        "/fs/nexus-scratch/anubhav/merit_envs/m8/bin/python coord/master.py view 2>/dev/null | tail -1"], timeout=240)
    try:
        v = json.loads(out.strip().splitlines()[-1])
    except Exception:
        alerts.append(dict(pool="board", job="—", what="jobs dataset not readable", state="alert", since=now(),
                           note="coord/master.py view failed (nx down or token problem)"))
        return
    for c in v["claims"]:
        if c["state"] in ("running", "lease lapsed", "failed") or c["finished"] >= time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400)):
            rows.append(dict(pool=c["pool"], job=c["task"], what=f"{c['task']} (board worker on {c['host']})",
                             state={"lease lapsed": "alert"}.get(c["state"], c["state"]), since=c["started"].replace("T", " ").replace("Z", " UTC"),
                             note=(f"{c['gpu']}; " + (f"{c['seconds'] // 60} min" if c.get("seconds") else "running")
                                   + (f"; {c['note'][-120:]}" if c["state"] == "failed" else "")
                                   + ("; worker stopped renewing its lease: the task is free again" if c["state"] == "lease lapsed" else ""))))
    for b, arm, v_ in v["built"]:
        built.add((b, arm, v_))
    for name, h in v["pools"].items():
        rows.append(dict(pool=name, job="worker", what=f"board worker on {h.get('host')}", state="active", since=(h.get("updated") or "").replace("T", " ").replace("Z", " UTC"),
                         note=f"slots {h.get('slots')}, {h.get('running')} running; last heartbeat {h.get('updated')}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--durations", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, alerts, built = [], [], set()
    zaratan(rows, built, alerts)
    arm00(rows, built, alerts)
    nexus(rows, alerts)
    board(rows, built, alerts)
    rows.append(dict(pool="nexus-vatsal", job="—", what="Vatsal's agents", state="not visible", since="",
                     note="We can't see Vatsal's jobs or agents until his pool runs a board worker"))
    os.makedirs(a.out, exist_ok=True)
    json.dump(alerts + rows, open(os.path.join(a.out, "running.json"), "w"), indent=0)
    open(os.path.join(a.out, "built.txt"), "w").write("".join(f"{b} {r} {v}\n" for b, r, v in sorted(built)))
    rc, out = sh(["/opt/homebrew/bin/python3", os.path.join(HERE, "board_snapshot.py"), "--durations", a.durations,
                  "--running", os.path.join(a.out, "running.json"), "--built", os.path.join(a.out, "built.txt"),
                  "--out", a.out])
    if rc != 0:
        raise SystemExit(f"board_snapshot failed: {out}")
    for name in ("running", "tasks", "pools", "progress", "decisions", "tranches"):
        p = os.path.join(a.out, f"{name}.json")
        print(name, hashlib.sha256(open(p, "rb").read()).hexdigest(), p)
    print("alerts", len(alerts), "| running", sum(r["state"] == "running" for r in rows),
          "| queued", sum(r["state"] == "queued" for r in rows), "| built", len(built))


if __name__ == "__main__":
    main()
