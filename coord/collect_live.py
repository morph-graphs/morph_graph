#!/usr/bin/env python3
"""Read our live jobs from every pool we can see and rebuild the board's data files.

Runs on the Mac (it uses nx and ssh): the job board (coord/board_read.py on Nexus), our Nexus and Zaratan
queues, the arm00 build loop and the MERIT memories built off the board. board_views.py turns that into the
dashboard's ten JSON datasets in --out; this prints one line per dataset with its sha256 so the publisher
uploads only what changed.

    python3 coord/collect_live.py --out <dir>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ARM = "anubhav@vulcan-arm00.umiacs.umd.edu"
ARM_RUNS = "/fs/cfar-projects/mem_open_vocab/anubhav/merit_runs"
ZR_RUNS = "/scratch/zt1/project/abhinav2-prj/user/anubhav/merit_runs"
SRC = {}    # (baseline, arm, video) -> where the memory was seen first: zaratan, arm00, a pool, or the results dataset
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
    # plain ssh over the existing master connection, not `zr`: zr flushes the file sync first, and a busy sync
    # (new worktrees, 2026-11-13) made every zr command hang, hiding Zaratan's jobs from the board
    rc, out = sh(SSH + ["-o", "RemoteCommand=none", "-o", "RequestTTY=no", "zaratan", "squeue -u $USER -h -r -o '%i|%j|%T|%R|%M'; echo ===; "
                        f"ls {ZR_RUNS}/results/m27 2>/dev/null | while read v; do "
                        f"grep -qs '\"stage\": \"build\"' {ZR_RUNS}/results/m27/$v/runs.jsonl && echo $v; done; echo ===; "
                        "sacct -n -X -S now-6hours -o JobID%20,JobName%20,State%14 | grep merit- | grep -vE 'COMPLETED|PENDING|RUNNING|CANCELLED' || true"])
    if rc != 0 or "===" not in out:
        alerts.append(dict(pool="zaratan", what="Zaratan not readable", state="alert", since=now(),
                           note="ssh to Zaratan failed: the login may have lapsed. Run `! zr login` (Duo); until then Zaratan jobs are not shown"))
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
        rows.append(dict(pool="zaratan", kind="other job", what=name, state=state, since="",
                         where=f"job {jid}: {j.get('RUNNING', 0)} running, {j.get('PENDING', 0)} waiting {j['reason'].strip('()')}"))
    for v in b.split():
        built.add(("merit", "m27", v))
        SRC[("merit", "m27", v)] = "zaratan"
    for line in failed.splitlines():
        if line.strip():
            alerts.append(dict(pool="zaratan", what=f"Zaratan job {line.split()[0]} {' '.join(line.split()[1:2])}", state="alert",
                               since=now(), note="ended " + line.split()[-1] + " in the last 6 h"))


def arm00(rows, built, alerts):
    script = (f"tmux has-session -t merit-arm 2>/dev/null && echo LOOP=up || echo LOOP=down; "
              f"nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; "
              f"for v in $(ls {ARM_RUNS}/results/m27 2>/dev/null); do grep -qs '\"stage\": \"build\"' {ARM_RUNS}/results/m27/$v/runs.jsonl && echo BUILT=$v; done; "
              f"f=$(ls -t {ARM_RUNS}/logs/arm-build-*.log 2>/dev/null | head -1); [ -n \"$f\" ] && echo CUR=$(basename $f .log | sed s/arm-build-//) && "
              f"echo LAST=$(grep '^===' $f | tail -1) && echo TAIL=$(tail -c 400 $f | tr '\\r' '\\n' | grep -v '^\\s*$' | tail -1 | cut -c1-90) && echo AGE=$(( $(date +%s) - $(stat -c %Y $f) ))")
    rc, out = sh(SSH + [ARM, script], timeout=60)
    kv = [l.split("=", 1) for l in out.splitlines() if "=" in l]
    if rc != 0 or not kv:
        alerts.append(dict(pool="arm00", what="arm00 not reachable", state="alert", since=now(),
                           note="ssh failed (UMIACS VPN down?); its loop is not shown"))
        return
    d = {}
    for k, v in kv:
        if k == "BUILT":
            built.add(("merit", "m27", v))
            SRC[("merit", "m27", v)] = "arm00"
        else:
            d[k] = v
    cur, last, age = d.get("CUR", "?"), d.get("LAST", ""), int(d.get("AGE", "0") or 0)
    done = last.endswith("done")
    if d.get("LOOP") == "down":
        if not done:   # a loop that stopped mid-video needs a person; one that finished its list does not
            alerts.append(dict(pool="arm00", what="arm00 build loop stopped mid-video", state="alert", since=now(),
                               note=f"last video {cur}: {last}"))
        return
    gpu = [l for l in out.splitlines() if "," in l and "=" not in l]
    mem, util = (gpu[0].split(",") + ["?", "?"])[:2] if gpu else ("?", "?")
    gb = int(float(mem)) // 1024 if mem.strip().replace(".", "").isdigit() else mem
    rows.append(dict(pool="arm00", kind="loop", what=f"MERIT 27B · memory build · {cur}", state="running", since="",
                     where=f"GH200, {gb} GB used, {util.strip()}% busy; log updated {age // 60} min ago"))
    if not done and age > 3 * 3600:
        alerts.append(dict(pool="arm00", what=f"arm00: no progress on {cur}", state="alert", since=now(),
                           note=f"log unchanged for {age // 3600} h"))


def nexus(rows, alerts, nxq):
    """Our Nexus jobs: board tasks' jobs (named board-<task id>) go to nxq for the views; other jobs are rows."""
    rc, out = sh(["nx", "squeue -u anubhav -h -o '%i|%j|%P|%T|%M|%R' | grep -v keepalive"], timeout=120)
    if rc not in (0, 1):
        alerts.append(dict(pool="nexus", what="Nexus not readable", state="alert", since=now(),
                           note="nx returned nothing: its login lapsed. Run `! nx login` (Duo); until then Nexus jobs show as claimed"))
        return
    for line in out.splitlines():
        if line.count("|") < 5:
            continue
        jid, name, part, state, el, reason = line.split("|")[:6]
        if name.startswith("board-worker"):
            continue                      # the workers themselves: their heartbeats are on the board
        if name.startswith("board-"):
            nxq[name] = dict(state=state, partition=part, elapsed=el, where=reason.strip("()") if state != "RUNNING" else reason)
            continue
        rows.append(dict(pool="nexus", kind="other job", what=name, state="running" if state == "RUNNING" else "queued",
                         since="", where=f"{part} on {reason}, {el} elapsed" if state == "RUNNING" else f"{part}, waiting {reason.strip('()')}"))


def board(alerts, out_dir):
    """The board as coord/board_read.py prints it; on failure the last good read, with an alert."""
    rc, out = sh(["nx", "cd /fs/cfar-projects/mem_open_vocab/morph_graphs/anubhav/pkg/wt-coord && HF_TOKEN=$(cat ~/.config/hf_write_token) "
                        "HF_HOME=/fs/cfar-projects/mem_open_vocab/anubhav/hf_home /fs/nexus-scratch/anubhav/merit_envs/m8/bin/python coord/board_read.py 2>/dev/null | tail -1"], timeout=300)
    cache = os.path.join(out_dir, "board_raw.json")
    try:
        v = json.loads(out.strip().splitlines()[-1])
        json.dump(v, open(cache, "w"))
    except Exception:
        v = json.load(open(cache)) if os.path.exists(cache) else None
        alerts.append(dict(pool="board", what="job board not read this cycle", state="alert", since=now(),
                           note="coord/board_read.py failed (nx down or token problem); "
                                + (f"showing the read from {v['read']}" if v else "no earlier read to show")))
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--durations", help="unused (kept for the refresher's command line)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, alerts, built, nxq = [], [], set(), {}
    os.makedirs(a.out, exist_ok=True)
    zaratan(rows, built, alerts)
    arm00(rows, built, alerts)
    nexus(rows, alerts, nxq)
    raw = board(alerts, a.out)
    import board_views
    n = board_views.update(a.out, raw, alerts + rows, nxq, SRC)
    for name in ("running", "tasks", "pools", "progress", "decisions", "tranches", "matrix", "people", "attention", "events"):
        p = os.path.join(a.out, f"{name}.json")
        print(name, hashlib.sha256(open(p, "rb").read()).hexdigest(), p)
    print("alerts", len(alerts), "|", " ".join(f"{k} {v}" for k, v in n.items()))


if __name__ == "__main__":
    main()
