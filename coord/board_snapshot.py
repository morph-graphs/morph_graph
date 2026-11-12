#!/usr/bin/env python3
"""First job-board snapshot for the monitoring dashboard, before coord's store exists.

Builds the task list from what is known today (2026-11-12): the batch10 tranche's finished runs, the 146
videos without memories (baselines/merit/build_order.txt), measured and estimated GPU-hour rates, the pools
and their state. Writes tasks / pools / progress / decisions / tranches as JSON rows for the dashboard.
`master.py` replaces this once the store and workers exist; the row shapes stay the same.

    python3 coord/board_snapshot.py --durations <video -> seconds json> --out <dir>
"""
import argparse
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_ORDER = os.path.join(HERE, "..", "baselines", "merit", "build_order.txt")
BATCH10 = ["P01-20240204-142301", "P01-20240203-123350", "P02-20240211-175138", "P03-20240217-131219",
           "P05-20240425-201444", "P06-20240510-090328", "P07-20240531-201703", "P08-20240617-130401",
           "P08-20240617-184909", "P09-20240624-160737"]

# GPU-hours per video-hour of memory build. measured: from our runs; est: scaled or guessed (design/11)
BUILD_RATE = {
    ("merit", "m8"): (0.44, "48gb", "measured"),
    ("merit", "m27"): (1.45, "hopper", "est"),
    ("magic", "m8"): (3.0, "48gb", "est"),
    ("magic", "m27"): (4.5, "hopper", "est"),
    ("worldmm", "w8"): (3.3, "48gb", "est"),
    ("worldmm", "w27"): (5.0, "hopper", "est"),
    ("m3agent", "m3"): (2.0, "48gb", "est"),
}
NAMES = {"merit": "MERIT", "magic": "MAGIC-Video", "worldmm": "WorldMM", "m3agent": "M3-Agent", "qwenplug": "Qwen-MM-Plugins"}

# pools: who runs them, what they can take, whether they can take work now
POOLS = [
    dict(pool="nexus", person="Anubhav's agent", hardware="A6000 / L40S / A5000 pairs (about 12 at once)",
         families="48gb", launcher="Slurm fan-out over scavenger, cml-scavenger, vulcan-scavenger, tron high, vulcan-ampere",
         state="active", capacity=12, note="Main pool for 8B arms. Preemptible tiers; jobs resume."),
    dict(pool="nexus-vatsal", person="Vatsal's agent", hardware="Nexus GPUs on Vatsal's account",
         families="48gb", launcher="Slurm (his MAGIC-Video and WorldMM ports)", state="active", capacity=6,
         note="Runs MAGIC-Video and WorldMM until they move into morph_graph."),
    dict(pool="exempt-sailon", person="Anubhav's agent", hardware="vulcan-exempt and sailon queues",
         families="48gb,hopper", launcher="Slurm", state="waiting for access", capacity=0,
         note="Access requested; no jobs until Anubhav confirms."),
    dict(pool="wriva", person="Archana and Aaron", hardware="2-4 H100-SXM (cml-wriva, cml-wriva-high), indefinitely",
         families="hopper,48gb", launcher="Slurm on cml-wriva, under their own accounts", state="needs onboarding", capacity=3,
         note="They are not in the openvocab group, so they can't read our videos yet. Needs their usernames, the public image and their own Hugging Face tokens."),
    dict(pool="arm00", person="Anubhav's agent", hardware="1x GH200 96 GB (aarch64), shared",
         families="hopper", launcher="direct, one job at a time when idle", state="busy", capacity=1,
         note="Busy at 21:20 UTC 2026-11-12: another user's jobs hold 78 of 98 GB at 100% load. 27B arm environment exists (Vatsal's); x86 Docker images don't run here."),
    dict(pool="zaratan", person="Anubhav's agent", hardware="H100 / A100",
         families="hopper,48gb", launcher="zr sbatch; uploads from the login node", state="active", capacity=8,
         note="AAC approved by Anubhav 2026-11-12 (CMSC is empty). H100 = 144 SU/h. Videos and models already on scratch; waits for the public image."),
    dict(pool="max", person="Max's agent", hardware="Max's GPUs (types to confirm)",
         families="48gb", launcher="Docker images from morph_graph main", state="busy with Stage A", capacity=4,
         note="MERIT images ready; takes MERIT builds after Stage A."),
]


def task(**k):
    base = dict(kind="gpu", baseline="", arm="", stage="", video="", tranche="", status="ready", pool="",
                person="", level=3, est_gpu_h=0.0, video_min=None, family="", blocked_by="", evidence="", note="")
    base.update(k)
    base["method"] = NAMES.get(base["baseline"], base["baseline"])
    return base


def build_tasks(dur):
    order = [v.strip() for v in open(BUILD_ORDER) if v.strip()]
    rows = []
    for i, v in enumerate(order):
        vh = dur.get(v, 0) / 3600
        for (b, arm), (rate, fam, ev) in BUILD_RATE.items():
            blocked, status = "", "ready"
            if b == "magic":
                blocked, status = "questions for this video (its narrative chains are built per question time)", "blocked"
            elif b == "worldmm":
                blocked, status = "WorldMM timing pilot", "blocked"
            elif b == "m3agent":
                blocked, status = "M3-Agent port", "blocked"
            elif (b, arm) == ("merit", "m27"):
                blocked, status = "", "ready"
            rows.append(task(baseline=b, arm=arm, stage="build", video=v, tranche="all videos", status=status,
                             level=3, est_gpu_h=round(rate * vh, 2), family=fam, blocked_by=blocked, evidence=ev,
                             video_min=round(vh * 60, 1), note=f"build order #{i + 1}; {vh * 60:.0f} min of video"))
    return rows


def tranche1_tasks(dur):
    rows = []
    for v in BATCH10:
        for arm, fam in (("m8", "48gb"), ("m27", "hopper")):
            rows.append(task(baseline="merit", arm=arm, stage="build", video=v, tranche="all videos", status="done",
                             level=3, family=fam, evidence="measured", video_min=round(dur.get(v, 0) / 60, 1),
                             note="memory in the results dataset"))
    for v in BATCH10:
        vm = dur.get(v, 0) / 60
        for arm in ("m8", "m27"):
            rows.append(task(baseline="merit", arm=arm, stage="qa (frames)", video=v, tranche="t1", status="done",
                             level=1, family="48gb" if arm == "m8" else "hopper", evidence="measured",
                             note="598-question main table; in the results dataset"))
        rows.append(task(baseline="magic", arm="m8", stage="build + qa", video=v, tranche="t1", status="done", level=1,
                         family="48gb", evidence="measured", note="Vatsal's run (10 completed jobs in Slurm accounting)"))
        rows.append(task(baseline="magic", arm="m27", stage="build + qa", video=v, tranche="t1",
                         status="unverified", level=1, family="hopper", est_gpu_h=round(4.5 * vm / 60 + 1.5, 1),
                         evidence="est", note="2 of 10 videos finished on Nexus by Slurm accounting; GH200 runs not visible from here"))
        for arm in ("w8", "w27"):
            rows.append(task(baseline="worldmm", arm=arm, stage="build + qa", video=v, tranche="t1",
                             status="unverified", level=1, family="48gb" if arm == "w8" else "hopper", evidence="est",
                             note="Vatsal's port; status not visible from here (ask Vatsal)"))
    rows.append(task(baseline="merit", arm="m27", stage="M2 memory-only", video="P07-20240531-201703", tranche="t1",
                     status="held", level=4, family="hopper", est_gpu_h=4.0, evidence="measured",
                     blocked_by="M2 probes are in the research phase (Anubhav, 2026-11-12)",
                     note="Still pending on Nexus as job 8367939; Anubhav to cancel"))
    for name, b, arm, fam, gh in [("WorldMM 8B local", "worldmm", "w8", "48gb", 8),
                                  ("WorldMM 27B local", "worldmm", "w27", "hopper", 10),
                                  ("M3-Agent", "m3agent", "m3", "48gb", 10)]:
        rows.append(task(baseline=b, arm=arm, stage="timing pilot (2 videos)", video="P07 + P03-20240217", tranche="t1",
                         status="blocked" if b == "m3agent" else "ready", level=2, family=fam, est_gpu_h=gh,
                         blocked_by="M3-Agent port" if b == "m3agent" else "", evidence="est",
                         note=f"Turns the {name} guess into a measurement before scaling"))
    return rows


def eng_tasks():
    E = lambda **k: task(kind="engineering", status=k.pop("status", "ready"), **k)
    return [
        E(stage="Stable question keys in MERIT answers (ID = bank key)", person="Anubhav's agent", level=1,
          note="Prerequisite for answering only new questions when a tranche lands"),
        E(stage="Job store + master.py + worker.py (MERIT first)", person="Anubhav's agent", level=1,
          blocked_by="the jobs dataset (decision)", status="blocked",
          note="Design in morph_graph coord/DESIGN.md"),
        E(stage="Apptainer images from the MERIT Docker images; aarch64 environment for arm00", person="Anubhav's agent",
          level=2, note="Lets Nexus, wriva and Zaratan run the same environment as Max"),
        E(stage="vLLM answering for 27B arms", person="Anubhav's agent", level=2,
          note="Expected 5-10x on 27B answering (not measured); halves the H100 budget"),
        E(stage="Move MAGIC-Video and WorldMM into morph_graph (run.sh + images)", person="Vatsal's agent", level=2,
          note="Same entry point and output layout as MERIT"),
        E(stage="Port M3-Agent (local swaps for Gemini diarization and OpenAI embeddings)", person="unassigned", level=3,
          note="Only method with open weights for memory and QA; needs audio"),
        E(stage="Onboard wriva friends (image, video access, token, worker)", person="Anubhav's agent", level=2,
          blocked_by="wriva details (decision)", status="blocked", note="Draft the note; Anubhav approves sending"),
        E(stage="Docker launcher for Max's worker", person="Anubhav's agent", level=3,
          blocked_by="Job store", status="blocked", note="After the MVP runs on our pools"),
    ]


def project(rows, pools):
    """Projected owner per ready GPU task: the pull rule played forward over a 72 h horizon."""
    left = {p["pool"]: p["capacity"] * 72.0 for p in pools if p["state"] in ("active", "check if idle")}
    fam = {p["pool"]: p["families"].split(",") for p in pools}
    person = {p["pool"]: p["person"] for p in pools}
    special = {"magic": ("nexus-vatsal", "arm00"), "worldmm": ("nexus-vatsal", "arm00")}   # Vatsal's ports
    for r in sorted((r for r in rows if r["kind"] == "gpu" and r["status"] == "ready"),
                    key=lambda r: (r["level"], -r["est_gpu_h"])):
        cands = [p for p in left if r["family"] in fam[p] and (r["baseline"] not in special or p in special[r["baseline"]])]
        cands = [p for p in cands if r["baseline"] in special or p != "nexus-vatsal"]
        if not cands:
            r["pool"], r["status"] = "—", "blocked"
            r["blocked_by"] = f"no free {'Hopper (H100 / GH200)' if r['family'] == 'hopper' else '48 GB'} pool: wriva needs onboarding, Zaratan needs approval, arm00 busy"
            continue
        p = max(cands, key=lambda p: left[p])
        if left[p] < r["est_gpu_h"]:
            r["pool"] = "after 72 h"
            continue
        left[p] -= r["est_gpu_h"]
        r["pool"], r["person"] = p, person[p]
    for r in rows:
        if not r["pool"]:
            r["pool"] = "—"


QUEUED = {}   # video -> (pool, note) for tasks submitted outside the board


def mark_queued(rows):
    for r in rows:
        q = QUEUED.get((r["baseline"], r["arm"], r["stage"], r["video"]))
        if q:
            r["status"], r["pool"], r["person"], r["note"] = "queued", q[0], "Anubhav's agent", q[1] + "; " + r["note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dur = json.JSONDecoder().raw_decode(open(a.durations).read())[0]
    rows = tranche1_tasks(dur) + build_tasks(dur) + eng_tasks()
    wave = os.path.join(HERE, "..", "..", "..", "morph", "pkg", "zaratan", "wave1.txt")
    if os.path.exists(os.path.expanduser("~/Work/morph/pkg/zaratan/wave1.txt")):
        for v in open(os.path.expanduser("~/Work/morph/pkg/zaratan/wave1.txt")).read().split():
            QUEUED[("merit", "m27", "build", v)] = ("zaratan", "Zaratan wave 1: prep 24047789 (scavenger), build 24047790 (H100, AAC)")
    mark_queued(rows)
    project(rows, POOLS)
    for i, r in enumerate(rows):
        r["id"] = f"T{i + 1:04d}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    for p in POOLS:
        mine = [r for r in rows if r["pool"] == p["pool"]]
        p["next_tasks"] = len(mine)
        p["next_gpu_h"] = round(sum(r["est_gpu_h"] for r in mine), 1)
        p["running"] = None   # filled by workers' heartbeats once they exist
        p["snapshot"] = now
    progress = []
    def add(exp, tranche, done, total, unit, state, note=""):
        progress.append(dict(experiment=exp, tranche=tranche, done=done, total=total, unit=unit, state=state, note=note))
    add("MERIT 8B · main (with frames)", "t1", 598, 598, "questions", "done")
    add("MERIT 27B · main (with frames)", "t1", 598, 598, "questions", "done")
    add("MAGIC-Video 8B · main", "t1", 10, 10, "videos", "done", "Vatsal's port")
    add("MAGIC-Video 27B · main", "t1", 2, 10, "videos", "unverified", "Nexus jobs only; GH200 runs not visible")
    add("WorldMM 8B · main", "t1", 0, 10, "videos", "unverified", "ask Vatsal")
    add("WorldMM 27B · main", "t1", 0, 10, "videos", "unverified", "ask Vatsal")
    add("M3-Agent · main", "t1", 0, 10, "videos", "blocked", "port needed")
    add("Qwen-MM-Plugins · main", "t1", 0, 10, "videos", "blocked", "API only: decision")
    add("M2 probes · 8B no context", "t1", 1513, 1513, "probes", "done")
    add("M2 probes · 27B no context", "t1", 1513, 1513, "probes", "done")
    add("M2 probes · MERIT 8B memory only", "t1", 1513, 1513, "probes", "done")
    add("M2 probes · MERIT 8B with frames", "t1", 805, 1513, "probes", "unverified", "6 of 10 videos at 07:45")
    add("M2 probes · MERIT 8B right clip", "t1", 0, 1513, "probes", "unverified", "running since this morning")
    add("M2 probes · MERIT 27B memory only", "t1", 0, 1513, "probes", "held", "probes are in the research phase")
    add("Memories · MERIT 8B", "all videos", 10, 156, "videos", "ready")
    add("Memories · MERIT 27B", "all videos", 10, 156, "videos", "queued", "17 queued on Zaratan (wave 1)")
    for exp in ("MERIT 8B · main (with frames)", "MERIT 27B · main (with frames)", "MAGIC-Video 8B · main",
                "M2 probes · MERIT 8B memory only"):
        add(exp, "t2", 0, 129, "videos", "waiting for questions", "Max's Stage A in progress")
    tranches = [
        dict(tranche="t1", name="batch10_agree2", videos=10, questions=598, probes=1513, state="frozen",
             source="epic-morph tag bank/batch10_agree2 (0f30904)"),
        dict(tranche="t2", name="Stage A on the remaining 129 videos", videos=129, questions=None, probes=None,
             state="building questions", source="Max: Stage A now, B stages later; short videos last"),
    ]
    decisions = [
        dict(id="D1", question="Bill Zaratan's AAC account for MERIT 27B memory builds?", owner="Anubhav", status="decided",
             why="CMSC has 3.4 kSU left.", options="Decided 2026-11-12: yes, AAC approved."),
        dict(id="D2", question="Who runs wriva, and how many H100s?", owner="Anubhav", status="decided",
             why="Sizing their queue.", options="Decided: Archana and Aaron, 2-4 H100s indefinitely. Still needed: their Nexus usernames."),
        dict(id="D3", question="Make the MERIT container images public?", owner="Anubhav", status="needs action",
             why="Checked safe: only the public PyTorch base image and public pip packages, no code, data or credentials.",
             options="Approved. Anubhav (org owner) flips it: github.com/orgs/morph-graphs/packages, package merit, Package settings, Change visibility, Public."),
        dict(id="D4", question="Create the private dataset morph-graphs/epic-morph-jobs?", owner="Anubhav", status="decided",
             why="Job claims apart from results.", options="Done: created and reachable with our token."),
        dict(id="D5", question="Which GPUs may run each model size?", owner="Anubhav", status="decided",
             why="Captions differ slightly across GPU models.",
             options="Decided: anywhere the model fits, whichever starts first; every output records its GPU and the paper gets one cross-GPU check per arm."),
        dict(id="D6", question="Qwen-MM-Plugins through OpenRouter?", owner="Anubhav", status="open",
             why="We have an OpenRouter key. Its default models (Qwen VL, ASR, omni, embeddings) must all exist there; frames leave our clusters.",
             options="Check model availability on OpenRouter, then a capped pilot"),
        dict(id="D7", question="M3-Agent: replace its Gemini diarization and OpenAI embeddings with local models?",
             owner="Anubhav", status="open", why="Keeps everything on our GPUs, but it is then not exactly the published method.",
             options="Local swaps, noted in the paper / Use the APIs"),
    ]
    os.makedirs(a.out, exist_ok=True)
    for name, data in [("tasks", rows), ("pools", POOLS), ("progress", progress), ("decisions", decisions),
                       ("tranches", tranches)]:
        json.dump(data, open(os.path.join(a.out, f"{name}.json"), "w"), indent=0)
    print({k: len(v) for k, v in [("tasks", rows), ("pools", POOLS), ("progress", progress)]},
          "ready GPU-h by pool:", {p["pool"]: p["next_gpu_h"] for p in POOLS})


if __name__ == "__main__":
    main()
