#!/usr/bin/env python3
"""The job board's master: works out what should exist, subtracts what does, and lists the rest as tasks.

    python coord/master.py publish     # rewrite tasks.json in the jobs dataset (one commit)
    python coord/master.py publish --dry-run   # the same tasks, counted and printed; nothing is committed
    python coord/master.py view        # print the board's view as JSON (pools, running, built), for the dashboard

Desired state (EXPERIMENTS below) x videos, minus memories already in the results dataset, minus nothing
else: claims and leases are the workers' business. A task's `pools` says which pools may take it; the order
of tasks.json is the priority. Reassignment is automatic: a claim whose lease lapsed is free again, and a
task that failed twice is held back and shown as a failure.

`qa` experiments answer a named epic-morph bank (one experiment per bank in BANKS): a task per video that has
questions in the bank and no answers_<bank>.json.gz in the results dataset, or one that is stale, depending on the same
video's build task while that is pending. A file is stale when some question of the bank has no answer in it as the
bank asks it now (unanswered(): the reuse rule's key, wording, options, gold letter and query time), e.g. after the bank
moved a question's t_ask, or when it holds an "Error" or does not parse. A stale file's task gets the id
<task>-redo-<hash of the video's questions>, since the plain id may already be done on a pool (held for good): it is
listed once per state of the bank, so an answer that fails again is not retried every cycle, and not while the task
or a redo of it still runs or waits for its relay (what it uploads may be current). The master reads the
answers files to check, through a cache of their summaries by content hash (only files that changed download).
The bank is generated afresh each cycle (v3/export_bank.py <bank>, deterministic per commit) in the epic-morph
checkout EPIC_MORPH_DIR, the master's own (not a worker's, which may be reading its bank files). publish stops
without EPIC_MORPH_DIR: a tasks.json without the qa tasks would drop them from the board.

Arms that call a paid API (NEEDS_ENV) carry `needs_env`, the variables a worker must have set to take them ("A|B":
either one), and go to pool `max` only, or also to `nexus` when an OpenRouter key is all they need (ROUTER_POOLS).
The reported-configuration arms (REPORTED: a gpt-5-mini memory per video, answered by several models) are listed right after the API tasks, and while PILOT is non-empty only the tasks it names are listed.
The other paid rows (PAID: Ego-R1, EGAgent, EgoRAG, HippoMM and MAGIC-Video with their published models) follow the
reported-configuration arms, for batch10 only, on pool `max` only, behind the same PILOT gate.
"""
import argparse
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, constants, hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(ROOT, "baselines", "common"),
                os.path.join(ROOT, "baselines", "merit", "epic")]
import jobs  # noqa: E402
from earlier import load  # noqa: E402
from prepare_epic import LETTERS, qhash, row_qhash  # noqa: E402

RESULTS = os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results")
BUILD_ORDER = os.path.join(ROOT, "baselines", "merit", "build_order.txt")
# summaries of the answers files read (summary()), one per content hash: <hash>.json
ANSWERS_CACHE = os.path.join(constants.HF_HOME, "coord", "answers")
FAILED = ("", "Error", "Unable to generate answer")   # no answer: earlier.reuse's failures, and WorldMM's own

# (baseline, arm, stage, pools that may run it, bank). A qa task answers its bank on the build's GPU family, so it
# takes the build's pools. Every baseline's m8 needs one GPU of 40 GB or more. m27: MERIT one 80 GB GPU (H100 / H200 /
# GH200) or 2x 48 GB; MAGIC one 80 GB GPU or 3+ GPUs; WorldMM one 80 GB GPU or 2x 48 GB. arm00 (GH200, aarch64) runs
# MERIT only: the images are x86, and only MERIT has a native arm environment there. A worker takes only the
# baselines it was started with (worker.py --baselines), so listing a baseline here runs nothing until a pool opts in.
# A person's one worker over several queues (worker.py --queues: Anubhav's `nexus`, Vatsal's `vatsal`, both on Nexus)
# is listed for both arms: the worker itself claims only the tasks one of its queues fits, so it takes a 27B only
# through an 80 GB queue (the H200). nexus-tron, nexus-sailon and nexus-scavenger are the per-queue workers `nexus`
# replaces; they stay listed until their last claims are finished.
M8 = ["max", "pulkit", "nexus", "nexus-tron", "nexus-sailon", "nexus-scavenger", "wriva", "vatsal", "zaratan"]   # zaratan: H100 jobs
M27 = ["wriva", "zaratan", "max", "pulkit", "nexus", "vatsal"]
POOLS = {("merit", "m27"): M27 + ["arm00"], ("merit", "m8"): M8,
         ("magic", "m27"): M27, ("magic", "m8"): M8,
         ("worldmm", "m27"): M27, ("worldmm", "m8"): M8}
# the banks every baseline and arm answers, in priority order: batch10 and epic-morph tranches 2 to 5
# (v3/batch10.txt, v3/tranche2.txt ... v3/tranche5.txt), disjoint by video, then two second banks of videos that
# already have one: the episode rescue (v3/rescue.txt) and the extra templates (v3/extra.txt; export_bank.py leaves
# out what the video's first bank asks). A video's answers under one bank are reused by another (hf_results.py
# earlier-answers), so a question is never answered twice. A bank a baseline has already answered (MERIT's batch10)
# lists nothing while its answers are current: they are in the dataset
BANKS = ["batch10_agree2", "t2_agree2", "t3_agree2", "t4_agree2", "t5_agree2", "rescue_agree2", "extra_agree2"]
# Closed-source API models (baselines/api): no GPU and no memory, one qa task per video and config. They need the
# provider's key (GEMINI_API_KEY / OPENAI_API_KEY in the worker's environment), which only Max's machine has, so no
# other pool may take them. Listed first: 30 short tasks that should not wait behind the memory builds.
API_CONFIGS = ["gemini-3.1-flash-lite-high", "gemini-3.6-flash-high", "gpt-5-f1024"]
# The reported-configuration arms (baselines/<b>/README.md): g5m is the memory written by gpt-5-mini, once per video;
# the g5m-* arms answer from it with another reasoning model and keep answers plus a reference to that memory
# (<video>_memory_ref.json) in their own folders. MEMORY_ARM: the arm whose build an arm's qa waits for (default itself).
MEMORY_ARM = {("worldmm", "g5m-q8"): "g5m", ("worldmm", "g5m-g5"): "g5m",
              ("merit", "g5m-q8"): "g5m", ("merit", "g5m-q27"): "g5m", ("merit", "g5m-g5"): "g5m"}
# Arms that call a paid API: every task of one carries needs_env (a worker with any of them unset leaves the task) and
# is listed for KEY_POOLS only, whatever its EXPERIMENTS line or --only-pools says: nobody but Max has the keys.
# WorldMM's answer arms both call gpt-5-mini (its retriever); MERIT's Qwen answer arms (g5m-q8, g5m-q27) call nothing
# at answer time, so they run on the GPU pools of m8 / m27 once the memory is in the dataset.
KEY_POOLS = ["max", "pulkit"]   # Max and Pulkit (NVIDIA) run with their own API keys
# A name of the form "A|B" is met by either variable. MERIT's and WorldMM's gpt-5-mini calls go to OpenAI with
# OPENAI_API_KEY, or through OpenRouter with OPENROUTER_API_KEY alone (each README, "Through OpenRouter").
# A task whose every need an OpenRouter key meets is also offered to our own Nexus worker (ROUTER_POOLS), which has
# that key and no other; the worker still takes it only with the key in its environment. The gpt-5 answer arms
# (g5m-g5) stay with Max and OpenAI's key (Anubhav, 2026-11-14), although the code could route them too.
GPT = "OPENAI_API_KEY|OPENROUTER_API_KEY"
ROUTER_POOLS = KEY_POOLS + ["nexus"]
NEEDS_ENV = {("api", c): ["GEMINI_API_KEY" if c.startswith("gemini") else "OPENAI_API_KEY"] for c in API_CONFIGS} | {
    ("worldmm", "g5m"): [GPT], ("worldmm", "g5m-q8"): [GPT], ("worldmm", "g5m-g5"): ["OPENAI_API_KEY"],
    ("merit", "g5m"): [GPT], ("merit", "g5m-g5"): ["OPENAI_API_KEY"]}
# OpenAI shuts gpt-5 and gpt-5-mini down on 2026-12-11, and a built memory can be reused afterwards: the g5m memories
# are built for ALL videos, as every memory is (desired: build_order.txt plus the batch10 videos it lacks, batch10's
# first, then each bank's of BANKS in order, then the rest); the answer arms answer every one of these banks.
REPORTED_BANKS = BANKS
REPORTED = [("worldmm", "g5m", "build", KEY_POOLS, None), ("merit", "g5m", "build", KEY_POOLS, None)] + [
    (b, arm, "qa", pools, bank) for bank in REPORTED_BANKS for b, arm, pools in (
        ("worldmm", "g5m-q8", KEY_POOLS), ("worldmm", "g5m-g5", KEY_POOLS),
        ("merit", "g5m-q8", M8), ("merit", "g5m-q27", POOLS["merit", "m27"]), ("merit", "g5m-g5", KEY_POOLS))]
# The pilot gate: while PILOT is non-empty, these are the only tasks of the gated arms (GATED below: the
# reported-configuration arms and the paid rows) that are listed; all their other tasks are withheld from tasks.json
# (the API tasks and the local-model tasks are not affected). Empty it (a one-line PR) once the pilot's token usage has
# been read, and the rest is listed.
PILOT = ["worldmm-g5m-build-P05-20240425-201444"]
# The other paid rows of the sheet, each baseline with the models it was published with (baselines/<b>/README.md).
# Decided by Anubhav: batch10 only (a build line with a bank builds only that bank's videos: 10 memories, no tranche,
# no 156-video builds), and run by Max: every line is for KEY_POOLS, also a stage that calls no API (HippoMM's build,
# local on a 48 GB GPU), which gets no needs_env but still goes nowhere else. In this order: the rows that use
# Gemini 2.5 Pro first (Ego-R1, EGAgent's Gemini arm), since Google stops it on 2026-12-01.
#   egor1    gemini25pro: Gemini 2.5 Pro captions + GPT-4.1 summaries (build); agent tools GPT-4o / Gemini, answer GPT-4.1 (qa)
#   egagent  one GPT-4.1 memory, built under gpt41; gemini25pro's qa takes it from the dataset (MEMORY_ARM) and keeps a
#            copy with its answers. It names OPENAI_API_KEY too: without the fetched memory it would have to build it
#   egorag   egogpt_gpt4o: local EgoGPT captions (48 GB GPU), then GPT-4o and DeepSeek summaries (build); GPT-4o (qa)
#   hippomm  q25vl7b-gpt4o: the build is local (Qwen2.5-VL-7B, ImageBind, Whisper); qa reasons with gpt-4o
#   magic    oss120 is the memory, oss120-flash / oss120-q27 answer from it: every stage calls OpenRouter (row q27 its
#            search loop), and oss120-q27 also serves Qwen3.8-27B, so it needs one of Max's 80 GB GPUs
# Stages the board does not run (worker.py runs build and qa, then upload): Ego-R1's `firstturn` and EGAgent's `check`
# (checks without an API), HippoMM's `count` (a cost count, part of `all`), and the paid `blind` floors.
PAID_BANK = "batch10_agree2"
PAID = [(b, arm, stage, KEY_POOLS, PAID_BANK) for b, arm, stage in (
    ("egor1", "gemini25pro", "build"), ("egor1", "gemini25pro", "qa"),
    ("egagent", "gpt41", "build"), ("egagent", "gemini25pro", "qa"), ("egagent", "gpt41", "qa"),
    ("egorag", "egogpt_gpt4o", "build"), ("egorag", "egogpt_gpt4o", "qa"),
    ("hippomm", "q25vl7b-gpt4o", "build"), ("hippomm", "q25vl7b-gpt4o", "qa"),
    ("magic", "oss120", "build"), ("magic", "oss120-flash", "qa"), ("magic", "oss120-q27", "qa"))]
MEMORY_ARM |= {("egagent", "gemini25pro"): "gpt41", ("magic", "oss120-flash"): "oss120", ("magic", "oss120-q27"): "oss120"}
# what each of their stages needs: a key (baseline, arm, stage) where the stages differ, else (baseline, arm)
NEEDS_ENV |= {
    ("egor1", "gemini25pro"): ["GEMINI_API_KEY", "OPENAI_API_KEY"],
    ("egagent", "gpt41"): ["OPENAI_API_KEY"], ("egagent", "gemini25pro"): ["GEMINI_API_KEY", "OPENAI_API_KEY"],
    ("egorag", "egogpt_gpt4o", "build"): ["OPENAI_API_KEY", "DEEPSEEK_API_KEY"], ("egorag", "egogpt_gpt4o", "qa"): ["OPENAI_API_KEY"],
    ("hippomm", "q25vl7b-gpt4o", "qa"): ["OPENAI_API_KEY"],
    ("magic", "oss120"): ["OPENROUTER_API_KEY"], ("magic", "oss120-flash"): ["OPENROUTER_API_KEY"],
    ("magic", "oss120-q27"): ["OPENROUTER_API_KEY"]}
# The arms behind the pilot gate: every arm of the REPORTED and PAID lines. A new paid baseline joins the gate by
# being listed in one of them, not by its name.
GATED = {(b, arm) for b, arm, *_ in REPORTED + PAID}
# The plain video model (baselines/direct, no memory): qa lines only, every bank, an arm per model and frame budget N.
# Nothing is built, so a task waits on nothing and is done when its answers_<bank>.json.gz is in the dataset. It runs
# in MERIT's image of the model's environment, so the pools are those of that arm: the 8B fits a 48 GB GPU at every N
# (measured peak 39.2 GB at N = 768); the 27B needs what MERIT's m27 does at N = 32 and 128 (2x 48 GB or one 80 GB GPU).
# Its N = 768 is not sized yet (75k video tokens), so only the pools with an H200 (141 GB) queue get it. Listed after
# the paid rows and before the memory arms: short tasks that should not wait behind the memory builds.
DIRECT = {"qwen3vl-8b-f32": M8, "qwen3vl-8b-f128": M8, "qwen3vl-8b-f768": M8,
          "qwen38-27b-f32": M27, "qwen38-27b-f128": M27, "qwen38-27b-f768": ["nexus", "vatsal"],
          # Qwen3.5-9B and Molmo2-8B fit a 48 GB GPU like the 8B (the 9B runs in the m27 image); Qwen3.5-35B-A3B needs
          # the 27B pools' GPUs, and its N = 768 is not sized, so it is not listed
          "qwen35-9b-f32": M8, "qwen35-9b-f128": M8, "qwen35-9b-f768": M8,
          "molmo2-8b-f32": M8, "molmo2-8b-f128": M8, "molmo2-8b-f384": M8,
          "qwen35-35b-a3b-f32": M27, "qwen35-35b-a3b-f128": M27}
EXPERIMENTS = [("api", c, "qa", KEY_POOLS, "batch10_agree2") for c in API_CONFIGS] + REPORTED + PAID + [
    ("direct", arm, "qa", pools, bank) for bank in BANKS for arm, pools in DIRECT.items()] + [
    (b, arm, stage, POOLS[b, arm], bank)
    for b in ("merit", "magic", "worldmm") for stage in ("build", "qa") for arm in ("m27", "m8")
    for bank in (BANKS if stage == "qa" else [None])]
# the file whose presence in <baseline>/<arm>/<video>/ (next to runs.jsonl) means the memory is built: the last
# one each baseline writes ({v}: the video)
MEMORY_DONE = {"merit": "{v}_30sec_4_key.json", "magic": "graph.pkl", "worldmm": "{v}_visual.pkl",
               "egor1": "{v}_1day.json", "egagent": "{v}_graph.json", "egorag": "{v}_l3_day.json",
               "hippomm": "{v}_memory.json"}
# a build that starts from another baseline's memory waits for that build while it is listed: MAGIC's 30 s captions
# are MERIT's (same backbone), fetched from the dataset; without the wait it would caption the video itself
BUILD_ON = {"magic": "merit"}


def bank_questions(bank):
    """{video: its questions} of the epic-morph bank, for the videos with at least one."""
    epic = os.environ.get("EPIC_MORPH_DIR")
    if not epic:
        raise SystemExit(f"set EPIC_MORPH_DIR to an epic-morph checkout with bank {bank} (qa tasks need it)")
    return exported(epic, bank)


def bank_videos(bank):
    """Videos with at least one question in the epic-morph bank."""
    return list(bank_questions(bank))


@functools.lru_cache(maxsize=None)   # one export per bank and run, however many experiments answer it
def exported(epic, bank):
    d = os.path.join(epic, "outputs", "v3", "banks", bank)
    # banks are generated, not committed: regenerate every time, so an interrupted export or an older commit's
    # files never stand in for the checkout's bank (a few seconds)
    subprocess.run([sys.executable, "v3/export_bank.py", bank], cwd=epic, check=True, stdout=subprocess.DEVNULL)
    out = {}
    for f in sorted(os.listdir(d)):
        qs = json.load(open(os.path.join(d, f)))["questions"] if f.endswith("_questions.json") else []
        if qs:
            out[f[:-len("_questions.json")]] = qs
    return out


def asked(q):
    """(key, qhash of wording and options, gold letter, t_ask or None at the end) of a bank question."""
    mcq = q["gemini_mcq"]
    return (q["key"], qhash(q["q"], mcq["options"]), LETTERS[mcq["answer_index"]],
            float(q["t_ask"]) if q.get("streaming") and "t_ask" in q else None)


def ask_time(r):
    """The second an answer row says its question was asked at: its t_ask (API, paid rows), else MERIT's query_time
    {date, time: HHMMSSff} or MAGIC / WorldMM's int DHHMMSSff; None when it records neither."""
    if isinstance(r.get("t_ask"), (int, float)):
        return float(r["t_ask"])
    qt = r.get("query_time")
    qt = str(qt.get("time") if isinstance(qt, dict) else qt)
    if not qt.isdigit():
        return None
    qt = qt.zfill(8)[-8:]
    return int(qt[:2]) * 3600 + int(qt[2:4]) * 60 + int(qt[4:6])


def summary(rows):
    """[key, qhash, gold letter, ask time] of each row of an answers file that has an answer (response, or EGAgent's
    pred, not one of FAILED); a field the row does not record is None."""
    out = []
    for r in rows:
        if not isinstance(r, dict) or str(r.get("response", r.get("pred")) or "").strip() in FAILED:
            continue
        h = r.get("qhash")
        if not h and "question" in r and ("choices" in r or "choice_a" in r):
            h = row_qhash(r)
        elif not h and "question" in r and "options" in r:
            h = qhash(r["question"], r["options"])
        out.append([r.get("key"), h, r.get("answer"), ask_time(r)])
    return out


def unanswered(questions, rows):
    """Keys of the bank questions that no answer (summary() rows; None: the file did not parse) answers as the bank asks
    them now, by the reuse rule (hf_results.py earlier-answers, prepare_epic.match_answers): the same key (a row from
    before keys: the same wording, options and gold letter), wording and options, gold letter, and query time. A
    baseline asks at a whole second, floor(t_ask) since #16 and ceil(t_ask) before (MERIT's batch10), or at the video's
    end, ceil(duration), for a t_ask in its last second: a row within a second of t_ask counts. A field the row does
    not record is not compared."""
    out = []
    for q in questions:
        k, h, gold, t = asked(q)
        if not any((rk == k or (rk is None and rh == h)) and rh in (h, None) and rg in (gold, None)
                   and (rt is None or t is None or abs(rt - t) < 1) for rk, rh, rg, rt in rows or []):
            out.append(k)
    return out


def redo_id(tid, questions):
    """The id of a task listed again for stale answers: the plain id may be done on a pool, which holds it for good."""
    return f"{tid}-redo-{hashlib.sha1(json.dumps(sorted(map(asked, questions))).encode()).hexdigest()[:8]}"


def listing(api):
    """(commit, {path: content hash}) of the results dataset's files at its current commit."""
    commit = api.dataset_info(RESULTS).sha
    files = {f.path: f.lfs.sha256 if f.lfs else f.blob_id
             for f in api.list_repo_tree(RESULTS, repo_type="dataset", recursive=True, revision=commit)
             if hasattr(f, "blob_id")}   # files, not folders
    return commit, files


def built(api):
    """({(baseline, arm, video)} whose build is recorded in the results dataset, {path: content hash} of its files)."""
    return built_from(listing(api)[1])


def read_answers(commit, files):
    """{path: summary()} of every answers_<bank>.json.gz of the qa banks in the dataset (files: listing()), None for a file
    that does not parse. A summary is cached by the file's content hash, so only new or changed files download (to a
    temporary folder, not the Hub cache: MERIT's answers are ~2 MB a video)."""
    names = {f"answers_{bank}.json.gz" for _, _, stage, _, bank in EXPERIMENTS if stage == "qa"}
    paths = [p for p in files if p.count("/") == 3 and p.rsplit("/", 1)[1] in names]
    os.makedirs(ANSWERS_CACHE, exist_ok=True)
    cached = lambda p: os.path.join(ANSWERS_CACHE, f"{files[p]}.json")

    def fetch(p):
        tmp = tempfile.mkdtemp()
        try:
            f = hf_hub_download(RESULTS, p, repo_type="dataset", revision=commit, local_dir=tmp)
            try:
                s = summary(load(f))
            except Exception:   # not gzip, not JSON: unreadable, and it stays so (the hash is its content)
                s = None
            json.dump(s, open(cached(p) + ".tmp", "w"))
            os.replace(cached(p) + ".tmp", cached(p))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    with ThreadPoolExecutor(8) as pool:   # one download per content hash (two identical files share a summary)
        list(pool.map(fetch, {files[p]: p for p in paths if not os.path.exists(cached(p))}.values()))
    return {p: json.load(open(cached(p))) for p in paths}


def built_from(files):
    """A folder with runs.jsonl and its baseline's last memory file counts as built."""
    out = set()
    for f in files:
        parts = f.split("/")
        if len(parts) == 4 and parts[3] == "runs.jsonl":
            out.add(tuple(parts[:3]))
    return {k for k in out if k[0] in MEMORY_DONE and f"{k[0]}/{k[1]}/{k[2]}/{MEMORY_DONE[k[0]].format(v=k[2])}" in files}, files


def reported(t):
    """A task behind the pilot gate: one of a reported-configuration or paid-row arm (GATED), not baselines/api."""
    return (t["baseline"], t["arm"]) in GATED


def needs(b, arm, stage):
    """The environment variables a task of this arm and stage needs (NEEDS_ENV), or None."""
    return NEEDS_ENV.get((b, arm, stage), NEEDS_ENV.get((b, arm)))


def key_pools(need):
    """The pools a task with these needs is listed for: ROUTER_POOLS when an OpenRouter key meets all of them."""
    return ROUTER_POOLS if all("OPENROUTER_API_KEY" in n.split("|") for n in need) else KEY_POOLS


def desired(have, files, answers=None, busy=()):
    """The tasks: builds missing from `have`, and qa where `files` (the dataset's paths) lacks the answers file or its
    summary in `answers` (read_answers(); a file without one is not checked) leaves a question unanswered, unless the
    task (its plain id) is `busy`: it or a redo of it running, or waiting for the relay (what it uploads may be current)."""
    answers = answers or {}
    order = [v.strip() for v in open(BUILD_ORDER) if v.strip()]
    # every memory is built for all videos: each bank's (batch10's first: build_order.txt lacks them, as MERIT built
    # them before the board), then the rest of the build order; the reported arms are also ranked by that group
    videos, group = [], {}
    for g, vs in enumerate([bank_videos(bank) for bank in REPORTED_BANKS] + [order]):
        for v in vs:
            if group.setdefault(v, g) == g and v not in videos:
                videos.append(v)
    tasks, ids = [], set()
    for b, arm, stage, pools, bank in EXPERIMENTS:
        env = {"needs_env": needs(b, arm, stage)} if needs(b, arm, stage) else {}
        pools = key_pools(env["needs_env"]) if env else pools
        mem = MEMORY_ARM.get((b, arm), arm)
        if mem != arm:   # for the worker: the arm whose dataset folder holds this task's memory (worker.py --prefetch)
            env["memory_arm"] = mem
        if stage == "build":
            for v in bank_videos(bank) if bank else videos:   # a build line with a bank: that bank's videos only
                if (b, arm, v) not in have:
                    base = f"{BUILD_ON.get(b)}-{arm}-build-{v}"
                    tasks.append(dict(id=f"{b}-{arm}-build-{v}", baseline=b, arm=arm, stage="build", video=v, bank=None,
                                      pools=pools, deps=[base] if base in ids else [], **env))
                    ids.add(tasks[-1]["id"])
            continue
        for v in bank_videos(bank):
            tid, f, stale = f"{b}-{arm}-{stage}-{bank}-{v}", f"{b}/{arm}/{v}/answers_{bank}.json.gz", {}
            if f in files:
                todo = unanswered(bank_questions(bank)[v], answers[f]) if f in answers and tid not in busy else []
                if not todo:
                    continue
                tid, stale = redo_id(tid, bank_questions(bank)[v]), {"stale": len(todo)}   # questions to ask again
            # wait for the video's build task (its memory arm's) while it is listed; a memory already built (on the
            # board or off it) needs no wait, and with neither, qa builds the memory itself first
            build = f"{b}-{mem}-build-{v}"
            tasks.append(dict(id=tid, baseline=b, arm=arm, stage=stage, video=v, bank=bank,
                              pools=pools, deps=[build] if build in ids else [], **env, **stale))
            ids.add(tasks[-1]["id"])
    # the reported arms by video (batch10: both memories, then its answers; then tranche 2; then the other memories),
    # so the first rows of the sheet do not wait behind 300 builds; every other task keeps its EXPERIMENTS order
    # (the paid rows, batch10 only, keep their EXPERIMENTS order like the rest: after all of these, before the local arms)
    rank = {id(t): i for i, t in enumerate(tasks)}
    by_group = {(b, arm) for b, arm, *_ in REPORTED}
    first = min([rank[id(t)] for t in tasks if (t["baseline"], t["arm"]) in by_group], default=0)
    tasks.sort(key=lambda t: (first, group[t["video"]], rank[id(t)]) if (t["baseline"], t["arm"]) in by_group
               else (rank[id(t)], 0, 0))
    for i, t in enumerate(tasks):
        t["priority"] = i
    return tasks


def gate(tasks):
    """(tasks to list, how many PILOT withholds): of the gated arms (GATED) only PILOT's tasks, while it is non-empty."""
    keep = [t for t in tasks if not PILOT or not reported(t) or t["id"] in PILOT]
    return keep, len(tasks) - len(keep)


def restrict(tasks, only_pools):
    """--only-pools (a test): every task to these pools, except that a task needing a key, or listed for the key
    pool only, never leaves its own."""
    for t in tasks:
        own = t.get("needs_env") or t["pools"] == KEY_POOLS   # (a keyless stage of a paid row is Max's too)
        t["pools"] = [p for p in only_pools if p in t["pools"]] if own else only_pools


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["publish", "view", "reset"])
    ap.add_argument("--task", nargs="*", default=[], help="reset: task ids whose failed claims to clear (after fixing the cause)")
    ap.add_argument("--first", nargs="*", default=[], help="task video ids to put at the top (e.g. a test video)")
    ap.add_argument("--only-pools", nargs="*", help="publish: restrict every task to these pools (for a test)")
    ap.add_argument("--dry-run", action="store_true", help="publish: print the task counts, commit nothing")
    a = ap.parse_args()
    first = os.path.join(os.path.dirname(os.path.abspath(__file__)), "first_videos.txt")
    if not a.first and os.path.exists(first):   # videos to put at the top (e.g. a collaborator's test)
        a.first = open(first).read().split()
    api = HfApi()
    commit, files = listing(api)
    have, _ = built_from(files)
    head, old, pools = jobs.read()
    if a.action == "reset":   # clear failed claims so the task can be claimed again
        files = {}
        for name, p in pools.items():
            gone = [t for t in a.task if p.get("claims", {}).get(t, {}).get("status") == "failed"]
            for t in gone:
                del p["claims"][t]
            if gone:
                files[f"pools/{name}.json"] = p
        ok = files and jobs.commit(files, head, f"master: reset {', '.join(a.task)}")
        print(f"reset {a.task} in {sorted(files)}" if ok else "nothing reset (or board moved; rerun)")
    elif a.action == "publish":
        held = jobs.taken(pools)
        # running or waiting for the relay, by plain id: a task's first run or a redo of it
        busy = {tid.split("-redo-")[0] for tid, (_, c) in held.items() if c["status"] != "done"}
        tasks, withheld = gate(desired(have, files, read_answers(commit, files), busy))
        if a.first:
            # the API tasks stay on top (they are few, short and only Max's pool can take them), then the reported
            # arms in their own order (batch10 first), then the first videos
            tasks.sort(key=lambda t: (0 if t["baseline"] == "api" else 1 if reported(t) else 2,
                                      not reported(t) and t["video"] not in a.first, t["priority"]))
        for i, t in enumerate(tasks):
            t["priority"] = i
        if a.only_pools is not None:
            restrict(tasks, a.only_pools)
        if a.dry_run:
            n, wait, redo, qs = Counter(), Counter(), Counter(), Counter()
            for t in tasks:
                k = (t["baseline"], t["arm"], t["stage"], t["bank"] or "-")
                n[k], wait[k], redo[k], qs[k] = n[k] + 1, wait[k] + bool(t["deps"]), redo[k] + ("stale" in t), qs[k] + t.get("stale", 0)
            for k, c in sorted(n.items()):
                print(f"{' '.join(k):30} {c:4} tasks ({wait[k]} waiting on a build"
                      + (f"; {redo[k]} listed again for stale answers, {qs[k]} questions to ask again)" if redo[k] else ")"))
            print(f"stale answers: {sum(redo.values())} tasks listed again, {sum(qs.values())} questions to ask again; "
                  f"{sum(held.get(t['id'], (0, {}))[1].get('status') == 'done' for t in tasks if 'stale' in t)} of them "
                  "done already and still stale (the worker's bank is older than the master's, or the questions failed again)")
            print(f"pilot gate: {withheld} tasks of the reported-configuration and paid-row arms withheld"
                  + (f"; listed of them: {[t['id'] for t in tasks if reported(t)]}" if PILOT else " (PILOT is empty)"))
            print(f"need a key (pool {', '.join(KEY_POOLS)} only; nexus too where an OpenRouter key is enough): "
                  f"{sum(1 for t in tasks if t.get('needs_env'))} tasks")
            print(f"dry run, nothing committed: {len(tasks)} tasks, {len(have)} memories already built; "
                  f"first: {[t['id'] for t in tasks[:5]]}")
            return
        ok = jobs.commit({"tasks.json": tasks, "master.json": {"updated": jobs.iso(), "tasks": len(tasks), "built": len(have)}},
                         head, f"master: {len(tasks)} tasks")
        print(f"{'published' if ok else 'board moved; rerun'}: {len(tasks)} tasks, {len(have)} memories already built")
    else:
        t = jobs.now()
        rows = []
        for name, p in pools.items():
            for tid, c in p.get("claims", {}).items():
                live = c["status"] == "running" and c.get("lease_until", 0) > t
                state = "running" if live else ("lease lapsed" if c["status"] == "running" else c["status"])
                rows.append(dict(pool=name, host=p.get("host", ""), task=tid, state=state, gpu=c.get("gpu", ""),
                                 started=c.get("started", ""), finished=c.get("finished", ""), seconds=c.get("seconds"),
                                 note=(c.get("note") or "")[-200:]))
        heart = {n: dict(updated=p.get("updated"), running=p.get("running"), slots=p.get("slots"), host=p.get("host"))
                 for n, p in pools.items()}
        print(json.dumps(dict(claims=rows, pools=heart, built=sorted(have), tasks=len(old))))


if __name__ == "__main__":
    main()
