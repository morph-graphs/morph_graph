#!/usr/bin/env python3
"""The job board's master: works out what should exist, subtracts what does, and lists the rest as tasks.

    python coord/master.py publish     # rewrite tasks.json in the jobs dataset (one commit)
    python coord/master.py publish --dry-run   # the same tasks, counted and printed; nothing is committed
    python coord/master.py view        # print the board's view as JSON (pools, running, built), for the dashboard

Desired state (EXPERIMENTS below) x videos, minus memories already in the results dataset, minus nothing
else: claims and leases are the workers' business. A task's `pools` says which pools may take it; the order
of tasks.json is the priority (`priority`: the index), and `rank` says where a task stands for each pool that may take
it (rank()). Reassignment is automatic: a claim whose lease lapsed is free again, and a
task that failed twice is held back and shown as a failure.

`qa` experiments answer a named epic-morph bank (one experiment per bank in BANKS): a task per video that has
questions in the bank and no answers_<bank>.json.gz in the results dataset, or one that is stale, depending on the same
video's build task while that is pending. A file is stale when some question of the bank has no answer in it as the
bank asks it now (unanswered(): the reuse rule's key, wording, options, gold letter and query time), e.g. after the bank
moved a question's t_ask, or when it holds an "Error" or does not parse, or (TRACED baselines) when an answer row
has no `trace`. The banks are the _end variants (BANKS: only the questions asked at the video's end), and the answers
in the base bank's answers_<base>.json.gz count for them too: a video they already cover is not listed. A stale file's task gets the id
<task>-redo-<hash of the video's questions>, since the plain id may already be done on a pool (held for good): it is
listed once per state of the bank, so an answer that fails again is not retried every cycle, and not while the task
or a redo of it still runs or waits for its relay (what it uploads may be current). The master reads the
answers files to check, through a cache of their summaries by content hash (only files that changed download).
The bank is generated afresh each cycle (v3/export_bank.py <bank>, deterministic per commit) in the epic-morph
checkout EPIC_MORPH_DIR, the master's own (not a worker's, which may be reading its bank files). publish stops
without EPIC_MORPH_DIR: a tasks.json without the qa tasks would drop them from the board.

Arms that call a paid API (NEEDS_ENV) carry `needs_env`, the variables a worker must have set to take them ("A|B":
either one), and go to pools `max` and `pulkit` only (ROUTER_POOLS once also took `nexus` for OpenRouter-only arms).
The reported-configuration arms (REPORTED: a gpt-5-mini memory per video, answered by several models) are listed right after the API tasks, and while PILOT is non-empty only the tasks it names are listed.
The other paid rows (PAID: Ego-R1, EGAgent, EgoRAG, HippoMM and MAGIC-Video, with the models Max's and Pulkit's
gateway serves) follow the reported-configuration arms, for every bank of BANKS, on pools `max` and `pulkit` only,
behind the same PILOT gate (MAGIC-Video's OpenRouter arms also on `nexus`: ALSO_POOLS).
While a task of a PRIORITY arm is listed, the tasks of every other arm are withheld (gate()).
The lines of FIRST lead the list, ahead of the API tasks: MAGIC-Video's tasks, then MERIT's with Qwen3.8-27B.
The stitched videos' memory builds (STITCHED): one video's (CANARY) ahead of everything, the others after the URGENT lines.
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
# summaries of the answers files read (summary()), one per content hash: <hash>.json. "answers-v2": since its rows
# record whether the answer row carries a `trace`; a v1 summary (without that) would hide a file that needs a redo
ANSWERS_CACHE = os.path.join(constants.HF_HOME, "coord", "answers-v2")
FAILED = ("", "Error", "Unable to generate answer")   # no answer: earlier.reuse's failures, and WorldMM's own
# baselines whose answer rows record how they walked their memory (`trace`): a row without one is answered again
TRACED = {"magic", "worldmm", "egagent", "m3agent"}

# (baseline, arm, stage, pools that may run it, bank; a build line: None, or the banks whose videos it builds). A qa task answers its bank on the build's GPU family, so it
# takes the build's pools. Every baseline's m8 needs one GPU of 40 GB or more. m27: MERIT one 80 GB GPU (H100 / H200 /
# GH200) or 2x 48 GB; MAGIC one 80 GB GPU or 3+ GPUs; WorldMM one 80 GB GPU or 2x 48 GB. arm00 (GH200, aarch64) runs
# MERIT only: the images are x86, and only MERIT has a native arm environment there. A worker takes only the
# baselines it was started with (worker.py --baselines), so listing a baseline here runs nothing until a pool opts in.
# A person's one worker over several queues (worker.py --queues: Anubhav's `nexus`, Vatsal's `vatsal`, both on Nexus)
# is listed for both arms: the worker itself claims only the tasks one of its queues fits, so it takes a 27B only
# through an 80 GB queue (the H200). nexus-tron, nexus-sailon and nexus-scavenger are the per-queue workers `nexus`
# replaces.
# Since 2026-11-14 (Anubhav) the Qwen work leaves Nexus, to free it for other research: Max and Pulkit (NVIDIA,
# GB200/GB300) take it first, Zaratan, wriva and arm00 help; Molmo2-8B (a public download) goes the same way. Claims
# already running on Nexus finish: a worker renews its claims whatever the task's pools say.
# Leftovers (Anubhav, 2026-11-14): Nexus should not sit idle, so a task no pool has held for LEFTOVER_H hours (since it
# was listed, or since a claim on it ended) is offered to LEFTOVER_POOLS too (our worker and Vatsal's, each over all of
# its queues); new work goes to the pools below first. Not a keyed task, nor one listed for KEY_POOLS only (Max's and
# Pulkit's paid rows).
LEFTOVER_POOLS = ["nexus", "vatsal"]
LEFTOVER_H = 2
M8 = ["max", "pulkit", "wriva", "zaratan"]   # zaratan: H100 jobs
M27 = ["wriva", "zaratan", "max", "pulkit"]
POOLS = {("merit", "m27"): M27 + ["arm00"], ("merit", "m8"): M8,
         ("magic", "m27"): M27, ("magic", "m8"): M8,
         ("worldmm", "m27"): M27, ("worldmm", "m8"): M8}
# the banks every baseline and arm answers, in priority order: batch10 and epic-morph tranches 2 to 5
# (v3/batch10.txt, v3/tranche2.txt ... v3/tranche5.txt), disjoint by video, then two second banks of videos that
# already have one: the episode rescue (v3/rescue.txt) and the extra templates (v3/extra.txt; export_bank.py leaves
# out what the video's first bank asks). A video's answers under one bank are reused by another (hf_results.py
# earlier-answers), so a question is never answered twice. A bank a baseline has already answered (MERIT's batch10)
# lists nothing while its answers are current: they are in the dataset
# End of the video only (Anubhav, 2026-11-15, every baseline): each bank's _end variant (epic-morph export_bank.py
# <bank>_end: the same bank, only its questions asked at the video's end, ask_at "end"), never the mid-video ones
# (ask_at "t"): memories are built over the whole video, so answers asked mid-video are not comparable. An _end bank's
# questions are its base bank's (base_bank()), so the answers already in answers_<base>.json.gz count for it (desired)
BANKS = [f"{b}_end" for b in ("batch10_agree2", "t2_agree2", "t3_agree2", "t4_agree2", "t5_agree2", "rescue_agree2",
                              "extra_agree2")]
# Closed-source API models (baselines/api): no GPU and no memory, one qa task per video and config. They need the
# provider's key (GEMINI_API_KEY / OPENAI_API_KEY in the worker's environment), which only Max's machine has, so no
# other pool may take them. Listed first: short tasks that should not wait behind the memory builds. Every bank since
# 2026-11-14 (Anubhav; batch10 only before).
# GPT-5 stand-ins (Anubhav, 2026-11-15): no pool reaches gpt-5 through an OpenAI key (Max's and Pulkit's gateway serves
# it from Azure only, 50 images a request), and the gateway has GPT-5.1 to 5.5. The GPT row is GPT-5.5 from 384 frames
# (gpt-5.5-f384: the newest, a model row), and MERIT's and WorldMM's GPT-5 answer rows are answered by GPT-5.1
# (g5m-g51: the nearest to the published model). The gpt-5 arms (gpt-5-f1024, g5m-g5) stay in the code, off the board.
# Frame limit (Anubhav, 2026-11-15): through NVIDIA's gateway GPT-5.1 and GPT-5.5 run out of context beyond 384 frames in
# a request, so no row sends more: what would need more (the 1,024-frame row) is cut to 384. MERIT's and WorldMM's
# answer requests stay below it by their own settings (each README).
API_CONFIGS = ["gemini-3.1-flash-lite-high", "gemini-3.6-flash-high", "gpt-5.5-f384"]
# The reported-configuration arms (baselines/<b>/README.md): g5m is the memory written by gpt-5-mini, once per video;
# the g5m-* arms answer from it with another reasoning model and keep answers plus a reference to that memory
# (<video>_memory_ref.json) in their own folders. MEMORY_ARM: the arm whose build an arm's qa waits for (default itself).
MEMORY_ARM = {("worldmm", "g5m-q8"): "g5m", ("worldmm", "g5m-g51"): "g5m",
              ("merit", "g5m-q8"): "g5m", ("merit", "g5m-q27"): "g5m", ("merit", "g5m-g51"): "g5m"}
# Arms that call a paid API: every task of one carries needs_env (a worker with any of them unset leaves the task) and
# is listed for KEY_POOLS only, whatever its EXPERIMENTS line or --only-pools says: nobody but Max has the keys.
# WorldMM's answer arms both call gpt-5-mini (its retriever); MERIT's Qwen answer arms (g5m-q8, g5m-q27) call nothing
# at answer time, so they run on the GPU pools of m8 / m27 once the memory is in the dataset.
KEY_POOLS = ["max", "pulkit"]   # Max and Pulkit (NVIDIA) run with their own API keys
# A name of the form "A|B" is met by either variable. MERIT's and WorldMM's gpt-5-mini calls go to OpenAI with
# OPENAI_API_KEY, or through OpenRouter with OPENROUTER_API_KEY alone (each README, "Through OpenRouter").
# ROUTER_POOLS once also offered a task whose every need an OpenRouter key meets to our own Nexus worker, which has
# that key and no other; since 2026-11-14 it is KEY_POOLS alone (the Qwen work left Nexus). The gpt-5 answer arms
# (g5m-g5, now its stand-in g5m-g51) stay with Max and OpenAI's key (Anubhav, 2026-11-14), although the code could route them too.
GPT = "OPENAI_API_KEY|OPENROUTER_API_KEY"
ROUTER_POOLS = KEY_POOLS   # was + ["nexus"]: the OpenRouter arm left is WorldMM's g5m-q8, a Qwen model, so off Nexus (2026-11-14)
NEEDS_ENV = {("api", c): ["GEMINI_API_KEY" if c.startswith("gemini") else "OPENAI_API_KEY"] for c in API_CONFIGS} | {
    ("worldmm", "g5m"): [GPT], ("worldmm", "g5m-q8"): [GPT], ("worldmm", "g5m-g51"): ["OPENAI_API_KEY"],
    ("merit", "g5m"): [GPT], ("merit", "g5m-g51"): ["OPENAI_API_KEY"]}
# OpenAI shuts gpt-5 and gpt-5-mini down on 2026-12-11, and a built memory can be reused afterwards: the g5m memories
# are built for ALL videos, as every memory is (desired: build_order.txt plus the batch10 videos it lacks, batch10's
# first, then each bank's of BANKS in order, then the rest); the answer arms answer every one of these banks.
REPORTED_BANKS = BANKS
REPORTED = [("worldmm", "g5m", "build", KEY_POOLS, None), ("merit", "g5m", "build", KEY_POOLS, None)] + [
    (b, arm, "qa", pools, bank) for bank in REPORTED_BANKS for b, arm, pools in (
        ("worldmm", "g5m-q8", KEY_POOLS), ("worldmm", "g5m-g51", KEY_POOLS),
        ("merit", "g5m-q8", M8), ("merit", "g5m-q27", POOLS["merit", "m27"]), ("merit", "g5m-g51", KEY_POOLS))]
# The pilot gate: while PILOT is non-empty, these are the only tasks of the gated arms (GATED below: the
# reported-configuration arms and the paid rows) that are listed; all their other tasks are withheld from tasks.json
# (the API tasks and the local-model tasks are not affected). Empty it (a one-line PR) once the pilot's token usage has
# been read, and the rest is listed.
PILOT = []   # emptied 2026-11-14 (Anubhav): the pilot, worldmm-g5m-build-P05-20240425-201444, cost $2.71 through OpenRouter
# Withheld whatever PILOT says: (baseline, arm, stage) lines whose work is running off the board, which a board copy
# would pay for twice. A qa task keeps its wait on such a build until the memory is uploaded. Empty since 2026-11-14:
# WorldMM's gpt-5-mini memories, built by Slurm arrays on Nexus, are in the dataset for all 156 videos.
# The three GPT-5 rows were held here on 2026-11-14 for want of GPT-5 access; since 2026-11-15 they run on stand-ins
# (API_CONFIGS above), so nothing is held.
HELD = set()
# Baselines whose answering is frozen, every arm and bank: their qa tasks are withheld, their memory builds are listed as
# usual, and a qa claim already running finishes. Used on 2026-11-14/15 while the walk through the memory was being
# logged for MAGIC-Video, WorldMM, EGAgent and M3-Agent (#73), and for MAGIC-Video until its time-named searches were
# cut at t_ask (#88: they could return captions from after a question's ask time). Empty since.
QA_FROZEN = set()
# The other paid rows of the sheet (baselines/<b>/README.md). Max and Pulkit run them through their gateway, which has
# GPT-4.1, GPT-5-mini, Gemini 3.6 Flash and DeepSeek but no GPT-4o and no Gemini 2.5 Pro, so each row is listed with a
# stand-in arm it can run (Anubhav, 2026-11-14). The published arms' code stays in each baseline; their lines are off
# the board (egor1 gemini25pro, egagent gemini25pro, egorag egogpt_gpt4o, hippomm q25vl7b-gpt4o's qa).
# Every bank of BANKS (Anubhav, 2026-11-14; batch10 only before): a build line names its banks and builds only their
# videos, the ones with questions, not all 156. Every line is for KEY_POOLS, also a stage that calls no API (HippoMM's
# build, local on a 48 GB GPU), which gets no needs_env but still goes nowhere else.
#   egor1    gemini36flash: Gemini 3.6 Flash captions + GPT-4.1 summaries (build). gemini36flash-gpt41 answers from that
#            memory: video_llm Gemini 3.6 Flash; rag, vlm and the answer GPT-4.1 (qa)
#   egagent  one GPT-4.1 memory, built under gpt41; gemini36flash's qa (a Gemini 3.6 Flash agent) takes that memory
#            from the dataset and keeps a copy with its answers. The GPT-4.1 agent's answer row is off the board
#            (Anubhav, 2026-11-14: reasoning with Gemini 3.6 Flash only, for now)
#   egorag   egogpt_gpt41: local EgoGPT captions (48 GB GPU), then GPT-4.1 and DeepSeek summaries (build); GPT-4.1 (qa)
#   hippomm  the memory is local (Qwen2.5-VL-7B, ImageBind, Whisper), built under the published arm q25vl7b-gpt4o;
#            q25vl7b-gpt41 answers from it with gpt-4.1
#   magic    oss120 is the memory, oss120-flash / oss120-q27 answer from it: every stage calls OpenRouter (row q27 its
#            search loop), and oss120-q27 also serves Qwen3.8-27B, so it needs one of Max's 80 GB GPUs
# Stages the board does not run (worker.py runs build and qa, then upload): Ego-R1's `firstturn` and EGAgent's `check`
# (checks without an API), HippoMM's `count` (a cost count, part of `all`), and the paid `blind` floors.
PAID_BANKS = BANKS
# (baseline, the arm that builds the memory, the arms that answer from it)
PAID_ROWS = [("egor1", "gemini36flash", ["gemini36flash-gpt41"]), ("egagent", "gpt41", ["gemini36flash"]),
             ("egorag", "egogpt_gpt41", ["egogpt_gpt41"]), ("hippomm", "q25vl7b-gpt4o", ["q25vl7b-gpt41"]),
             ("magic", "oss120", ["oss120-flash", "oss120-q27"])]
PAID = [line for b, mem, answer in PAID_ROWS for line in
        [(b, mem, "build", KEY_POOLS, tuple(PAID_BANKS))]
        + [(b, arm, "qa", KEY_POOLS, bank) for bank in PAID_BANKS for arm in answer]]
MEMORY_ARM |= {(b, arm): mem for b, mem, answer in PAID_ROWS for arm in answer if arm != mem}
# what each of their stages needs: a key (baseline, arm, stage) where the stages differ, else (baseline, arm)
NEEDS_ENV |= {
    ("egor1", "gemini36flash"): ["GEMINI_API_KEY", "OPENAI_API_KEY"],
    ("egor1", "gemini36flash-gpt41"): ["GEMINI_API_KEY", "OPENAI_API_KEY"],
    # EGAgent's memory: OpenAI's key, a gateway's, or OpenRouter's (run_egagent.sh "Through OpenRouter")
    ("egagent", "gpt41"): ["OPENAI_API_KEY|OPENROUTER_API_KEY"], ("egagent", "gemini36flash"): ["GEMINI_API_KEY"],
    ("egorag", "egogpt_gpt41", "build"): ["OPENAI_API_KEY", "DEEPSEEK_API_KEY"], ("egorag", "egogpt_gpt41", "qa"): ["OPENAI_API_KEY"],
    ("hippomm", "q25vl7b-gpt41"): ["OPENAI_API_KEY"],
    ("magic", "oss120"): ["OPENROUTER_API_KEY"], ("magic", "oss120-flash"): ["OPENROUTER_API_KEY"],
    ("magic", "oss120-q27"): ["OPENROUTER_API_KEY"]}
# Pools added to an arm's key pools: MAGIC-Video's Qwen3.5-Flash row calls OpenRouter only, a key that Max's and
# Pulkit's gateway does not have and our Nexus worker does (Anubhav, 2026-11-14). Zaratan answers it too (Anubhav,
# 2026-11-15: Nexus GPUs were full, Zaratan idle); its compute nodes reach OpenRouter over IPv6
# EGAgent's GPT-4.1 memory on Zaratan too, through OpenRouter to OpenAI on our key (Anubhav, 2026-11-15): Azure's
# content filter, behind Max's gateway, refused frames of 37 of its builds (ContentPolicyViolationError)
ALSO_POOLS = {("magic", "oss120"): ["nexus", "zaratan"], ("magic", "oss120-flash"): ["nexus", "zaratan"],
              ("egagent", "gpt41"): ["zaratan", "nexus"]}   # through OpenRouter (Nexus: Anubhav, 2026-11-16)
# The arms behind the pilot gate: every arm of the REPORTED and PAID lines. A new paid baseline joins the gate by
# being listed in one of them, not by its name.
GATED = {(b, arm) for b, arm, *_ in REPORTED + PAID}
# The plain video model (baselines/direct, no memory): qa lines only, every bank, an arm per model and frame budget N.
# Nothing is built, so a task waits on nothing and is done when its answers_<bank>.json.gz is in the dataset. It runs
# in MERIT's image of the model's environment, so the pools are those of that arm: the 8B fits a 48 GB GPU at every N
# (measured peak 39.2 GB at N = 768); the 27B needs what MERIT's m27 does at N = 32 and 128 (2x 48 GB or one 80 GB GPU).
# Qwen3-VL-8B and Qwen3.8-27B are run at N = 768 only (Anubhav, 2026-11-14: no 32- and 128-frame rows for them; what
# was answered at those budgets stays in the dataset). The 27B's N = 768 (75k video tokens) is not sized yet. Listed
# after the paid rows and before the memory arms: short tasks that should not wait behind the memory builds.
DIRECT = {"qwen3vl-8b-f768": M8, "qwen38-27b-f768": M27,
          # Qwen3.5-9B and Molmo2-8B fit a 48 GB GPU like the 8B (the 9B runs in the m27 image); Qwen3.5-35B-A3B needs
          # the 27B pools' GPUs, and its N = 768 is not sized, so it is not listed
          "qwen35-35b-a3b-f128": M27}
# Paused (Anubhav, 2026-11-17): Molmo2-8B at 384 frames, so the main benchmark and the evidence clips go first; its answers so
# far stay in the dataset and the sheet (unlike DROPPED_ARMS). Back on: add "molmo2-8b-f384": M8 above.
# Dropped (Anubhav, 2026-11-16): Qwen3.5-9B everywhere, and every 32- and 128-frame row, except Qwen3.5-35B-A3B at
# 128 for now (its 768 is not sized). Their answers stay in the dataset; the results sheet leaves them out (dropped()).
DROPPED_ARMS = {"qwen35-9b-f32", "qwen35-9b-f128", "qwen35-9b-f768", "qwen35-9b-evclips", "molmo2-8b-f32", "molmo2-8b-f128",
                "qwen35-35b-a3b-f32", "qwen3vl-8b-f32", "qwen3vl-8b-f128", "qwen38-27b-f32", "qwen38-27b-f128"}
dropped = lambda b, arm: b == "direct" and arm in DROPPED_ARMS
# ReKV (baselines/rekv, arm m8: Qwen3-VL-8B through the vendored port; it has no m27 arm): qa lines only, every bank.
# Its memory is the video's KV-cache, held in host RAM while qa runs and never stored, so nothing is built: as with the
# plain video model, a task waits on nothing and is done when its answers_<bank>.json.gz is in the dataset. No key.
# One 48 GB GPU and up to 36 GB of host RAM (the cache grows with the video; measured on the longest one), so the
# m8 pools, and a queues file's plain m8 names it too. After the plain video model's short tasks.
# VST (baselines/vst, arm vst7b: the released Catalan258/VST-7B, Qwen2.5-VL-7B weights, a public download; no key): qa
# lines only, every bank. Its memory is its own streaming thoughts, which depend on where the video is cut: they are
# made in qa per question cutoff and kept with the answers (thoughts/), so nothing is built and there is no
# MEMORY_DONE entry, as for direct and ReKV. One GPU with >= 24 GB (measured peak 18.7 GiB) and 64 GB of host RAM
# (measured peak 54.4 GiB: the loader holds the sampled source frames as floats), so the m8 pools' GPUs fit it; a
# queues file names it vst/vst7b, on a queue with 64G (the plain m8 does not name it: those queues may have 40G).
# M3-Agent (baselines/m3agent, arm m3: the published checkpoints with local ASR and embeddings, no key): its two
# stages need different GPUs, so its lines differ in pools. build (the memory graph, every video) fits one GPU of
# 40 GB or more: the m8 pools. qa serves the 32.8B control model: 2x 48 GB or one 80 GB GPU, the m27 pools (a queues
# file names the stages: m3agent/m3/build, m3agent/m3/qa). Listed last. Its arm m3api calls paid APIs (Gemini speech
# segments, OpenAI embeddings) and is not listed: it would be a paid row (PAID, NEEDS_ENV).
M3AGENT = {"build": M8, "qa": M27}
# Baselines whose answering is never listed without its memory (Anubhav, 2026-11-15): a qa task of one is listed only
# when the video's memory is in the dataset, or its build task is listed too (the qa then waits on it: deps). For the
# other baselines a qa task with neither builds the memory itself first; M3-Agent's qa job is sized for the control
# model (the m27 pools), not for building a memory, so it must never do that.
NEEDS_MEMORY = {"m3agent"}
# Urgent (Anubhav, 2026-11-14): MAGIC-Video's memories (its graph.pkl) are what the structure matching of the memory
# evaluation (M3) needs first, so its builds go right after the API tasks, ahead of everything else, 8B then 27B, and
# to the Nexus pools at once as well (no LEFTOVER_H wait).
# The 27B memories still missing (8 of 156 on 2026-11-15) go to Pulkit first (Anubhav, 2026-11-15: his GB200/GB300
# pool, on priority); after LEFTOVER_H hours unclaimed they are offered to the Nexus pools too, like any task.
# Zaratan takes them too (Anubhav, 2026-11-15), on two H100s: on one 80 GB GPU the 27B server and MAGIC's graph step
# ran out of memory together (2026-11-14), and on two the server spreads over both
URGENT = [("magic", "m8", "build", POOLS["magic", "m8"] + LEFTOVER_POOLS, None),
          ("magic", "m27", "build", ["pulkit", "zaratan"], None)]
# The rows to finish first (Anubhav, 2026-11-14: time and compute are short). While a task of one of these arms is
# listed, the tasks of every other arm are withheld from tasks.json (gate()); a claim already running on a withheld
# task finishes, since a worker renews its claims whatever is listed. Results-sheet rows (tab `Epic Morph Results`):
# 4, 6, 7 the API models; 12, 14 Qwen3-VL-8B and Qwen3.8-27B from 768 frames; 18 VST; 22 Ego-R1; 23 EgoRAG; 24, 25
# MERIT on the gpt-5-mini memory, answered by Qwen 8B and by GPT-5; 34 M3-Agent; 35, 36 WorldMM likewise; 39 EGAgent
# (Gemini 3.6 Flash agent); 40 MAGIC-Video (Qwen3.5-Flash); with the arms that build their memories. The GPT-5 rows
# (7, 25, 36) run on GPT-5.5 / GPT-5.1 stand-ins (API_CONFIGS above). Empty the set to
# list everything again.
PRIORITY = {("api", c) for c in API_CONFIGS} | {
    ("direct", "qwen3vl-8b-f768"), ("direct", "qwen38-27b-f768"), ("vst", "vst7b"),
    ("egor1", "gemini36flash"), ("egor1", "gemini36flash-gpt41"), ("egorag", "egogpt_gpt41"),
    ("merit", "g5m"), ("merit", "g5m-q8"), ("merit", "g5m-g51"), ("m3agent", "m3"),
    ("worldmm", "g5m"), ("worldmm", "g5m-q8"), ("worldmm", "g5m-g51"),
    ("egagent", "gpt41"), ("egagent", "gemini36flash"), ("magic", "oss120"), ("magic", "oss120-flash"),
    ("magic", "m27", "build"),   # a (baseline, arm, stage) entry: that stage only (MAGIC-Video's 27B memories, not its answers)
    ("merit", "m27")}            # MERIT with Qwen3.8-27B: its answers, every bank (Anubhav, 2026-11-15), with the memory they
#                                  wait for (a qa task whose build is withheld would wait for good; all 156 are built)
# What leads the list (Anubhav, 2026-11-15): MAGIC-Video's tasks first, then MERIT with Qwen3.8-27B (its answers; a
# memory still to build comes before them), both ahead of the API tasks and of the other priority rows. An entry is
# (baseline,), (baseline, arm) or (baseline, arm, stage); the list's order is the order of the groups, and inside a
# group the tasks keep the order they had (MAGIC: its 8B and 27B memory builds, then the GPT-OSS-120B memory with its
# answers, video by video, then its 27B and 8B answers). It only orders: which rows are listed at all is PRIORITY's
# and the gates' business.
FIRST = [("magic",), ("merit", "m27"), ("merit", "m8", "qa")]   # MERIT m8 answers on every free A6000 (Anubhav, 2026-11-16)
# ... on Nexus only (Anubhav, 2026-11-16): first in the order must not change what Max, Pulkit, wriva or Zaratan take,
# so these lines name only Nexus, and no leftover pools are added to them
ONLY_POOLS = {("merit", "m8", "qa"): ["nexus"],
              # GPT-4.1 on the evidence clips through our OpenRouter key (#122): Max's jobs cannot read the evidence-clips
              # dataset (404, 39 failures on 2026-11-16) and Pulkit's route takes 50 images at most
              ("api", "gpt-4.1-f384-evclips", "qa"): ["nexus"],
              # Gemini 3.6 Flash and 3.1 Flash-Lite, the benchmark rows and their evidence clips, on our own Gemini key on
              # Nexus's high-memory CPU nodes (Anubhav, 2026-11-17): reserved for us
              ("api", "gemini-3.6-flash-high", "qa"): ["nexus"], ("api", "gemini-3.1-flash-lite-high", "qa"): ["nexus"],
              # ... the evidence-clips ones with Max's pools too, once his token could read the evidence-clips dataset
              # (Anubhav, 2026-11-17; Pulkit's route has no Gemini Files API)
              ("api", "gemini-3.6-flash-high-evclips", "qa"): ["nexus", "max", "max-cpu"],
              ("api", "gemini-3.1-flash-lite-high-evclips", "qa"): ["nexus", "max", "max-cpu"]}
# ... and lines that Nexus takes besides their usual pools (Anubhav, 2026-11-16): MAGIC m27 answers on 2x A6000 at 24G
# or an H200 at 16G, WorldMM m27 answers on an H200 (smoke tests: process memory under 10 GB, answers as before)
EXTRA_POOLS = {("magic", "m27", "qa"): ["nexus"], ("worldmm", "m27", "qa"): ["nexus"], ("merit", "m27", "qa"): ["nexus"]}
EXPERIMENTS = [("api", c, "qa", KEY_POOLS, bank) for bank in BANKS for c in API_CONFIGS] + URGENT + REPORTED + PAID + [
    ("direct", arm, "qa", pools, bank) for bank in BANKS for arm, pools in DIRECT.items()] + [
    ("rekv", "m8", "qa", M8, bank) for bank in BANKS] + [
    ("vst", "vst7b", "qa", M8, bank) for bank in BANKS] + [
    (b, arm, stage, POOLS[b, arm], bank)
    for b in ("merit", "magic", "worldmm") for stage in ("build", "qa") for arm in ("m27", "m8")
    for bank in (BANKS if stage == "qa" else [None]) if (b, stage) != ("magic", "build")] + [
    ("m3agent", "m3", stage, pools, bank) for stage, pools in M3AGENT.items()
    for bank in (BANKS if stage == "qa" else [None])]
# The evidence-clips ablation (Anubhav, 2026-11-15): each reasoning model the methods use answers every question from
# only its evidence clips, stitched into one short video per question (baselines/common/evclips.py; the union of the
# bank's evidence.clips ranges, merged, no padding), shown whole: no memory, nothing built, an upper bound of the model
# given the right evidence. One row per model, not per method; each model takes the video through its own usual path
# and settings: the local models Direct's (<model>-evclips: the stitched video's 1 fps frames, stamped with source
# times), VST its own streaming loop (vst7b-evclips), the paid ones epic_eval's evidence mode with the stitched video as
# the media (baselines/api/configs/*-evclips.json: Gemini the video file, GPT and Qwen3.5-Flash frames of it). A task
# fetches the videos it needs from the dataset (or EVCLIPS_DIR). Local rows take the pools of their Direct arms (the 8B
# class one 48 GB GPU; the 27B and 35B-A3B the 27B pools; VST its m8 pools); the paid ones need their key (NEEDS_ENV:
# Max's and Pulkit's pools), except Qwen3.5-Flash, which calls OpenRouter on our key as MAGIC-Video's oss120-flash does
# (ALSO_POOLS). M3-Agent has no row: its control model (Qwen3-32B, Qwen3ForCausalLM) reads no images.
# GPT-4.1 runs on our Nexus worker too, through OpenRouter pinned to OpenAI (backends/openai.py; Anubhav, 2026-11-16:
# Azure's content filter behind Max's gateway refused EGAgent's frames), and is a PRIORITY row; the other paid
# evidence-clips rows stay behind the gate.
EVCLIPS = [("direct", f"{m}-evclips", pools) for m, pools in (
    ("qwen3vl-8b", M8), ("qwen38-27b", M27), ("molmo2-8b", M8), ("qwen35-35b-a3b", M27))] + [
    ("vst", "vst7b-evclips", M8)] + [("api", c, KEY_POOLS) for c in (
    "gpt-4.1-f384-evclips", "gemini-3.6-flash-high-evclips", "gpt-5.1-f384-evclips", "gemini-3.1-flash-lite-high-evclips",
    "gpt-5.5-f384-evclips", "qwen3.5-flash-f384-evclips")]
NEEDS_ENV |= {("api", c): [k] for c, k in (
    ("gpt-4.1-f384-evclips", GPT), ("gpt-5.1-f384-evclips", "OPENAI_API_KEY"), ("gpt-5.5-f384-evclips", "OPENAI_API_KEY"),
    ("gemini-3.6-flash-high-evclips", "GEMINI_API_KEY"), ("gemini-3.1-flash-lite-high-evclips", "GEMINI_API_KEY"),
    ("qwen3.5-flash-f384-evclips", "OPENROUTER_API_KEY"))}
ALSO_POOLS |= {("api", "qwen3.5-flash-f384-evclips"): ["nexus", "zaratan"], ("api", "gpt-4.1-f384-evclips"): ["nexus"]}
PRIORITY |= {("api", "gpt-4.1-f384-evclips")}
# and the two Gemini rows: the Evid. column of the paper's Table 2 (Anubhav, 2026-11-16 evening); only Qwen3.5-Flash's
# paid evidence-clips row stays behind the gate
PRIORITY |= {("api", "gemini-3.6-flash-high-evclips"), ("api", "gemini-3.1-flash-lite-high-evclips")}
EXPERIMENTS += [(b, arm, "qa", pools, bank) for bank in BANKS for b, arm, pools in EVCLIPS]
# ... not a priority row but for GPT-4.1 (Anubhav, 2026-11-16: back from #103); its local-GPU rows pass the gate anyway (gate())
# the file whose presence in <baseline>/<arm>/<video>/ (next to runs.jsonl) means the memory is built: the last
# one each baseline writes ({v}: the video)
MEMORY_DONE = {"merit": "{v}_30sec_4_key.json", "magic": "graph.pkl", "worldmm": "{v}_visual.pkl",
               "egor1": "{v}_1day.json", "egagent": "{v}_graph.json", "egorag": "{v}_l3_day.json",
               "hippomm": "{v}_memory.json", "m3agent": "{v}_m3graph.pkl"}
# a build that starts from another baseline's memory waits for that build while it is listed: MAGIC's 30 s captions
# are MERIT's (same backbone), fetched from the dataset; without the wait it would caption the video itself
BUILD_ON = {"magic": "merit"}
# The stitched videos (Anubhav, 2026-11-15): one per kitchen, its recording sessions in order with a 4 s date card
# before each (epic-morph v3/stitched, Vatsal; 41.5 h in all), id <kitchen>-stitched, found like any other video at
# <videos>/<kitchen>/<kitchen>-stitched.mp4. A build line for each line of EXPERIMENTS that builds, built on the
# stitched video itself (not put together from its sessions' memories), and qa lines on their end-of-video bank
# (STITCHED_EOV, below). STITCHED is the bank a build line names for them: its videos are
# STITCHED_VIDEOS, not an epic-morph export. A pool runs them only once it has the files (worker.py: a stitched build
# whose video is not there is left), so a task waits rather than fails.
# Pools (Anubhav, 2026-11-15): a build is one job of 9 to 19 h on the longest video, so only the line's pools among
# STITCHED_POOLS take it (Max's and Pulkit's machines have no time limit; Nexus only on its long queue, worker.py), not
# Zaratan, wriva or arm00, and not the leftover pools.
# Order: CANARY's builds first of all, ahead of the FIRST groups, so they are claimed at once and errors show early
# (Anubhav, 2026-11-15: some picked up right now); the others after the URGENT lines (lead()).
STITCHED = "stitched"
STITCHED_VIDEOS = [f"P0{k}-stitched" for k in range(1, 10)]
STITCHED_POOLS = ["max", "pulkit", "nexus"]
# ... and per line (Anubhav, 2026-11-16): MAGIC-Video's stitched builds on Zaratan's H100s too, whichever pool starts first
# (measured on the 156 normal videos: build wall time ~0.21x the video's length, so ~1.5 h on the 7.2 h video)
STITCHED_ALSO = {("magic", "oss120"): ["zaratan"], ("magic", "oss120-flash"): ["zaratan"]}
# Max's second worker, CPU only (pool max-cpu; Anubhav, 2026-11-16): the answers of the paid rows that need no GPU. Two
# workers in one pool overwrite each other's pool file and fail each other's claims, so it is a pool of its own.
CPU_ROWS = {("api", "gemini-3.6-flash-high-stitched-f2048", "qa"), ("api", "gemini-3.6-flash-high", "qa"),
            ("api", "gemini-3.1-flash-lite-high", "qa"), ("egagent", "gemini36flash", "qa"), ("egorag", "egogpt_gpt41", "qa")}
CANARY = "P05-stitched"   # the shortest, 3.48 h
# MERIT's lines first: MAGIC-Video's builds wait for MERIT's of the same video, which must be listed before them
# Only the systems Vatsal's stitched hand-off asks for (Anubhav, 2026-11-16): MERIT with Qwen3.8-27B, WorldMM with the
# GPT-5-mini memory (answers: GPT-5.1 for GPT-5, as on the normal banks), MAGIC-Video's GPT-OSS-120B memory (its answers
# wait, STITCHED_QA_LATER); the other methods' stitched builds and answers are off the board
# MAGIC-Video's stitched memory is built once over the whole video too (Anubhav, 2026-11-16): everything its build writes
# is filtered at a question's t_ask (graph filter_by_time); only its event chains read across time, and run_qa builds
# those per cutoff (prototype on P05-stitched: no time after either anchor in its chains or filtered graph)
# Frozen (Anubhav, 2026-11-17): every stitched build and answer, so the main benchmark and the evidence clips go
# first. Was: {("merit", "m27"), ("worldmm", "g5m"), ("magic", "oss120")}
STITCHED_BUILDS = set()
# Answering on the stitched videos is frozen (Anubhav, 2026-11-16) while their protocol changes to Vatsal's: each card at
# its anchor time, from memory units that end before it; builds go on. The end-of-video lines (STITCHED_EOV, below) were
# {("merit", "m27"), ("worldmm", "g5m-g51")}; they stay off.
STITCHED_EOV_ANSWERS = set()
# Vatsal's protocol (Anubhav, 2026-11-16): his stitched bank (epic-morph outputs/v3/banks/stitched: 630 questions, plain
# wording, streaming, asked at t_ask, mid-video), answered from the memory built once over the whole video (the stitched
# build lines above), each question seeing only the units that END by its t_ask (MERIT's 30 s clips, WorldMM's every
# caption level, semantic snapshot and visual clip: each baseline's README), with the kitchen's sessions started by
# t_ask in its prompt (baselines/common/stitched_sessions.py). The memory arms answer in t_ask order on one memory per
# video; the long-context Gemini 3.6 Flash row (STITCHED_API) gets [0, t_ask] sampled uniformly to its frame budget
# (baselines/api/configs/<STITCHED_API>.json; a kitchen that runs out of context goes to the -f1024 config). Its lines:
# STITCHED_ANSWER_LINES; listed once STITCHED_ANSWERS names them (empty: frozen). The bank's name is STITCHED, as the
# build lines' marker, but a qa line names it as a string and a build line as the tuple (STITCHED,).
# Its files are committed in epic-morph as they are (the same as the results dataset's stitched/banks/), with no
# export_bank.py entry: exported() reads them without generating (COMMITTED_BANKS)
COMMITTED_BANKS = {STITCHED}
STITCHED_API = "gemini-3.6-flash-high-stitched-f2048"
STITCHED_ANSWER_LINES = [("merit", "m27"), ("worldmm", "g5m-g51"), ("api", STITCHED_API),
                         ("magic", "oss120-flash")]   # MAGIC (Anubhav, 2026-11-16): chains per t_ask (#110), from the oss120 memory
# On again for three lines (Anubhav, 2026-11-17): MERIT m27, WorldMM g5m-g51 and Gemini 3.6 Flash, whose memories are
# built (MERIT's and WorldMM's) or need none. MAGIC-Video waits for its chain-length fix (empty topic replies at 4,096
# tokens) and its three missing stitched builds. All on: set(STITCHED_ANSWER_LINES)
STITCHED_ANSWERS = {("merit", "m27"), ("worldmm", "g5m-g51"), ("api", STITCHED_API)}
NEEDS_ENV[("api", STITCHED_API)] = ["GEMINI_API_KEY"]
EXPERIMENTS += sorted([(b, arm, "build", pools, (STITCHED,)) for b, arm, stage, pools, _ in EXPERIMENTS
                       if stage == "build" and (b, arm) in STITCHED_BUILDS],
                      key=lambda e: e[0] not in BUILD_ON.values())
# The stitched videos' end-of-video bank (Anubhav, 2026-11-15; epic-morph v3/export_bank.py stitched_eov: 623 questions,
# each asked at the end of the video with the "As of video time ..." clause naming the moment its answer is as of). A qa
# line for every arm that answers the normal banks from a memory (MEMORY_DONE: not the plain video models, the API rows,
# ReKV or VST, whose 7 h video is another question), on the stitched pools (as the builds), and like NEEDS_MEMORY never
# listed without its stitched memory built or listed to wait on. The CANARY's answers right after its builds, the other
# videos' after their builds (lead()). Not yet (STITCHED_QA_LATER): MAGIC-Video, whose searches that name a time read
# seconds against HHMMSSff stamps (README: they return the video's first minutes), and every question here names a
# time, and whose event-chain step puts every 3 min summary (~144 on the 7.2 h video) in one 4,000-token reply;
# EgoRAG, whose memory and question time are on a clock that starts at 10:00:00, while the clause gives video time.
STITCHED_EOV = "stitched_eov"
STITCHED_QA_LATER = {"magic", "egorag"}
EXPERIMENTS += [(b, arm, "qa", list(pools), STITCHED_EOV) for (b, arm), pools in dict.fromkeys(
    ((b, arm), tuple(pools)) for b, arm, stage, pools, bank in EXPERIMENTS
    if stage == "qa" and b in MEMORY_DONE and b not in STITCHED_QA_LATER and (b, arm) in STITCHED_EOV_ANSWERS)]
# ... and the lines of Vatsal's stitched bank (STITCHED_ANSWER_LINES), after the stitched builds (lead()); a memory arm's
# never listed without its stitched memory built or listed to wait on, as on STITCHED_EOV
EXPERIMENTS += [(b, arm, "qa", KEY_POOLS if b == "api" else list(POOLS.get((b, arm), KEY_POOLS)), STITCHED)
                for b, arm in STITCHED_ANSWER_LINES if (b, arm) in STITCHED_ANSWERS]
PRIORITY |= {(b, arm) for b, arm in STITCHED_ANSWER_LINES if (b, arm) in STITCHED_ANSWERS}
# The M2 fact probes (Anubhav, 2026-11-16; epic-morph v3/export_bank.py probes_v2, design/13_m2_probe_sample.md): 3,844
# probes of a fixed sample of 1,565 benchmark questions over 107 normal videos, every one asked mid-video at its t_ask
# (the memory cut there, as for the benchmark's streaming questions). Answered by the memory methods only (MEMORY_DONE:
# not the plain video models, API rows, ReKV or VST): a qa line for every arm that answers BANKS, on its pools, from the
# memory it already has of the video (never listed without it built or its build listed to wait on). Each arm's lines
# follow its own PRIORITY gate. Listed after every memory build and the evidence-clips rows, ahead of the other answers
# (lead()). Not benchmark accuracy: the sheet keeps them out of its "all" rows (NOT_IN_ALL).
PROBES = "probes_v2"
# Which arms answer them, and in what order (Anubhav, 2026-11-16): PROBES_FIRST lead the probe tasks, in this order, then
# the other probe arms (MAGIC-Video's GPT-OSS-120B + Qwen3.5-Flash row); PROBES_FROZEN have no probe line at all (a claim
# already running finishes, and answers already in the dataset stay; their normal and stitched lines are untouched)
PROBES_FIRST = [("merit", "m8"), ("merit", "m27"), ("merit", "g5m-q8"), ("merit", "g5m-q27"), ("worldmm", "g5m-q8"),
                ("worldmm", "g5m-g51"), ("egagent", "gemini36flash"), ("egor1", "gemini36flash-gpt41"),
                ("egorag", "egogpt_gpt41"), ("m3agent", "m3")]
# PROBES_NOW (Anubhav, 2026-11-17): these arms' probe tasks lead the whole list, ahead of the stitched tasks, on every
# pool that takes them: WorldMM's GPT-5.1 row had its last videos free with pulkit's worker quiet and max busy elsewhere
PROBES_NOW = {("worldmm", "g5m-g51")}
PROBES_FROZEN = {("magic", "m8"), ("magic", "m27"), ("magic", "oss120-q27"), ("worldmm", "m8"), ("worldmm", "m27"),
                 ("hippomm", "q25vl7b-gpt41")}
EXPERIMENTS += [(b, arm, "qa", list(pools), PROBES) for (b, arm), pools in dict.fromkeys(
    ((b, arm), tuple(pools)) for b, arm, stage, pools, bank in EXPERIMENTS
    if stage == "qa" and bank in BANKS and b in MEMORY_DONE and (b, arm) not in PROBES_FROZEN)]
# banks the sheet scores on their own rows but never folds into its "all" rows (sheet_update.py)
NOT_IN_ALL = {PROBES, STITCHED_EOV}


# Paused (Anubhav, 2026-11-17): HippoMM, its memory builds (q25vl7b-gpt4o) and answers (q25vl7b-gpt41), not wanted for
# this round. Taken off every line here, after all of them are made; tasks already running finish, answers already in
# the dataset stay. Delete an entry to turn it back on.
PAUSED = {("hippomm", "q25vl7b-gpt4o"), ("hippomm", "q25vl7b-gpt41")}
EXPERIMENTS = [e for e in EXPERIMENTS if (e[0], e[1]) not in PAUSED]


def bank_questions(bank):
    """{video: its questions} of the epic-morph bank, for the videos with at least one."""
    epic = os.environ.get("EPIC_MORPH_DIR")
    if not epic:
        raise SystemExit(f"set EPIC_MORPH_DIR to an epic-morph checkout with bank {bank} (qa tasks need it)")
    return exported(epic, bank)


def base_bank(bank):
    """The bank an _end bank (export_bank.py <bank>_end) takes its questions from, whose answers count for it; else None."""
    return bank[:-len("_end")] if bank and bank.endswith("_end") else None


def bank_videos(bank):
    """Videos with at least one question in the epic-morph bank."""
    return list(bank_questions(bank))


@functools.lru_cache(maxsize=None)   # one export per bank and run, however many experiments answer it
def exported(epic, bank):
    d = os.path.join(epic, "outputs", "v3", "banks", bank)
    # banks are generated, not committed: regenerate every time, so an interrupted export or an older commit's
    # files never stand in for the checkout's bank (a few seconds)
    if bank not in COMMITTED_BANKS:
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
    """[key, qhash, gold letter, ask time, has a trace] of each row of an answers file that has an answer (response, or
    EGAgent's pred, not one of FAILED); a field the row does not record is None."""
    out = []
    for r in rows:
        if not isinstance(r, dict) or str(r.get("response", r.get("pred")) or "").strip() in FAILED:
            continue
        h = r.get("qhash")
        if not h and "question" in r and ("choices" in r or "choice_a" in r):
            h = row_qhash(r)
        elif not h and "question" in r and "options" in r:
            h = qhash(r["question"], r["options"])
        out.append([r.get("key"), h, r.get("answer"), ask_time(r), "trace" in r])
    return out


def unanswered(questions, rows, trace=False):
    """Keys of the bank questions that no answer (summary() rows; None: the file did not parse) answers as the bank asks
    them now, by the reuse rule (hf_results.py earlier-answers, prepare_epic.match_answers): the same key (a row from
    before keys: the same wording, options and gold letter), wording and options, gold letter, and query time. A
    baseline asks at a whole second, floor(t_ask) since #16 and ceil(t_ask) before (MERIT's batch10), or at the video's
    end, ceil(duration), for a t_ask in its last second: a row within a second of t_ask counts. A field the row does
    not record is not compared. trace=True (a TRACED baseline): a row without a trace does not count."""
    out = []
    for q in questions:
        k, h, gold, t = asked(q)
        if not any((rk == k or (rk is None and rh == h)) and rh in (h, None) and rg in (gold, None)
                   and (rt is None or t is None or abs(rt - t) < 1) and (not trace or bool(tr and tr[0]))
                   for rk, rh, rg, rt, *tr in rows or []):
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
    names = {f"answers_{x}.json.gz" for _, _, stage, _, bank in EXPERIMENTS if stage == "qa"
             for x in (bank, base_bank(bank)) if x}   # an _end bank's base bank's too: its answers count (desired)
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


def lead(t):
    """Where a task's group stands in the list: the CANARY's builds (-2) and answers (-1), the index of its FIRST entry,
    then the API tasks (API_CONFIGS'), the URGENT lines, the other stitched videos' builds and their answers, every
    other memory build, the evidence-clips rows (EVCLIPS), the M2 probes (PROBES), the rest. PROBES_NOW's probe tasks lead all
    of these (-3)."""
    k = (t["baseline"], t["arm"], t["stage"])
    if t.get("bank") == PROBES and k[:2] in PROBES_NOW and not t.get("deps"):
        return -3   # ahead of everything (PROBES_NOW); one that waits on a listed build keeps its place after it
    if t["video"] in STITCHED_VIDEOS:
        qa = t["stage"] == "qa"
        return -2 + qa   # every stitched video's builds, then their answers, ahead of all else (Anubhav, 2026-11-16)
    if t.get("bank") == PROBES:   # PROBES_FIRST's in their order, then the other probe arms
        i = PROBES_FIRST.index((t["baseline"], t["arm"])) if (t["baseline"], t["arm"]) in PROBES_FIRST else len(PROBES_FIRST)
        return len(FIRST) + 6 + i / 100
    for i, f in enumerate(FIRST):
        if k[:len(f)] == f:
            return i
    api = k[0] == "api" and k[1] in API_CONFIGS   # the evidence-clips configs (EVCLIPS) have their own place
    evclips = k[1].endswith("-evclips")   # after every memory build, ahead of the other answers (Anubhav, 2026-11-16)
    return len(FIRST) + (1 if api else 2 if k in {(b, arm, stage) for b, arm, stage, *_ in URGENT}
                         else 4 if k[2] == "build" else 5 if evclips else 7)


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
    summary in `answers` (read_answers(); a file without one is not checked), with the base bank's file of an _end bank,
    leaves a question unanswered, unless the
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
        pools = ONLY_POOLS.get((b, arm, stage), pools)
        pools = pools + [p for p in EXTRA_POOLS.get((b, arm, stage), []) if p not in pools]
        env = {"needs_env": needs(b, arm, stage)} if needs(b, arm, stage) else {}
        pools = key_pools(env["needs_env"]) + ALSO_POOLS.get((b, arm), []) if env else pools
        pools = ONLY_POOLS.get((b, arm, stage), pools)   # also over a paid line's key pools
        if bank in ((STITCHED,), STITCHED_EOV, STITCHED):
            pools = [p for p in pools if p in STITCHED_POOLS + STITCHED_ALSO.get((b, arm), [])]
        if (b, arm, stage) in CPU_ROWS:
            pools = pools + ["max-cpu"]
        mem = MEMORY_ARM.get((b, arm), arm)
        if mem != arm:   # for the worker: the arm whose dataset folder holds this task's memory (worker.py --prefetch)
            env["memory_arm"] = mem
        if stage == "build":
            # a build line with banks: their videos only, each once, in the banks' order
            for v in dict.fromkeys(v for x in bank for v in (STITCHED_VIDEOS if x == STITCHED else bank_videos(x))) if bank else videos:
                if (b, arm, v) not in have:
                    base = f"{BUILD_ON.get(b)}-{arm}-build-{v}"
                    tasks.append(dict(id=f"{b}-{arm}-build-{v}", baseline=b, arm=arm, stage="build", video=v, bank=None,
                                      pools=pools, deps=[base] if base in ids else [], **env))
                    ids.add(tasks[-1]["id"])
            continue
        for v in bank_videos(bank):
            tid, f, stale = f"{b}-{arm}-{stage}-{bank}-{v}", f"{b}/{arm}/{v}/answers_{bank}.json.gz", {}
            # an _end bank's questions are its base bank's: the answers there count, with the _end file's or alone (a
            # video whose base answers cover every question is not listed; one they do not cover, listed while no
            # _end file exists, is a plain task: the baseline reuses them, hf_results.py earlier-answers)
            base = f"{b}/{arm}/{v}/answers_{base_bank(bank)}.json.gz" if base_bank(bank) else None
            if f in files and (f not in answers or tid in busy):
                continue
            if f in files or (base in files and base in answers and tid not in busy):
                todo = unanswered(bank_questions(bank)[v], [r for p in (f, base) if p in files for r in answers.get(p) or []],
                                  trace=b in TRACED)
                if not todo:
                    continue
                if f in files:
                    tid, stale = redo_id(tid, bank_questions(bank)[v]), {"stale": len(todo)}   # questions to ask again
            # wait for the video's build task (its memory arm's) while it is listed; a memory already built (on the
            # board or off it) needs no wait, and with neither, qa builds the memory itself first
            build = f"{b}-{mem}-build-{v}"
            if (b in NEEDS_MEMORY or (bank in (STITCHED_EOV, STITCHED, PROBES) and b in MEMORY_DONE)) and (b, mem, v) not in have \
                    and build not in ids:
                continue                         # no memory and no build to wait on: not listed
            tasks.append(dict(id=tid, baseline=b, arm=arm, stage=stage, video=v, bank=bank,
                              pools=pools, deps=[build] if build in ids else [], **env, **stale))
            ids.add(tasks[-1]["id"])
    # the reported arms by video (batch10: both memories, then its answers; then tranche 2; then the other memories),
    # so the first rows of the sheet do not wait behind 300 builds; the paid rows after them, by video in the same way
    # (batch10's memories and answers of every row, then tranche 2's, ...); every other task keeps its EXPERIMENTS order
    rank = {id(t): i for i, t in enumerate(tasks)}
    block = {id(t): i for t in tasks for i, lines in enumerate((REPORTED, PAID))
             if (t["baseline"], t["arm"]) in {(b, arm) for b, arm, *_ in lines}}
    first = {i: min(rank[k] for k in block if block[k] == i) for i in set(block.values())}
    # the FIRST groups lead, then the API tasks, then the URGENT lines, whatever their rank (lead())
    tasks.sort(key=lambda t: (lead(t),)
               + ((first[block[id(t)]], group.get(t["video"], len(group)), rank[id(t)]) if id(t) in block else (rank[id(t)], 0, 0)))
    for i, t in enumerate(tasks):
        t["priority"] = i
    return tasks


def gate(tasks):
    """(tasks to list, how many are withheld): of the gated arms (GATED) only PILOT's tasks, while it is non-empty,
    none of HELD and no qa of a QA_FROZEN baseline; then, while a task of a PRIORITY arm is among them, only those and
    the tasks that need no key (local GPU arms)."""
    keep = [t for t in tasks if (t["baseline"], t["arm"], t["stage"]) not in HELD
            and not (t["stage"] == "qa" and t["baseline"] in QA_FROZEN)
            and (not PILOT or not reported(t) or t["id"] in PILOT)]
    pri = lambda t: (t["baseline"], t["arm"]) in PRIORITY or (t["baseline"], t["arm"], t["stage"]) in PRIORITY
    # ... and, with them, every task that needs no key (a local GPU arm; Anubhav, 2026-11-16): the paid rows outside
    # PRIORITY wait, the local ones fill the GPUs the paid rows can't use (Nexus, Zaratan, wriva, arm00)
    first = [t for t in keep if pri(t) or not t.get("needs_env")] if any(pri(t) for t in keep) else []
    return first or keep, len(tasks) - len(first or keep)


def leftovers(tasks, old, now, claimed=()):
    """Add LEFTOVER_POOLS to a task nobody has held for LEFTOVER_H hours. listed_t, carried over from the previous
    tasks.json, is when it was last free: reset while a pool holds it (claimed: the ids running or waiting for a
    relay), so a claim that lapses or fails gives the other pools another LEFTOVER_H hours first."""
    since = {t.get("id"): t.get("listed_t") for t in old}
    n = 0
    for t in tasks:
        t["listed_t"] = now if t["id"] in claimed else since.get(t["id"]) or now
        if (t["pools"] and t["pools"] != KEY_POOLS and not t.get("needs_env") and t["video"] not in STITCHED_VIDEOS
                and (t["baseline"], t["arm"], t["stage"]) not in ONLY_POOLS and now - t["listed_t"] >= LEFTOVER_H * 3600):
            t["pools"] = t["pools"] + [p for p in LEFTOVER_POOLS if p not in t["pools"]]
            n += 1
    return n


def rank(tasks, pools, t=None):
    """The per-pool priority, by the worker's own claim rule (worker.py; pools: the pool files). A task free to claim gets
    `rank` {pool: n} for each pool it is listed for: it is the n-th (1: the next) of the free tasks that pool may take,
    in the list's order, so two pools read the same list as two queues. Any other task gets an empty `rank` and
    `waiting`, why no pool can take it now: "claimed" (a pool holds it: running, or done and not yet off the list),
    "deps" (a task it waits for is not done), "failed" (failed twice: held until `reset`; or every pool listed for it
    has failed it, which a worker does not retry), "no video" (no pool listed for it has failed it, and
    those that could take it lack the video file: `no_video` in their pool files, worker.py), "no pool" (listed for none). What a worker decides from its own
    setup is not known here (--baselines, --arms, its queues, the keys in its environment), so a pool's n is an upper
    bound: it takes its lowest-ranked task that it can run."""
    held, fails = jobs.taken(pools, t), jobs.failures(pools)
    done = {tid for tid, (_, c) in held.items() if c["status"] == "done"}
    n = Counter()
    for x in tasks:
        x.pop("waiting", None)
        x["rank"] = {}
        if x["id"] in held:
            x["waiting"] = "claimed"
        elif fails.get(x["id"], 0) >= 2:
            x["waiting"] = "failed"
        elif any(d not in done for d in x["deps"]):
            x["waiting"] = "deps"
        else:
            novideo = [p for p in x["pools"] if x["video"] in pools.get(p, {}).get("no_video", [])]
            for p in x["pools"]:
                if p not in novideo and pools.get(p, {}).get("claims", {}).get(x["id"], {}).get("status") != "failed":
                    n[p] += 1
                    x["rank"][p] = n[p]
            if not x["rank"]:
                failed = any(pools.get(p, {}).get("claims", {}).get(x["id"], {}).get("status") == "failed" for p in x["pools"])
                x["waiting"] = "no video" if novideo and not failed else "failed" if x["pools"] else "no pool"
    return n


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
    ap.add_argument("--top", type=int, default=5, help="--dry-run: how many of the first tasks, and of each pool's next ones, to print")
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
            # the FIRST groups stay on top, in their own order, then the API tasks (few, short, and only the key pools
            # can take them), then the URGENT lines, then the reported arms in their own order (batch10 first), then
            # the first videos
            tasks.sort(key=lambda t: (lead(t),) + ((False, False) if lead(t) < len(FIRST) else
                                                   (not reported(t), not reported(t) and t["video"] not in a.first))
                                     + (t["priority"],))
        for i, t in enumerate(tasks):
            t["priority"] = i
        left = leftovers(tasks, old, int(jobs.now()), {tid for tid, (_, c) in held.items() if c["status"] != "done"})
        if a.only_pools is not None:
            restrict(tasks, a.only_pools)
        free = rank(tasks, pools)   # after the pools are final (leftovers, --only-pools)
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
            print(f"gate: {withheld} tasks withheld (PRIORITY: every arm but the {len(PRIORITY)} priority ones while they have tasks; "
                  f"PILOT: the reported-configuration and paid-row arms; HELD: {sorted(HELD)}; qa frozen: {sorted(QA_FROZEN)})"
                  + (f"; listed of them: {[t['id'] for t in tasks if reported(t)]}" if PILOT else " (PILOT is empty)"))
            print(f"leftovers: {left} tasks listed {LEFTOVER_H} h or more ago, offered to {', '.join(LEFTOVER_POOLS)} too")
            print(f"need a key (pool {', '.join(KEY_POOLS)} only): "
                  f"{sum(1 for t in tasks if t.get('needs_env'))} tasks")
            print(f"waiting, no rank: {dict(Counter(t['waiting'] for t in tasks if 'waiting' in t))}")
            for p, c in sorted(free.items()):
                print(f"next for {p} ({c} free tasks listed for it): "
                      f"{[t['id'] for t in tasks if 0 < t['rank'].get(p, 0) <= a.top]}")
            print(f"dry run, nothing committed: {len(tasks)} tasks, {len(have)} memories already built; "
                  f"first: {[t['id'] for t in tasks[:a.top]]}")
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
