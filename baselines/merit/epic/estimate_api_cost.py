#!/usr/bin/env python3
"""What the g5m arms will cost at the OpenAI API, from the Qwen runs already in the results dataset. No API call.

    python epic/estimate_api_cost.py --dest <scratch folder> [--proxy m27] [--bank batch10_agree2] [--top-k 10]
                                     [--count <epic-morph>/outputs/v3/banks/t2_agree2 ...]

Measured: clips per video and caption / key lengths (the proxy arm's memories, every video in the dataset); per
question the search rounds, the clips each retrieved, the filter replies and the answers (the proxy arm's
answers_<bank>.json.gz), from which each request's prompt is rebuilt as MERIT builds it and counted with tiktoken
(o200k_base). Assumed, and printed with the result: gpt-5 retrieving --top-k clips a round where Qwen retrieved 5
(text and frames of a round scaled by that, up to the clips visible at the ask time), the same number of rounds,
GPT captions as long as the proxy's, and REASONING hidden reasoning tokens per call, which the API bills as
output (shown at 0x, 1x and 3x). Prices and image tokens: OpenAI's pricing and images-vision pages, read 2026-11-13.
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
from functools import lru_cache

import tiktoken
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "MERIT", "src"))
sys.path.insert(0, HERE)
from merit.llm.templates.final_qa_egolife import merit_qa_system  # noqa: E402
from merit.llm.templates.memory_retrieval_round_reasoning import memory_reasoning_system  # noqa: E402
from merit.llm.templates.multikey_extract import get_multikey_extract_prompt  # noqa: E402
from merit.llm.templates.neighbor_filter import NEIGHBOR_FILTER_SYSTEM  # noqa: E402

# $ per 1M tokens (input, output), standard tier; flex and batch are half. https://developers.openai.com/api/docs/pricing
PRICE = {"gpt-5-mini": (0.25, 2.00), "gpt-5": (1.25, 10.00)}
# tokens of one 448x448 frame: gpt-5-mini ceil(14*14 patches * 1.2); gpt-5 70 + 140 * 1 tile
# (https://developers.openai.com/api/docs/guides/images-vision)
IMAGE = {"gpt-5-mini": 236, "gpt-5": 210}
REASONING = {"captions": 500, "keys": 300, "search": 500, "filter": 800, "answer": 800}   # assumed, per call
ROLE_MODEL = {"captions": "gpt-5-mini", "keys": "gpt-5-mini", "search": "gpt-5", "filter": "gpt-5", "answer": "gpt-5"}
ENC = tiktoken.get_encoding("o200k_base")


@lru_cache(maxsize=None)
def tok(text):
    return len(ENC.encode(text or ""))


def secs(t):   # HHMMSS00
    t = str(t).zfill(8)
    return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def add(total, role, calls=0, inp=0, out=0):
    r = total.setdefault(role, [0, 0, 0])
    r[0] += calls; r[1] += inp; r[2] += out


def build(mem, prompt):
    t = {}
    for c in mem:
        frames = max(1, min(30, secs(c["end_time"]) - secs(c["start_time"])))
        add(t, "captions", 1, frames * IMAGE["gpt-5-mini"] + tok(prompt), tok(c["text"]))
        add(t, "keys", 1, tok(get_multikey_extract_prompt(c["text"])),
            tok("\n".join(c.get(k) or "" for k in ("event_key", "dialogue_key", "object_key", "sum_key"))))
    return t


def question(row, mem, top_k):
    """One question's gpt-5 requests, rebuilt from a Qwen answer row (top-k 5) and scaled to top_k."""
    t = {}
    visible = [c for c in mem if int("1" + str(c["end_time"]).zfill(8)) <= row["query_time_int"]]
    start = {secs(c["start_time"]): i for i, c in enumerate(visible)}
    choices = " ".join(f"({k}) {v}" for k, v in sorted(row["choices"].items()))
    query = f"Query: {row['question']}\nChoices: {choices}"
    hist, left = [], len(visible)   # per search round: (history tokens, answer-prompt tokens, frames), scaled
    for h in row["round_history"]:
        n5 = len(h["retrieved_content"])
        if not n5:
            hist.append((30, 0, 0))
            continue
        scale = min(top_k, left) / n5
        left = max(0, left - top_k)
        text = tok("\n\n".join(h["retrieved_content"]))
        neigh = 0
        for s in h["retrieved_content"]:   # "topN: [DAY1 hh:mm:ss - ...]\n<caption>": its +-2 visible neighbours
            m = re.search(r"\[DAY1 (\d\d):(\d\d):(\d\d)", s)
            i = start.get(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])) if m else None
            around = visible[max(0, i - 2): i + 3] if i is not None else []
            neigh += sum(tok(c["text"]) + 25 for c in around)
        add(t, "filter", 1, tok(NEIGHBOR_FILTER_SYSTEM) + tok(query) + 60 + neigh * scale,
            tok(json.dumps(h.get("filtered_infos", []))) * scale)
        r, n = h["round_num"], n5 * scale
        hist.append((text * scale + 30, text * scale, (6 * n if r == 1 else 6 * min(n, top_k / 2) if r == 2 else n)))
    for j, raw in enumerate(x for x in row.get("raw_responses", []) if x["type"] == "reasoning"):
        add(t, "search", 1, tok(memory_reasoning_system) + tok(query) + 60 + sum(h[0] for h in hist[:j]), tok(raw["raw_response"]))
    add(t, "answer", 1, tok(merit_qa_system) + tok(query) + 40 + sum(h[1] for h in hist) + IMAGE["gpt-5"] * sum(h[2] for h in hist),
        tok(row["response"]))
    return t


def dollars(t, rx=1.0, tier=1.0):
    d = 0
    for role, (calls, inp, out) in t.items():
        pi, po = PRICE[ROLE_MODEL[role]]
        d += tier * (inp * pi + (out + rx * REASONING[role] * calls) * po) / 1e6
    return d


def merge(a, b):
    for role, (c, i, o) in b.items():
        add(a, role, c, i, o)
    return a


def show(name, t):
    cells = "  ".join(f"{r} {c:.0f} calls {i / 1e6:.2f}M in {o / 1e6:.3f}M out" for r, (c, i, o) in sorted(t.items()))
    print(f"{name:34} ${dollars(t):8.2f}  (no reasoning ${dollars(t, 0):.2f}, 3x ${dollars(t, 3):.2f}; flex ${dollars(t, 1, .5):.2f})  {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", required=True, help="where the dataset's memories and answers are downloaded")
    ap.add_argument("--proxy", default="m27", help="the Qwen arm whose runs stand in (m27 has every video; m8 for a second view)")
    ap.add_argument("--bank", default="batch10_agree2", help="the bank the proxy arm has answered")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--count", nargs="*", default=[], help="bank folders (<video>_questions.json) to price at the mean per question")
    a = ap.parse_args()
    from caption_clips_qwen import PROMPT
    root = snapshot_download(os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results"), repo_type="dataset", local_dir=a.dest,
                             allow_patterns=[f"merit/{a.proxy}/*/*_30sec_4_key.json", f"merit/{a.proxy}/*/answers_{a.bank}.json.gz"])
    mems = {os.path.basename(os.path.dirname(f)): json.load(open(f))
            for f in sorted(glob.glob(os.path.join(root, "merit", a.proxy, "*", "*_30sec_4_key.json")))}
    print(f"proxy {a.proxy}: {len(mems)} memories, {sum(map(len, mems.values()))} clips; reasoning tokens assumed per call {REASONING}")
    builds = {v: build(m, PROMPT) for v, m in mems.items()}
    qa, nq = {}, {}
    for f in sorted(glob.glob(os.path.join(root, "merit", a.proxy, "*", f"answers_{a.bank}.json.gz"))):
        v = os.path.basename(os.path.dirname(f))
        rows = json.load(gzip.open(f, "rt"))
        nq[v] = len(rows)
        qa[v] = {}
        for r in rows:
            merge(qa[v], question(r, mems[v], a.top_k))
    print(f"\nper video of {a.bank} (build g5m; qa g5m-g5 at top-k {a.top_k}):")
    for v in qa:
        show(f"{v} build {len(mems[v])} clips", builds[v])
        show(f"{v} qa {nq[v]} questions", qa[v])
    allqa = {}
    for v in qa:
        merge(allqa, qa[v])
    n = sum(nq.values())
    bank_build = {}
    for v in qa:
        merge(bank_build, builds[v])
    every = {}
    for v in builds:
        merge(every, builds[v])
    print("\ntotals:")
    show(f"build, {a.bank} ({len(qa)} videos)", bank_build)
    show(f"build, all {len(builds)} videos", every)
    show(f"qa g5m-g5, {a.bank} ({n} q)", allqa)
    per_q = {r: [c / n, i / n, o / n] for r, (c, i, o) in allqa.items()}
    rounds = sum(c for r, (c, i, o) in allqa.items() if r == "filter") / n
    print(f"per question: ${dollars(per_q):.4f} ({per_q['search'][0]:.2f} search, {rounds:.2f} filter, 1 answer request; "
          f"{(per_q['search'][1] + per_q['filter'][1] + per_q['answer'][1]) / 1e3:.1f}k input tokens)")
    for d in a.count:
        k = sum(len(json.load(open(f))["questions"]) for f in glob.glob(os.path.join(d, "*_questions.json")))
        vs = [os.path.basename(f)[:-len("_questions.json")] for f in glob.glob(os.path.join(d, "*_questions.json"))]
        tb = {}
        for v in vs:
            if v in builds:
                merge(tb, builds[v])
        show(f"build, {os.path.basename(d.rstrip('/'))} ({len(vs)} videos)", tb)
        show(f"qa g5m-g5, {os.path.basename(d.rstrip('/'))} ({k} q, at the mean)", {r: [c * k, i * k, o * k] for r, (c, i, o) in per_q.items()})


if __name__ == "__main__":
    main()
