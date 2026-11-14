#!/usr/bin/env python3
"""Count the paid API calls and tokens EgoRAG needs for one video, without making any call.

From the video's EgoGPT captions (and, with --bank_file, its questions) this walks the calls upstream makes
(gen_event.py for the memory; RagAgent.query / extract_evidence / get_answer per question) and counts their
input tokens from upstream's real prompts and the real captions. What a call's input takes from an earlier
call's output, and every output, is not known before running: those use the ASSUMED lengths below.
Tokens are counted with tiktoken's o200k_base (GPT-4o's; also used for the DeepSeek model, an approximation).

    python estimate_api.py --video <V> --captions <V>_30sec.json [--bank_file <V>_questions.json ...] [--out est.json]
    --gpt gpt-4.1                       price the GPT calls as arm egogpt_gpt41 (default gpt-4o)
    --like <W>_30sec.json --seconds S   a video without captions yet: video W's captions, repeated over S seconds,
                                        stand in for its own (ASSUMES W's caption lengths; the questions are real)
    --sum est1.json est2.json ...       add up --out files (one per video): the cost of a set of videos
"""
import argparse
import inspect
import json
import math
import os
import re
import sys

import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "merit", "epic")]
import build_memory   # noqa: E402,F401  (sqlite for Chroma, which upstream's gen_event imports)
from prepare_clips import CLOCK0, video_sec   # noqa: E402
from prepare_epic import convert_questions            # noqa: E402

# ASSUMED output lengths, in tokens (upstream's caps: max_tokens 2048 for the summaries, 2200 elsewhere)
OUT = {"l1": 500,          # a 10 min summary, "one paragraph per major event"
       "l2": 650 + 1500,   # a 1 h summary under 500 words, plus the DeepSeek model's reasoning, assumed billed as output
       "l3": 2048,         # "around 1500 words", cut by max_tokens
       "tail": 300,        # the merged event of the captions since the last hour summary
       "day": 4, "hour": 14, "keywords": 30, "pick_id": 16,
       "evidence": 60, "answer": 2}
# List prices, USD per million tokens (input, output), read 2026-11-13: OpenAI gpt-4o; gpt-4.1 (arm egogpt_gpt41,
# read 2026-11-14); DeepSeek's own API, deepseek-v4-pro at the peak rate (cache miss). Not checked at run time.
PRICE = {"gpt-4o": (2.50, 10.00), "gpt-4.1": (2.00, 8.00), "deepseek": (1.32, 3.96)}
ENC = tiktoken.get_encoding("o200k_base")
tok = lambda s: len(ENC.encode(str(s)))


def sum_up(files):
    """--sum: the per-video estimates added up, per stage and call."""
    ests = [json.load(open(f)) for f in files]
    print(f"{len(ests)} videos, {sum(e['minutes'] for e in ests) / 60:.1f} h, {sum(e['clips'] for e in ests)} clips, "
          f"{sum(e['questions'] for e in ests)} questions; {sum(1 for e in ests if e.get('captions_like'))} with stand-in captions")
    print(f"{'stage':6} {'call':9} {'model':12} {'calls':>7} {'input tok':>12} {'output tok':>11} {'USD':>9}")
    for stage in ("build", "qa"):
        for call in OUT:
            rows = [e["stages"][stage][call] for e in ests if call in e["stages"].get(stage, {})]
            if rows:
                print(f"{stage:6} {call:9} {rows[0]['model']:12} {sum(r['calls'] for r in rows):7d} {sum(r['input_tokens'] for r in rows):12d} "
                      f"{sum(r['output_tokens'] for r in rows):11d} {sum(r['usd'] for r in rows):9.2f}")
        print(f"{stage:6} total USD {sum(e['stages'].get(stage, {}).get('usd', 0) for e in ests):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video")
    ap.add_argument("--captions")
    ap.add_argument("--like", help="another video's <W>_30sec.json, when this one has no captions yet (with --seconds)")
    ap.add_argument("--seconds", type=float, help="with --like: this video's length")
    ap.add_argument("--bank_file", nargs="+", default=[], help="the video's questions, of one or more banks")
    ap.add_argument("--gpt", default="gpt-4o", choices=["gpt-4o", "gpt-4.1"], help="the arm's GPT model, for the prices")
    ap.add_argument("--sum", nargs="+", help="add up these --out files instead")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.sum:
        return sum_up(a.sum)
    from egorag.utils import agent_fun, gen_event, prompts as P
    caps = [c for c in json.load(open(a.captions or a.like)) if c.get("text")]
    # the sentences of each clip, as create_database_from_json splits them
    for c in caps:
        c["sent"] = [s.strip() for s in re.split(r"(?<=[.!?])\s*", c["text"]) if s.strip()]
        c["s"], c["e"] = video_sec(c["start_time"]), video_sec(c["end_time"])
    if not a.captions:   # stand-in captions: the other video's, in order and repeated, on this video's 30 s clips
        end = math.ceil(a.seconds)
        caps = [{**caps[i % len(caps)], "s": 30 * i, "e": min(30 * (i + 1), end)} for i in range(math.ceil(end / 30))]
    calls = []   # (stage, call, model, input tokens, output tokens)
    add = lambda stage, call, model, tin: calls.append((stage, call, a.gpt if model == "gpt-4o" else model, tin, OUT[call]))

    # memory (gen_event.py): 10 min windows of captions -> hours of those -> the day
    hours = {}
    for w in sorted({c["s"] // 600 for c in caps}):
        ctx = [s for c in caps if c["s"] // 600 == w for s in c["sent"]]
        add("build", "l1", "gpt-4o", tok(gen_event.system_message_min) + tok(f"All descriptions: {ctx}"))
        hours.setdefault(w // 6, []).append(w)
    for h, ws in hours.items():
        add("build", "l2", "deepseek", tok(gen_event.system_message_hour) + len(ws) * OUT["l1"] + 20)
    add("build", "l3", "gpt-4o", tok(gen_event.system_message_day) + len(hours) * 650 + 20)
    # an hour summary ends where its last 10 min window ends, and is used by questions asked at or after that
    hour_end = {h: (max(ws) + 1) * 600 for h, ws in hours.items()}

    nq = 0
    if a.bank_file:
        rows = [r for f in a.bank_file for r in convert_questions(f, a.video, caps[-1]["e"])]
        nq = len(rows)
        cap_tok = sum(tok(" ".join(c["sent"][:21])) for c in caps) / len(caps)   # a clip's caption, +-10 sentences
        for r in rows:
            t = video_sec(r["query_time"]["time"]) + CLOCK0
            q = r["question"]
            opts = "\n".join(f"{L}. {r[f'choice_{L.lower()}']}" for L in "ABCD" if f"choice_{L.lower()}" in r)
            if "last time" in q.lower() or "the last" in q.lower():
                add("qa", "keywords", "gpt-4o", tok(P.query_prompt) + tok(q) + tok(opts) + 30)
                seen = [c for c in caps if c["e"] <= t]
                near = 50 * 3 * (sum(tok(s) for c in seen for s in c["sent"]) / max(1, sum(len(c["sent"]) for c in seen)))
                add("qa", "pick_id", "gpt-4o", tok(inspect.getsource(agent_fun.get_ids)) + tok(q) + int(near) + 50 * 45)
                n_ev = 1
            else:
                used = [h for h in hours if hour_end[h] <= t]
                since = max([hour_end[h] for h in used], default=0)
                ctx = [s for c in caps if c["s"] >= since and c["e"] <= t for s in c["sent"]]
                summaries = len(used) * 650
                if since < t:
                    add("qa", "tail", "gpt-4o", tok(P.gen_event_prompt) + tok(f"all descriptions: {ctx}"))
                    summaries += OUT["tail"]
                add("qa", "day", "gpt-4o", tok(P.get_day_prompt) + tok(q) + summaries + 40)
                add("qa", "hour", "gpt-4o", tok(P.get_hour_prompt) + tok(q) + summaries + 40)
                n_ev = 3
            for _ in range(n_ev):
                add("qa", "evidence", "gpt-4o", tok(P.text_evidence_prompt) + tok(q) + int(cap_tok) + 40)
            add("qa", "answer", "gpt-4o", tok(P.with_evidence) + tok(q) + tok(opts) + n_ev * (OUT["evidence"] + 25) + 30)

    res = {"video": a.video, "clips": len(caps), "minutes": round(caps[-1]["e"] / 60, 1), "questions": nq,
           "caption_tokens_per_clip": round(sum(tok(c["text"]) for c in caps) / len(caps), 1),
           "captions_like": None if a.captions else a.like, "stages": {}}
    print(f"{a.video}: {len(caps)} clips ({res['minutes']} min), {res['caption_tokens_per_clip']} caption tokens per clip, {nq} questions")
    print(f"{'stage':6} {'call':9} {'model':12} {'calls':>6} {'input tok':>10} {'output tok':>10} {'USD':>8}")
    for stage in ("build", "qa"):
        total = 0
        for call in OUT:
            rows_ = [c for c in calls if c[0] == stage and c[1] == call]
            if not rows_:
                continue
            model, tin, tout = rows_[0][2], sum(c[3] for c in rows_), sum(c[4] for c in rows_)
            usd = (tin * PRICE[model][0] + tout * PRICE[model][1]) / 1e6
            total += usd
            res["stages"].setdefault(stage, {})[call] = {"model": model, "calls": len(rows_), "input_tokens": tin,
                                                        "output_tokens": tout, "usd": round(usd, 4)}
            print(f"{stage:6} {call:9} {model:12} {len(rows_):6d} {tin:10d} {tout:10d} {usd:8.3f}")
        res["stages"].setdefault(stage, {})["usd"] = round(total, 4)
        print(f"{stage:6} total USD {total:.3f}" + (f" ({total / nq:.4f} per question)" if stage == "qa" and nq else ""))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
