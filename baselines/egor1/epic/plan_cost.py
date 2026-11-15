#!/usr/bin/env python3
"""What Ego-R1 would call and cost on EPIC-Morph, per arm. Calls nothing: arithmetic from the videos' lengths and
the questions of one or more banks, at list prices.

Counts that are exact given the inputs: clips to caption (30 s each), summaries (one per 10 min / hour / day
window), cutoff views (three partial-window summaries per distinct question time before the video's end, shared
by the banks of a video), one final-answer call per question. A video in several banks has one memory.

The sizes of the calls and the agent's tool use are MEASURED for Gemini 3.6 Flash, in the one run there is: arm
gemini36flash on P05-20240425-201444, its memory and 43 of its 59 batch10 questions (2026-11-14, stopped there).
mid is the mean of those 43 questions; low and high are the 10th and 90th percentile of each count (TOOLS). rag
and vlm ran on GPT-4o in that run; arm gemini36flash-gpt41 sends the same requests to GPT-4.1, so the token
counts are taken to be the same (same tokenizer; the image is assumed to count the same). For Gemini 2.5 Pro
(arm gemini25pro, never run: Google refuses it on our key) the sizes are ASSUMED.

    python plan_cost.py --videos <HD-EPIC videos dir> --bank_dir <epic-morph>/outputs/v3/banks/batch10_agree2 [more banks]
        [--arm gemini36flash-gpt41]
"""
import argparse
import collections
import glob
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apis   # noqa: E402

# $ per million tokens (input, output with thinking tokens), standard paid tier, list prices
PRICE = {
    # Google's pricing page, read 2026-11-14: these through 2026-12-31, then 1.50 / 7.50. One input rate (no
    # separate rate for video or audio), no prompt-size tier; thinking tokens are billed as output
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-2.5-pro": (1.25, 10.0), "gemini-2.5-pro>200k": (2.5, 15.0),   # read 2026-11-13
    # OpenAI's list prices, and what OpenRouter charged for both in the run. Cached input (cheaper: 61% of the rag
    # input was billed as cached in the run) is not counted: an upper bound for a gateway without that discount
    "gpt-4o": (2.5, 10.0), "gpt-4.1": (2.0, 8.0)}
# per Gemini model: input tokens per second of video (frames and audio, as the API counted them), a caption's
# output and thinking tokens, a video_llm reply's output and thinking tokens
GEMINI = {"gemini-3.6-flash": dict(tok_s=90, cap_out=125, cap_think=650, vid_out=60, vid_think=990),       # MEASURED
          "gemini-2.5-pro": dict(tok_s=300, cap_out=350, cap_think=1000, vid_out=300, vid_think=1000)}      # ASSUMED
CAPTION_TEXT = 135        # a caption as input of a summary (its 125 tokens and its time span)        MEASURED
SUMMARY_OUT = 700         # tokens of a summary (the port's limit is 1,000, reached by 8 of 23)        MEASURED
CUTOFF = (1375, 730)      # GPT-4.1 input, output per summary of a cutoff view                         MEASURED
RAG = (1610, 150)         # per call: the retrieval prompt and the log entries of its level            MEASURED
VLM = (815, 30)           # per call: one frame (765 tokens) and the question                           MEASURED
ANSWER = (3165, 115)      # GPT-4.1: the trace                                                          MEASURED
# per question: rag, video_llm and vlm calls, and the seconds of video per video_llm call               MEASURED
# (mean: 2.5 / 1.0 / 1.7 calls and 44 s; the most seen in one question: 4 rag, 2 video_llm, 3 vlm, a 249 s range;
# one range of 1,329 s was refused by upstream's tool, which takes at most 10 minutes, without a call)
TOOLS = {"low": (1, 1, 1, 1), "mid": (2.5, 1.0, 1.7, 44), "high": (4, 1, 3, 79)}


def usd(model, tin, tout):
    if model == "gemini-2.5-pro" and tin > 200000:
        model += ">200k"
    return (tin * PRICE[model][0] + tout * PRICE[model][1]) / 1e6


def memory(dur, M):
    """{model: usd} and the calls of one video's memory build."""
    g = GEMINI[M["caption"]]
    n30, n10, n1h = math.ceil(dur / 30), math.ceil(dur / 600), math.ceil(dur / 3600)
    cost = {M["caption"]: n30 * usd(M["caption"], 30 * g["tok_s"], g["cap_out"] + g["cap_think"]),
            M["summary"]: (n10 * usd(M["summary"], 20 * CAPTION_TEXT + 200, SUMMARY_OUT)
                           + n1h * usd(M["summary"], 6 * SUMMARY_OUT + 200, SUMMARY_OUT)
                           + usd(M["summary"], n1h * SUMMARY_OUT + 200, SUMMARY_OUT))}
    return cost, {"captions": n30, "summaries": n10 + n1h + 1}


def question(case, M):
    """{model: usd} of one question."""
    rag, vid, vlm, rng = TOOLS[case]
    g, c = GEMINI[M["video_llm"]], collections.Counter()
    c[M["rag"]] += rag * usd(M["rag"], *RAG)
    c[M["video_llm"]] += vid * usd(M["video_llm"], rng * g["tok_s"] + 100, g["vid_out"] + g["vid_think"])
    c[M["vlm"]] += vlm * usd(M["vlm"], *VLM)
    c[M["answer"]] += usd(M["answer"], *ANSWER)
    return c


def table(c):
    return ", ".join(f"{m} ${v:,.2f}" for m, v in sorted(c.items())) + f" = ${sum(c.values()):,.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", required=True, help="HD-EPIC videos dir (<P>/<video>.mp4)")
    ap.add_argument("--bank_dir", required=True, nargs="+", help="one or more bank folders (<video>_questions.json)")
    ap.add_argument("--arm", default="gemini36flash-gpt41", choices=[x for x in apis.ARMS if x != "llava"])
    a = ap.parse_args()
    apis.use_arm(a.arm)
    M = apis.MODELS
    print(f"arm {a.arm}: " + ", ".join(f"{r} {M[r]}" for r in ("caption", "summary", "rag", "video_llm", "vlm", "answer")))
    dur, nq, cuts = {}, {}, set()
    for d in a.bank_dir:
        bank = os.path.basename(d.rstrip("/"))
        for f in sorted(glob.glob(os.path.join(d, "*_questions.json"))):
            v = os.path.basename(f).replace("_questions.json", "")
            if v not in dur:
                dur[v] = float(subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                     os.path.join(a.videos, v.split("-")[0], v + ".mp4")], check=True, capture_output=True, text=True).stdout)
            qs = json.load(open(f))["questions"]
            nq[bank] = nq.get(bank, 0) + len(qs)
            cuts |= {(v, math.floor(q["t_ask"])) for q in qs if q.get("streaming") and "t_ask" in q and q["t_ask"] <= dur[v] - 1}
        print(f"{bank}: {nq.get(bank, 0)} questions")
    q = sum(nq.values())
    mem, calls = collections.Counter(), collections.Counter()
    for v in dur:
        c, n = memory(dur[v], M)
        mem.update(c)
        calls.update(n)
    cut = collections.Counter({M["summary"]: len(cuts) * 3 * usd(M["summary"], *CUTOFF)})
    print(f"\n== {len(a.bank_dir)} bank(s): {len(dur)} videos, {sum(dur.values()) / 3600:.1f} h, {q} questions, {len(cuts)} question times before the end")
    print(f"memory build ({calls['captions']} captions, {calls['summaries']} summaries): {table(mem)}")
    print(f"cutoff views ({3 * len(cuts)} summaries): {table(cut)}")
    for case in TOOLS:
        r, vi, vl, rng = TOOLS[case]
        one = question(case, M)
        ans = collections.Counter({m: q * x for m, x in one.items()})
        print(f"answering, {case:4s} ({r} rag + {vi} video_llm of {rng} s + {vl} vlm + 1 final per question, ${sum(one.values()):.4f}): {table(ans)}")
        print(f"   total, {case}: {table(mem + cut + ans)}")
    if any(m.startswith("gemini-3.6-flash") for m in M.values()):
        print("\nGemini 3.6 Flash doubles on 2027-01-01 (1.50 / 7.50): its part above doubles for calls made from that day.")
    print("Arm llava: video_llm is local, so its part of answering drops out (the memory still needs Gemini).")


if __name__ == "__main__":
    main()
