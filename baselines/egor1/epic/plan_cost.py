#!/usr/bin/env python3
"""What Ego-R1 (arm gemini25pro) would call and cost on EPIC-Morph. Calls nothing: arithmetic from the videos'
lengths and a bank's questions.

Counts that are exact given the inputs: clips to caption (30 s each), summaries (one per 10 min / hour / day
window), cutoff views (up to three partial-window summaries per distinct question time before the video's end),
one final-answer call per question. Everything about the agent's tool use is an ASSUMPTION until a real run:
the tool mix per question (TOOLS; at most MAX_TURNS - 1 = 6 tool calls fit) and the length of the video_llm
ranges (upstream allows up to 10 minutes). low / mid / high span that. Token sizes and list prices are the
constants below (prices per million tokens, as last listed for these models; check before budgeting).

    python plan_cost.py --videos <HD-EPIC videos dir> --bank_dir <epic-morph>/outputs/v3/banks/batch10_agree2
"""
import argparse
import glob
import json
import math
import os
import subprocess

# Gemini 2.5 Pro: Google's pricing page, standard paid tier, read 2026-11-13; output includes thinking tokens
PRICE = {"gemini-2.5-pro": (1.25, 10.0), "gemini-2.5-pro>200k": (2.5, 15.0), "gpt-4o": (2.5, 10.0), "gpt-4.1": (2.0, 8.0)}
THINK = 1000              # thinking tokens per Gemini 2.5 Pro call (it cannot be switched off)            ASSUMED
VIDEO_TOK_S = 300         # Gemini: ~258 tokens per frame at 1 fps + 32 per second of audio
CAPTION_OUT, SUMMARY_OUT = 350, 500      # tokens; the released EgoLife logs average ~2,000 characters per entry
# per question: (rag, video_llm, vlm) calls and the video_llm range in seconds        ASSUMED
TOOLS = {"low": (3, 1, 1, 60), "mid": (3, 2, 1, 240), "high": (2, 3, 1, 600)}
RAG_IN, RAG_OUT = 4000, 300          # GPT-4o: retrieval prompt + up to six 10-min entries (or the hour / day entries)
VLM_IN, VLM_OUT = 1100, 200          # GPT-4o: one 704 px frame (high detail, 765 tokens) + the question
ANSWER_IN, ANSWER_OUT = 8000, 30     # GPT-4.1: the trace (system prompt ~1,300 tokens + turns and tool results)


def usd(model, tin, tout):
    if model == "gemini-2.5-pro" and tin > 200000:
        model += ">200k"
    return (tin * PRICE[model][0] + tout * PRICE[model][1]) / 1e6


def memory(dur):
    """(calls, usd) of one video's memory build."""
    n30, n10, n1h = math.ceil(dur / 30), math.ceil(dur / 600), math.ceil(dur / 3600)
    cap = n30 * usd("gemini-2.5-pro", 30 * VIDEO_TOK_S + 100, CAPTION_OUT + THINK)
    summ = (n10 * usd("gpt-4.1", 20 * CAPTION_OUT + 200, SUMMARY_OUT) + n1h * usd("gpt-4.1", 6 * SUMMARY_OUT + 200, SUMMARY_OUT)
            + usd("gpt-4.1", n1h * SUMMARY_OUT + 200, SUMMARY_OUT))
    return {"gemini-2.5-pro captions": n30, "gpt-4.1 summaries": n10 + n1h + 1}, cap + summ


def question(case):
    rag, vid, vlm, rng = TOOLS[case]
    return (rag * usd("gpt-4o", RAG_IN, RAG_OUT) + vid * usd("gemini-2.5-pro", rng * VIDEO_TOK_S + 100, 300 + THINK)
            + vlm * usd("gpt-4o", VLM_IN, VLM_OUT) + usd("gpt-4.1", ANSWER_IN, ANSWER_OUT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", required=True, help="HD-EPIC videos dir (<P>/<video>.mp4)")
    ap.add_argument("--bank_dir", required=True)
    a = ap.parse_args()
    dur = {}
    for f in sorted(glob.glob(os.path.join(a.videos, "*", "*.mp4"))):
        dur[os.path.basename(f)[:-4]] = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f],
            check=True, capture_output=True, text=True).stdout)
    bank, nq, ncut = [], 0, 0
    for f in sorted(glob.glob(os.path.join(a.bank_dir, "*_questions.json"))):
        v = os.path.basename(f).replace("_questions.json", "")
        qs = json.load(open(f))["questions"]
        cuts = {math.floor(q["t_ask"]) for q in qs if q.get("streaming") and "t_ask" in q and q["t_ask"] <= dur[v] - 1}
        bank.append(v); nq += len(qs); ncut += len(cuts)
        print(f"{v}: {dur[v] / 60:5.1f} min, {len(qs)} questions, {len(cuts)} question times before the end")
    cut_usd = 3 * usd("gpt-4.1", 20 * CAPTION_OUT + 200, SUMMARY_OUT)     # upper bound per cutoff view
    for name, vids, q, cuts in [(f"the bank ({len(bank)} videos)", bank, nq, ncut),
                                (f"all {len(dur)} videos (questions and their times scaled by video hours)", list(dur),
                                 nq * sum(dur.values()) / sum(dur[v] for v in bank),
                                 ncut * sum(dur.values()) / sum(dur[v] for v in bank))]:
        calls, mem = {}, 0.0
        for v in vids:
            c, d = memory(dur[v])
            mem += d
            for k, n in c.items():
                calls[k] = calls.get(k, 0) + n
        print(f"\n== {name}: {sum(dur[v] for v in vids) / 3600:.1f} h of video, {q:.0f} questions")
        print(f"memory build: {calls} -> ${mem:,.0f}")
        print(f"cutoff views: up to {3 * cuts:.0f} gpt-4.1 calls -> up to ${cuts * cut_usd:,.0f}")
        for case in TOOLS:
            r, vi, vl, rng = TOOLS[case]
            print(f"answering, {case:4s} ({r} rag + {vi} video_llm of {rng} s + {vl} vlm + 1 final per question): "
                  f"${question(case):.3f} per question -> ${q * question(case):,.0f}; "
                  f"total ${mem + cuts * cut_usd + q * question(case):,.0f}")
        if name.startswith("the bank"):
            print(f"calls per question (mid): {TOOLS['mid'][0]} gpt-4o (rag), {TOOLS['mid'][1]} gemini-2.5-pro (video_llm), "
                  f"{TOOLS['mid'][2]} gpt-4o (vlm), 1 gpt-4.1 (final answer)")
    print("\narm llava: video_llm is local, so the video_llm part of answering drops out (the memory still needs Gemini).")


if __name__ == "__main__":
    main()
