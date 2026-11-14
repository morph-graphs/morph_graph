#!/usr/bin/env python3
"""What EGAgent's paid API calls would cost on EPIC-Morph, from the calls the code makes (no API call here).

    estimate_cost.py --hours 5.66 --questions 598          # batch10_agree2: 10 videos
    estimate_cost.py --hours 5.66 --questions 598 --frame_px 512

Calls are counted from the code; tokens per call are estimates (image tokens from the providers' published
formulas, text and output lengths assumed), to be replaced by the `total_tokens` that
run_qa.py and caption_epic.py record once a first video has run. Prices are list prices per million tokens.
"""
import argparse
import math

PRICE = {"gpt-4.1": (2.00, 8.00), "gemini-2.5-pro": (1.25, 10.00)}   # $ per 1M input, output tokens


def image_tokens(px):
    """Tokens of one square px x px frame, by each provider's published formula."""
    s = min(px, 768)                                   # OpenAI, detail high: shortest side at most 768, 512 px tiles
    gpt = 85 + 170 * math.ceil(s / 512) ** 2
    unit = max(256, min(768, px // 1.5))               # Gemini: 258 a tile; one tile up to 384 px
    gem = 258 if px <= 384 else 258 * math.ceil(px / unit) ** 2
    return {"gpt-4.1": gpt, "gemini-2.5-pro": gem}


CLIP_S, CHUNK_S, FRAMES_PER_SEARCH = 30, 300, 50


def usd(model, tin, tout):
    return (tin * PRICE[model][0] + tout * PRICE[model][1]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--questions", type=int, required=True)
    ap.add_argument("--frame_px", type=int, default=1408, help="frame size sent to the APIs (EGAGENT_FRAME_PX; HD-EPIC's own is 1408)")
    a = ap.parse_args()
    IMG = image_tokens(a.frame_px)
    print(f"frames of {a.frame_px} px: {IMG['gpt-4.1']} tokens each for GPT-4.1, {IMG['gemini-2.5-pro']} for Gemini 2.5 Pro")
    sec = a.hours * 3600
    # memory (GPT-4.1): a caption per 30 s (30 frames + prompt in, ~300 tokens out); per 300 s chunk one graph
    # extraction (10 captions + LLMGraphTransformer's prompt in, ~1.5k out) and one time-stamping call
    ncap, nchunk = sec / CLIP_S, sec / CHUNK_S
    cap_in, cap_out = ncap * (CLIP_S * IMG["gpt-4.1"] + 100), ncap * 300
    g_in, g_out = nchunk * (4000 + 4800), nchunk * 3000
    mem = usd("gpt-4.1", cap_in + g_in, cap_out + g_out)
    print(f"memory, {a.hours:.1f} h of video: {ncap:.0f} caption calls ({cap_in / 1e6:.1f}M in, {cap_out / 1e6:.2f}M out), "
          f"{2 * nchunk:.0f} graph calls ({g_in / 1e6:.2f}M in, {g_out / 1e6:.2f}M out): ${mem:.0f} (${mem / a.hours:.2f} per video hour)")
    # a question: 1 planner call; per plan step (<= 5) each chosen tool (frame search = 1 query-writing call +
    # 1 analysis call with <= 50 frames; graph = 1 SQL-writing call; transcript = 1 call) and 1 grader call;
    # 1 answer call. Frame analysis dominates: typical = 3 frame searches, most = 5.
    for model, out_tok in (("gpt-4.1", 2500), ("gemini-2.5-pro", 10000)):   # Gemini bills its thinking as output
        lo, hi = (usd(model, n * FRAMES_PER_SEARCH * IMG[model] + 10000, out_tok * n / 3) for n in (3, 5))
        print(f"answers, agent {model}: ~{3 * FRAMES_PER_SEARCH * IMG[model] + 10000:,} input tokens and ~12 calls per question "
              f"(most: ~{5 * FRAMES_PER_SEARCH * IMG[model] + 10000:,}, ~22 calls): ${lo:.2f} to ${hi:.2f} per question, "
              f"${lo * a.questions:,.0f} to ${hi * a.questions:,.0f} for {a.questions} questions; "
              f"with the memory ${mem + lo * a.questions:,.0f} to ${mem + hi * a.questions:,.0f}")
    print(f"blind: {a.questions} answer calls of ~300 tokens: under $1")


if __name__ == "__main__":
    main()
