#!/usr/bin/env python3
"""What the official arms (g5m memory; g5m-q8 and g5m-g5 answers) cost at the OpenAI API: tokens and dollars.

    python cost_official.py --durations <file: "video seconds" per line> --banks <epic-morph>/outputs/v3/banks \
        [--set name=<file of video ids> ...] [--reasoning 1.0]

Call counts follow the pipeline (README "Official arms"); sizes are measured, not guessed, wherever the Qwen runs
and the mock run could measure them (constants below, with their sources). What no run without the API can
measure is how many reasoning tokens gpt-5-mini and gpt-5 spend per call (billed as output; WorldMM sets no
reasoning effort, so the API's default applies): REASONING holds an assumption per call kind, and --reasoning
scales it (the table prints 0.4x, 1x and 2.5x). The first real video replaces the assumption: its runs.jsonl
records reasoning_tokens per step.
"""
import argparse
import json
import math
import os

# $ per 1M tokens (input, output), standard tier; flex and batch are half. https://developers.openai.com/api/docs/pricing
# read 2026-11-13. Cached-input discounts are ignored (prompts here share only a short system prefix).
PRICE = {"gpt-5-mini": (0.25, 2.00), "gpt-5": (1.25, 10.00)}
# tokens of one 512 x 512 frame, detail auto (https://developers.openai.com/api/docs/guides/images-vision, read
# 2026-11-13): gpt-5-mini 16 x 16 patches x 1.2; gpt-5 70 + 140 x 1 tile. Frame sizes as the mock run received them.
FRAME = {"gpt-5-mini": 308, "gpt-5": 210}
# fixed text of each call (system prompt, examples, instructions) in o200k tokens: WorldMM's templates rendered
# empty and counted with tiktoken (2026-11-13); the mock run's requests (test_official.sh) carry the same text
FIXED = dict(caption=243, multiscale=510, ner=132, triples=479, semantic_extraction=736, semantic_consolidation=361,
             fact_filter=2559, multiscale_filter=322, reasoning=900, answer=138)
# content sizes in tokens (chars / 4), from the Qwen3-VL-8B memory and answers of P05-20240425-201444 (163 captions,
# 59 questions; worldmm_runs/results/m8, read 2026-11-13)
CAPTION = {"10sec": 90, "30sec": 218, "3min": 366, "10min": 1209}     # one caption of each granularity
ENTITIES = {"10sec": 25, "30sec": 75, "3min": 110, "10min": 215}        # NER reply (5, 15, 22, 43 entities)
TRIPLES = {"10sec": 65, "30sec": 194, "3min": 272, "10min": 573}        # triple reply (6, 19, 30, 63 triples)
SEMANTIC_PER_GROUP = 19.2     # semantic triples extracted per group of 10 captions (327 / 17) = consolidation calls
ROUNDS, SEARCHES = 4.41, {"episodic": 2.76, "semantic": 0.53, "visual": 0.68}   # per question (260, 163, 31, 40 over 59)
RETRIEVED_TEXT = 1940         # tokens of retrieved text a question accumulates (457,633 chars / 59 / 4)
FRAMES_PER_VISUAL = 33        # 3 clips x 11 frames per visual search
QUESTION = 60                 # question and options
# ASSUMED reasoning tokens per call at the API's default effort (not measurable without the API)
REASONING = dict(caption=500, multiscale=600, ner=250, triples=400, semantic_extraction=800, semantic_consolidation=300,
                 fact_filter=500, multiscale_filter=500, reasoning=800, answer=1000)
BUILD_STEPS = ("captions", "multiscale", "triples", "semantic")


def build(seconds, r):
    """{step: [calls, input tokens, output tokens]} of one video's gpt-5-mini memory."""
    n = {"10sec": math.ceil(seconds / 10)}
    n["30sec"], n["3min"], n["10min"] = (math.ceil(n["10sec"] / k) for k in (3, 18, 60))
    out = {s: [0, 0, 0] for s in BUILD_STEPS}

    def add(step, kind, calls, tin, tout):
        out[step][0] += calls
        out[step][1] += calls * (FIXED[kind] + tin)
        out[step][2] += calls * (tout + r * REASONING[kind])
    add("captions", "caption", n["10sec"], 10 * FRAME["gpt-5-mini"], CAPTION["10sec"])
    # each level summarises the level below it: 3 x 10 s, 6 x 30 s, 3.3 x 3 min
    for g, below, k in (("30sec", "10sec", 3), ("3min", "30sec", 6), ("10min", "3min", 10 / 3)):
        add("multiscale", "multiscale", n[g], k * CAPTION[below], CAPTION[g])
    # NER + triples per caption: on the 10 s captions (the episodic triples; HippoRAG's 10 s OpenIE then repeats
    # the same prompts and is served by WorldMM's reply cache), and HippoRAG's OpenIE on the three coarser levels
    for g in n:
        add("triples", "ner", n[g], CAPTION[g], ENTITIES[g])
        add("triples", "triples", n[g], CAPTION[g] + ENTITIES[g], TRIPLES[g])
    groups = math.ceil(n["10sec"] / 10)
    add("semantic", "semantic_extraction", groups, 10 * TRIPLES["10sec"], SEMANTIC_PER_GROUP * 12)
    add("semantic", "semantic_consolidation", max(0, groups - 1) * SEMANTIC_PER_GROUP, 10 * 12, 15)
    return out


def qa(seconds, r):
    """{(role, model): [calls, input, output]} of one question: the retriever (gpt-5-mini, both arms) and the gpt-5
    respond model (g5m-g5 only; g5m-q8's respond model is local)."""
    n10min = min(5, math.ceil(seconds / 600))
    # the multiscale filter reads the candidates of every level: 5 x 10 s, 10 x 30 s, 5 x 3 min, up to 5 x 10 min
    cands = 5 * CAPTION["10sec"] + 10 * CAPTION["30sec"] + 5 * CAPTION["3min"] + n10min * CAPTION["10min"]
    e = SEARCHES["episodic"]
    retr = [e * 5, e * (4 * (FIXED["fact_filter"] + QUESTION + 5 * 12) + FIXED["multiscale_filter"] + cands * 1.3),
            e * (4 * (60 + r * REASONING["fact_filter"]) + 30 + r * REASONING["multiscale_filter"])]
    # round k re-reads the k-1 earlier searches' text; the final call reads all of it plus the visual frames
    per_search = RETRIEVED_TEXT / sum(SEARCHES.values())
    history = per_search * ROUNDS * (ROUNDS - 1) / 2
    resp = [ROUNDS + 1,
            ROUNDS * (FIXED["reasoning"] + QUESTION) + history
            + FIXED["answer"] + QUESTION + RETRIEVED_TEXT + SEARCHES["visual"] * FRAMES_PER_VISUAL * FRAME["gpt-5"],
            ROUNDS * (40 + r * REASONING["reasoning"]) + 5 + r * REASONING["answer"]]
    return {("retriever", "gpt-5-mini"): retr, ("respond", "gpt-5"): resp}


def dollars(model, tin, tout):
    return (tin * PRICE[model][0] + tout * PRICE[model][1]) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--durations", required=True)
    ap.add_argument("--banks", required=True, help="folder of banks: <bank>/<video>_questions.json")
    ap.add_argument("--set", action="append", default=[], help="name=<bank name>: the videos of a bank and its questions; "
                    "'all' (every video of --durations, the questions of every bank named) is always printed")
    ap.add_argument("--reasoning", type=float, default=1.0, help="scale of the assumed reasoning tokens")
    a = ap.parse_args()
    dur = {l.split()[0]: float(l.split()[1]) for l in open(a.durations) if l.strip()}
    sets, everyq = {}, {}
    for name in a.set:
        d = os.path.join(a.banks, name)
        q = {f[:-len("_questions.json")]: len(json.load(open(os.path.join(d, f)))["questions"])
             for f in os.listdir(d) if f.endswith("_questions.json")}
        sets[name] = {v: n for v, n in q.items() if n}
        for v, n in sets[name].items():
            everyq[v] = everyq.get(v, 0) + n
    sets["all videos"] = {v: everyq.get(v, 0) for v in dur}
    for name, qs in sets.items():
        secs = sum(dur[v] for v in qs)
        print(f"\n== {name}: {len(qs)} videos, {secs / 3600:.1f} h, {sum(qs.values())} questions")
        print(f"{'':34} {'calls':>8} {'input tok':>12} {'output tok':>11} {'$':>8}   $ at reasoning x0.4 / x2.5")
        total = {}

        def line(label, model, f):
            c, i, o = f(a.reasoning)
            lo, hi = (dollars(model, *f(k)[1:]) for k in (0.4, 2.5))
            print(f"{label:34} {c:8.0f} {i:12,.0f} {o:11,.0f} {dollars(model, i, o):8.2f}   {lo:.2f} / {hi:.2f}")
            return dollars(model, i, o), lo, hi
        for step in BUILD_STEPS:
            total[step] = line(f"build {step} (gpt-5-mini)", "gpt-5-mini",
                               lambda r, s=step: [sum(build(dur[v], r)[s][k] for v in qs) for k in range(3)])
        for (role, model), arms in ((("retriever", "gpt-5-mini"), "each answer arm"), (("respond", "gpt-5"), "g5m-g5 only")):
            total[role] = line(f"qa {role} ({model}; {arms})", model,
                               lambda r, key=(role, model): [sum(qa(dur[v], r)[key][k] * n for v, n in qs.items()) for k in range(3)])
        b = [sum(total[s][k] for s in BUILD_STEPS) for k in range(3)]
        q8, g5 = total["retriever"], [total["retriever"][k] + total["respond"][k] for k in range(3)]
        both = [b[k] + q8[k] + g5[k] for k in range(3)]
        for label, x in (("memory (g5m), once", b), ("answers g5m-q8", q8), ("answers g5m-g5", g5), ("TOTAL, both sheet rows", both)):
            print(f"{label:34} {'':8} {'':12} {'':11} {x[0]:8.2f}   {x[1]:.2f} / {x[2]:.2f}")
        if len(qs):
            print(f"per video: memory ${b[0] / len(qs):.2f}; per question: g5m-q8 ${q8[0] / max(1, sum(qs.values())):.4f}, "
                  f"g5m-g5 ${g5[0] / max(1, sum(qs.values())):.4f}")


if __name__ == "__main__":
    main()
