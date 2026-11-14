#!/usr/bin/env python3
"""Answer one video's questions of an EPIC-Morph bank with HippoMM's QARecallSystem, from a built memory.

Streaming: a question asked at t_ask gets the memory as it stood then (hm_common.at_cutoff): frames, key-frame
captions and transcription lines up to t_ask, and a one-sentence summary made by Qwen2.5-VL from those captions
with upstream's replay prompt (cached per cutoff in <memory>/summaries.json; the build's own summary serves
questions asked at the end). Detailed recall re-reads the video only up to t_ask (memory.max_time).

The question is given to upstream as text: the question, then the options as "A. ...", one per line. Upstream's
prompts ask for one letter on multiple-choice questions; the first standalone option letter of its answer is
the prediction (none -> wrong, the raw answer is kept).

Modes:
  --count    no GPU, no API: how many gpt-4o calls and input tokens the questions need, as a range (see README)
  --blind    upstream's final answer prompt with no memory at all: one gpt-4o call per question
  (default)  answer every question. Needs the Qwen2.5-VL endpoint, a GPU (ImageBind, Whisper) and
             OPENAI_API_KEY in the environment for upstream's reasoning model, gpt-4o.

Writes --answers (.json.gz): one row per question in bank order, keyed by the bank's `key`. Resumes from it: a
row is done when it has an answer from this VERSION, this memory (memory_hash) and the same question (qhash).
Every row has the question's gpt-4o calls and token usage (`api`), read from the API's responses; --usage gets
this run's total, with the model ids the API returned (run_hippomm.sh puts it in runs.jsonl).
Exit code 2 when some questions are still unanswered.

    python run_qa.py --video <V> --bank_file <V>_questions.json --memory <results dir> --video_file <V>.mp4 \
        --store <work>/store --answers answers_<bank>.json.gz
"""
import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import time
from types import SimpleNamespace

import hm_common as hm

VERSION = "1"   # bump when a change alters answers (prompts, cutoff rule, question format)
LETTERS = "ABCD"
USD_PER_M = {"input": 2.50, "output": 10.00}   # gpt-4o list price per million tokens, as assumed here: check before budgeting
log = logging.getLogger("qa")


def questions(bank_file, video_end):
    rows = []
    for q in json.load(open(bank_file))["questions"]:
        opts = [str(o) for o in q["gemini_mcq"]["options"]]
        rows.append({"key": q["key"], "qhash": hashlib.sha1(json.dumps([q["q"], opts]).encode()).hexdigest()[:12],
                     "type": q.get("form"), "question": q["q"], "choices": dict(zip(LETTERS, opts)),
                     "answer": LETTERS[q["gemini_mcq"]["answer_index"]], "t_ask": q.get("t_ask"),
                     "cutoff": hm.cutoff(q, video_end),
                     "text": q["q"] + "\n" + "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, opts))})
    assert len({r["key"] for r in rows}) == len(rows), f"duplicate question keys in {bank_file}"
    return rows


def letter(resp, choices):
    """The option letter of upstream's answer: the answer when it is just a letter ("B", "B.", "(B) text"), else
    the letter named as the answer ("the answer is C"), else the only option whose text the answer contains. A
    leading "A" followed by a word is the article, not the option."""
    L = "".join(choices)
    resp = re.sub(r"^\s*ANSWER\s*:", "", resp or "", flags=re.I).strip()
    # the letter itself is case-sensitive: "the answer is a spoon" names no option
    m = (re.match(r"\(?([%s])\)?(?:[.)]|\s*$)" % L, resp)
         or re.search(r"(?i:(?:answer|option|choice)(?:\s+is)?)\s*:?\s*\(?([%s])\)?(?![A-Za-z'])" % L, resp))
    if m:
        return m.group(1)
    hits = [k for k, o in choices.items() if o.lower() in resp.lower()]
    return hits[0] if len(hits) == 1 else None


def memory_hash(out, video):
    h = hashlib.sha1()
    for p in hm.memory_paths(out, video):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def count(rows, J, F):
    """gpt-4o input tokens per question, as a range over upstream's paths. Every path starts with the
    classification and the direct answer from the summary and all key-frame captions (VIDEO questions; AUDIO
    ones get the transcription instead, VIDEO+AUDIO both). `low`: that answer is confident (2 calls). `high`:
    it is not, ImageBind retrieval is weak too, and the two answers disagree (7 calls, the caption list sent
    three times: direct answer, frame selection, final answer). Templates are counted as fixed token sizes
    measured from upstream's prompt strings; upstream caps a prompt's context at 120k tokens."""
    import tiktoken
    enc = tiktoken.encoding_for_model(hm.REASONING)
    n = lambda s: len(enc.encode(s))
    cap_tok = [n(c) for c in J["frame_captions"]]
    tot = {"questions": len(rows), "low_calls": 2 * len(rows), "high_calls": 7 * len(rows), "low_in": 0, "high_in": 0}
    for r in rows:
        m = hm.at_cutoff(J, F, r["cutoff"])
        C = min(sum(cap_tok[:m["nk"]]) + m["nk"], 120000)
        q = n(r["text"])
        r["count"] = {"key_frames": m["nk"], "caption_tokens": C, "low_in": 600 + 2 * q + C, "high_in": 2100 + 7 * q + 3 * C}
        tot["low_in"] += r["count"]["low_in"]
        tot["high_in"] += r["count"]["high_in"]
    tot["low_out"], tot["high_out"] = 20 * len(rows), 400 * len(rows)
    for k in ("low", "high"):
        tot[f"{k}_usd"] = round((tot[f"{k}_in"] * USD_PER_M["input"] + tot[f"{k}_out"] * USD_PER_M["output"]) / 1e6, 2)
    return tot


def meter(client, flush):
    """Count what goes through an OpenAI client's chat.completions.create: calls, the usage each response
    reports, the model ids it names. Returns the running total (a dict that keeps updating); `flush(total)`
    runs after every call, so a run that is killed mid-question still leaves what it spent."""
    tot = {"requested": hm.REASONING, "models": [], "calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    create = client.chat.completions.create

    def metered(*args, **kw):
        r = create(*args, **kw)
        u = getattr(r, "usage", None)
        det = getattr(u, "completion_tokens_details", None)
        tot["calls"] += 1
        tot["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        tot["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
        tot["reasoning_tokens"] += getattr(det, "reasoning_tokens", 0) or 0
        if getattr(r, "model", None) and r.model not in tot["models"]:
            tot["models"].append(r.model)
        flush(tot)
        return r
    client.chat.completions.create = metered
    return tot


def spent(tot, before):
    """What one question used: the total now minus the total before it."""
    return {k: tot[k] - before[k] for k in ("calls", "input_tokens", "output_tokens", "reasoning_tokens")}


def summary_at(client, model, J, m):
    """Upstream's replay summary for the memory at a cutoff (its messages, its over-length fallback)."""
    caps = m["captions"]
    def ask(caps):
        content = [{"type": "text", "text": f"Image descriptions: {' '.join(caps)}"}]
        if m["na"]:
            content.append({"type": "text", "text": "Audio transcription: " + " ".join(t["text"] for t in m["transcription"])})
        content.append({"type": "text", "text": hm.SUMMARY_PROMPT[(True, bool(m["na"]))]})
        return client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}],
                                              temperature=0).choices[0].message.content
    try:
        return ask(caps)
    except Exception as e:
        if "maximum context length" not in str(e):
            raise
        return ask(caps[::max(1, len(caps) // 1000)][:1000])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--bank_file", required=True, help="epic-morph outputs/v3/banks/<bank>/<video>_questions.json")
    ap.add_argument("--memory", help="the results folder holding the memory (not needed with --blind)")
    ap.add_argument("--video_file", help="the video (detailed recall re-reads frames and audio from it)")
    ap.add_argument("--store", help="upstream's memory_store folder (work files)")
    ap.add_argument("--answers", help="output .json.gz (with --count: a .json of the per-question counts)")
    ap.add_argument("--usage", help="write this run's gpt-4o calls, tokens and returned model ids here (.json)")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N unanswered questions (smoke tests)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    V = a.video
    J = F = mh = None
    if not a.blind:
        J, F = hm.load_memory(a.memory, V)
        mh = memory_hash(a.memory, V)
    rows = questions(a.bank_file, J["duration"] if J else 0.0)
    if a.count:
        tot = count(rows, J, F)
        if a.answers:
            json.dump({"total": tot, "questions": [{"key": r["key"], "cutoff": r["cutoff"], **r["count"]} for r in rows]},
                      open(a.answers, "w"), indent=1)
        print(f"# {V}: {json.dumps(tot)}", flush=True)
        return

    done = {}
    if os.path.exists(a.answers):
        qh = {r["key"]: r["qhash"] for r in rows}   # a question edited under the same key is redone
        done = {r["key"]: r for r in json.load(gzip.open(a.answers, "rt")) if r.get("version") == VERSION
                and r.get("memory_hash") == mh and r.get("qhash") == qh.get(r["key"]) and (r.get("response") or "").strip()}
    todo = [r for r in rows if r["key"] not in done]
    todo = todo[:a.limit] if a.limit else todo
    log.info(f"{V}: {len(rows)} questions, {len(done)} done, {len(todo)} to answer (memory {mh})")

    def save():
        tmp = a.answers + ".tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump([done[r["key"]] for r in rows if r["key"] in done], f, indent=1)
        os.replace(tmp, a.answers)

    def row(r, resp, extra, t0):
        pred = letter(resp, r["choices"])
        out = {k: r[k] for k in ("key", "qhash", "type", "question", "choices", "answer", "t_ask", "cutoff")}
        out.update(response=resp, pred=pred, evaluate=pred == r["answer"], **extra,
                   seconds=round(time.time() - t0, 1), version=VERSION, memory_hash=mh)
        log.info(f"{r['key']} @ {r['cutoff']:.1f}s: {resp[:60]!r} -> {pred} gold {r['answer']} ({out['seconds']}s)")
        return out

    if todo and not os.environ.get("OPENAI_API_KEY"):
        sys.exit(f"answering needs OPENAI_API_KEY in the environment: upstream's reasoning model is {hm.REASONING} "
                 "(run with --count for the number of calls and tokens)")
    cfg = hm.config(os.path.abspath(a.store or "."))
    from hippomm.core.hippocampal_memory import HippocampalMemory, QARecallSystem

    asked = []

    def usage(tot):   # this run's spend so far, rewritten after every API call
        if a.usage:
            json.dump({**tot, "questions": len(asked)}, open(a.usage, "w"))
    if a.blind and todo:   # upstream's final answer prompt over an empty memory
        from openai import OpenAI
        client, empty = OpenAI(api_key=cfg["api"]["reasoning"]["api_key"]), SimpleNamespace(memory=SimpleNamespace(long_term_store=[]))
        tot = meter(client, usage)
        for r in todo:
            t0, before = time.time(), dict(tot)
            asked.append(r["key"])
            prompt = QARecallSystem._format_final_answer_prompt(empty, r["text"], [], "video")
            try:
                resp = client.chat.completions.create(model=hm.REASONING, messages=[{"role": "user", "content": prompt}],
                                                      temperature=0).choices[0].message.content
            except Exception as e:
                log.exception(f"{r['key']}: {type(e).__name__}: {e}")
                continue
            done[r["key"]] = row(r, resp or "", {"api": spent(tot, before)}, t0)
            save()
    elif todo:
        store = cfg["storage"]["base_dir"]
        mem = HippocampalMemory(cfg)   # ImageBind, Whisper, the Qwen2.5-VL client
        mem.video_index[V] = {"metadata": {"path": os.path.abspath(a.video_file), "duration": J["duration"]}}
        qa = QARecallSystem(mem, cfg)
        tot = meter(qa.thinking_client, usage)
        spath = os.path.join(a.memory, "summaries.json")
        summaries = json.load(open(spath)) if os.path.exists(spath) else {}
        if summaries.get("memory_hash") != mh:
            summaries = {"memory_hash": mh}
        for r in sorted(todo, key=lambda r: r["cutoff"]):
            t0, cut, before = time.time(), r["cutoff"], dict(tot)
            m = hm.at_cutoff(J, F, cut)
            if m["nk"] == len(J["frames"]) and m["nv"] == len(F["vision_times"]):
                summary = J["summary"]          # everything was seen: the build's own summary
            else:
                ck = f"{cut:.2f}"
                if ck not in summaries:
                    summaries[ck] = summary_at(mem.qwen.client, mem.qwen.model_name, J, m)
                    json.dump(summaries, open(spath, "w"), indent=1)
                summary = summaries[ck]
            mem.long_term_store = [hm.theta_event(J, F, cut, summary, store)]
            mem.max_time = cut                  # detailed recall never reads the video past the question
            asked.append(r["key"])
            try:
                res = qa.answer_question(r["text"])
            except Exception as e:
                log.exception(f"{r['key']}: {type(e).__name__}: {e}")
                continue
            done[r["key"]] = row(r, res.answer or "", {
                "summary": summary, "indexed": {k: m[k] for k in ("nv", "nk", "na")},
                "question_type": res.question_type, "primary_modality": res.primary_modality,
                "confidence": res.confidence, "reasoning": res.reasoning, "used_direct_answer": res.used_direct_answer,
                "used_reflection": res.used_reflection,
                "retrieved": [[s.start_time, s.end_time] for s in res.retrieved_segments or []],
                "api": spent(tot, before)}, t0)
            save()
    right = sum(bool(done[r["key"]]["evaluate"]) for r in rows if r["key"] in done)
    missing = [r["key"] for r in rows if r["key"] not in done]
    print(f"# {len(rows) - len(missing)}/{len(rows)} answered, {right} right, {len(missing)} missing -> {a.answers}", flush=True)
    sys.exit(2 if missing and not a.limit else 0)


if __name__ == "__main__":
    main()
