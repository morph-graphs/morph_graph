#!/usr/bin/env python3
"""Answer one video's questions of an EPIC-Morph bank with EgoRAG, from a built memory. PAID API CALLS (GPT-4o).

Per question, upstream's own steps (EgoRAG/main.py, stages query and answer; RagAgent as released):
  query             the LLM picks a day, then a time range, from the hour summaries that ended by the question
                    plus its own summary of the captions since; the top 3 caption sentences in that range by
                    OpenCLIP text similarity. ("last time" questions: LLM keywords and search time, top 50
                    sentences, the LLM picks one.)
  extract_evidence  modality "text", as main.py: the LLM reads each retrieved clip's caption for evidence
  get_answer        the LLM answers from the evidence cards (or the captions when there are none)

What the adapter adds:
- Streaming. Upstream filters the summaries by the question time itself, but the time range the LLM returns
  is used unchecked; here every retrieval is also limited to clips that ended by the question time
  (build_memory.SeenOnly), and that is asserted on what came back.
- The query string. EgoLifeQA rows carry hand-written `keywords` (e.g. "use screwdriver"), which upstream embeds
  as the retrieval query. EPIC-Morph has none, so the question text is the query (what upstream itself falls
  back to in its "last time" branch).
- No silent drops. Upstream skips a question it fails on and answers at random after an API error; here such
  a question is retried (3 tries) and otherwise left unanswered (exit code 2: run again).

Writes --answers (.json.gz): one row per question in bank order, keyed by the bank's `key`, with the chosen
option, correctness, and what was retrieved (times in video seconds), rewritten after every answer. Resumes from it.
EGORAG_QA_WORKERS=n answers n questions at a time (threads; default 1, one at a time): a question's steps share
nothing with another's but the index, whose time cutoff is per thread (build_memory.SeenOnly); only the main
thread writes the file.

    python run_qa.py --video <V> --bank_file <V>_questions.json --memory <memory dir> --work <dir> --answers <file>
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "merit", "epic"), os.path.join(HERE, "..", "..", "common")]
from build_memory import MEMORY, open_agent, paid_api   # noqa: E402
from prepare_clips import CLOCK0, clock, video_sec       # noqa: E402
from prepare_epic import convert_questions               # noqa: E402
from stitched_sessions import context                    # noqa: E402

VERSION = "1"   # bump when a change alters answers


def answer(agent, r, hour_docs, date_docs):
    qt = r["query_time"]
    agent.database_t.cutoff = int(qt["time"])
    # a stitched video's question is asked with its session table first (as MERIT, WorldMM and EGAgent ask it); the
    # retrieval query (r["keywords"]) and the saved row keep the bank's wording
    asked = dict(r, question=f"{r['context']}\n\nQuestion: {r['question']}") if r.get("context") else r
    result, question, options = agent.query(asked, hour_docs, date_docs)
    docs = result[0]["docs"]
    got = [{"id": i, "t_start": video_sec(m["start_time"]), "t_end": video_sec(m["end_time"])}
           for i, m in zip(docs["ids"], docs["metadatas"])]
    assert all(m["end_time"] <= int(qt["time"]) for m in docs["metadatas"]), f"retrieved after the question: {got}"
    cards, captions = agent.extract_evidence(question, result, modality="text")
    raw, option = agent.get_answer(question, f"{qt['date']} {qt['time']}", options, cards, caption_results=captions)
    if raw is None or raw.startswith("answer error") or raw == "error":
        raise RuntimeError("the answer call failed")
    rng = result[0]["query_range"]
    return {"response": option or "none", "raw": raw, "branch": "range" if rng else "last",
            "range": [video_sec(rng[1]), video_sec(rng[2])] if rng else None, "retrieved": got, "evidence_cards": cards}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--bank_file", required=True, help="epic-morph outputs/v3/banks/<bank>/<video>_questions.json")
    ap.add_argument("--memory", required=True, help="the memory folder (build_memory.py --out)")
    ap.add_argument("--work", required=True, help="where the Chroma index lives (build_memory.py --work)")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--limit", type=int, help="answer at most this many unanswered questions (a smoke test)")
    a = ap.parse_args()
    V = a.video
    a.memory, a.work, a.answers, a.bank_file = map(os.path.abspath, (a.memory, a.work, a.answers, a.bank_file))
    mem = [os.path.join(a.memory, m.format(video=V)) for m in MEMORY]
    caps, _, hour_docs, date_docs = [json.load(open(p)) for p in mem]
    mh = hashlib.sha1(b"".join(open(p, "rb").read() for p in mem)).hexdigest()[:16]
    rows = convert_questions(a.bank_file, V, video_sec(caps[-1]["end_time"]))
    for r in rows:   # into the clock of the memory; the retrieval query is the question (no EgoLifeQA keywords)
        t = video_sec(r["query_time"]["time"])
        r["query_time"]["time"] = clock(t + CLOCK0)
        r["keywords"] = r["question"]
        r["context"] = context(V, t, day=True, offset=CLOCK0)   # a stitched video's sessions, on the memory's clock; else ""
    done = {}
    if os.path.exists(a.answers):
        qh = {r["key"]: r["qhash"] for r in rows}
        done = {r["key"]: r for r in json.load(gzip.open(a.answers, "rt")) if r.get("version") == VERSION
                and r.get("memory_hash") == mh and r.get("qhash") == qh.get(r["key"])}
    todo = [r for r in rows if r["key"] not in done][: a.limit]
    print(f"{V}: {len(rows)} questions, {len(done)} done, {len(todo)} to answer (memory {mh})", flush=True)
    if todo:
        paid_api(deepseek=False)
    agent = open_agent(V, mem[0], a.work) if todo else None

    def save():
        with gzip.open(a.answers + ".tmp", "wt") as f:
            json.dump([done[r["key"]] for r in rows if r["key"] in done], f, indent=1)
        os.replace(a.answers + ".tmp", a.answers)

    def attempt(r):
        """(result or None, the last error, seconds): up to 3 tries."""
        t0, res, err = time.time(), None, None
        for _ in range(3):
            try:
                res = answer(agent, r, hour_docs, date_docs)
                break
            except AssertionError:
                raise
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        return res, err, round(time.time() - t0, 1)

    def keep(n, r, res, err, seconds):
        """The main thread only: the answer into `done` and the answers file rewritten, in bank order."""
        if res is None:
            print(f"[{n}/{len(todo)}] {r['key']}: failed 3 times ({err})", flush=True)
            return
        choices = {L: r[f"choice_{L.lower()}"] for L in "ABCD" if f"choice_{L.lower()}" in r}
        done[r["key"]] = {"key": r["key"], "qhash": r["qhash"], "ID": r["ID"], "type": r["type"], "question": r["question"],
                          **({"context": r["context"]} if r["context"] else {}), "choices": choices, "answer": r["answer"], **res, "evaluate": res["response"] == r["answer"],
                          "t_ask": r["meta"]["t_ask"], "cutoff": video_sec(r["query_time"]["time"]),
                          "seconds": seconds, "version": VERSION, "memory_hash": mh}
        print(f"[{n}/{len(todo)}] {r['key']} @ {done[r['key']]['cutoff']}s: {res['response']!r} gold {r['answer']} "
              f"({res['branch']}, {len(res['retrieved'])} clips, {len(res['evidence_cards'])} evidence)", flush=True)
        save()

    workers = max(1, int(os.environ.get("EGORAG_QA_WORKERS") or 1))
    if workers == 1:
        for n, r in enumerate(todo, 1):
            keep(n, r, *attempt(r))
    elif todo:
        print(f"answering {workers} questions at a time", flush=True)
        pool = ThreadPoolExecutor(workers)
        try:
            futures = {pool.submit(attempt, r): r for r in todo}
            for n, f in enumerate(as_completed(futures), 1):
                keep(n, futures[f], *f.result())
        finally:   # a retrieval after the question time (AssertionError) stops the questions not yet started
            pool.shutdown(wait=True, cancel_futures=True)
    missing = [r["key"] for r in todo if r["key"] not in done]   # with --limit: of what this run was asked
    right = sum(done[r["key"]]["evaluate"] for r in rows if r["key"] in done)
    print(f"# {len(done)}/{len(rows)} answered, {right} right, {len(missing)} missing -> {a.answers}", flush=True)
    sys.exit(2 if missing else 0)


if __name__ == "__main__":
    main()
