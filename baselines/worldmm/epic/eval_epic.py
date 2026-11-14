#!/usr/bin/env python3
"""WorldMM answers one video's bank questions from its memory, streaming: each question sees only the memory
that ends by its t_ask.

The same loop as WorldMM's eval/eval.py + eval_egolife.py (WorldMemory.answer with until_time = the question
time, the egocentric QA template, WorldMM's evaluate_prediction), with the memory restricted to the video up to
each question's t_ask:
  - questions run in t_ask order; WorldMemory.index only grows, so each question sees exactly the captions,
    the semantic snapshot and the visual clips that end by its t_ask;
  - HippoRAG starts empty in --cache_dir and takes OpenIE rows from the memory's OpenIE files only for the
    passages being indexed (the fork's EpisodicMemory.openie_sources).
The memory is read from the results folder (build_epic.py, or fetched from the dataset); its captions' video
path is replaced by --video_file, the local 512 px copy, so a memory built elsewhere answers here.

--blind: the same final answer prompt with no context (WorldMemory with max_rounds 0, nothing indexed): what
the model gets from the question and options alone, the no-memory floor of this prompt.

Rows go to --rows (jsonl, one per answered question, resumable; a failed question is redone on the next run);
when every question has a valid answer, --out gets them all in bank order (.gz: gzipped), keyed by the bank's
question key, with the rounds and the retrieved items' time spans (for evidence-retrieval scoring).
--prev: the video's answers under other banks (same memory): a question with the same key, wording, options, gold
letter and query time takes its answer from there instead of being asked again.
"""
import argparse
import gzip
import json
import os
import re
import shutil
import sys
import time
import traceback

import wmm_paths  # noqa: F401  (first)
import openai_usage  # noqa: F401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from earlier import reuse  # noqa: E402
from wmm_paths import GRANULARITIES
from eval import build_choices, evaluate_prediction, extract_choice_letter
from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory

FAILED = ("", "Error", "Unable to generate answer")   # the last is WorldMemory.answer's own failure string
SPAN = re.compile(r"DAY\d+ (\d\d):(\d\d):(\d\d) - DAY\d+ (\d\d):(\d\d):(\d\d)")


def qtime(row):
    return int(row["query_time"]["date"][-1] + row["query_time"]["time"].zfill(8))


def spans(text):
    """[start, end] seconds of every "DAYd HH:MM:SS - DAYd HH:MM:SS" range in a retrieved item."""
    return [[int(h) * 3600 + int(m) * 60 + int(s), int(h2) * 3600 + int(m2) * 60 + int(s2)]
            for h, m, s, h2, m2, s2 in SPAN.findall(text)]


def retrieved(rounds):
    """The items each search round retrieved: episodic captions and visual clips carry time spans; semantic
    triples are consolidated over the video and have none."""
    out = []
    for r in rounds:
        texts = r.get("retrieved_clips") if r["memory_type"] == "visual" else [r.get("retrieved_content") or ""]
        out.append({"round": r["round_num"], "memory": r["memory_type"], "query": r["search_query"],
                    "spans": [s for t in texts or [] for s in spans(t)]})
    return out


def load_memory(wm, mem, v, video_file):
    def captions(g):
        rows = json.load(open(os.path.join(mem, f"{v}_{g}.json")))
        for r in rows:
            r["video_path"] = video_file
        return rows
    wm.load_episodic_captions(caption_data={g: captions(g) for g in GRANULARITIES})
    wm.load_semantic_triples(file_path=os.path.join(mem, f"{v}_semantic.json"))
    wm.load_visual_clips(embeddings_path=os.path.join(mem, f"{v}_visual.pkl"), clips_data=captions("10sec"))
    wm.episodic_memory.openie_sources = {g: os.path.join(mem, f"{v}_openie_{g}.json") for g in GRANULARITIES}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--qa_file", required=True, help="prepare_epic.py questions")
    p.add_argument("--memory_dir", help="results folder holding the memory files (not needed with --blind)")
    p.add_argument("--video_file", help="the local 512 px video (prepare_epic.py video)")
    p.add_argument("--model", default="qwen3vl-8b", help="the retriever model (HippoRAG's fact filter, the multiscale filter)")
    p.add_argument("--respond_model", help="the model that decides each round's search and answers (WorldMM's "
                   "respond model); default --model, as WorldMM's qwen3vl-8b setting")
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache_dir", help="HippoRAG's working folder for this run (emptied at start)")
    p.add_argument("--max_rounds", type=int, default=5)
    p.add_argument("--max_errors", type=int, default=5)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--prev", nargs="*", default=[], help="the video's answers files under other banks (same memory)")
    p.add_argument("--final_pass", action="store_true", help="a question that stops on an assertion inside WorldMM "
                   "(HippoRAG's retrieval) is kept as answered \"Error\", i.e. wrong, as WorldMM's eval.py records it, "
                   "instead of holding back the video's other answers; any other failure is still left for a rerun")
    a = p.parse_args()
    v = a.video

    rows = json.load(open(a.qa_file))
    rows.sort(key=lambda r: (qtime(r), int(r["ID"])))
    done = {}
    if os.path.exists(a.rows):
        for line in open(a.rows):
            r = json.loads(line)
            if (r.get("response") or "").strip() not in FAILED or r.get("error_final"):   # failed questions are redone
                done[r["ID"]] = r
    if a.prev and not a.blind:
        want = {r["key"]: (r["question"], build_choices(r), r["answer"], qtime(r)) for r in rows if r.get("key") and r["ID"] not in done}
        got = reuse(want, a.prev, lambda o: (o.get("question"), o.get("choices"), o.get("answer"), o.get("query_time")), FAILED)
        done.update({r["ID"]: {**got[r["key"]], "ID": r["ID"]} for r in rows if r.get("key") in got})
        print(f"{v}: {len(got)} answers reused from other banks", flush=True)
    todo = [r for r in rows if r["ID"] not in done]
    print(f"{v}: {len(rows)} questions, {len(todo)} to answer{' (blind)' if a.blind else ''}", flush=True)

    if todo:
        # as WorldMM's eval.py: LLMModel(retriever), LLMModel(respond, fps=1); one object when they are the same
        respond = LLMModel(model_name=a.respond_model or a.model, fps=1)
        same = a.blind or (a.respond_model or a.model) == a.model   # blind makes no retriever call
        retriever = respond if same else LLMModel(model_name=a.model)
        retriever.model.usage_role, respond.model.usage_role = "retriever", "respond"   # same object: all "respond"
        wm = WorldMemory(embedding_model=EmbeddingModel(), retriever_llm_model=retriever, respond_llm_model=respond,
                         prompt_template_manager=PromptTemplateManager(), episodic_granularities=GRANULARITIES,
                         qa_template_name="qa_egolife", max_rounds=0 if a.blind else a.max_rounds,
                         max_errors=a.max_errors)
        wm.set_retrieval_top_k(episodic=3, semantic=10, visual=3)   # 4_eval.sh defaults
        if not a.blind:
            shutil.rmtree(a.cache_dir, ignore_errors=True)   # fresh HippoRAG: nothing indexed beyond a question's time
            wm.episodic_memory.save_dir_root = a.cache_dir
            load_memory(wm, a.memory_dir, v, os.path.abspath(a.video_file))

    os.makedirs(os.path.dirname(os.path.abspath(a.rows)), exist_ok=True)
    with open(a.rows, "a") as f:
        for n, row in enumerate(todo, 1):
            t0 = time.time()
            choices = build_choices(row)
            until = qtime(row)
            try:
                res = wm.answer(query=row["question"], choices=choices, until_time=None if a.blind else until)
                response, hist, nr = res.answer, res.round_history, res.num_rounds
            except Exception as e:
                scrub = lambda x: re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-...", x)   # an API error can quote the key
                print(f"ERROR on {row['ID']}: {type(e).__name__}: {scrub(str(e))}\n{scrub(traceback.format_exc())[-1200:]}", flush=True)
                response, hist, nr = "Error", [], 0
                err = scrub(f"{type(e).__name__}: {e}")[:300]
                final = a.final_pass and isinstance(e, AssertionError)
            else:
                err, final = None, False
            ok = evaluate_prediction(response, row["answer"], choices)
            rec = {"key": row.get("key"), "ID": row["ID"], "video": v, "type": row["type"], "question": row["question"],
                   "choices": choices, "answer": row["answer"], "response": response,
                   "pred": extract_choice_letter(response or ""), "evaluate": ok, "t_ask": row["t_ask"],
                   "query_time": until, "num_rounds": nr, "retrieved": retrieved(hist), "round_history": hist,
                   "sec": round(time.time() - t0, 1)}
            if not a.blind:
                rec["indexed"] = ({g: len(wm.episodic_memory.indexed_entries[g]) for g in GRANULARITIES}
                                  | {"visual": len(wm.visual_memory.indexed_entries),
                                     "semantic_ts": wm.semantic_memory.indexed_timestamp})
            if err:
                rec["error"] = err
            if final:
                rec["error_final"] = True
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if response.strip() not in FAILED or final:
                done[row["ID"]] = rec
            print(f"[{n}/{len(todo)}] ID {row['ID']} t={row['t_ask']:.0f}s pred={response!r} gold={row['answer']} "
                  f"ok={ok} rounds={nr} {rec['sec']}s | acc {sum(bool(r['evaluate']) for r in done.values())}/{len(done)}",
                  flush=True)

    missing = [r["ID"] for r in rows if r["ID"] not in done]
    if missing:
        raise SystemExit(f"{len(missing)} questions without a valid answer ({' '.join(missing)}): rerun to retry them")
    final = [done[r["ID"]] for r in sorted(rows, key=lambda r: int(r["ID"]))]
    with (gzip.open(a.out + ".tmp", "wt") if a.out.endswith(".gz") else open(a.out + ".tmp", "w")) as g:
        json.dump(final, g, indent=1, ensure_ascii=False)
    os.replace(a.out + ".tmp", a.out)
    print(f"final: {sum(bool(r['evaluate']) for r in final)}/{len(final)} correct -> {a.out}")


if __name__ == "__main__":
    main()
