#!/usr/bin/env python3
"""Answer one video's questions of an EPIC-Morph bank with MAGIC-Video, from a built memory.

Streaming (as MERIT): a question asked at t_ask sees only the video before ceil(t_ask). The memory graph is
filtered to nodes ending before that time (MAGIC's own query_time filter), the semantic snapshot is the last
one before it, and the Narrative Memory Chains are built from the memory before that cutoff (chains.py,
cached per cutoff in <memory>/chains/, so banks asking at the same second share them). Questions are grouped
by cutoff; each group gets its own index.

Questions: MERIT's conversion of the bank (prepare_epic.convert_questions: ID = 1-based bank row, multiple
choice, query time ceil(t_ask)), plus the bank's stable `key` and a hash of the question text and options.
Hyperparameters are eval/eval_egolife.py's paper defaults (unified graph, chain_mode=facts).

Writes --answers (.json.gz): one row per question in bank order, keyed by `key`, with the chosen option,
correctness and MAGIC's retrieval rounds (what was retrieved, with its time stamps). Resumes from it, then from
--prev (the video's answers under other banks): a row counts as done when it has a valid answer from this VERSION,
this memory (memory_hash), the same key and question (qhash), gold letter and query time. Exit code 2 when
some questions are still unanswered (MAGIC returned an error): run it again to retry them.

    python run_qa.py --video <V> --bank_file <V>_questions.json --memory <memory dir> --clip_root <clips> \
        --answers answers_<bank>.json.gz --cache_dir <dir> [--prev <other banks' answers files>]
"""
import argparse
import gzip
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "merit", "epic"), os.path.join(HERE, "..", "..", "common")]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("qa")

from build_memory import SRC, TAG, TEXT_EMB, VIS_EMB, jload, memory_paths, secs   # noqa: E402
from chains import ensure_chains                                                   # noqa: E402
from earlier import reuse                                                          # noqa: E402
from prepare_epic import convert_questions                                         # noqa: E402

VERSION = "1"   # bump when a change alters answers (prompts, settings, chain building)
QWEN_REASONING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                  "repetition_penalty": 1.0}   # eval_egolife.py: Qwen3.5 non-thinking sampling for the responder


def questions(bank_file, video, end_s):
    raw = json.load(open(bank_file))["questions"]
    rows = convert_questions(bank_file, video, end_s)
    for r, q in zip(rows, raw):
        r["key"] = q["key"]
        r["qhash"] = hashlib.sha1(json.dumps([q["q"], q["gemini_mcq"]["options"]]).encode()).hexdigest()[:12]
    return rows


def until(r):
    """The question's query time as MAGIC's index takes it (and its answer rows record it)."""
    return int(r["query_time"]["date"][-1] + r["query_time"]["time"].zfill(8))


def done_rows(rows, paths, mh):
    """{key: answer row} reusable from the answer files: this VERSION and memory, the question unchanged."""
    want = {r["key"]: (VERSION, mh, r["qhash"], r["answer"], until(r)) for r in rows}
    sig = lambda o: (o.get("version"), o.get("memory_hash"), o.get("qhash"), o.get("answer"), o.get("query_time"))
    ids = {r["key"]: r["ID"] for r in rows}
    return {k: {**o, "ID": ids[k]} for k, o in reuse(want, paths, sig).items()}


def memory_hash(mem, video):
    h = hashlib.sha1()
    for p in memory_paths(mem, video):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--bank_file", required=True, help="epic-morph outputs/v3/banks/<bank>/<video>_questions.json")
    ap.add_argument("--memory", required=True, help="the memory folder (build_memory.py --out)")
    ap.add_argument("--clip_root", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--cache_dir", required=True, help="WorldMM's per-video index cache (not part of the memory)")
    ap.add_argument("--prev", nargs="*", default=[], help="the video's answers files under other banks (same memory)")
    ap.add_argument("--parallel", type=int, default=8, help="questions in flight per cutoff group (eval_egolife default 8)")
    a = ap.parse_args()
    a.memory, a.answers, a.cache_dir, a.bank_file = map(os.path.abspath, (a.memory, a.answers, a.cache_dir, a.bank_file))
    sys.path.insert(0, os.path.join(SRC, "eval"))
    from _common import evaluate_prediction
    from worldmm.embedding import EmbeddingModel
    from worldmm.llm import LLMModel, PromptTemplateManager
    from worldmm.memory import WorldMemory

    V, M = a.video, a.memory
    caps30 = jload(os.path.join(M, f"{V}_30sec.json"))
    rows = questions(a.bank_file, V, secs(caps30[-1]["end_time"]))
    mh = memory_hash(M, V)
    done = done_rows(rows, [a.answers] + a.prev, mh)   # a question edited under the same key is redone
    todo = [r for r in rows if r["key"] not in done]
    log.info(f"{V}: {len(rows)} questions, {len(done)} done, {len(todo)} to answer (memory {mh})")
    os.chdir(a.clip_root)   # the memory's clip paths are relative to it

    def save():
        out = [done[r["key"]] for r in rows if r["key"] in done]
        tmp = a.answers + ".tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, a.answers)

    groups = {}
    for r in todo:
        groups.setdefault(secs(r["query_time"]["time"]), []).append(r)
    chain_dirs = ensure_chains(M, V, list(groups), mh) if groups else {}
    emb = EmbeddingModel(text_model_name=TEXT_EMB, vis_model_name=VIS_EMB) if groups else None
    ptm = PromptTemplateManager()
    for cut, grp in sorted(groups.items()):
        cdir = chain_dirs[cut]
        wm = WorldMemory(
            embedding_model=emb,
            retriever_llm_model=LLMModel(model_name=TAG),
            respond_llm_model=LLMModel(model_name=TAG, **QWEN_REASONING),
            prompt_template_manager=ptm, max_rounds=5, max_errors=5,
            retrieval_backend="unified_graph", graph_path=os.path.join(M, "graph.pkl"),
            cache_dir=a.cache_dir, chain_mode="facts",
            topic_chain_facts_path=os.path.join(cdir, "topic_chains.json"),
            event_chain_path=os.path.join(cdir, "step3_enriched_chains.json"))
        wm.bm25_weight, wm.damping = 0.3, 0.85
        wm.chain_min_hits, wm.chain_max_topics, wm.chain_max_events = 1, 3, 1
        wm.chain_topic_sim = wm.chain_event_sim = 0.7
        wm.chain_event_min_hits, wm.chain_event_granularities = 1, ("30sec", "3min")
        wm.set_retrieval_top_k(episodic=3, semantic=10, visual=3)
        wm.load_episodic_captions(caption_files={g: os.path.join(M, f"{V}_{g}.json") for g in ["30sec", "3min", "10min", "1h"]})
        wm.load_semantic_triples(data=jload(os.path.join(M, f"semantic_consolidation_results_{TAG}.json")))
        wm.load_visual_clips(embeddings_path=os.path.join(M, "visual_embeddings.pkl"), clips_data=caps30)
        until_t = until(grp[0])
        wm.index(until_t)     # graph filtered to nodes ending <= ceil(t_ask); shared by the group's questions

        def one(r):
            choices = {L: r[f"choice_{L.lower()}"] for L in "ABCD" if r.get(f"choice_{L.lower()}")}
            t0, res = time.time(), None
            try:
                res = wm.answer(query=r["question"], choices=choices, until_time=until_t)
                resp = (res.answer or "").strip() or "Error"
            except Exception as e:
                log.error(f"{r['key']} (ID {r['ID']}): {type(e).__name__}: {e}")
                resp = "Error"
            return {"key": r["key"], "qhash": r["qhash"], "ID": r["ID"], "type": r["type"], "question": r["question"],
                    "choices": choices, "answer": r["answer"], "response": resp,
                    "evaluate": evaluate_prediction(resp, r["answer"], choices),
                    "t_ask": r["meta"]["t_ask"], "query_time": until_t, "cutoff": cut,
                    "num_rounds": res.num_rounds if res else 0, "round_history": res.round_history if res else [],
                    "seconds": round(time.time() - t0, 1), "version": VERSION, "memory_hash": mh}
        with ThreadPoolExecutor(a.parallel) as ex:
            for res in ex.map(one, grp):
                if res["response"] != "Error":
                    done[res["key"]] = res
                log.info(f"{res['key']} ID {res['ID']} @ {cut}s: {res['response']!r} gold {res['answer']} -> {res['evaluate']} "
                         f"({res['num_rounds']} rounds, {res['seconds']}s)")
        save()
        wm.cleanup()
    save()
    right = sum(bool(done[r["key"]]["evaluate"]) for r in rows if r["key"] in done)
    missing = [r["key"] for r in rows if r["key"] not in done]
    print(f"# {len(rows) - len(missing)}/{len(rows)} answered, {right} right, {len(missing)} missing -> {a.answers}", flush=True)
    sys.exit(2 if missing else 0)


if __name__ == "__main__":
    main()
