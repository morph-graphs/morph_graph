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
correctness and MAGIC's retrieval rounds (what was retrieved, with its time stamps). Resumes from it: a row
counts as done when it has a valid answer from this VERSION, this memory (memory_hash) and the same question
(qhash). With the reported arms (MAGIC_ROLES=1, roles.py) the retriever is the memory role, the search/answer loop
the control role (the memory role's model unless MAGIC_CONTROLLER=reason), the final answer and the time parser
the reason role; rows also carry `reason_model` and `control_model` (other ones redo them), and when the answers
live in another folder than the memory, memory.json there names the memory they were made from. Exit code 2 when
some questions are still unanswered (MAGIC returned an error): run it again to retry them.

    python run_qa.py --video <V> --bank_file <V>_questions.json --memory <memory dir> --clip_root <clips> \
        --answers answers_<bank>.json.gz --cache_dir <dir>
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
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "merit", "epic")]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("qa")

import roles                                                                       # noqa: E402
from build_memory import SRC, TAG, TEXT_EMB, VIS_EMB, jload, llm, memory_paths, secs   # noqa: E402
from chains import ensure_chains                                                   # noqa: E402
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
    ap.add_argument("--parallel", type=int, default=8, help="questions in flight per cutoff group (eval_egolife default 8)")
    a = ap.parse_args()
    a.memory, a.answers, a.cache_dir, a.bank_file = map(os.path.abspath, (a.memory, a.answers, a.cache_dir, a.bank_file))
    sys.path.insert(0, os.path.join(SRC, "eval"))
    from _common import evaluate_prediction
    from worldmm.embedding import EmbeddingModel
    import worldmm.memory.memory as wmem
    from worldmm.llm import PromptTemplateManager
    from worldmm.memory import WorldMemory

    V, M = a.video, a.memory
    caps30 = jload(os.path.join(M, f"{V}_30sec.json"))
    rows = questions(a.bank_file, V, secs(caps30[-1]["end_time"]))
    mh = memory_hash(M, V)
    reason = {"reason_model": roles.endpoint("reason")[0], "control_model": roles.endpoint("control")[0]} if roles.ROUTED else {}
    if roles.ROUTED:
        wmem._time_extract_llm = llm("reason")   # upstream hard-codes qwen/qwen3.5-flash-02-23 here: the reason role
        if os.path.dirname(a.answers) != M:      # an answer arm's folder: its "memory" file for upload (a rebuilt
            with open(os.path.join(os.path.dirname(a.answers), "memory.json"), "w") as f:   # memory replaces the answers)
                json.dump({"memory": os.environ.get("MAGIC_MEMORY_REPO_PATH", M), "memory_hash": mh}, f)
    done = {}
    if os.path.exists(a.answers):
        old = json.load(gzip.open(a.answers, "rt"))
        qh = {r["key"]: r["qhash"] for r in rows}   # a question edited under the same key is redone
        done = {r["key"]: r for r in old if r.get("version") == VERSION and r.get("memory_hash") == mh
                and r.get("qhash") == qh.get(r["key"]) and (r.get("response") or "").strip() not in ("", "Error")
                and all(r.get(k) == reason.get(k) for k in ("reason_model", "control_model"))}
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
    # the reported arms split the respond model by call (roles.responder); the loop's prompt is WorldMemory's default
    respond = lambda: (roles.responder(TAG, ptm.render("memory_reasoning_unified_egolife"), **QWEN_REASONING) if roles.ROUTED
                       else llm("reason", **QWEN_REASONING))
    for cut, grp in sorted(groups.items()):
        cdir = chain_dirs[cut]
        wm = WorldMemory(
            embedding_model=emb,
            retriever_llm_model=llm("memory"),   # upstream's --retriever-model; the unified graph never calls it
            respond_llm_model=respond(),
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
        until = int(grp[0]["query_time"]["date"][-1] + grp[0]["query_time"]["time"].zfill(8))
        wm.index(until)     # graph filtered to nodes ending <= ceil(t_ask); shared by the group's questions

        def one(r):
            choices = {L: r[f"choice_{L.lower()}"] for L in "ABCD" if r.get(f"choice_{L.lower()}")}
            t0, res = time.time(), None
            try:
                res = wm.answer(query=r["question"], choices=choices, until_time=until)
                resp = (res.answer or "").strip() or "Error"
            except Exception as e:
                log.error(f"{r['key']} (ID {r['ID']}): {type(e).__name__}: {e}")
                resp = "Error"
            return {"key": r["key"], "qhash": r["qhash"], "ID": r["ID"], "type": r["type"], "question": r["question"],
                    "choices": choices, "answer": r["answer"], "response": resp,
                    "evaluate": evaluate_prediction(resp, r["answer"], choices),
                    "t_ask": r["meta"]["t_ask"], "query_time": until, "cutoff": cut,
                    "num_rounds": res.num_rounds if res else 0, "round_history": res.round_history if res else [],
                    "seconds": round(time.time() - t0, 1), "version": VERSION, "memory_hash": mh, **reason}
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
