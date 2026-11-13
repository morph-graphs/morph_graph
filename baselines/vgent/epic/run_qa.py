#!/usr/bin/env python3
"""Answer one video's questions of an EPIC-Morph bank with Vgent (upstream vgent_rag.py, one video, one GPU).

Streaming: a question asked at t_ask sees only the frames at or before t_ask (all frames when it is not a
streaming question). For those frames the pipeline is upstream's, call for call: the graph of that prefix of
the video (construct_graph over the cached chunk extractions; only a last chunk cut short by t_ask needs a
backbone call, cached in <video>_cutoff_nodes.jsonl), or no graph under chunk_size * n_retrieval frames, then
extract_keywords, retrieve_nodes, refine_nodes, aggregate_nodes. The frames keep the size prepare_frames.py
gave them for the whole video (upstream on a cut video would size them from its own length).

Questions are MERIT's conversion of the bank (ID = the bank's key), shown to Vgent in its MLVU format:
candidates "A. <option>", prompt "Question: ...\nOptions:\n(A) ...". --blind asks the backbone the same final
prompt with no video (the no-memory floor).

Writes --answers (.json.gz; .json for --blind): one row per question in bank order with Vgent's letter
(`response`), the backbone's text behind it (`raw`), the retrieved and refined chunks and every backbone
call's time. Resumes: a row counts as done for the same VERSION, memory (memory_hash) and question (qhash).
Exit code 2 when some questions failed: run it again to retry them.

    python run_qa.py --video <V> --arm qwenvl25_7b --bank_file <V>_questions.json --frames <frames dir> \
        --memory <results dir> --answers answers_<bank>.json.gz
"""
import argparse
import gzip
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "merit", "epic")]
from build_graph import MEMORY, SETTINGS, construct, header, load_frames, load_vgent, memory_hash, read_nodes  # noqa: E402
from prepare_epic import convert_questions   # noqa: E402

VERSION = "1"   # bump when a change alters answers (prompts, settings, frame preparation)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--arm", default="qwenvl25_7b")
    ap.add_argument("--bank_file", required=True, help="epic-morph outputs/v3/banks/<bank>/<video>_questions.json")
    ap.add_argument("--frames", help="prepare_frames.py --out (not needed with --blind)")
    ap.add_argument("--memory", help="the results folder of this video and arm (build_graph.py --out)")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--blind", action="store_true", help="no video and no memory: the backbone on the question alone")
    ap.add_argument("--max_questions", type=int, help="answer only the first N unanswered questions (smoke test)")
    a = ap.parse_args()
    V = a.video
    frames, meta = (None, None) if a.blind else load_frames(a.frames, V)
    times = [] if a.blind else meta["times"]
    rows = convert_questions(a.bank_file, V, 0)   # its query time (whole seconds) is not used: t_ask is
    mh = None if a.blind else memory_hash(a.memory, V)
    opener = open if a.blind else gzip.open
    done = {}
    if os.path.exists(a.answers):
        qh = {r["key"]: r["qhash"] for r in rows}   # a question edited under the same key is redone
        done = {r["key"]: r for r in json.load(opener(a.answers, "rt"))
                if r.get("version") == VERSION and r.get("memory_hash") == mh and r.get("arm") == a.arm
                and r.get("qhash") == qh.get(r["key"])
                and (r.get("response") or "").strip() not in ("", "Error")}
    todo = [r for r in rows if r["key"] not in done][:a.max_questions]
    print(f"{V}: {len(rows)} questions, {len(done)} done, {len(todo)} to answer (memory {mh})", flush=True)

    def save():
        tmp = a.answers + ".tmp"
        with opener(tmp, "wt") as f:
            json.dump([done[r["key"]] for r in rows if r["key"] in done], f, indent=1)
        os.replace(tmp, a.answers)

    groups = {}   # questions by the number of frames they may see
    for r in todo:
        m = r["meta"]
        n = 0 if a.blind else sum(t <= m["t_ask"] for t in times) if m.get("streaming") and m.get("t_ask") is not None else len(times)
        groups.setdefault(n, []).append(r)
    if todo:
        vg = load_vgent(a.arm)
        from utils.prompts import PRED_PROMPT
    if todo and not a.blind:
        head = header(V, a.arm, meta)
        nodes = read_nodes(os.path.join(a.memory, MEMORY[0].format(video=V)), head)
        cut_file = os.path.join(a.memory, f"{V}_cutoff_nodes.jsonl")
        if not os.path.exists(cut_file):
            open(cut_file, "w").write(json.dumps(head) + "\n")
        nodes.update(read_nodes(cut_file, head))
    for n, grp in sorted(groups.items()):
        fr = None if a.blind else frames[:n]
        video_graph = entity_graph = None
        if not a.blind and n >= SETTINGS["chunk_size"] * SETTINGS["n_retrieval"]:   # as vgent_rag.py
            video_graph, entity_graph = construct(vg, fr, nodes, cut_file)
        for r in grp:
            letters = [L for L in "ABCD" if f"choice_{L.lower()}" in r]
            choices = {L: r[f"choice_{L.lower()}"] for L in letters}
            candidates = [f"{L}. {choices[L]}" for L in letters]                   # utils/data.py, MLVU
            prompt = f"Question: {r['question']}\nOptions:\n" + "".join(f"({L}) {choices[L]}\n" for L in letters)
            line = {"question": r["question"], "candidates": candidates, "letters": letters}
            t0, info, ncall = time.time(), {}, len(vg.calls)
            torch.manual_seed(0)
            try:
                if a.blind:
                    raw = vg.mllm_response(vg.video_llm, vg.processor, vg.image_processor, prompt + PRED_PROMPT, None, None, max_new_tokens=10)
                    resp = raw.strip("()").strip()
                    resp = resp if resp in letters else "?"
                else:
                    if n == 0:
                        raise ValueError("no frame at or before t_ask")
                    vi = [fr]
                    query_list, llm_info = vg.extract_keywords(r["question"], candidates, vi)
                    retrieved = vg.retrieve_nodes(r["question"], query_list, vi, candidates, video_graph, entity_graph, None, llm_info)
                    info = {"llm_info": llm_info, "retrieved": [int(x) for x in retrieved["nodes"]]}
                    refined, sql_check, check_result = vg.refine_nodes(retrieved, r["question"], llm_info, candidates, vi, None, None)
                    info.update(nodes=[int(x) for x in refined["nodes"][:SETTINGS["n_refine"]]], sql_check=sql_check,
                                check_result={str(k): v for k, v in (check_result or {}).items()})
                    try:
                        resp = vg.aggregate_nodes(refined, llm_info, vi, vi, None, None, prompt, line, video_graph,
                                                  sql_check, check_result, meta["fps"])
                    except IndexError:   # upstream answers the third option when it cannot read a letter: there is none
                        if len(letters) > 2:
                            raise
                        resp = "?"
                    raw = vg.calls[-1]["output"]
            except Exception as e:
                print(f"{r['key']}: {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache()
                resp, raw = "Error", None
            calls = [{k: c[k] for k in ("frames", "seconds")} for c in vg.calls[ncall:]]
            res = {"key": r["key"], "qhash": r["qhash"], "ID": r["ID"], "type": r["type"], "question": r["question"],
                   "choices": choices, "answer": r["answer"], "response": resp, "raw": raw, "evaluate": resp == r["answer"],
                   "t_ask": r["meta"].get("t_ask"), "cutoff_frames": n, "cutoff": times[n - 1] if n else None,
                   "graph": video_graph is not None, **info, "calls": calls, "seconds": round(time.time() - t0, 1),
                   "arm": a.arm, "version": VERSION, "memory_hash": mh}
            if resp != "Error":
                done[r["key"]] = res
            print(f"{r['key']} @ {n} frames: {raw!r} -> {resp} gold {r['answer']} ({len(calls)} calls, {res['seconds']}s, "
                  f"chunks {info.get('nodes')})", flush=True)
            save()
    save()
    right = sum(bool(done[r["key"]]["evaluate"]) for r in rows if r["key"] in done)
    missing = [r["key"] for r in rows if r["key"] not in done]
    print(f"# {len(rows) - len(missing)}/{len(rows)} answered, {right} right, {len(missing)} missing -> {a.answers}", flush=True)
    sys.exit(2 if [r for r in todo if r["key"] in missing] else 0)


if __name__ == "__main__":
    main()
