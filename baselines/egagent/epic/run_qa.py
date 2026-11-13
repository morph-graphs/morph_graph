#!/usr/bin/env python3
"""EGAgent answers one video's questions of a bank, each from the memory up to its t_ask. PAID API (the arm's
agent backbone: GPT-4.1 or Gemini 2.5 Pro), except --retrieval_check.

    run_qa.py --video V --bank_file <bank>/<V>_questions.json --memory <results dir> --frames_dir <work>/frames/<V> \
              --db_root <work>/dbs/<arm>_<V> --rows <work>/rows/....jsonl --out <results>/answers_<bank>.json.gz
    --blind             upstream's final answer prompt with no retrieved context (the no-memory floor)
    --retrieval_check   no LLM call: build each cutoff's two SQLite tables and run upstream's frame search tool
                        with the question as the text query (checks the tables, the embedder and the cutoff)

The agent is upstream's Video-MME workflow (one video, elapsed time): planner -> per step the chosen tools
(frame search + analysis, entity graph SQL, transcript search) -> early-exit grader -> answer. Streaming: for
each cutoff floor(t_ask) (the video's end for a non-streaming question) the frames named for seconds <= cutoff
and the graph chunks that end <= cutoff are written to upstream's two tables under a video name unique to the
cutoff, so no tool can return anything later; the tables are deleted after the cutoff's questions. HD-EPIC
has no transcript: the transcript tool gets upstream's "NO TRANSCRIPTS AVAILABLE".
Rows are appended to --rows as they finish (resume); --out is written when every question has an answer
(exit 2 otherwise, so the run script retries).
"""
import argparse
import gzip
import json
import math
import os
import shutil
import sqlite3
import time

import numpy as np

import eg_paths
from eg_paths import clock, hhmmss

LETTERS = "ABCD"
EG_CONFIG = "dtonly_concatwith_fused_dt_and_llava-video-7bcaptions"   # the folder upstream's graph tool reads


def load_questions(bank_file, n_frames):
    rows = []
    for i, q in enumerate(json.load(open(bank_file))["questions"], 1):
        opts = [str(o) for o in q["gemini_mcq"]["options"]]
        assert 2 <= len(opts) <= 4, (i, opts)
        streaming = q.get("streaming") and "t_ask" in q
        rows.append({"ID": str(i), "key": q.get("key"), "question": q["q"], "options": opts,
                     "candidates": ", ".join(f"{l}. {o}" for l, o in zip(LETTERS, opts)),   # Video-MME's option strings
                     "answer": LETTERS[q["gemini_mcq"]["answer_index"]], "t_ask": q.get("t_ask"),
                     "cutoff": min(math.floor(q["t_ask"]), n_frames) if streaming else n_frames,
                     "meta": {k: q.get(k) for k in ("form", "hops", "streaming", "origin")}})
    return rows


def seconds(ts):
    """'HH:MM:SS,mmm' or 'HH:MM:SS' -> seconds (float), None when it is not a time."""
    try:
        h, m, s = ts.strip().replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return None


def write_tables(db_root, name, cutoff, embs, frames_dir, files, graph):
    """Upstream's two SQLite tables (create_db_visual_frames.py, create_db_entity_graph.py: Video-MME schemas)
    holding only what exists at `cutoff` seconds. Returns (frames, graph rows) written."""
    fdb = os.path.join(db_root, "videomme", f"videomme_frames_{name}.db")
    gdb = os.path.join(db_root, "videomme", EG_CONFIG, f"videomme_{name}.db")
    os.makedirs(os.path.dirname(gdb), exist_ok=True)
    for p in (fdb, gdb):
        if os.path.exists(p):
            os.remove(p)
    n = min(cutoff, len(files))   # frame i (0-based) is second i + 1
    con = sqlite3.connect(fdb)
    con.execute("CREATE TABLE frames (id INTEGER PRIMARY KEY, day TEXT, timestamp TEXT, path TEXT, embedding BLOB)")
    con.executemany("INSERT INTO frames (day, timestamp, path, embedding) VALUES (?, ?, ?, ?)",
                    [("day1", files[i][:-4], os.path.join(frames_dir, files[i]), embs[i].astype(np.float32).tobytes())
                     for i in range(n)])
    con.commit()
    con.close()
    rows = []
    for ch in graph["chunks"]:
        if ch["end"] > cutoff:
            continue
        for rel in ch["relationships"]:
            for iv in rel.get("intervals") or []:
                s, e = seconds(iv.get("start_t") or ""), seconds(iv.get("end_t") or "")
                if s is None or e is None:
                    continue
                s, e = max(ch["start"], min(s, ch["end"])), max(ch["start"], min(e, ch["end"]))   # a chunk's rows stay in it
                rows.append((name, clock(s), clock(e), iv.get("transcript") or "", rel["source_id"], rel["source_type"],
                             rel["target_id"], rel["target_type"], rel["rel_type"]))
    con = sqlite3.connect(gdb)
    con.execute("CREATE TABLE entity_graph_table (id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT, start_t TEXT, "
                "end_t TEXT, transcript TEXT, source_id TEXT, source_type TEXT, target_id TEXT, target_type TEXT, rel_type TEXT)")
    con.executemany("INSERT INTO entity_graph_table (video_id, start_t, end_t, transcript, source_id, source_type, "
                    "target_id, target_type, rel_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return n, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--bank_file", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--memory", help="results folder with the memory files (not needed for --blind)")
    ap.add_argument("--db_root", required=True)
    ap.add_argument("--rows", help="per-question rows, appended (resume)")
    ap.add_argument("--out")
    ap.add_argument("--backbone", default="gpt-4.1", help="the agent's model (eg_paths.BACKBONES)")
    ap.add_argument("--tscript_search", default="llm", choices=["llm", "bm25"])
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--retrieval_check", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (in cutoff order)")
    a = ap.parse_args()

    files = eg_paths.frame_files(a.frames_dir)
    qs = load_questions(a.bank_file, len(files))
    os.environ.update(EGAGENT_DATASET="videomme", EGAGENT_BACKBONE=a.backbone, EGAGENT_DB_ROOT=os.path.abspath(a.db_root))
    os.environ.setdefault("EGAGENT_MODEL_ROOT", os.path.dirname(eg_paths.EMBEDDER))
    order = sorted(qs, key=lambda r: (r["cutoff"], int(r["ID"])))
    if a.limit:
        order = order[:a.limit]

    if not a.blind:
        embs = np.load(os.path.join(a.memory, f"{a.video}_frames.npy"))
        assert embs.shape[0] == len(files), (embs.shape, len(files))
        gfile = os.path.join(a.memory, f"{a.video}_graph.json")
        if a.retrieval_check and not os.path.exists(gfile):
            graph = {"chunks": []}
        else:
            graph = json.load(open(gfile))

    if a.retrieval_check:
        from langgraph_agent import frame_retriever_sql   # loads SigLIP 2; builds the LLM clients, calls none
        for r in order:
            name = f"{a.video}_c{r['cutoff']:06d}"
            nf, ng = write_tables(a.db_root, name, r["cutoff"], embs, a.frames_dir, files, graph)
            got = frame_retriever_sql.invoke({"selected_video": name, "queries": [r["question"][:200]], "topk": 50,
                                              "start_t": "000000", "end_t": "235959"})
            times = sorted(os.path.basename(p)[:6] for p in got)
            assert all(int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:]) <= r["cutoff"] for t in times), (r["ID"], times[-1])
            print(f"Q{r['ID']:>3} t_ask {r['t_ask']} cutoff {r['cutoff']:>5}: {nf} frames, {ng} graph rows in the tables; "
                  f"top-50 search returned {len(got)} frames, latest {times[-1] if times else '-'} (cutoff {hhmmss(r['cutoff'])})")
        shutil.rmtree(a.db_root, ignore_errors=True)
        return

    done = {}
    if os.path.exists(a.rows):
        for r in map(json.loads, open(a.rows)):
            if r.get("pred") in list(LETTERS):
                done[r["key"] or r["ID"]] = r
    todo = [r for r in order if (r["key"] or r["ID"]) not in done]
    print(f"{len(qs)} questions, {len(done)} answered, {len(todo)} to do ({a.backbone}{', blind' if a.blind else ''})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.rows)), exist_ok=True)
    if todo:
        if a.blind:
            from langgraph_agent import generate_finalanswer
        else:
            from run_egagent_on_videomme import build_app, run_agentic_inference
            app = build_app(a.tscript_search)
    name = None
    for r in todo:
        t0 = time.time()
        cut = hhmmss(r["cutoff"])
        row = {k: r[k] for k in ("ID", "key", "question", "options", "answer", "t_ask", "cutoff", "meta")}
        try:
            if a.blind:
                ans = generate_finalanswer.invoke({"context": "", "question": r["question"], "candidates": r["candidates"],
                                                   "query_time": f"Query Time: {clock(r['cutoff'])} on DAY0"})
            else:
                if name != f"{a.video}_c{r['cutoff']:06d}":
                    shutil.rmtree(a.db_root, ignore_errors=True)
                    name = None   # set only once the tables are written
                    new = f"{a.video}_c{r['cutoff']:06d}"
                    indexed = dict(zip(("frames", "graph_rows"),
                                       write_tables(a.db_root, new, r["cutoff"], embs, a.frames_dir, files, graph)))
                    name = new
                row["indexed"] = indexed
                # upstream's prepare_videomme_example, with the question's cutoff as the video's end; the time is
                # HHMMSScc (EgoLife's form), which upstream's time formatter prints whole
                value = run_agentic_inference(app, r["question"], r["candidates"], "000001", cut, "NO TRANSCRIPTS AVAILABLE ",
                                              {"date": "DAY0", "time": cut + "00"}, {"DAY0": {"start": "000001", "end": cut}},
                                              name, "")
                ans = value["answer"]
                row.update(plan=value["plan"], working_memory=value["working_memory"], total_tokens=value["total_tokens"])
            row.update(pred=ans.mcq_prediction, justification=ans.justification)
            row["evaluate"] = row["pred"] == r["answer"]
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        row["seconds"] = round(time.time() - t0, 1)
        open(a.rows, "a").write(json.dumps(row) + "\n")
        if row.get("pred") in list(LETTERS):   # the same test as on resume
            done[r["key"] or r["ID"]] = row
        print(f"Q{r['ID']:>3} cutoff {r['cutoff']:>5} gold {r['answer']} pred {row.get('pred', '?')} "
              f"{row.get('error', '')} {row['seconds']} s", flush=True)
    shutil.rmtree(a.db_root, ignore_errors=True)
    out = [done.get(r["key"] or r["ID"]) for r in (order if a.limit else qs)]
    if not all(out):
        print(f"{sum(o is None for o in out)} questions unanswered")
        raise SystemExit(2)
    k = sum(o["evaluate"] for o in out)
    print(f"accuracy {k}/{len(out)} = {k / len(out):.4f}")
    if a.limit:
        return   # a partial run never writes the answers file
    tmp = a.out + ".tmp"
    with (gzip.open if a.out.endswith(".gz") else open)(tmp, "wt") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
