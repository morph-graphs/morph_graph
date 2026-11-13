#!/usr/bin/env python3
"""EgoRAG's memory for one EPIC-Morph video, after the captions: the index and the event summaries.

Runs upstream's own code (EgoRAG/create_database.py), unpatched, in two steps:

  db      RagAgent.create_database_from_json: the captions split into sentences, embedded with Chroma's default
          OpenCLIP text encoder (ViT-B-32, laion2b_s34b_b79k), into a Chroma collection. No LLM. The collection
          is an index of the captions, kept in --work and rebuilt from them when missing.
  events  gen_event: 10 min summaries (GPT-4o), 1 h summaries of those (DeepSeek-R1), a day summary of those
          (GPT-4o)  -> <video>_l1_10min.json, <video>_l2_1h.json, <video>_l3_day.json in --out.
          PAID API CALLS: runs only with EGORAG_PAID_API=1 and upstream's Azure variables set (paid_api()).

Nothing here depends on the questions. Streaming: the summaries cover the whole video, but at question time
upstream uses an hour summary only when it ends at or before the question, and summarises the captions since
then itself (RagAgent.filter_event_data), so a question never reads a summary of video after it.

    python build_memory.py --video <V> --captions <V>_30sec.json --out <memory dir> --work <dir> --step db events
    python build_memory.py --memory_files        # the memory's file names, with {video}
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

try:   # Chroma needs sqlite >= 3.35; where the system's is older (RHEL 8), use the pysqlite3 wheel instead
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

MEMORY = ["{video}_30sec.json", "{video}_l1_10min.json", "{video}_l2_1h.json", "{video}_l3_day.json"]
# upstream reads these (egorag/utils/util.py). Its other branches cannot run as released: call_gpt4's
# OPENAI_API_KEY branch calls openai.ChatCompletion (removed in the openai 1.x it pins), and call_deepseek's
# DEEPSEEK_API_KEY branch indexes the response object like a dict.
GPT_VARS = ["AZURE_API_KEY", "AZURE_API_ENDPOINT"]             # a GPT-4o deployment's chat completions URL
DEEPSEEK_VARS = ["AZURE_DEEPSEEK_KEY", "AZURE_DEEPSEEK_ENDPOINT"]   # DeepSeek-R1 on Azure AI


def paid_api(deepseek):
    """Stop (exit 4) unless the paid calls were explicitly allowed and upstream's endpoints are set."""
    need = GPT_VARS + (DEEPSEEK_VARS if deepseek else [])
    missing = [v for v in need if not os.environ.get(v)]
    if os.environ.get("EGORAG_PAID_API") != "1" or missing:
        print("this step calls paid APIs (GPT-4o" + (", DeepSeek-R1" if deepseek else "") + "): set EGORAG_PAID_API=1"
              f" and {', '.join(need)} to run it (missing: {', '.join(missing) or 'none'}). estimate_api.py"
              " counts the calls and tokens first.", file=sys.stderr)
        sys.exit(4)


def open_agent(video, captions, work):
    """Upstream's RagAgent over the video's Chroma collection in <work>/chroma_db (built from the captions
    when missing or made from other captions). The process stays in <work>: upstream's paths are relative."""
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    caps = [c for c in json.load(open(captions)) if c.get("text")]
    h = hashlib.sha1(open(captions, "rb").read()).hexdigest()
    mark = os.path.join("chroma_db", video, "captions.sha1")
    if os.path.exists(os.path.join("chroma_db", video)) and (not os.path.exists(mark) or open(mark).read() != h):
        shutil.rmtree(os.path.join("chroma_db", video))
        shutil.rmtree("events_diary", ignore_errors=True)   # summaries made from the old index
    from egorag.agents.RagAgent import RagAgent
    from egorag.database.Chroma import Chroma

    class SeenOnly(Chroma):
        """With .cutoff set (a clock integer), every retrieval is limited to clips that ended by then."""
        cutoff = None

        def custom_query(self, query_texts, n_results=3, where=None, filter_first=True):
            if self.cutoff is not None:
                seen = {"end_time": {"$lte": self.cutoff}}
                where = {"$and": [where, seen]} if where else seen
            return super().custom_query(query_texts, n_results, where, filter_first)

    agent = RagAgent(database_t=SeenOnly(name=video), name=video, video_base_dir=".")
    clip = lambda i: i.rsplit("_", 1)[0]
    want = {f"{c['date']}-{c['start_time']}-{c['end_time']}" for c in caps}
    if {clip(i) for i in agent.database_t.collection.get(include=[])["ids"]} != want:
        agent.create_database_from_json(captions)   # resumes; it skips an entry it fails on, so check
        ids = agent.database_t.collection.get(include=[])["ids"]
        assert {clip(i) for i in ids} == want, f"index has {len({clip(i) for i in ids})} clips, captions {len(want)}"
        open(mark, "w").write(h)
        print(f"index: {len(ids)} sentences of {len(want)} clips in {os.path.join(work, 'chroma_db', video)}")
    return agent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--memory_files", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--captions", help="<video>_30sec.json (caption_clips_egogpt.py)")
    ap.add_argument("--out", help="the memory folder")
    ap.add_argument("--work", help="the Chroma index and upstream's events_diary folder")
    ap.add_argument("--step", nargs="+", default=["db", "events"], choices=["db", "events"])
    a = ap.parse_args()
    if a.memory_files:
        print(" ".join(MEMORY))
        return
    V = a.video
    a.captions, a.out, a.work = map(os.path.abspath, (a.captions, a.out, a.work))
    files = {lvl: os.path.join(a.out, m.format(video=V)) for lvl, m in zip(["30sec", "l1", "l2", "l3"], MEMORY)}
    if a.captions != files["30sec"]:
        shutil.copy(a.captions, files["30sec"])
    agent = open_agent(V, files["30sec"], a.work)
    levels = ["l1", "l2", "l3"]
    if "events" not in a.step or all(os.path.exists(files[k]) and os.path.getsize(files[k]) > 2 for k in levels):
        return
    paid_api(deepseek=True)
    del agent
    from egorag.utils.gen_event import gen_event
    gen_event(V, "events_diary")   # resumes per level; a level it fails on is missing or empty below
    for n, k in enumerate(levels):
        src = os.path.join("events_diary", f"{V}_{k}", f"{k}_day1.json")
        rows = json.load(open(src)) if os.path.exists(src) else []
        if not rows or any((r.get("generated_text") or "").strip() in ("", "error") for r in rows):
            for j in levels[n:]:   # upstream would skip the level's file on a rerun: remove it and what was made from it
                shutil.rmtree(os.path.join("events_diary", f"{V}_{j}"), ignore_errors=True)
            sys.exit(f"{k} summaries missing or failed ({src}): rerun to retry")
        shutil.copy(src, files[k])
        print(f"{k}: {len(rows)} summaries -> {files[k]}")


if __name__ == "__main__":
    main()
