#!/usr/bin/env python3
"""EGAgent's entity scene graph of one video, from its captions. PAID API (OpenAI GPT-4.1).

    build_graph.py --captions <results>/<V>_captions.json --out <results>/<V>_graph.json [--chunk_s 300] [--dry]

Upstream's prepare_datasources/create_entity_graph.py, step by step and with its own functions and prompts:
LangChain's LLMGraphTransformer (nodes Person / Location / Object; relations TALKS_TO / INTERACTS_WITH /
MENTIONS / USES) over a chunk of captions, then the relationship time-stamper over the same chunk. HD-EPIC has
no transcript, so captions are not fused with one and the time-stamper's transcript list is empty (upstream's
path for a video without subtitles).

Streaming: upstream extracts one graph per hour (EgoLife) or per video (Video-MME), which would let a
relationship before t_ask depend on captions after it. Here the graph is extracted per chunk of --chunk_s
seconds, each from its own captions only, and a question sees the chunks that end by its t_ask (run_qa.py).
Writes {"chunk_s", "chunks": [{"start", "end", "relationships": [upstream's rows, with intervals]}]};
resumes from a .partial file. --dry prints the calls it would make and makes none.
"""
import argparse
import asyncio
import contextvars
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import eg_paths
from eg_paths import clock


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_s", type=int, default=300)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1, help="chunks extracted at once (each chunk's two calls stay in order)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    caps = json.load(open(a.captions))
    assert a.chunk_s % eg_paths.CLIP_S == 0
    chunks = {}
    for c in caps:   # a caption belongs to the chunk it ends in, so a chunk is complete at its end time
        chunks.setdefault((c["end"] - 1) // a.chunk_s, []).append(c)
    if a.dry:
        print(f"{len(chunks)} chunks of {a.chunk_s} s: {2 * len(chunks)} {eg_paths.MEMORY_LLM} calls "
              f"(graph extraction + time stamps), {sum(len(c['caption']) for c in caps)} caption characters in all")
        return
    part = a.out + ".partial"
    done = {r["start"]: r for r in map(json.loads, open(part))} if os.path.exists(part) else {}
    from create_entity_graph import generate_graph_for_hour, get_rel_timestamper_llm, parse_relationships
    eg_paths.watch_api("graph", eg_paths.MEMORY_LLM)
    stamper = get_rel_timestamper_llm(model=eg_paths.MEMORY_LLM, config="fused_dt_and_captions")

    def extract(k):
        cs = chunks[k]
        start, end = cs[0]["start"], cs[-1]["end"]
        # upstream's EgoLife caption format (add_start_and_end_times_to_egolife_captions)
        context = [{f"clip_{c['start']:06d}": f"{clock(c['start'], True)} --> {clock(c['end'], True)} : {c['caption']}"} for c in cs]
        for t in range(a.tries):
            try:
                parsed = parse_relationships(asyncio.run(generate_graph_for_hour(context)))
                stamped = stamper.invoke({"relationships": parsed, "transcripts": "", "captions": context})
                lookup = {r["rel_id"]: r for r in parsed}
                rels = []
                for e in (x.model_dump() for x in stamped.relationships):   # as upstream merges them
                    info = lookup.get(e["relationship_id"])
                    if info:
                        rels.append({**e, **{f: info[f] for f in ("source_id", "source_type", "target_id", "target_type", "rel_type")}})
                return {"start": start, "end": end, "relationships": rels}
            except Exception as ex:
                print(f"chunk {start}-{end} try {t + 1}: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
        raise SystemExit(f"chunk {start}-{end} failed {a.tries} times; rerun to resume")

    # each chunk in a copy of this context (a new thread starts without the API watch); a finished chunk is saved at
    # once by this thread, and on a failure the chunks not yet started are dropped (as caption_epic.py)
    todo = [k for k in sorted(chunks) if chunks[k][0]["start"] not in done]
    ex = ThreadPoolExecutor(max(1, a.workers))
    futs = [ex.submit(contextvars.copy_context().run, extract, k) for k in todo]
    try:
        with open(part, "a") as f:
            for fut in as_completed(futs):
                row = fut.result()
                done[row["start"]] = row
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"chunk {row['start']}-{row['end']}: {len(row['relationships'])} relationships, e.g. {json.dumps(row['relationships'][:1])[:200]}", flush=True)
    except BaseException:
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown()
    out = [done[chunks[k][0]["start"]] for k in sorted(chunks)]
    json.dump({"chunk_s": a.chunk_s, "chunks": out}, open(a.out + ".tmp", "w"), indent=1)
    os.replace(a.out + ".tmp", a.out)
    if os.path.exists(part):
        os.remove(part)
    print(f"{len(out)} chunks, {sum(len(c['relationships']) for c in out)} relationships -> {a.out}")


if __name__ == "__main__":
    main()
