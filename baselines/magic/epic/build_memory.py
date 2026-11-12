#!/usr/bin/env python3
"""Build MAGIC-Video's memory for one EPIC-Morph video: everything that does not depend on the questions.

Every LLM call goes to the local server at $MAGIC_LLM_BASE_URL (epic/hf_server.py, started by run_magic.sh);
the text (Qwen3-Embedding-4B) and visual (VLM2Vec-V2.0) embedders run in this process. Steps (each skips
existing output, in order), all written flat into --out:

  captions  MERIT's 30 s captions for the arm (same backbone)        -> <video>_30sec.json
  multi     3 min / 10 min / 1 h summaries (EgoRAG gen_event prompts) -> <video>_{3min,10min,1h}.json
  openie    NER + triples per 30 s caption (WorldMM OpenIE)           -> openie_results_local.json
  semantic  semantic extraction per 10-caption group, then incremental consolidation (a snapshot per group)
                                                                      -> semantic_{extraction,consolidation}_results_local.json
  visual    VLM2Vec embedding per 30 s clip (16 frames)               -> visual_embeddings.pkl
  graph     the Multimodal Memory Graph (episode / visual / entity / semantic nodes, text embeddings) -> graph.pkl

Streaming: each piece reads only video before its own end time, and at question time the graph is filtered
to nodes ending <= ceil(t_ask) and the semantic snapshot to the last group before it (MAGIC's own
query_time filter). The Narrative Memory Chains are the one whole-video component, so they are not built
here: run_qa.py builds them per question cutoff (chains.py).

Clip paths in the memory are relative to --clip_root (as in MERIT's captions); the process works from there.

    python build_memory.py --video <V> --out <memory dir> --merit_captions <V>_30sec.json --clip_root <clips>
    python build_memory.py --memory_files        # the memory's file names, with {video}
"""
import argparse
import ast
import copy
import json
import logging
import os
import pickle
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build")

STEPS = ["captions", "multi", "openie", "semantic", "visual", "graph"]
TAG = "local"   # the model tag in WorldMM's file names (openie_results_<tag>.json, ...)
GRAN = ["30sec", "3min", "10min", "1h"]
MEMORY = [f"{{video}}_{g}.json" for g in GRAN] + [
    f"openie_results_{TAG}.json", f"semantic_extraction_results_{TAG}.json",
    f"semantic_consolidation_results_{TAG}.json", "visual_embeddings.pkl", "graph.pkl"]
# MAGIC's embedders; run_magic.sh points these at local copies when MAGIC_MODELS is set
TEXT_EMB = os.environ.get("MAGIC_TEXT_EMB", "Qwen/Qwen3-Embedding-4B")
VIS_EMB = os.environ.get("MAGIC_VIS_EMB", "VLM2Vec/VLM2Vec-V2.0")   # LoRA on Qwen2-VL-2B-Instruct
SRC = os.environ.get("MAGIC_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MAGIC-video"))


def jload(p):
    return json.load(open(p))


def jsave(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def secs(hhmmssff):
    s = str(hhmmssff).zfill(8)
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


def hhmmssff(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}{sec % 3600 // 60:02d}{sec % 60:02d}00"


def llm():
    from worldmm.llm import LLMModel
    return LLMModel(model_name=TAG)


def memory_paths(out, video):
    return [os.path.join(out, m.format(video=video)) for m in MEMORY]


def cap(out, video, g):
    return os.path.join(out, f"{video}_{g}.json")


def into(out, name, write):
    """Run write(tmp dir) (a WorldMM step that saves <name> into a directory) and move the file into out
    only when it is complete, so an interrupted step is redone rather than taken as finished."""
    tmp = os.path.join(out, f".tmp_{name}")
    shutil.rmtree(tmp, ignore_errors=True)
    write(tmp)
    os.replace(os.path.join(tmp, name), os.path.join(out, name))
    shutil.rmtree(tmp)


# ----------------------------------------------------------------------------- captions
def step_captions(a):
    """MERIT's 30 s captions (made with the arm's backbone): MAGIC's EgoLife caption layout."""
    caps = jload(a.merit_captions)
    assert caps and all(c.get("text") for c in caps), f"uncaptioned clips in {a.merit_captions}"
    for c in caps:
        c.pop("frame_dir", None)                  # MERIT's frames; MAGIC reads the clips
        assert os.path.exists(c["video_path"]), f"missing clip {a.clip_root}/{c['video_path']}"
    jsave(caps, cap(a.out, a.video, "30sec"))
    log.info(f"captions: {len(caps)} clips from {a.merit_captions}")


# ----------------------------------------------------------------------------- multi
def egorag_prompts():
    """The 3 min / 10 min / 1 h system prompts from EgoRAG's gen_event.py (read, not imported: it needs chromadb)."""
    tree = ast.parse(open(os.path.join(SRC, "src", "EgoRAG", "egorag", "utils", "gen_event.py")).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id.startswith("system_message_"):
            out[node.targets[0].id] = "".join(v.value for v in node.value.values if isinstance(v, ast.Constant))
    return out


def step_multi(a):
    sm, model = egorag_prompts(), llm()
    caps = jload(cap(a.out, a.video, "30sec"))

    def gpt(system, prompt, **kw):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return (model.generate(msgs, **kw) or "").strip()

    # 3 min: aligned 180 s windows over the 30 s captions (gen_event.process_day)
    win = defaultdict(list)
    for c in caps:
        win[secs(c["start_time"]) // 180].append(c)

    def one3(k):
        ctx = [c["text"] for c in win[k]]
        return {"text": gpt(sm["system_message_3m"], f"All descriptions: {ctx}", max_tokens=2048, temperature=0.7, top_p=0.95),
                "date": "DAY1", "start_time": hhmmssff(k * 180), "end_time": hhmmssff(k * 180 + 180)}
    with ThreadPoolExecutor(16) as ex:
        m3 = list(ex.map(one3, sorted(win)))

    # 10 min / 1 h: consecutive batches by start time (gen_event.process_json_batches / get_event_1h)
    def batches(items, span):
        out, b = [], []
        for it in items:
            if b and secs(it["start_time"]) >= secs(b[0]["start_time"]) + span:
                out.append(b)
                b = []
            b.append(it)
        return out + ([b] if b else [])

    def agg(b, system, suffix, **kw):
        ctx = "\n".join(x["text"] for x in b)
        return {"text": gpt(system, f"All descriptions: {ctx}. {suffix}", **kw), "date": "DAY1",
                "start_time": b[0]["start_time"], "end_time": b[-1]["end_time"]}
    with ThreadPoolExecutor(16) as ex:
        m10 = list(ex.map(lambda b: agg(b, sm["system_message_10m"], "Strictly output below 300 words in total."), batches(m3, 600)))
    m1h = [agg(b, sm["system_message_1h"], "The output should be around 1500 words in total.", max_tokens=2048, temperature=0.7, top_p=0.95)
           for b in batches(m10, 3600)]
    for g, x in [("3min", m3), ("10min", m10), ("1h", m1h)]:   # 1h last: it marks the step done
        jsave(x, cap(a.out, a.video, g))
    log.info(f"multi: 3min {len(m3)}, 10min {len(m10)}, 1h {len(m1h)}; sample 3min: {m3[0]['text'][:200]}")


# ----------------------------------------------------------------------------- openie
def step_openie(a):
    from worldmm.memory.episodic.openie import OpenIE
    caps = jload(cap(a.out, a.video, "30sec"))
    into(a.out, f"openie_results_{TAG}.json", lambda d: OpenIE(llm()).batch_openie([c["text"] for c in caps], output_dir=d))
    oie = jload(os.path.join(a.out, f"openie_results_{TAG}.json"))
    log.info(f"openie: {sum(map(len, oie['triple_results'].values()))} triples over {len(caps)} captions")


# ----------------------------------------------------------------------------- semantic
def step_semantic(a, emb):
    ext = os.path.join(a.out, f"semantic_extraction_results_{TAG}.json")
    from worldmm.memory.episodic.utils import compute_mdhash_id
    from worldmm.memory.semantic import SemanticConsolidation, SemanticExtraction
    caps = jload(cap(a.out, a.video, "30sec"))
    oie = jload(os.path.join(a.out, f"openie_results_{TAG}.json"))["triple_results"]
    if not os.path.exists(ext):
        batch = {}   # extract_semantic_triples.py: groups of period=10 captions keyed by the group's end
        for i in range(0, len(caps), 10):
            chunk = caps[i:i + 10]
            key = chunk[-1]["date"][-1] + chunk[-1]["end_time"].zfill(8)
            batch[key] = [t for c in chunk for t in oie[compute_mdhash_id(c["text"], prefix="chunk-")]]
        into(a.out, os.path.basename(ext), lambda d: SemanticExtraction(llm()).batch_semantic_extraction(batch, output_dir=d))
    sem = jload(ext)
    # consolidate_semantic_memory.py, verbatim logic: incremental, one snapshot per group timestamp
    cons = SemanticConsolidation(llm(), emb)
    acc_t, acc_e, snaps = [], [], {}
    for ts in sorted(sem["semantic_triples"]):
        cur_t = sem["semantic_triples"].get(ts, [])
        cur_e = [[f"{ts}_{i}" for i in ev] for ev in sem["episodic_evidence"].get(ts, [])]
        new_t, new_e, rm = cons.batch_semantic_consolidation((acc_t.copy(), acc_e.copy()), (cur_t, cur_e))
        rm = {(tuple(t), tuple(e)) for t, e in rm}
        keep = [(t, e) for t, e in zip(acc_t, acc_e) if (tuple(t), tuple(e)) not in rm]
        acc_t = [t for t, _ in keep] + list(new_t)
        acc_e = [e for _, e in keep] + list(new_e)
        snaps[ts] = {"consolidated_semantic_triples": copy.deepcopy(acc_t)}
    jsave(snaps, os.path.join(a.out, f"semantic_consolidation_results_{TAG}.json"))
    log.info(f"semantic: {sum(map(len, sem['semantic_triples'].values()))} extracted -> {len(acc_t)} consolidated, {len(snaps)} snapshots")


# ----------------------------------------------------------------------------- visual
def step_visual(a, emb):
    caps = jload(cap(a.out, a.video, "30sec"))
    emb.load_model(model_type="vision")
    out = {c["video_path"]: emb.encode_video([c["video_path"]], nframes=a.num_frames)[0] for c in caps}
    p = os.path.join(a.out, "visual_embeddings.pkl")
    pickle.dump(out, open(p + ".tmp", "wb"))
    os.replace(p + ".tmp", p)
    log.info(f"visual: {len(out)} clip embeddings, dim {next(iter(out.values())).shape}")


# ----------------------------------------------------------------------------- graph
def step_graph(a, emb):
    from worldmm.memory.unified.graph import MultimodalGraph
    caps = jload(cap(a.out, a.video, "30sec"))
    g = MultimodalGraph()
    g.add_episode_nodes({x: cap(a.out, a.video, x) for x in GRAN})
    g.add_visual_nodes(caps, pickle.load(open(os.path.join(a.out, "visual_embeddings.pkl"), "rb")))
    g.add_entity_nodes(os.path.join(a.out, f"openie_results_{TAG}.json"), caps, person="I")
    g.add_semantic_nodes(os.path.join(a.out, f"semantic_consolidation_results_{TAG}.json"), person="I")
    emb.load_model(model_type="text")
    g.compute_text_embeddings(emb, batch_size=64)
    p = os.path.join(a.out, "graph.pkl")
    g.save(p + ".tmp")
    os.replace(p + ".tmp", p)
    log.info(f"graph: {g.stats()}")


DONE = {"captions": ["{video}_30sec.json"], "multi": ["{video}_1h.json"], "openie": [f"openie_results_{TAG}.json"],
        "semantic": [f"semantic_consolidation_results_{TAG}.json"], "visual": ["visual_embeddings.pkl"], "graph": ["graph.pkl"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video")
    ap.add_argument("--out", help="the memory folder (<results>/<arm>/<video>)")
    ap.add_argument("--merit_captions", help="MERIT's <video>_30sec.json for the arm")
    ap.add_argument("--clip_root", help="the 30 s clips; the captions' video_path is relative to it")
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--memory_files", action="store_true", help="print the memory's file names and exit")
    a = ap.parse_args()
    if a.memory_files:
        print(" ".join(MEMORY))
        return
    a.out, a.merit_captions = os.path.abspath(a.out), os.path.abspath(a.merit_captions)
    os.makedirs(a.out, exist_ok=True)
    os.chdir(a.clip_root)
    emb = None
    for st in STEPS:
        if st not in a.steps or all(os.path.exists(os.path.join(a.out, f.format(video=a.video))) for f in DONE[st]):
            continue
        log.info(f"=== {st} ({a.video})")
        if st in ("semantic", "visual", "graph"):
            if emb is None:
                from worldmm.embedding import EmbeddingModel
                emb = EmbeddingModel(text_model_name=TEXT_EMB, vis_model_name=VIS_EMB)
            globals()[f"step_{st}"](a, emb)
        else:
            globals()[f"step_{st}"](a)
    missing = [p for p in memory_paths(a.out, a.video) if not os.path.exists(p)]
    if set(a.steps) == set(STEPS) and missing:
        sys.exit(f"memory incomplete, missing {missing}")


if __name__ == "__main__":
    main()
