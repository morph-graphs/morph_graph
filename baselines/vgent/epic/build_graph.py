#!/usr/bin/env python3
"""Vgent's offline graph for one HD-EPIC video (upstream vgent_graph.py, one video, one GPU), and the helpers
run_qa.py shares.

Vgent's code is used as released (utils/vgent.py Vgent: construct_graph, extract_keywords, retrieve_nodes,
refine_nodes, aggregate_nodes; its backbone wrapper models/qwenvl.py; its prompts). What this adds around it:
- the frames come from prepare_frames.py (memory-mapped uint8) instead of load_video;
- each 64-frame chunk's extraction (generate_entities: the one backbone call per chunk) is cached in
  <out>/<video>_nodes.jsonl, so a build resumes, and run_qa.py rebuilds the graph of any prefix of the video
  with construct_graph and no backbone call for the full chunks;
- compute_text_similarity embeds each string once (upstream re-embeds every graph key at every comparison, on
  the CPU); the similarities are the same up to float rounding.

Memory files: <video>_nodes.jsonl (a header line, then one line per chunk) and <video>_graph.pkl (what
upstream saves: {"video_graph", "entity_graph"}; both None for a video under chunk_size * n_retrieval
frames, where upstream builds no graph).

    python build_graph.py --video <V> --arm qwenvl25_7b --frames <frames dir> --out <results dir>
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.environ.get("VGENT_SRC") or os.path.join(HERE, "..", "Vgent"))
MEMORY = ["{video}_nodes.jsonl", "{video}_graph.pkl"]
# upstream defaults (vgent_graph.py / utils/config.py); task only matters for one MLVU-specific rule
SETTINGS = dict(chunk_size=64, n_retrieval=20, n_refine=5, fps=1.0, total_pixels=16384, uniform_frame=450, task="epic")
# the bug in refine_nodes (see README "Upstream as released"): VGENT_FIX=1 applies this one-line fix
REFINE_BUG = ("info.update(json.loads(", "info = (json.loads(")


def load_frames(frame_dir, video):
    meta = json.load(open(os.path.join(frame_dir, video + ".json")))
    frames = torch.from_numpy(np.load(os.path.join(frame_dir, video + ".npy"), mmap_mode="r"))   # read-only: torch warns, nothing writes
    assert len(frames) == meta["n"] == len(meta["times"])
    return frames, meta


def load_vgent(arm, fix=False):
    """Upstream's Vgent with the arm's backbone (a key of its MODEL_MAP) and BAAI/bge-large-en-v1.5."""
    sys.path.insert(0, SRC)
    import utils.retrieval
    if fix:   # utils/vgent.py with the one line replaced, loaded under its own name
        import importlib.util
        spec = importlib.util.spec_from_file_location("utils.vgent", os.path.join(SRC, "utils", "vgent.py"))
        code = open(spec.origin).read()
        assert code.count(REFINE_BUG[0]) == 1, "upstream refine_nodes changed: check REFINE_BUG"
        uv = sys.modules["utils.vgent"] = importlib.util.module_from_spec(spec)
        exec(compile(code.replace(*REFINE_BUG), spec.origin, "exec"), uv.__dict__)
    else:
        import utils.vgent as uv
    torch.manual_seed(0)
    vg = uv.Vgent(argparse.Namespace(model_name=arm, **SETTINGS))
    vg.video_llm.eval()

    # frames are stored uint8; upstream hands the processor floats. Every backbone call is logged.
    vg.calls, call = [], vg.mllm_response
    def mllm_response(llm, tok, proc, text, images, video, *a, **k):
        t0 = time.time()
        out = call(llm, tok, proc, text, images, None if video is None else video.float(), *a, **k)
        vg.calls.append({"frames": 0 if video is None else len(video), "seconds": round(time.time() - t0, 2), "output": out})
        return out
    vg.mllm_response = mllm_response

    emb, tok, cache = vg.embedding_model.cuda().eval(), vg.embedding_tokenizer, {}
    def embed(texts):
        new = [t for t in dict.fromkeys(texts) if t not in cache]
        for i in range(0, len(new), 64):
            enc = tok(new[i:i + 64], padding=True, truncation=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                e = torch.nn.functional.normalize(emb(**enc)[0][:, 0], p=2, dim=1).cpu()
            cache.update(zip(new[i:i + 64], e))
        return torch.stack([cache[t] for t in texts]) if texts else torch.zeros(0, emb.config.hidden_size)
    def compute_text_similarity(query_list, key_list, embedding_model, tokenizer, return_all=False):
        sims = embed(query_list) @ embed(key_list).T
        return sims if return_all else torch.mean(sims)
    vg.exact_similarity = utils.retrieval.compute_text_similarity   # upstream's, for a check
    uv.compute_text_similarity = utils.retrieval.compute_text_similarity = compute_text_similarity
    return vg


def header(video, arm, meta):
    return {"video": video, "arm": arm, "frames": meta["n"], "size": [meta["H"], meta["W"]],
            **{k: SETTINGS[k] for k in ("chunk_size", "fps", "total_pixels")}}


def read_nodes(path, head):
    """{(chunk index, frames in it): node} from a nodes file whose header matches this video and settings."""
    if not os.path.exists(path):
        return {}
    lines = [json.loads(l) for l in open(path) if l.strip()]
    assert lines and lines[0] == head, f"{path} was made for {lines[:1]}, not {head}: remove it to rebuild"
    return {(r["chunk"], r["frames"]): r for r in lines[1:]}


def construct(vg, frames, nodes, write):
    """Upstream construct_graph on these frames, taking each chunk's extraction from `nodes` when it is there
    and appending new ones to the file `write`."""
    gen, i = vg.generate_entities, [0]
    def cached(prompt, video_input, max_new_tokens=512):
        k = (i[0], len(video_input)); i[0] += 1
        if k not in nodes:
            t0 = time.time()
            e, a, s = gen(prompt, video_input, max_new_tokens)
            nodes[k] = {"chunk": k[0], "frames": k[1], "entities": e, "actions": a, "scenes": s,
                        "seconds": round(time.time() - t0, 1)}
            with open(write, "a") as f:
                f.write(json.dumps(nodes[k]) + "\n")
            print(f"chunk {k[0]} ({k[1]} frames): {len(e)} entities, {len(a)} actions, {len(s)} scenes, "
                  f"{nodes[k]['seconds']}s", flush=True)
        return nodes[k]["entities"], nodes[k]["actions"], nodes[k]["scenes"]
    vg.generate_entities = cached
    try:
        return vg.construct_graph([frames], None)   # no subtitles: HD-EPIC comes without a transcript
    finally:
        vg.generate_entities = gen


def memory_hash(out, video):
    return hashlib.sha1(open(os.path.join(out, MEMORY[0].format(video=video)), "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--memory_files", action="store_true", help="print the memory's file patterns and exit")
    ap.add_argument("--video")
    ap.add_argument("--arm", default="qwenvl25_7b")
    ap.add_argument("--frames", help="prepare_frames.py --out")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.memory_files:
        print(" ".join(MEMORY))
        return
    os.makedirs(a.out, exist_ok=True)
    nodes_file, graph_file = (os.path.join(a.out, m.format(video=a.video)) for m in MEMORY)
    frames, meta = load_frames(a.frames, a.video)
    head = header(a.video, a.arm, meta)
    nodes = read_nodes(nodes_file, head)
    if not nodes and not os.path.exists(nodes_file):
        open(nodes_file, "w").write(json.dumps(head) + "\n")
    video_graph = entity_graph = None
    if len(frames) >= SETTINGS["chunk_size"] * SETTINGS["n_retrieval"]:   # upstream: no graph for a shorter video
        vg = load_vgent(a.arm)
        video_graph, entity_graph = construct(vg, frames, nodes, nodes_file)
        empty = [n for n, d in video_graph.nodes(data=True) if not (d["entities"] or d["actions"] or d["scenes"])]
        print(f"graph: {video_graph.number_of_nodes()} nodes, {video_graph.number_of_edges()} edges, "
              f"{len(entity_graph)} entity keys; {len(empty)} empty nodes (no valid JSON in 5 tries): {empty}")
    else:
        print(f"{len(frames)} frames < {SETTINGS['chunk_size'] * SETTINGS['n_retrieval']}: Vgent builds no graph")
    pickle.dump({"video_graph": video_graph, "entity_graph": entity_graph}, open(graph_file + ".tmp", "wb"))
    os.replace(graph_file + ".tmp", graph_file)
    print(f"memory {memory_hash(a.out, a.video)} -> {a.out}")


if __name__ == "__main__":
    main()
