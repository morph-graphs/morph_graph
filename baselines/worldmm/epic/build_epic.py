#!/usr/bin/env python3
"""Build WorldMM's memories for one video from its 10 s captions, with WorldMM's own code, then put them in
the results folder as flat files (wmm_paths.memory_files). No questions are involved: the memory covers the
whole video, and eval_epic.py lets each question see only what ends by its t_ask.

Follows WorldMM's script/videomme/3_build_memory.sh (its per-video pipeline) step by step, skip-existing:
  1. multiscale episodic captions 30sec/3min/10min   worldmm.memory.episodic.multiscale (egocentric prompts, as EgoLife)
  2. episodic triples (OpenIE on 10sec)              preprocess/build_memory.py run_episodic
  3. semantic triples + streaming consolidation      preprocess/build_memory.py run_semantic (snapshots keyed by
                                                     the end time of each caption group)
  4. visual embeddings (VLM2Vec-V2, 10 frames/10 s)  preprocess/build_memory.py's process_caption_dir
  5. HippoRAG OpenIE for every granularity, once per video (WorldMM's eval would run it lazily at question
     time, for every question again); eval_epic.py reads it back only for captions that end by t_ask.

    python build_epic.py --video V --caption_root <dir> --meta_root <dir> --model qwen3vl-8b --out <results>/<arm>/<V>
"""
import argparse
import gc
import glob
import os
import shutil
import time

import wmm_paths  # noqa: F401  (first)
from wmm_paths import GRANULARITIES, memory_files
from build_memory import run_episodic, run_semantic
from visual_memory.extract_visual_features import process_caption_dir
from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory.episodic.memory import EpisodicMemory
from worldmm.memory.episodic.multiscale import generate_multiscale_memory

END = int("1" + "23595999")   # DAY1 23:59:59: index every caption


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--caption_root", required=True, help="<caption_root>/<V>/10sec.json from caption_epic.py")
    p.add_argument("--meta_root", required=True, help="WorldMM's intermediate files (triples, caches)")
    p.add_argument("--model", default="qwen3vl-8b")
    p.add_argument("--num_frames", type=int, default=10, help="videomme/3_build_memory.sh NUM_FRAMES")
    p.add_argument("--out", required=True, help="results folder: the memory files go here")
    a = p.parse_args()
    v, cap = a.video, os.path.join(a.caption_root, a.video)
    t0 = time.time()

    print("== 1. multiscale", flush=True)
    generate_multiscale_memory(caption_dir=cap, model_name=a.model, base_name="10sec.json", windows=[30, 180, 600],
                               granularity_names=["30sec", "3min", "10min"], perspective="egocentric")
    llm = LLMModel(model_name=a.model)
    print(f"== 2. episodic triples ({time.time() - t0:.0f}s)", flush=True)
    run_episodic([v], a.caption_root, a.meta_root, a.model, llm)

    emb = EmbeddingModel()   # WorldMM's defaults: Qwen3-Embedding-4B, VLM2Vec-V2.0 (WORLDMM_MODEL_DIR to load locally)
    emb.load_model(model_type="text")
    print(f"== 3. semantic ({time.time() - t0:.0f}s)", flush=True)
    run_semantic([v], a.caption_root, a.meta_root, a.model, llm, emb)
    sem = os.path.join(a.meta_root, "semantic_memory", v, f"semantic_consolidation_results_{a.model}.json")
    assert os.path.exists(sem), f"semantic consolidation missing: {sem}"

    print(f"== 4. visual ({time.time() - t0:.0f}s)", flush=True)
    vis = os.path.join(a.meta_root, "visual_memory", v, "visual_embeddings.pkl")
    if not os.path.exists(vis):
        emb.load_model(model_type="vision")
        process_caption_dir(a.caption_root, os.path.join(a.meta_root, "visual_memory"), emb, num_frames=a.num_frames, video_ids=[v])
    assert os.path.exists(vis), vis
    emb._vis_model = None   # step 5 needs only the text model; VLM2Vec still loaded OOMs a 16 GB GPU
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    print(f"== 5. HippoRAG OpenIE ({time.time() - t0:.0f}s)", flush=True)
    hippo = os.path.join(a.meta_root, "hippo_openie", v)
    if not os.path.exists(os.path.join(hippo, "DONE")):
        ep = EpisodicMemory(embedding_model=emb, llm_model=llm, prompt_template_manager=PromptTemplateManager(),
                            granularities=GRANULARITIES, save_dir_root=hippo)
        ep.load_captions_from_files({g: os.path.join(cap, f"{g}.json") for g in GRANULARITIES})
        ep.index(END)
        open(os.path.join(hippo, "DONE"), "w").write("ok\n")

    # the memory, as flat files of the results folder (copied last, so a complete set means a finished build)
    src = [os.path.join(cap, f"{g}.json") for g in GRANULARITIES] + [sem, vis]
    for g in GRANULARITIES:
        oie = glob.glob(os.path.join(hippo, g, "openie_results_ner_*.json"))
        assert len(oie) == 1, (g, oie)
        src += oie
    os.makedirs(a.out, exist_ok=True)
    for s, name in zip(src, memory_files(v)):
        shutil.copy(s, os.path.join(a.out, name + ".partial"))
    for name in memory_files(v):
        os.replace(os.path.join(a.out, name + ".partial"), os.path.join(a.out, name))
    print(f"build done for {v} in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
