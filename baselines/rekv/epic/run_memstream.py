#!/usr/bin/env python3
"""Answer one video's questions of an EPIC-Morph bank with ReKV on Qwen3-VL, using the vendored port
(../vendor/MemStream: Vatsal Agarwal's ReKV fork, model/qwen3_vl_rekv.py), in its plain ReKV configuration.

One process, nothing on disk: the video is encoded once with sliding-window attention (the KV-cache of every
block is offloaded to host RAM, as upstream ReKV does), then every question is answered from it.

Plain ReKV = prefill_method "swa", retrieval_method "rekv", no MEMSTREAM_* research knob (refused when set),
with the paper's settings: 0.5 fps, 196 visual tokens per frame, window n_local 15000, retrieve 64. In the
port a block is one Qwen3-VL temporal patch = 2 sampled frames = 4 s of video.

Streaming: the port encodes a whole video per call (its Qwen3-VL path refuses incremental encoding), so the
cutoff is applied at retrieval: a question asked at t_ask may only retrieve blocks whose last frame is at or
before t_ask (ContextManager._calc_block_topk is wrapped to score only those; the wrapper also asserts it).
Attention is causal, so the cache of those blocks is what a stream stopped at t_ask would hold.

--blind answers the same prompt with the unpatched model and no video (the no-memory floor).

Writes --answers (.json.gz): one row per question in bank order, keyed by the bank's `key`. Resumes from it: a
row counts as done when it is from this VERSION, these settings and vendored files (memory_hash) and the same
question (qhash). A resumed run encodes the video again.

    python run_memstream.py --video <V> --video_file <V>.mp4 --bank_file <V>_questions.json \
        --answers answers_<bank>.json.gz --memory_json memory.json
"""
import argparse
import gzip
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "..", "vendor")
LETTERS = "ABCD"
VERSION = "m1"   # bump when a change alters answers
# The paper's settings (upstream README: --sample_fps 0.5 --n_local 15000 --retrieve_size 64; 196 tokens per frame).
# encode_chunk_size (frames per forward pass; upstream default 64) is lowered to fit one 48 GB GPU.
SETTINGS = {"sample_fps": 0.5, "frame_tokens": 196, "n_local": 15000, "retrieve_size": 64, "retrieve_chunk_size": 1,
            "encode_chunk_size": 16, "prefill_method": "swa", "retrieval_method": "rekv", "max_new_tokens": 16}
INIT_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"   # the port's load_model


def qhash(question, options):   # as baselines/merit/epic/prepare_epic.py
    return hashlib.sha1(json.dumps([question, [str(o) for o in options]]).encode()).hexdigest()[:12]


def letter(text):   # the port's (and upstream's) extract_characters_regex
    s = text.strip()
    return (s[s.index(")") - 1: s.index(")")] if ")" in s else s[:1]).upper()


def prompts(q, opts):   # the port's format_mcqa_prompt + get_prompt(mc=True)
    formatted = f"Question: {q}\nOptions:\n" + "\n".join(f"({L}) {o}" for L, o in zip(LETTERS, opts)) + "\nOnly give the best option."
    return {"question": q, "prompt": f"\n{formatted}<|im_end|><|im_start|>assistant\nBest option: ("}


def config_hash(model):
    """What the answers depend on besides the video: settings, version, model, and the vendored files."""
    src = open(os.path.join(VENDOR, "SOURCE.sha256")).read()
    return hashlib.sha1(json.dumps([VERSION, SETTINGS, os.path.basename(model.rstrip("/")), src]).encode()).hexdigest()[:16]


def read_video(video_file, sample_fps, tokens):
    """Every (fps / sample_fps)-th frame, as the port's load_video picks them (frame i is at i / sample_fps
    seconds), resized to the size that gives `tokens` visual tokens (square video: 448 x 448)."""
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,avg_frame_rate:format=duration", "-of", "json", video_file],
                                  capture_output=True, text=True, check=True).stdout)
    s = j["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    stride = int(round(float(num) / float(den)) / sample_fps)
    gw = max(1, round((tokens * s["width"] / s["height"]) ** 0.5))
    w, h = gw * 32, max(1, round(tokens / gw)) * 32
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video_file, "-vf",
                          f"select=not(mod(n\\,{stride})),scale={w}:{h}:flags=bicubic", "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), float(j["format"]["duration"])


def cap_retrieval():
    """Let a ContextManager with `_epic_visible = nb` retrieve only among its first nb blocks."""
    from model.attention.kv_cache_manager import ContextManager
    orig = ContextManager._calc_block_topk

    def capped(self, *args, **kwargs):
        nb = getattr(self, "_epic_visible", None)
        if nb is None:
            return orig(self, *args, **kwargs)
        assert self.init_exc and 0 < nb <= self.num_global_block, (self.init_exc, nb, self.num_global_block)
        total, lengths = self.num_global_block, [b.length for b in self.block_k]
        self.num_global_block = nb          # what the retrieval scores: the first nb representative keys
        for b in self.block_k:
            b.length = nb
        try:
            ret = orig(self, *args, **kwargs)
        finally:
            self.num_global_block = total
            for b, n in zip(self.block_k, lengths):
                b.length = n
        ret = [[int(i) for i in r if int(i) < nb] for r in ret]   # fewer than top-k visible: orig lists every block
        assert all(r and max(r) < nb for r in ret), "retrieval reached past the question's cutoff"
        self._epic_retrieved = ret[0]
        return ret
    ContextManager._calc_block_topk = capped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--video_file", help="not needed with --blind")
    ap.add_argument("--bank_file", required=True, help="epic-morph outputs/v3/banks/<bank>/<video>_questions.json")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--memory_json", help="where to write the description of the encoded video (blocks, sizes, times)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--blind", action="store_true", help="no video: the unpatched model answers the same prompt")
    ap.add_argument("--limit", type=int, default=0, help="answer only the first N unanswered questions (smoke tests)")
    a = ap.parse_args()
    knobs = sorted(k for k in os.environ if k.startswith(("MEMSTREAM_", "REKV_ROPE_")))
    assert not knobs, f"research knobs of the port are set, this would not be plain ReKV: {knobs}"
    torch.manual_seed(0)
    qs = json.load(open(a.bank_file))["questions"]
    assert all(q["video"] == a.video for q in qs), "bank file is for another video"
    mh = "blind" if a.blind else config_hash(a.model)
    done = {}
    if os.path.exists(a.answers):
        qh = {q["key"]: qhash(q["q"], q["gemini_mcq"]["options"]) for q in qs}   # a question edited under the same key is redone
        done = {r["key"]: r for r in json.load(gzip.open(a.answers, "rt"))
                if r.get("version") == VERSION and r.get("memory_hash") == mh and r.get("qhash") == qh.get(r["key"])}
    todo = [q for q in qs if q["key"] not in done][: a.limit or None]
    print(f"# {a.video}: {len(qs)} questions, {len(done)} done, {len(todo)} to answer ({mh})", flush=True)

    def save():
        with gzip.open(a.answers + ".tmp", "wt") as f:
            json.dump([done[q["key"]] for q in qs if q["key"] in done], f, indent=1)
        os.replace(a.answers + ".tmp", a.answers)

    period = 1 / SETTINGS["sample_fps"]   # seconds between sampled frames; block j holds frames 2j, 2j+1
    if todo and a.blind:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        tok = AutoProcessor.from_pretrained(a.model).tokenizer
        model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    elif todo:
        sys.path.insert(0, os.path.join(VENDOR, "MemStream"))
        from model.qwen3_vl_rekv import load_model
        cap_retrieval()
        frames, duration = read_video(a.video_file, SETTINGS["sample_fps"], SETTINGS["frame_tokens"])
        print(f"# {len(frames)} frames {frames.shape[2]}x{frames.shape[1]} at {SETTINGS['sample_fps']} fps of {duration:.0f} s", flush=True)
        model, _ = load_model(model_path=a.model, n_local=SETTINGS["n_local"], topk=SETTINGS["retrieve_size"],
                              encode_chunk_size=SETTINGS["encode_chunk_size"], chunk_size=SETTINGS["retrieve_chunk_size"],
                              sample_fps=SETTINGS["sample_fps"], target_frames=None, prefill_method=SETTINGS["prefill_method"],
                              retrieval_method=SETTINGS["retrieval_method"], min_tokens=SETTINGS["frame_tokens"],
                              max_tokens=SETTINGS["frame_tokens"], pin_memory=False)
        t0 = time.time()
        model.clear_cache()
        model.encode_init_prompt()
        model.encode_video(torch.from_numpy(frames))
        kv = model.kv_cache
        nblocks, sizes = kv[0].num_global_block, list(kv[0].adaptive_block_sizes)
        assert kv[0].init_exc and nblocks == len(sizes) == len(frames) // 2, \
            f"expected every block offloaded: init_exc {kv[0].init_exc}, offloaded {nblocks}, appended {len(sizes)}, frames {len(frames)}"
        assert kv[0].block_size == SETTINGS["frame_tokens"], f"{kv[0].block_size} visual tokens per block"
        mem = {"version": VERSION, "settings": SETTINGS, "model": os.path.basename(a.model.rstrip("/")), "video": a.video,
               "duration": duration, "frames": len(frames), "blocks": nblocks, "block_seconds": 2 * period,
               "visual_tokens_per_block": kv[0].block_size, "tokens_per_block": [min(sizes), max(sizes)],
               "kv_cache_gb": round(model.calc_memory_usage() / 2**30, 2), "encode_seconds": round(time.time() - t0),
               "gpu_peak_gb": round(sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30, 1),
               "memory_hash": mh}
        print(f"# encoded: {json.dumps(mem)}", flush=True)
        if a.memory_json:
            json.dump(mem, open(a.memory_json, "w"), indent=1)
    for n, q in enumerate(sorted(todo, key=lambda q: q["t_ask"]), 1):
        opts = [str(o) for o in q["gemini_mcq"]["options"]]
        gold, text_in, t0 = LETTERS[q["gemini_mcq"]["answer_index"]], prompts(q["q"], opts), time.time()
        cut = q["t_ask"] if q.get("streaming") and q.get("t_ask") is not None else None
        nb, hits = 0, {}
        if a.blind:
            ids = tok(INIT_PROMPT + text_in["prompt"], return_tensors="pt").input_ids.to(model.device)
            with torch.inference_mode():
                out = model.generate(input_ids=ids, max_new_tokens=SETTINGS["max_new_tokens"], do_sample=False)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        else:
            # block j is seen once its last frame, at (2j + 1) * period seconds, is at or before the question
            nb = nblocks if cut is None else min(nblocks, int((cut / period + 1) // 2))
            for layer in kv:
                layer._epic_visible, layer._epic_retrieved = nb, None
            text = model.question_answering(text_in, max_new_tokens=SETTINGS["max_new_tokens"])
            for layer in kv:
                for i in layer._epic_retrieved:
                    hits[i] = hits.get(i, 0) + 1
            assert cut is None or all((2 * i + 1) * period <= cut for i in hits), "a retrieved block ends after t_ask"
        resp = letter(text)
        done[q["key"]] = d = {
            "key": q["key"], "qhash": qhash(q["q"], opts), "type": q["form"], "question": q["q"],
            "choices": dict(zip(LETTERS, opts)), "answer": gold, "response": resp, "raw": text,
            "evaluate": resp == gold, "valid": bool(resp) and resp in LETTERS[: len(opts)], "t_ask": q.get("t_ask"), "cutoff": cut,
            "blocks_visible": nb, "seen_until": (2 * nb - 1) * period if nb else 0,
            # start second of every retrieved block -> number of layers that retrieved it
            "retrieved": {str(int(2 * i * period)): c for i, c in sorted(hits.items())},
            "seconds": round(time.time() - t0, 1), "version": VERSION, "memory_hash": mh}
        print(f"{q['key']} @ {cut}: {text!r} -> {resp} gold {gold} {'right' if d['evaluate'] else 'wrong'} "
              f"({nb} blocks visible, {len(hits)} retrieved by some layer, {d['seconds']}s)", flush=True)
        if n % 10 == 0:
            save()
    save()
    rows = [done[q["key"]] for q in qs if q["key"] in done]
    print(f"# {len(rows)}/{len(qs)} answered, {sum(x['evaluate'] for x in rows)} right, "
          f"{sum(not x['valid'] for x in rows)} without a valid option -> {a.answers}; "
          f"host RAM peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
