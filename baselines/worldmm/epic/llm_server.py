#!/usr/bin/env python3
"""Serve one local Qwen backbone to WorldMM over a unix socket, batching concurrent requests.

WorldMM calls its LLM from many threads at once (captioning, OpenIE, semantic extraction, consolidation,
HippoRAG reranking). With WORLDMM_SERVER set, the fork's Qwen3VLModel loads no model and sends every call here
(after its own prompt preprocessing); the GPU thread runs up to --max_batch of them per generate() (greedy,
left padded). The server is its own process so the backbone can use another transformers than WorldMM:
Qwen3.8-27B (model type qwen3_5) needs transformers 5.8, WorldMM's VLM2Vec code needs 4.x. Needs no WorldMM import.

    python llm_server.py --model_path <dir or HF id> --address /tmp/wmm.sock [--max_batch 8]

Request: {"messages": [...chat messages with PIL images...], "max_new_tokens": int|None}
Reply:   {"text": str} or {"error": str}
Env: WORLDMM_AUTHKEY (shared with the client), WORLDMM_DEVICE_MAP (default cuda:0), WORLDMM_MAX_MEMORY
("0:30GiB,1:44GiB", with a split device map), WORLDMM_SERVER_LOG=1 (one line per batch).
"""
import argparse
import os
import queue
import threading
import time
import traceback
from multiprocessing.connection import Listener

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--address", required=True)
    ap.add_argument("--max_batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=2048, help="WorldMM's Qwen default")
    ap.add_argument("--ready_file", default=None)
    a = ap.parse_args()

    torch.manual_seed(42)
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    device_map = os.environ.get("WORLDMM_DEVICE_MAP", "cuda:0")
    kw = {}
    if device_map in ("auto", "sequential", "balanced", "balanced_low_0") and os.environ.get("WORLDMM_MAX_MEMORY"):
        kw["max_memory"] = {int(k): v for k, v in (x.split(":") for x in os.environ["WORLDMM_MAX_MEMORY"].split(","))}
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(a.model_path, dtype=torch.bfloat16, attn_implementation=attn,
                                                        device_map=device_map, **kw).eval()
    proc = AutoProcessor.from_pretrained(a.model_path)
    proc.tokenizer.padding_side = "left"
    # Qwen3.8 is a thinking model: run it in instruct mode, as MERIT does
    tmpl_kw = {"enable_thinking": False} if model.config.model_type.startswith("qwen3_5") else {}
    dm = getattr(model, "hf_device_map", {}) or {}
    off = sorted({str(d) for d in dm.values() if str(d) in ("cpu", "disk")})
    if off:   # CPU/disk offload makes decoding ~50x slower; refuse rather than crawl
        raise SystemExit(f"model partly offloaded to {off}; give it more GPU memory (WORLDMM_MAX_MEMORY)")
    first = next(model.parameters()).device
    print(f"loaded {a.model_path} ({model.config.model_type}, {attn}, {device_map}) in {time.time() - t0:.0f}s", flush=True)

    jobs = queue.Queue()
    stats = {"calls": 0, "batches": 0, "gen_s": 0.0}

    def run(batch):
        t_run = time.time()
        msgs = [b[0]["messages"] for b in batch]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **tmpl_kw) for m in msgs]
        imgs, vids = process_vision_info(msgs, image_patch_size=16)[:2]
        inputs = proc(text=texts, images=imgs, videos=vids, padding=True, return_tensors="pt").to(first)
        mnt = max(b[0].get("max_new_tokens") or a.max_new_tokens for b in batch)
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 max_new_tokens=mnt)
        out = out[:, inputs.input_ids.shape[1]:]
        if os.environ.get("WORLDMM_SERVER_LOG"):
            print(f"[server] batch {len(batch)} in {inputs.input_ids.shape[1]} out {out.shape[1]} tok "
                  f"imgs {len(imgs or [])} {time.time() - t_run:.1f}s", flush=True)
        return [t.strip() for t in proc.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)]

    def gpu_loop():
        while True:
            batch = [jobs.get()]
            deadline = time.time() + 0.05
            while len(batch) < a.max_batch:
                try:
                    batch.append(jobs.get(timeout=max(0.0, deadline - time.time())))
                except queue.Empty:
                    break
            t = time.time()
            try:
                outs = [{"text": x} for x in run(batch)]
            except Exception:            # e.g. OOM on a big batch: retry one by one
                torch.cuda.empty_cache()
                outs = []
                for b in batch:
                    try:
                        outs.append({"text": run([b])[0]})
                    except Exception as e1:
                        torch.cuda.empty_cache()
                        outs.append({"error": f"{type(e1).__name__}: {e1}"})
                        print("generate failed:", traceback.format_exc()[-1500:], flush=True)
            stats["calls"] += len(batch)
            stats["batches"] += 1
            stats["gen_s"] += time.time() - t
            if stats["batches"] % 50 == 0:
                print(f"[server] {stats['calls']} calls in {stats['batches']} batches, {stats['gen_s']:.0f}s generating", flush=True)
            for (req, ev, box), o in zip(batch, outs):
                box["out"] = o
                ev.set()

    def serve(conn):
        try:
            while True:
                req = conn.recv()
                ev, box = threading.Event(), {}
                jobs.put((req, ev, box))
                ev.wait()
                conn.send(box["out"])
        except (EOFError, ConnectionResetError, BrokenPipeError):
            pass

    threading.Thread(target=gpu_loop, daemon=True).start()
    if os.path.exists(a.address):
        os.remove(a.address)
    lst = Listener(a.address, family="AF_UNIX", authkey=os.environ.get("WORLDMM_AUTHKEY", "worldmm").encode())
    if a.ready_file:
        open(a.ready_file, "w").write(str(os.getpid()))
    print("READY", a.address, flush=True)
    while True:
        threading.Thread(target=serve, args=(lst.accept(),), daemon=True).start()


if __name__ == "__main__":
    main()
