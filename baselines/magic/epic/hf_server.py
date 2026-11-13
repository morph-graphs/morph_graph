#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/chat/completions server over a local HF Qwen VLM.

MAGIC-Video sends every LLM call through an OpenAI client (OpenRouter in the paper). With
MAGIC_LLM_BASE_URL pointing here, the same calls go to the local backbone instead: Qwen3-VL-8B (arm m8,
the client environment) or Qwen3.8-27B (arm m27, its own environment with transformers 5.8). It runs in its
own process so the MAGIC process can keep transformers 4.57 for VLM2Vec. run_magic.sh starts it.

Requests from MAGIC's thread pools are batched (left padding) when they share sampling settings;
requests with many images run alone. Supported fields: messages (text, image_url data URLs),
max_tokens, temperature, top_p, top_k, min_p, repetition_penalty, seed, chat_template_kwargs.
presence_penalty has no HF generate equivalent and is ignored.

    python hf_server.py --model Qwen/Qwen3-VL-8B-Instruct --port 23456
"""
import argparse
import base64
import io
import json
import queue
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

Q = queue.Queue()
STATS = {"requests": 0, "batches": 0, "gen_tokens": 0, "busy_s": 0.0}


def to_hf_messages(messages):
    """OpenAI chat messages -> HF chat messages + the images in order."""
    out, images = [], []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str) or c is None:
            out.append({"role": m["role"], "content": [{"type": "text", "text": c or ""}]})
            continue
        parts = []
        for it in c:
            if it.get("type") == "text":
                parts.append({"type": "text", "text": it["text"]})
            elif it.get("type") == "image_url":
                url = it["image_url"]["url"]
                data = base64.b64decode(url.split(",", 1)[1]) if url.startswith("data:") else open(url, "rb").read()
                images.append(Image.open(io.BytesIO(data)).convert("RGB"))
                parts.append({"type": "image"})
            else:
                raise ValueError(f"unsupported content type {it.get('type')}")
        out.append({"role": m["role"], "content": parts})
    return out, images


def worker(model, proc, a):
    while True:
        first = Q.get()
        batch = [first]
        key = first["key"]
        if first["n_img"] <= a.solo_images:
            deadline = time.time() + a.wait_ms / 1000
            held = []
            while len(batch) < a.max_batch and time.time() < deadline:
                try:
                    r = Q.get(timeout=max(0.0, deadline - time.time()))
                except queue.Empty:
                    break
                fits = sum(x["n_tok"] + x["max_tokens"] for x in batch) + r["n_tok"] + r["max_tokens"] <= a.token_budget
                (batch if r["key"] == key and r["n_img"] <= a.solo_images and fits else held).append(r)
            for r in held:
                Q.put(r)
        t0 = time.time()
        run_batch(model, proc, batch)
        STATS["batches"] += 1
        STATS["busy_s"] += time.time() - t0
        for r in batch:
            r["done"].set()


def run_batch(model, proc, batch):
    """Generate for a batch; on CUDA OOM retry its halves (down to single requests)."""
    try:
        texts = [r["text"] for r in batch]
        imgs = [im for r in batch for im in r["images"]]
        inputs = proc(text=texts, images=imgs or None, padding=True, return_tensors="pt").to(model.device)
        p = batch[0]["params"]
        temp = float(p.get("temperature", 0.7) or 0.0)
        gen = dict(max_new_tokens=max(r["max_tokens"] for r in batch), do_sample=temp > 0,
                   repetition_penalty=float(p.get("repetition_penalty") or 1.0))
        if temp > 0:
            gen.update(temperature=temp, top_p=float(p.get("top_p", 0.8)), top_k=int(p.get("top_k", 20)))
            if p.get("min_p"):
                gen["min_p"] = float(p["min_p"])
        torch.manual_seed(int(p.get("seed", 0)))
        with torch.inference_mode():
            out = model.generate(**inputs, **gen)
        new = out[:, inputs["input_ids"].shape[1]:]
        for i, r in enumerate(batch):
            ids = new[i][: r["max_tokens"]]
            txt = proc.batch_decode([ids], skip_special_tokens=True)[0]
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
            n = int((ids != proc.tokenizer.pad_token_id).sum())
            STATS["gen_tokens"] += n
            r["result"] = {"text": txt, "prompt_tokens": int(inputs["attention_mask"][i].sum()), "completion_tokens": n,
                           "finish": "length" if n >= r["max_tokens"] else "stop"}
    except torch.cuda.OutOfMemoryError as e:
        inputs = out = new = None
        torch.cuda.empty_cache()
        if len(batch) > 1:
            print(f"OOM on a batch of {len(batch)}; splitting", flush=True)
            h = len(batch) // 2
            run_batch(model, proc, batch[:h])
            run_batch(model, proc, batch[h:])
        else:
            traceback.print_exc()
            batch[0]["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        for r in batch:
            r["error"] = f"{type(e).__name__}: {e}"


def make_handler(proc, a):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"object": "list", "data": [{"id": a.name, "object": "model"}], "stats": STATS})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                msgs, images = to_hf_messages(req["messages"])
                ctk = req.get("chat_template_kwargs") or {}
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **ctk)
                params = {k: req.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed")
                          if req.get(k) is not None}
                r = {"text": text, "images": images, "n_img": len(images), "params": params, "n_tok": len(text) // 3,
                     "key": json.dumps(params, sort_keys=True), "max_tokens": int(req.get("max_tokens") or a.max_tokens),
                     "done": threading.Event()}
                STATS["requests"] += 1
                Q.put(r)
                r["done"].wait()
                if "error" in r:
                    return self._send(500, {"error": {"message": r["error"]}})
                res = r["result"]
                self._send(200, {"id": f"chatcmpl-{STATS['requests']}", "object": "chat.completion", "created": int(time.time()),
                                 "model": a.name, "choices": [{"index": 0, "message": {"role": "assistant", "content": res["text"]},
                                                               "finish_reason": res["finish"]}],
                                 "usage": {"prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
                                           "total_tokens": res["prompt_tokens"] + res["completion_tokens"]}})
            except Exception as e:
                traceback.print_exc()
                self._send(400, {"error": {"message": f"{type(e).__name__}: {e}"}})
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="local")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--device_map", default="cuda:0", help="'auto' to spread a big model over all visible GPUs")
    ap.add_argument("--max_batch", type=int, default=8)
    ap.add_argument("--solo_images", type=int, default=4, help="requests with more images than this run alone")
    ap.add_argument("--wait_ms", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--token_budget", type=int, default=24000, help="max sum of prompt + max_tokens over a batch (KV cache)")
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=a.attn,
                                                        device_map=a.device_map).eval()
    print(f"loaded {type(model).__name__} from {a.model} on {getattr(model, 'hf_device_map', model.device)}", flush=True)
    threading.Thread(target=worker, args=(model, proc, a), daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(proc, a)).serve_forever()


if __name__ == "__main__":
    main()
