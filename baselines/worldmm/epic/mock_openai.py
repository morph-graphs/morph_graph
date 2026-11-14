#!/usr/bin/env python3
"""A stand-in for the OpenAI Responses API, to run the official arms end to end without paying (test only).

    python mock_openai.py --port 8765 --log calls.jsonl      then, for the pipeline:
    OPENAI_BASE_URL=http://127.0.0.1:8765/v1 OPENAI_API_KEY=mock bash run_worldmm.sh build <video> g5m

Answers POST /v1/responses with canned content in the shape each WorldMM call expects, and logs one line per call
(kind, model, images and their sizes, text tokens, the time markers the prompt carried, the request's path, its
`provider` field and its images' `detail`). Under /openrouter/ it answers as OpenRouter does: the reply also
names the provider that served the call and the usage has a cost (OPENROUTER_BASE_URL=http://127.0.0.1:8765/openrouter/api/v1). Every canned caption
names the seconds it covers as CAP<start-end>, and the entities and triples made from it as tm<start>x<end>, so
whatever later reaches a retriever or answer call shows which part of the video it came from: the test checks that
none of it ends after the question's t_ask. The reasoning replies walk through episodic, semantic, visual by time
range (MOCK_VISUAL_RANGE, one that runs past the first question's t_ask), visual by text, then answer.
Usage is estimated the way the API bills: text tokens (tiktoken when installed, else chars / 4) plus image tokens
from each image's real size (gpt-5-mini: 32 px patches x 1.2; gpt-5: 70 + 140 per 512 px tile).
"""
import argparse
import base64
import io
import json
import math
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

try:
    import tiktoken
    ENC = tiktoken.get_encoding("o200k_base")
except Exception:
    ENC = None
MARK = re.compile(r"CAP<(\d+)-(\d+)>|tm(\d+)x(\d+)")
RANGE = os.environ.get("MOCK_VISUAL_RANGE", "DAY1 00:04:30 - DAY1 00:05:30")
LOCK = threading.Lock()


def tokens(text):
    return len(ENC.encode(text)) if ENC else math.ceil(len(text) / 4)


def image_tokens(model, w, h):
    """OpenAI's image input rules (images-vision guide, read 2026-11-13), detail auto, for images <= 2048 px."""
    if "mini" in model:
        return math.ceil(math.ceil(w / 32) * math.ceil(h / 32) * 1.2)
    if min(w, h) > 768:
        w, h = w * 768 / min(w, h), h * 768 / min(w, h)
    return 70 + 140 * math.ceil(w / 512) * math.ceil(h / 512)


def clock(s):
    h, m, sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


def reply(last, schema):
    """(kind of call, canned reply text)."""
    spans = [(int(a or c), int(b or d)) for a, b, c, d in MARK.findall(last)]
    lo, hi = (min(s for s, _ in spans), max(e for _, e in spans)) if spans else (0, 0)
    if schema == "NerRawOutput":
        return "ner", json.dumps({"named_entities": ["person", "kitchen", f"tm{lo}x{hi}"]})
    if schema == "TripleRawOutput":
        return "triples", json.dumps({"triples": [["person", "is in", "kitchen"], ["person", "acts during", f"tm{lo}x{hi}"]]})
    if schema == "SemanticRawOutput":
        return "semantic_extraction", json.dumps({"semantic_triples": [["person", "habitually acts during", f"tm{lo}x{hi}"]],
                                                  "episodic_evidence": [[0]]})
    if schema == "ConsolidationRawOutput":
        return "semantic_consolidation", json.dumps({"updated_triple": ["person", "habitually acts during", f"tm{lo}x{hi}"],
                                                     "triples_to_remove": []})
    if "Segment window:" in last:
        s, e = re.search(r"Segment window: (\S+) to (\S+)", last).groups()
        return "caption", f"CAP<{clock(s)}-{clock(e)}> The person works at the kitchen counter from second {clock(s)} to {clock(e)}."
    if last.startswith("All descriptions:"):
        return "multiscale", f"CAP<{lo}-{hi}> **Kitchen work** The person works at the counter between second {lo} and {hi}."
    if "[[ ## fact_before_filter ## ]]" in last:   # HippoRAG's fact filter: keep the best two candidates, never all
        facts = json.loads(last.split("[[ ## fact_before_filter ## ]]")[1].split("Respond with")[0])["fact"]
        keep = json.dumps({"fact": facts[:max(1, min(2, len(facts) - 1))]})
        return "fact_filter", f"[[ ## fact_after_filter ## ]]\n{keep}\n\n[[ ## completed ## ]]"
    if "Retrieved Captions:" in last:
        return "multiscale_filter", json.dumps(re.findall(r'"id": "([^"]+)"', last)[:3])
    if "Round History:" in last:
        steps = [("episodic", "what the person does at the counter"), ("semantic", "person kitchen"),
                 ("visual", RANGE), ("visual", "the kitchen counter")]
        n = last.count("### Round ")
        if n >= len(steps):
            return "reasoning", json.dumps({"decision": "answer", "reason": "enough context"})
        return "reasoning", json.dumps({"decision": "search", "selected_memory": {"memory_type": steps[n][0], "search_query": steps[n][1]}})
    if "Context:" in last:
        return "answer", "A"
    return "other", "ok"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model, texts, sizes, last, details = body["model"], [], [], "", set()
        for msg in body["input"]:
            parts = msg["content"] if isinstance(msg["content"], list) else [{"type": "input_text", "text": msg["content"]}]
            mine = []
            for part in parts:
                if part["type"] == "input_image":
                    details.add(part.get("detail"))
                    sizes.append(Image.open(io.BytesIO(base64.b64decode(part["image_url"].split(",", 1)[1]))).size)
                else:
                    mine.append(part["text"])
            texts += mine
            if msg["role"] == "user":
                last = "\n".join(mine)
        schema = ((body.get("text") or {}).get("format") or {}).get("name")
        kind, out = reply(last, schema)
        text_in, img_in = sum(tokens(t) for t in texts), sum(image_tokens(model, w, h) for w, h in sizes)
        ends = [int(b or d) for _, b, _, d in MARK.findall(last)]
        q = re.search(r"^Query: (.*)$", last, re.M)
        row = {"kind": kind, "model": model, "images": len(sizes), "image_size": sorted(set(sizes))[-1:] and list(sorted(set(sizes))[-1]),
               "text_tokens": text_in, "image_tokens": img_in, "output_tokens": tokens(out), "max_end": max(ends, default=None),
               "query": q.group(1) if q else None, "tier": body.get("service_tier"), "path": self.path,
               "provider": body.get("provider"), "detail": sorted(details, key=str),
               "auth": "bearer" if self.headers.get("Authorization", "").startswith("Bearer ") else "none"}
        with LOCK:
            LOG.write(json.dumps(row) + "\n")
            LOG.flush()
        resp = {"id": "resp_mock", "object": "response", "created_at": 0, "model": model, "status": "completed",
                "output": [{"id": "msg_mock", "type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": out, "annotations": []}]}],
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "service_tier": body.get("service_tier") or "default",
                "usage": {"input_tokens": text_in + img_in, "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens": row["output_tokens"], "output_tokens_details": {"reasoning_tokens": 0},
                          "total_tokens": text_in + img_in + row["output_tokens"]}}
        if self.path.startswith("/openrouter/"):
            resp["provider"] = "OpenAI"
            resp["usage"].update(cost=0.001, is_byok=False)
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    LOG = open(a.log, "a")
    print(f"mock OpenAI on 127.0.0.1:{a.port}, log {a.log}, tokenizer {'tiktoken' if ENC else 'chars/4'}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
