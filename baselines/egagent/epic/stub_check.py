#!/usr/bin/env python3
"""Run EGAgent's paid scripts end to end against fake APIs (no key, no network, no cost) and check the
streaming rule on what they send.

    stub_check.py --video V --frames_dir <work>/frames/<V> --embeddings <results>/<V>_frames.npy \
                  --bank_file <bank>/<V>_questions.json --tmp <scratch dir> --arm gpt41|gemini25pro [--questions 3]

The real caption_epic.py, build_graph.py and run_qa.py (qa and blind) run unchanged in this process, with the
OpenAI client's HTTP layer (httpx) and the Gemini client's generate_content replaced by fakes that answer
every request with a reply that fits the schema it asks for. The fakes log each request's text and frames
(recognised by their bytes) and plant markers: a caption names the seconds of its frames, a graph node names
the end of the captions it came from. Afterwards, for every question: no frame after its cutoff was sent, and
no marker from a caption or graph chunk ending after it reached any prompt. Last, the fake answers as another
model and the run must stop (exit 4). Everything is written under --tmp; nothing here is a result.
"""
import argparse
import base64
import collections
import hashlib
import json
import os
import re
import runpy
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CALLS = []                      # every fake API call: {"phase", "kind", "texts", "seconds"}
STATE = {"phase": None, "model": {"openai": "gpt-4.1-2025-04-14", "gemini": "gemini-2.5-pro"}}
FRAME = {}                      # sha1 of a frame file's bytes -> its second


def fill(schema, text, name="", defs=None):
    """A value that fits a JSON schema, with the few fields the pipeline's flow depends on set on purpose."""
    defs = defs if defs is not None else schema.get("$defs", {})
    if "$ref" in schema:
        return fill(defs[schema["$ref"].split("/")[-1]], text, name, defs)
    if "anyOf" in schema:
        return fill(next(s for s in schema["anyOf"] if s.get("type") != "null"), text, name, defs)
    t = schema.get("type")
    t = [x for x in t if x != "null"][0] if isinstance(t, list) else t
    if t == "object":
        return {k: fill(v, text, k, defs) for k, v in (schema.get("properties") or {}).items()}
    if t == "array":
        items = schema.get("items") or {}
        if name == "tools" and items.get("enum"):
            return list(items["enum"])                      # a plan step that uses every tool
        if name == "sql_queries":
            return ["SELECT * FROM entity_graph_table"]     # everything the question's graph table holds
        return [fill(items, text, name, defs)]
    if "enum" in schema:
        return "no" if "no" in schema["enum"] else schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    if t in ("integer", "number"):
        return 1
    if t == "boolean":
        return True
    ends = [int(m) for m in re.findall(r"CAPTION_\d{6}_(\d{6})", text)]
    span = re.search(r"(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})", text)
    if name in ("id", "source_node_id", "target_node_id"):
        return f"ENT_{max(ends, default=0):06d}_{'B' if name.startswith('target') else 'A'}"
    if name in ("start_t", "end_t"):
        if span and "relationship_id" in text:              # the time-stamper: the first caption's interval
            return span.group(1 if name == "start_t" else 2)
        return "000000" if name == "start_t" else "235959"  # frame search: the whole video
    if name == "binary_score":
        return "no"
    return "stub"


def answer(texts, images, schema, phase):
    """(reply value, kind): a caption for a caption call, else a value fitting the schema."""
    secs = sorted(FRAME.get(hashlib.sha1(b).hexdigest(), -1) for b in images)
    assert -1 not in secs, "a frame was sent that is not one of the video's frames"
    text = "\n".join(texts)
    CALLS.append({"phase": phase, "kind": (schema or {}).get("title", "caption" if phase == "captions" else "text"),
                  "text": text, "seconds": secs})
    if schema is None:
        return f"CAPTION_{secs[0]:06d}_{secs[-1]:06d} the camera wearer cooks." if secs else "stub"
    return fill(schema, text)


def fake_openai():
    import httpx

    def reply(request):
        assert request.url.path.endswith("/chat/completions"), f"unexpected request to {request.url}"
        body = json.loads(request.content)
        texts, images = [], []
        for m in body["messages"]:
            for part in ([{"type": "text", "text": m["content"]}] if isinstance(m["content"], str) else m["content"]):
                if part["type"] == "text":
                    texts.append(part["text"])
                else:
                    images.append(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))
        tool = (body.get("tools") or [None])[0]
        rf = body.get("response_format") or {}
        schema = tool["function"]["parameters"] if tool else (rf.get("json_schema") or {}).get("schema")
        if schema is not None and tool:
            schema = {"title": tool["function"]["name"], **schema}
        value = answer(texts, images, schema, STATE["phase"])
        msg = {"role": "assistant", "content": value if schema is None else json.dumps(value)}
        if tool:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_stub", "type": "function", "function": {"name": tool["function"]["name"], "arguments": json.dumps(value)}}]}
        n_in = len("".join(texts)) // 4 + 765 * len(images)
        return httpx.Response(200, request=request, json={
            "id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": STATE["model"]["openai"],
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool else "stop"}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": 50, "total_tokens": n_in + 50}})

    async def areply(self, request, **kw):
        return reply(request)
    httpx.Client.send = lambda self, request, **kw: reply(request)
    httpx.AsyncClient.send = areply


def fake_gemini():
    from google.ai import generativelanguage_v1beta as glm

    def to_json_schema(s):   # the client's Schema message -> JSON schema
        t = str(s.get("type_") or s.get("type") or "STRING")
        t = {"1": "string", "2": "number", "3": "integer", "4": "boolean", "5": "array", "6": "object"}.get(t, t.lower().split(".")[-1])
        out = {"type": t}
        if s.get("enum"):
            out["enum"] = list(s["enum"])
        if s.get("properties"):
            out["properties"] = {k: to_json_schema(v) for k, v in s["properties"].items()}
        if s.get("items"):
            out["items"] = to_json_schema(s["items"])
        return out

    def generate_content(self, request=None, **kw):
        texts, images = [], []
        contents = list(request.contents) + ([request.system_instruction] if request.system_instruction.parts else [])
        for c in contents:
            for p in c.parts:
                if p.inline_data.data:
                    images.append(bytes(p.inline_data.data))
                elif p.text:
                    texts.append(p.text)
        decl = request.tools[0].function_declarations[0] if request.tools else None
        schema = None
        if decl is not None:
            schema = {"title": decl.name, **to_json_schema(type(decl.parameters).to_dict(decl.parameters))}
        assert decl is not None or not request.generation_config.response_schema, "a response schema without a tool: not faked"
        value = answer(texts, images, schema, STATE["phase"])
        part = glm.Part(function_call=glm.FunctionCall(name=decl.name, args=value)) if decl is not None else glm.Part(text=value)
        n_in = len("".join(texts)) // 4 + 1032 * len(images)
        return glm.GenerateContentResponse(
            candidates=[glm.Candidate(content=glm.Content(role="model", parts=[part]), finish_reason=glm.Candidate.FinishReason.STOP)],
            usage_metadata=glm.GenerateContentResponse.UsageMetadata(prompt_token_count=n_in, candidates_token_count=50, total_token_count=n_in + 50),
            model_version=STATE["model"]["gemini"])
    glm.GenerativeServiceClient.generate_content = generate_content


def run(phase, script, *argv):
    """One of the real scripts, as its own __main__; returns its exit code."""
    STATE["phase"] = phase
    sys.argv = [script, *map(str, argv)]
    try:
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (1 if e.code else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--bank_file", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--arm", required=True, choices=["gpt41", "gemini25pro"])
    ap.add_argument("--questions", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    import eg_paths
    backbone = eg_paths.BACKBONES[a.arm]
    mem = os.path.join(a.tmp, "memory")
    os.makedirs(mem, exist_ok=True)
    os.environ.update(OPENAI_API_KEY="stub-not-a-key", GEMINI_API_KEY="stub-not-a-key", HF_HUB_OFFLINE="1",
                      EGAGENT_USAGE=os.path.join(a.tmp, f"api_usage_{a.arm}.jsonl"),
                      # upstream's modules read these once, at their first import (here: by the caption script)
                      EGAGENT_DATASET="videomme", EGAGENT_BACKBONE=backbone,
                      EGAGENT_DB_ROOT=os.path.abspath(os.path.join(a.tmp, f"dbs_{a.arm}")),
                      EGAGENT_MODEL_ROOT=os.path.dirname(eg_paths.EMBEDDER))
    for f in eg_paths.frame_files(a.frames_dir):
        FRAME[hashlib.sha1(open(os.path.join(a.frames_dir, f), "rb").read()).hexdigest()] = int(f[:2]) * 3600 + int(f[2:4]) * 60 + int(f[4:6])
    fake_openai()
    fake_gemini()
    emb = os.path.join(mem, f"{a.video}_frames.npy")
    if not os.path.exists(emb):
        shutil.copy(a.embeddings, emb)
    caps, graph = os.path.join(mem, f"{a.video}_captions.json"), os.path.join(mem, f"{a.video}_graph.json")
    if not os.path.exists(graph):   # the memory is GPT-4.1's for both arms: made once, by the gpt41 pass
        assert a.arm == "gpt41", "run --arm gpt41 first: it makes the stub memory"
        assert run("captions", "caption_epic.py", "--frames_dir", a.frames_dir, "--out", caps) == 0
        assert run("graph", "build_graph.py", "--captions", caps, "--out", graph) == 0
    g = json.load(open(graph))
    print(f"memory: {len(json.load(open(caps)))} captions, {len(g['chunks'])} graph chunks, "
          f"{sum(len(c['relationships']) for c in g['chunks'])} relationships; e.g. {json.dumps(g['chunks'][0]['relationships'][:1])[:260]}")

    common = ["--video", a.video, "--bank_file", a.bank_file, "--frames_dir", a.frames_dir, "--backbone", backbone,
              "--db_root", os.path.join(a.tmp, f"dbs_{a.arm}"), "--limit", a.questions]
    rows = os.path.join(a.tmp, f"rows_{a.arm}.jsonl")
    for f in (rows, rows + ".blind", rows + ".wrong"):
        if os.path.exists(f):
            os.remove(f)
    assert run("qa", "run_qa.py", *common, "--memory", mem, "--rows", rows, "--out", os.path.join(a.tmp, "unused.json.gz")) == 0
    assert run("blind", "run_qa.py", *common, "--blind", "--rows", rows + ".blind", "--out", os.path.join(a.tmp, "unused.json")) == 0

    print("calls per stage:")
    for phase in ("captions", "graph", "qa", "blind"):
        kinds = collections.Counter(c["kind"] for c in CALLS if c["phase"] == phase)
        if kinds:
            print(f"  {phase}: {sum(kinds.values())} calls, {sum(len(c['seconds']) for c in CALLS if c['phase'] == phase)} frames sent: {dict(kinds)}")

    # the streaming rule, per question: its calls run from one planner call to the next
    qrows = [json.loads(l) for l in open(rows)]
    assert len(qrows) == a.questions and all(r.get("pred") for r in qrows), [r.get("error") for r in qrows]
    qa_calls, groups = [c for c in CALLS if c["phase"] == "qa"], []
    for c in qa_calls:
        if c["kind"] == "MultiHopPlan":
            groups.append([])
        groups[-1].append(c)
    assert len(groups) == len(qrows), (len(groups), len(qrows))
    for r, calls in zip(qrows, groups):
        cut = r["cutoff"]
        frames = [s for c in calls for s in c["seconds"]]
        text = "\n".join(c["text"] for c in calls)
        ents = sorted({int(m) for m in re.findall(r"ENT_(\d{6})", text, flags=re.I)})
        capm = re.findall(r"CAPTION_\d{6}_(\d{6})", text)
        chunks_visible = sorted(c["end"] for c in g["chunks"] if c["end"] <= cut)
        assert frames and max(frames) <= cut, (r["ID"], max(frames, default=None), cut)
        assert ents == chunks_visible, (r["ID"], ents, chunks_visible)
        assert not capm, (r["ID"], capm[:3])
        assert "NO TRANSCRIPTS AVAILABLE" in text
        print(f"  Q{r['ID']} cutoff {cut} s: {len(calls)} calls {[c['kind'] for c in calls]}; {len(frames)} frames sent, "
              f"latest second {max(frames)}; graph markers in prompts from chunks ending {ents} (visible chunks {chunks_visible}); "
              f"indexed {r['indexed']}; tokens recorded for {len(r['total_tokens'])} calls")
    blind = [c for c in CALLS if c["phase"] == "blind"]
    assert len(blind) == a.questions and not any(c["seconds"] for c in blind) and not any("ENT_" in c["text"].upper() for c in blind)
    print(f"  blind: {len(blind)} calls, no frame and no memory text in any")

    # a reply from another model must stop the run
    STATE["model"] = {"openai": "gpt-6-luna", "gemini": "gemini-3.8-flash"}
    rc = run("wrong-model", "run_qa.py", *common, "--blind", "--rows", rows + ".wrong", "--out", os.path.join(a.tmp, "unused.json"))
    assert rc == 4 and not os.path.exists(rows + ".wrong"), (rc, os.path.exists(rows + ".wrong"))
    print(f"  wrong model id ({STATE['model']}): the run stopped with exit {rc}, no row written")
    print(f"STUB CHECK PASSED for arm {a.arm} ({backbone})")


if __name__ == "__main__":
    main()
