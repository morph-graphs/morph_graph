#!/usr/bin/env python3
"""A fake OpenAI-compatible endpoint for testing the reported arms without any paid call (README, "Test without
API calls"): canned replies in the format each of MAGIC's prompts expects, and a log of every request.

    python mock_llm.py serve --port 18000 --log requests.jsonl
    python mock_llm.py check --log requests.jsonl --memory <memory dir> --answers <answers.json.gz> [--runs <dir> ...]

Point the three roles at it with distinct model names, so the log shows which role sent what:
    MAGIC_{CAPTION,MEMORY,REASON}_BASE_URL=http://127.0.0.1:18000/v1  MAGIC_{CAPTION,MEMORY,REASON}_MODEL=mock-{caption,memory,reason}
Replies are derived from the prompt (entities are kitchen words found in it; chain replies repeat the latest
time stamp they were shown), so the memory graph and the chains are non-trivial and `check` can tell what a
prompt contained from what was stored. `check` asserts: each kind of call came from the right role; no answer-loop
call contains a time stamp after its question's "Current time"; no stored chain mentions a time after its cutoff;
the answer rows have the expected fields; the usage recorded in runs.jsonl equals the calls logged here.
"""
import argparse
import ast
import gzip
import json
import os
import re
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = ["pan", "bowl", "knife", "onion", "fridge", "sink", "plate", "spoon", "cupboard", "kettle", "scale", "counter",
         "drawer", "hob", "tap", "board", "pot", "lid", "cup", "bin"]
ROLE = {"caption": "caption", "ner": "memory", "triples": "memory", "semantic": "memory", "consolidate": "memory",
        "summary": "memory", "topic": "memory", "event1": "memory", "event2": "memory", "event2b": "memory",
        "event3": "memory", "search": "control", "answer": "reason", "time": "reason"}
LOCK, N = threading.Lock(), Counter()


def parts(messages):
    """(system text, last user text, all text, images, videos)."""
    text, last, sysm, img, vid = [], "", "", 0, 0
    for m in messages:
        c = m.get("content")
        items = [{"type": "text", "text": c or ""}] if not isinstance(c, list) else c
        t = "".join(i.get("text", "") for i in items if i.get("type") == "text")
        img += sum(i.get("type") == "image_url" for i in items)
        vid += sum(i.get("type") == "video_url" for i in items)
        text.append(t)
        if m["role"] == "system":
            sysm = t
        if m["role"] == "user":
            last = t
    return sysm, last, "\n".join(text), img, vid


def words(t):
    return [w for w in WORDS if w in t.lower()]


def reply(sysm, last, vid):
    """(kind of call, canned reply)."""
    if vid:
        with LOCK:
            N["caption"] += 1
            i = N["caption"]
        a, b, c = (WORDS[(i + k) % len(WORDS)] for k in (0, 3, 7))
        return "caption", f"I take the {a} from the {b} and put it next to the {c}. I wash the {WORDS[i % 5]} at the sink."
    if sysm.startswith("Your task is to extract named entities"):
        return "ner", json.dumps({"named_entities": words(last)})
    if sysm.startswith("Your task is to construct an RDF"):
        return "triples", json.dumps({"triples": [["I", "use", w] for w in words(last)]})
    if "extracting semantic knowledge from episodic triples" in sysm:
        objs = list(dict.fromkeys(m.group(2) for m in re.finditer(r"(\d+)\. \[.*?, '([^']*)'\]", last)))[:6]
        idx = {o: int(m.group(1)) for m in re.finditer(r"(\d+)\. \[.*?, '([^']*)'\]", last) for o in [m.group(2)]}
        return "semantic", json.dumps({"semantic_triples": [["I", "often use", o] for o in objs],
                                       "episodic_evidence": [[idx[o]] for o in objs]})
    if "updated_triple" in sysm:
        new = ast.literal_eval(re.search(r"New triple:\n(\[.*?\])\n", last).group(1))
        return "consolidate", json.dumps({"updated_triple": new, "triples_to_remove": []})
    if "Event Summary Documentation Specialist" in sysm:
        return "summary", "I work in the kitchen with the " + ", the ".join(words(last)[:6] or ["pan"]) + "."
    if 'Extract facts about "' in last:
        ent = re.search(r'Extract facts about "([^"]*)"', last).group(1)
        times = re.findall(r"\[(DAY1 \d\d:\d\d)\]", last)
        return "topic", json.dumps([{"time": t, "fact": f"The camera wearer used the {ent} on the counter"} for t in times[:1] + times[-1:]])
    if "For each major activity in the video so far" in last:
        t = re.findall(r"\[(\d\d:\d\d:\d\d)\]", last)
        return "event1", json.dumps([{"name": f"Washing up {i}", "time_range": f"{x}-{x}", "summary": "The camera wearer washed the pan.",
                                      "key_entities": ["pan", "sink"]} for i, x in enumerate(t[:1] + t[-1:])])
    if "find ADDITIONAL cross-time event chains" in last:
        return "event2b", "[]"
    if "Find activities that are related across different time periods" in last:
        t = [x.split("-")[0] for x in re.findall(r"- \[(\d\d:\d\d:\d\d-\d\d:\d\d:\d\d)\]", last)]
        return "event2", json.dumps([{"name": "Washing up", "key_entities": ["pan"], "steps": [
            {"day": "DAY1", "start_time": x, "end_time": x, "description": "The camera wearer washed the pan."} for x in t]}] if len(t) > 1 else [])
    if "This is one step of the event chain" in last:
        t = re.findall(r"\[(\d\d:\d\d:\d\d)\]", last)
        return "event3", json.dumps({"description": f"The camera wearer washed the pan; last caption at {t[-1] if t else 'none'}."})
    if last.startswith("Extract the specific video timestamp"):
        return "time", "null"
    if "reasoning agent for a multimodal video memory retrieval system" in sysm:
        n = last.count("### Round")
        q = re.search(r"Query: (.*)", last).group(1)
        if n >= 2:
            return "search", json.dumps({"decision": "answer"})
        return "search", json.dumps({"decision": "search", "search_query": q if n == 0 else f"What did I do with the {(words(q) or ['pan'])[0]}?"})
    if "answers questions about egocentric video" in sysm:
        return "answer", "A"
    return "unknown", ""


def serve(a):
    log = open(a.log, "a")

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
            self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            sysm, last, text, img, vid = parts(req["messages"])
            kind, out = reply(sysm, last, vid)
            pt, ct = len(text) // 4 + 200 * img + 5000 * vid, len(out) // 4 + 1
            with LOCK:
                log.write(json.dumps({"model": req.get("model"), "kind": kind, "chars": len(text), "images": img, "videos": vid,
                                      "prompt_tokens": pt, "completion_tokens": ct, "max_tokens": req.get("max_tokens"),
                                      "temperature": req.get("temperature"), "user": last, "reply": out}) + "\n")
                log.flush()
            if kind == "unknown":
                return self._send(400, {"error": {"message": "mock_llm: a prompt it has no reply for"}})
            self._send(200, {"id": "mock", "object": "chat.completion", "created": 0, "model": req.get("model"),
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


def stamps(t):
    """Seconds of every DAY1 HH:MM[:SS] in a text."""
    return [int(h) * 3600 + int(m) * 60 + int(s or 0) for h, m, s in re.findall(r"DAY1 (\d\d):(\d\d)(?::(\d\d))?", t)]


def hms(t):
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def check(a):
    rows = [json.loads(l) for l in open(a.log)]
    by = Counter((r["model"], r["kind"]) for r in rows)
    print("calls per role (model) and kind:")
    for (m, k), c in sorted(by.items()):
        sel = [r for r in rows if (r["model"], r["kind"]) == (m, k)]
        print(f"  {m:14} {k:12} {c:5} calls, prompt chars mean {sum(r['chars'] for r in sel) // c:6}, "
              f"images {sum(r['images'] for r in sel):5}, videos {sum(r['videos'] for r in sel)}")
    assert not by.get((None, "unknown")) and all(k != "unknown" for _, k in by), "prompts the mock does not know"
    # the control role is the memory role's endpoint unless MAGIC_CONTROLLER=reason (--controller)
    sent = lambda k: a.controller if ROLE[k] == "control" else ROLE[k]
    wrong = [(m, k) for m, k in by if m != "mock-" + sent(k)]
    assert not wrong, f"calls sent to the wrong role: {wrong}"
    # the answer loop: nothing after the question's current time
    loop = [r for r in rows if r["kind"] in ("search", "answer")]
    late = 0
    for r in loop:
        now = stamps(re.search(r"Current time: (DAY1 \d\d:\d\d:\d\d)", r["user"]).group(1))[0]
        seen = stamps(r["user"].replace("Current time:", ""))
        late += any(s > now for s in seen)
    ctl = [r for r in rows if r["kind"] == "search"]
    assert a.controller == "reason" or all(r["temperature"] == 0 for r in ctl), "a loop call without temperature 0"
    assert all(r["temperature"] == 1.0 for r in rows if r["kind"] == "answer"), "a final answer without the responder's sampling"
    assert loop and not late, f"{late} answer-loop calls show a time stamp after the question's current time"
    print(f"answer loop: {len(loop)} calls, {sum(1 for r in loop if 'DAY1' in r['user'].replace('Current time:', ''))} with retrieved "
          f"time stamps, {sum(r['images'] for r in loop)} frames, none after its question's current time")
    # stored chains: the mock repeated the latest time each chain prompt showed it
    cdir, n = os.path.join(a.memory, "chains"), 0
    for d in sorted(x for x in os.listdir(cdir) if re.fullmatch(r"c\d{5}", x)):
        cut = int(d[1:])
        topics = json.load(open(os.path.join(cdir, d, "topic_chains.json")))
        events = json.load(open(os.path.join(cdir, d, "step3_enriched_chains.json")))
        ts = [s for t in topics.values() for f in t["facts"] for s in stamps(f["time"])]
        ts += [hms(x) for e in events for st in e["steps"] for x in re.findall(r"\d\d:\d\d:\d\d", json.dumps(st))]
        assert all(s <= cut for s in ts), f"chains at {cut}s mention {max(ts)}s"
        n += 1
        print(f"  chains @ {cut}s: {sum(bool(t['facts']) for t in topics.values())} topic chains, {len(events)} event chains, "
              f"latest time {max(ts) if ts else '-'}s")
    assert n, "no chains stored"
    ans = json.load(gzip.open(a.answers, "rt"))
    need = {"key", "qhash", "ID", "type", "question", "choices", "answer", "response", "evaluate", "t_ask", "query_time", "cutoff",
            "num_rounds", "round_history", "seconds", "version", "memory_hash", "reason_model", "control_model"}
    for r in ans:
        assert need <= set(r), f"row lacks {need - set(r)}"
        assert r["response"] == "A" and abs(r["cutoff"] - r["t_ask"]) < 1 and r["num_rounds"] == 3 and r["reason_model"] == "mock-reason" and r["control_model"] == "mock-" + a.controller, r
    print(f"answers: {len(ans)} rows, fields ok; first: " + json.dumps({k: ans[0][k] for k in sorted(need - {'round_history', 'question', 'choices'})}))
    # runs.jsonl: the last line of each stage carries the stage's usage; together they equal the log
    used = Counter()
    for d in a.runs:
        last = {}
        for l in open(os.path.join(d, "runs.jsonl")):
            r = json.loads(l)
            last[r["stage"]] = r
        for st, r in last.items():
            for role, u in (r.get("usage") or {}).items():
                used[role] += u["calls"]
                want = "mock-" + (a.controller if role == "control" else role)
                assert u["model"] == want and u["returned"] == [want] and u["failed"] == 0 and u["prompt_tokens"] > 0, (st, role, u)
            print(f"  runs.jsonl {os.path.basename(os.path.dirname(d + '/'))} {st}: roles "
                  f"{ {k: v['model'] for k, v in (r.get('roles') or {}).items()} }, usage {r.get('usage')}")
    if a.runs:
        logged = Counter("reason" if ROLE[r["kind"]] == "control" and a.controller == "reason" else ROLE[r["kind"]] for r in rows)
        assert used == logged, f"usage in runs.jsonl {dict(used)} != calls logged {dict(logged)}"
    print("mock check passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["serve", "check"])
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--log", required=True)
    ap.add_argument("--memory", help="check: the memory folder (with chains/)")
    ap.add_argument("--answers", help="check: answers_<bank>.json.gz")
    ap.add_argument("--controller", default="memory", choices=["memory", "reason"], help="check: MAGIC_CONTROLLER of the run")
    ap.add_argument("--runs", nargs="*", default=[], help="check: folders whose runs.jsonl usage must add up to the log")
    a = ap.parse_args()
    serve(a) if a.action == "serve" else check(a)
