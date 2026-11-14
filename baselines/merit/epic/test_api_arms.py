#!/usr/bin/env python3
"""The g5m arms end to end on one real video with no OpenAI call: run_merit.sh runs unchanged against a local
stand-in for the Chat Completions endpoint (OPENAI_BASE_URL -> 127.0.0.1, a dummy key), which sees every request.

    PY=<python of the m8 environment> EPIC_MORPH_DIR=<epic-morph checkout> [HDEPIC_VIDEOS=...] \
        $PY baselines/merit/epic/test_api_arms.py --work <empty scratch folder> [--video P05-20240425-201444]
                                                  [--bank batch10_agree2] [--no-qa] [--qwen] [--gateway openrouter]

Checks (each prints a line; any failure is an assertion):
  build g5m      one caption call and one key call per clip, counted per role and model id in runs.jsonl; a rerun
                 makes no call; without OPENAI_API_KEY the stage stops before doing anything
  qa g5m-g5      (needs one GPU, for the embedding model) three of the bank's questions, asked at 95 s, 215 s and
                 the end: calls per role; no search, filter or answer request holds a clip ending after the
                 question's ask time; frames in an answer request = MERIT's rule for the clips retrieved; the
                 answers file has MERIT's row format and collect_answers.py takes it with nothing missing
  key            every request carried the dummy key, and no file written by the run contains it
  gateway        every request came on the gateway's path with its model ids. --gateway openai (default):
                 /v1/chat/completions, MERIT's snapshots, no `provider` field, with OPENAI_API_KEY (an
                 OPENROUTER_API_KEY that is also set is ignored). --gateway openrouter: only OPENROUTER_API_KEY is
                 set and the stand-in answers as OpenRouter under /openrouter/api/v1: openai/gpt-5-mini and
                 openai/gpt-5, pinned to provider OpenAI without fallbacks; runs.jsonl has the endpoint, provider, cost
  --qwen         qa g5m-q8 with the real Qwen3-VL-8B on the same memory: answers written, zero requests
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from openai_usage import PROVIDER, role  # noqa: E402

KEY = "sk-mock-not-a-real-key-0000"
LOG, WANT, LOCK = [], {}, threading.Lock()   # requests seen; search rounds wanted per question (1, 2, 3)
STAMP = re.compile(r"\[DAY1 (\d\d):(\d\d):(\d\d) - DAY1 (\d\d):(\d\d):(\d\d)\]")


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        parts = [p for m in body["messages"] if m["role"] == "user"
                 for p in (m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}])]
        text = "".join(p["text"] for p in parts if p["type"] == "text")
        images = sum(p["type"] == "image_url" for p in parts)
        r = role(body["messages"])
        with LOCK:
            n = len(LOG)
            LOG.append(dict(role=r, model=body["model"], images=images, text=text, path=self.path, provider=body.get("provider"),
                            key_ok=self.headers.get("Authorization") == f"Bearer {KEY}"))
            if r == "captions":
                reply = f"MOCKCAP{n:04d} I pick up item {n} from the counter and put it in the sink."
            elif r == "keys":
                i = re.findall(r"MOCKCAP(\d+)", text)[-1]
                reply = f"I move item {i} to the sink\nnone\nitem {i} on the counter\nitem {i} counter sink"
            elif r == "search":
                q = text.split("\n")[0]
                done = text.split("Round History:")[1].count("### Round")
                want = WANT.setdefault(q, len(WANT) % 3 + 1)
                reply = json.dumps({"decision": "search", "search_query": f"item {done} counter"} if done < want
                                   else {"decision": "answer"})
            elif r == "filter":
                reply = json.dumps({f"clip_{k}": "relevant (mock)" for k in range(1, text.count("=== Clip ") + 1)})
            else:
                reply = "ANSWER: A\nREASONING: mock"
        out = {
            "id": f"chatcmpl-mock{n}", "object": "chat.completion", "created": 0, "model": body["model"],
            "service_tier": body.get("service_tier") or "default",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": len(text) // 4 + 236 * images, "completion_tokens": len(reply) // 4 + 64,
                      "total_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0},
                      "completion_tokens_details": {"reasoning_tokens": 64}}}
        if self.path.startswith("/openrouter/"):   # what OpenRouter adds: who served the call, and what it charged
            out["provider"] = "OpenAI"
            out["usage"].update(cost=0.001, is_byok=False)
        out = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", required=True, help="scratch folder for clips, frames, results (a few hundred MB)")
    ap.add_argument("--video", default="P05-20240425-201444")
    ap.add_argument("--bank", default="batch10_agree2", help="the real bank the three questions are taken from")
    ap.add_argument("--no-qa", action="store_true", help="only the build checks (no GPU needed)")
    ap.add_argument("--qwen", action="store_true", help="also answer with the real Qwen3-VL-8B (g5m-q8)")
    ap.add_argument("--gateway", default="openai", choices=("openai", "openrouter"), help="which key the run gets")
    a = ap.parse_args()
    v, work = a.video, os.path.abspath(a.work)
    merit = os.path.dirname(HERE)
    real = os.environ["EPIC_MORPH_DIR"]

    # three real questions with set ask times: early (few clips visible), middle, end of the video
    qs = json.load(open(os.path.join(real, "outputs/v3/banks", a.bank, f"{v}_questions.json")))["questions"][:3]
    assert len(qs) == 3 and len({q["q"] for q in qs}) == 3, "need three distinct questions"
    for q, t in zip(qs, (95.0, 215.5, None)):
        q["streaming"] = t is not None
        q["t_ask"] = t
    epic = os.path.join(work, "epic")
    os.makedirs(os.path.join(epic, "outputs/v3/banks/mock3"), exist_ok=True)
    json.dump({"questions": qs}, open(os.path.join(epic, f"outputs/v3/banks/mock3/{v}_questions.json"), "w"))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # both stand-in URLs are always set; the keys decide. OpenAI's key wins over an OpenRouter key that is also there
    here = f"http://127.0.0.1:{server.server_port}"
    env = dict(os.environ, OPENAI_BASE_URL=here + "/v1", OPENROUTER_BASE_URL=here + "/openrouter/api/v1",
               OPENROUTER_API_KEY=KEY if a.gateway == "openrouter" else "sk-or-mock-must-not-be-used",
               EPIC_MORPH_DIR=epic, HDEPIC_VIDEOS=os.environ.get("HDEPIC_VIDEOS", os.path.join(real, "data/HD-EPIC/videos")),
               MERIT_WORK=os.path.join(work, "work"), MERIT_OUT=os.path.join(work, "results"), MERIT_FETCH="0",
               MERIT_API_THREADS="2")
    for k in ("MERIT_TAG", "MERIT_NO_FRAMES", "MERIT_TOP_K", "MERIT_SERVICE_TIER", "HF_TOKEN", "OPENAI_API_KEY"):
        env.pop(k, None)
    if a.gateway == "openai":
        env["OPENAI_API_KEY"] = KEY
        mini, big, path, pin = "gpt-5-mini-2025-08-07", "gpt-5-2025-08-07", "/v1/chat/completions", None
    else:
        mini, big, path, pin = "openai/gpt-5-mini", "openai/gpt-5", "/openrouter/api/v1/chat/completions", PROVIDER

    def run(*args, env=env, ok=True):
        p = subprocess.run(["bash", os.path.join(merit, "run_merit.sh"), *args], env=env, capture_output=True, text=True)
        open(os.path.join(work, "run.log"), "a").write(f"\n$ run_merit.sh {' '.join(args)}\n{p.stdout}{p.stderr}")
        assert (p.returncode == 0) == ok, f"run_merit.sh {' '.join(args)} exited {p.returncode}:\n{(p.stdout + p.stderr)[-3000:]}"
        return p.stdout + p.stderr

    def calls(start=0):
        c = {}
        for e in LOG[start:]:
            c[e["role"]] = c.get(e["role"], 0) + 1
        return c

    def last_run(arm, stage):
        rows = [json.loads(l) for l in open(os.path.join(work, "results", arm, v, "runs.jsonl"))]
        return [r for r in rows if r["stage"].startswith(stage)][-1]

    # --- build
    out = run("build", v, "g5m", env={k: x for k, x in env.items() if not k.endswith("_API_KEY")}, ok=False)
    assert "set OPENAI_API_KEY" in out and not LOG
    print("no key: build g5m stops before any request")
    run("build", v, "g5m")
    mem = json.load(open(os.path.join(work, "results/g5m", v, f"{v}_30sec_4_key.json")))
    n = len(mem)
    assert calls() == {"captions": n, "keys": n}, calls()
    assert all(c["text"].startswith("MOCKCAP") and c["event_key"] and c["object_key"] and c["sum_key"] for c in mem)
    assert all(1 <= e["images"] <= 30 if e["role"] == "captions" else e["images"] == 0 for e in LOG)
    use = last_run("g5m", "build")["openai"]
    assert {r: sum(m["calls"] for m in use[r].values()) for r in use} == {"captions": n, "keys": n}, use
    assert list(use["captions"]) == list(use["keys"]) == [mini], use
    row = use["captions"][mini]
    assert row["endpoint"].rstrip("/") == here + path[:-len("/chat/completions")], row
    assert (row.get("provider"), row.get("cost")) == (("OpenAI", round(0.001 * n, 6)) if pin else (None, None)), row
    print(f"build g5m: {n} clips, {n} caption calls ({LOG[0]['images']} frames in the first), {n} key calls; runs.jsonl: "
          f"{json.dumps(use)}")
    seen = len(LOG)
    run("build", v, "g5m")
    assert len(LOG) == seen
    print("build g5m again: 0 requests")

    if not a.no_qa:
        # --- qa with gpt-5
        run("qa", v, "g5m-g5", "mock3")
        c = calls(seen)
        qa = json.load(open(os.path.join(work, "work/qa/mock3", f"EgoLifeQA_{v}.json")))
        rows = json.load(gzip.open(os.path.join(work, "results/g5m-g5", v, "answers_mock3.json.gz"), "rt"))
        assert len(rows) == 3 and c["answer"] == 3 and c["search"] == 2 + 3 + 4 and 1 <= c.get("filter", 0) <= 6, c
        assert set(e["model"] for e in LOG[seen:]) == {big}
        secs = lambda h, m, s: int(h) * 3600 + int(m) * 60 + int(s)
        ends = [secs(c["end_time"][:2], c["end_time"][2:4], c["end_time"][4:6]) for c in mem]
        by_id = {r["ID"]: r for r in rows}
        for q in qa:
            row = by_id[q["ID"]]
            limit = secs(*re.match(r"(\d\d)(\d\d)(\d\d)", q["query_time"]["time"]).groups())
            mine = [e for e in LOG[seen:] if q["question"] in e["text"]]
            stamps = [secs(*m[3:]) for e in mine for m in STAMP.findall(e["text"])]
            assert stamps and max(stamps) <= limit, (q["ID"], limit, max(stamps, default=None))
            want = sum(6 * k if r == 1 else 6 * min(k, 5) if r == 2 else k
                       for r, k in ((h["round_num"], len(h["retrieved_content"])) for h in row["round_history"]))
            got = [e["images"] for e in mine if e["role"] == "answer"]
            assert got == [want], (q["ID"], got, want)
            assert {"ID", "key", "qhash", "pos", "question", "choices", "answer", "response", "parsed_answer", "evaluate",
                    "query_time", "query_time_int", "round_history", "raw_responses"} <= set(row), sorted(row)
            assert row["parsed_answer"] == "A" and row["qhash"] == q["qhash"]
            print(f"qa g5m-g5 {q['ID']}: asked at {limit} s ({sum(e <= limit for e in ends)} of {n} clips visible), "
                  f"{sum(e['role'] == 'search' for e in mine)} search / {sum(e['role'] == 'filter' for e in mine)} filter / 1 answer "
                  f"requests, latest clip end in any of them {max(stamps)} s, {want} frames in the answer request")
        assert sum(e <= 95 for e in ends) < n, "the video is too short for the early question to hide anything"
        out = subprocess.run([env["PY"], os.path.join(HERE, "collect_answers.py"), "--qa_file",
                              os.path.join(work, "work/qa/mock3", f"EgoLifeQA_{v}.json"), "--runs", os.path.join(work, "none/*"),
                              "--prev", os.path.join(work, "results/g5m-g5", v, "answers_mock3.json.gz"),
                              "--out", os.path.join(work, "collected.json.gz")], capture_output=True, text=True, check=True).stdout
        assert "3/3 answered" in out and out.strip().splitlines()[-1].startswith("#"), out
        r = last_run("g5m-g5", "qa")
        assert (r["respond_model"], r["memory_arm"], r["top_k"]) == ("gpt-5", "g5m", 10), r
        assert {k: sum(m["calls"] for m in x.values()) for k, x in r["openai"].items()} == c, (r["openai"], c)
        assert os.path.exists(os.path.join(work, "results/g5m-g5", v, f"{v}_memory_ref.json"))
        print(f"qa g5m-g5: requests {c}; collect_answers.py: {out.strip().splitlines()[0]}; runs.jsonl: {json.dumps(r['openai'])}")

    if a.qwen:
        seen = len(LOG)
        run("qa", v, "g5m-q8", "mock3")
        rows = json.load(gzip.open(os.path.join(work, "results/g5m-q8", v, "answers_mock3.json.gz"), "rt"))
        assert len(LOG) == seen and len(rows) == 3 and "openai" not in last_run("g5m-q8", "qa")
        print(f"qa g5m-q8 (real Qwen3-VL-8B): 3 answers ({[r['parsed_answer'] for r in rows]}), 0 requests")

    # --- the gateway and the key
    assert all(e["path"] == path and e["provider"] == pin and e["model"] in (mini, big) for e in LOG), \
        sorted({(e["path"], e["model"], json.dumps(e["provider"])) for e in LOG})
    print(f"gateway {a.gateway}: all {len(LOG)} requests on {path}, models {sorted({e['model'] for e in LOG})}, provider {pin}")
    assert LOG and all(e["key_ok"] for e in LOG)
    hits = subprocess.run(["grep", "-rlF", KEY, os.path.join(work, "work"), os.path.join(work, "results"),
                           os.path.join(work, "run.log")], capture_output=True, text=True).stdout.split()
    assert not hits, hits
    print(f"key: all {len(LOG)} requests carried the dummy key; no file under {work} contains it")
    print("ok")


if __name__ == "__main__":
    main()
