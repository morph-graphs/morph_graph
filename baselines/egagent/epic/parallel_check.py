#!/usr/bin/env python3
"""Check that answering in parallel (EGAGENT_QA_WORKERS) changes nothing, against stub_check.py's fake APIs (no key,
no network, no cost): the fakes answer as a fixed function of each request, so every question's prompts, trace,
working memory and answer must be the same with 1 worker and with n, and every call must still be counted.

    parallel_check.py --video V --frames_dir <work>/frames/<V> --memory <results>/<arm>/<V> \
                      --bank_file <bank>/<V>_questions.json --tmp <scratch dir> [--workers 6]

Runs run_qa.py over all the bank's questions twice in this process (one worker, then --workers), with the
video's real memory, and compares the rows question by question and the requests the fakes received.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stub_check   # noqa: E402
from stub_check import CALLS, FRAME, STATE, WATCHED, run   # noqa: E402

SAME = ("pred", "plan", "working_memory", "trace", "indexed", "total_tokens", "evaluate")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--memory", required=True)
    ap.add_argument("--bank_file", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    import eg_paths
    watch = eg_paths.watch_api
    eg_paths.watch_api = lambda *args: WATCHED.append(watch(*args)) or WATCHED[-1]
    STATE["route"] = stub_check.ROUTES["openai"]
    os.makedirs(a.tmp, exist_ok=True)
    os.environ.update(OPENAI_API_KEY="stub-not-a-key", HF_HUB_OFFLINE="1", EGAGENT_DATASET="videomme",
                      EGAGENT_BACKBONE="gpt-4.1", EGAGENT_USAGE=os.path.join(a.tmp, "api_usage.jsonl"),
                      EGAGENT_DB_ROOT=os.path.abspath(os.path.join(a.tmp, "dbs")),
                      EGAGENT_MODEL_ROOT=os.path.dirname(eg_paths.EMBEDDER))
    for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(k, None)
    for f in eg_paths.frame_files(a.frames_dir):
        FRAME[hashlib.sha1(open(os.path.join(a.frames_dir, f), "rb").read()).hexdigest()] = int(f[:2]) * 3600 + int(f[2:4]) * 60 + int(f[4:6])
    stub_check.fake_openai()

    got = {}
    # one worker first: the parallel run swaps upstream's trace lists for per-question ones. One --db_root for both:
    # upstream's modules read it once, at their first import
    for w in (1, a.workers):
        os.environ["EGAGENT_QA_WORKERS"] = str(w)
        rows, out = os.path.join(a.tmp, f"rows_w{w}.jsonl"), os.path.join(a.tmp, f"answers_w{w}.json.gz")
        for f in (rows, out):
            if os.path.exists(f):
                os.remove(f)
        n0, w0 = len(CALLS), len(WATCHED)
        rc = run("qa", "run_qa.py", "--video", a.video, "--bank_file", a.bank_file, "--frames_dir", a.frames_dir,
                 "--backbone", "gpt-4.1", "--db_root", os.path.join(a.tmp, "dbs"), "--memory", a.memory,
                 "--rows", rows, "--out", out)
        assert rc == 0, rc
        calls = CALLS[n0:]
        counted = sum(c["calls"] for m in WATCHED[w0:] for c in m.values())
        assert counted == len(calls), f"workers {w}: {len(calls)} calls made, {counted} counted by watch_api"
        got[w] = (json.load(gzip.open(out, "rt")), collections.Counter(hashlib.sha1(c["text"].encode()).hexdigest() for c in calls))
        print(f"workers {w}: {len(got[w][0])} answers, {len(calls)} calls, all counted; "
              f"{sum(len(r['trace']) for r in got[w][0])} trace steps")
    (r1, c1), (rn, cn) = got[1], got[a.workers]
    assert [r["key"] for r in r1] == [r["key"] for r in rn], "the answers are not in the same order"
    diff = [(r["ID"], k) for r, s in zip(r1, rn) for k in SAME if r.get(k) != s.get(k)]
    assert not diff, f"rows differ: {diff[:10]}"
    assert c1 == cn, f"the requests differ: {sum((c1 - cn).values())} only with 1 worker, {sum((cn - c1).values())} only with {a.workers}"
    # upstream's frame search never closes its SQLite connections: on NFS a deleted table stays as a .nfs* file
    # until the process ends, so only the tables' own names count
    left = [f for _, _, fs in os.walk(os.path.join(a.tmp, "dbs")) for f in fs if f.endswith(".db")]
    assert not left, f"tables left behind: {left[:5]}"
    print(f"PARALLEL CHECK PASSED: {len(r1)} questions, the same {', '.join(SAME)} per question and the same "
          f"{sum(c1.values())} requests with 1 and {a.workers} workers")


if __name__ == "__main__":
    main()
