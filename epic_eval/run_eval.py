"""Answer every question of one video with the configured model; resumable.

Questions are grouped by what the model may see: the full video, or [0, t_ask]
for streaming questions asked at t_ask. Each group gets one backend context
(one upload + one cache for Gemini) and one query per question. Up to
``runtime.max_concurrent_groups`` contexts are open at once and up to
``runtime.max_concurrent_requests`` queries are in flight, shared across groups.

Output: <run>/answers/<video>.jsonl, one row per answered question. A question
counts as done only when its row has a parsed letter; failures go to
<run>/errors/<video>.jsonl and are retried on the next run.

    python -m epic_eval.run_eval --run_dir <run> --video P09-20240621-093545 [--limit 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from .backends import get_backend
from .common import (LETTERS, build_prompt, group_by_cutoff, load_run_config, media_done_marker,
                     media_path, questions_for_run, system_instruction)


def read_jsonl(path) -> list[dict]:
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut off by preemption; that question is simply redone
    return rows


def done_qids(path) -> set:
    return {r["qid"] for r in read_jsonl(path) if r.get("pred_letter")}


def append_jsonl(path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_video(run_dir, video: str, limit: int | None = None, which: str = "all") -> tuple[int, int]:
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir)
    mcfg, video_on = cfg["model"], cfg["media"]["enabled"]
    backend = get_backend(mcfg["backend"])
    answers_path, errors_path = run_dir / "answers" / f"{video}.jsonl", run_dir / "errors" / f"{video}.jsonl"

    rows = [r for r in questions_for_run(cfg) if r["video"] == video]
    if which != "all":
        rows = [r for r in rows if r["streaming"] == (which == "streaming")]
    done = done_qids(answers_path)
    todo = [r for r in rows if r["qid"] not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{video}: {len(rows)} questions, {len(done & {r['qid'] for r in rows})} done, {len(todo)} to run")
    if not todo:
        return 0, 0

    media_dir = Path(cfg["resolved"]["media_dir"]) if video_on else None
    if video_on and not media_done_marker(media_dir, video).exists():
        raise FileNotFoundError(f"media for {video} not prepared ({media_done_marker(media_dir, video)}); run prep first")
    media_files = json.loads(media_done_marker(media_dir, video).read_text())["files"] if video_on else {}

    client = backend.make_client(mcfg)
    lock = threading.Lock()
    counts = {"ok": 0, "fail": 0}

    def record(path, row: dict, ok: bool) -> None:
        with lock:
            append_jsonl(path, row)
            counts["ok" if ok else "fail"] += 1

    def ask_one(r: dict, ctx: dict, cutoff, vpath, label: str) -> None:
        base = {"qid": r["qid"], "video": video, "cutoff_s": cutoff, "model": mcfg["name"], "time": now()}
        try:  # anything that goes wrong for one question stays with that question
            out = backend.ask(client, mcfg, ctx, build_prompt(r), len(r["options"]))
            gt = LETTERS[r["answer_index"]]
            row = {**base, **out, "gt_letter": gt,
                   "correct": out["pred_letter"] == gt if out["pred_letter"] else None,
                   "media_file": vpath.name if vpath else None,
                   "media_duration_s": media_files.get(vpath.name, {}).get("duration_s") if vpath else None,
                   "cache_tokens": ctx.get("cache_tokens"), "cache_error": ctx.get("cache_error")}
        except Exception as e:
            record(errors_path, {**base, "stage": "ask", "error": f"{type(e).__name__}: {e}",
                                 "traceback": traceback.format_exc()}, False)
            print(f"    [{label}] {r['qid']}: ask failed: {type(e).__name__}: {e}")
            return
        if out["pred_letter"]:
            record(answers_path, row, True)
        else:
            record(errors_path, {**row, "stage": "parse"}, False)
        print(f"    [{label}] {r['qid']} pred={out['pred_letter']} gt={gt} "
              f"finish={out.get('finish_reason')} {out.get('latency_s')}s "
              f"tok={out.get('usage', {}).get('total_token_count')}")

    def run_group(cutoff, group: list[dict], requests: ThreadPoolExecutor) -> None:
        vpath = media_path(media_dir, video, cutoff) if video_on else None
        sys_inst = system_instruction(cutoff is not None, video_on)
        label = "full" if cutoff is None else f"t_ask={cutoff:.1f}s"
        print(f"  [{label}] {len(group)} questions, input={vpath.name if vpath else 'text-only'}")
        ctx = None
        try:
            t0 = time.time()
            ctx = backend.open_context(client, mcfg, vpath, sys_inst, len(group))
            print(f"    [{label}] context ready in {time.time() - t0:.0f}s "
                  f"(cache={'yes' if ctx.get('cache') else 'no'}, cache_tokens={ctx.get('cache_tokens')})")
            # every question of the group is in flight at once, bounded by the shared request pool;
            # wait for all of them (never stop at the first error) before the context is closed
            futures = [requests.submit(ask_one, r, ctx, cutoff, vpath, label) for r in group]
            wait(futures)
            for f in futures:
                if f.exception() is not None:  # ask_one records its own failures; this is a bug
                    print(f"    [{label}] unexpected: {f.exception()!r}")
                    with lock:
                        counts["fail"] += 1
        except Exception as e:
            for r in group:
                record(errors_path, {"qid": r["qid"], "video": video, "cutoff_s": cutoff, "stage": "context",
                                     "time": now(), "error": f"{type(e).__name__}: {e}",
                                     "traceback": traceback.format_exc()}, False)
            print(f"    [{label}] context failed: {type(e).__name__}: {e}")
        finally:
            if ctx is not None:
                backend.close_context(client, ctx)

    runtime = cfg.get("runtime", {})
    n_groups = int(runtime.get("max_concurrent_groups", 1))
    n_requests = int(runtime.get("max_concurrent_requests", 1))
    print(f"  concurrency: {n_groups} contexts, {n_requests} requests in flight")
    with ThreadPoolExecutor(n_requests) as requests, ThreadPoolExecutor(n_groups) as groups:
        for f in [groups.submit(run_group, c, g, requests) for c, g in group_by_cutoff(todo).items()]:
            f.result()
    return counts["ok"], counts["fail"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--limit", type=int, help="answer at most this many pending questions (smoke tests)")
    p.add_argument("--which", choices=["all", "full", "streaming"], default="all")
    a = p.parse_args()
    n_ok, n_fail = run_video(a.run_dir, a.video, a.limit, a.which)
    print(f"done: {n_ok} answered, {n_fail} failed")
    if n_fail:
        sys.exit(1)  # surfaces as FAILED in sacct; rerunning retries only the failures


if __name__ == "__main__":
    main()
