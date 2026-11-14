#!/usr/bin/env python3
"""EGAgent's raw visual captions of one video: GPT-4.1 on each 30 s window of 1 fps frames. PAID API (OpenAI).

    caption_epic.py --frames_dir <work>/frames/<V> --out <results>/<V>_captions.json [--dry]

The paper's EgoLife setting ("GPT-4.1 on each 30 second raw video clip sampled at 1 FPS"), sent with upstream's
utils.query_multimodal (frames at their own resolution, temperature 0). Upstream released neither the
captioning script nor its prompt: PROMPT below is ours and is the one part of the memory that is not the
authors' text. Writes [{"start": s, "end": s, "caption": str}] in time order; resumes from a .partial file.
--dry prints the calls it would make and makes none.
"""
import argparse
import contextvars
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import eg_paths

SYSTEM = "You are an expert at describing egocentric (first-person) video."
PROMPT = ("These are {n} frames sampled at 1 frame per second from a {sec} second clip of a first-person video. "
          "Describe in detail what happens in the clip: the location, the people present, what the camera wearer "
          "does, and the objects they interact with.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    files = eg_paths.frame_files(a.frames_dir)
    C = eg_paths.CLIP_S
    wins = [(s, min(s + C, len(files))) for s in range(0, len(files), C)]   # frames s+1 .. e cover [s, e) seconds
    if a.dry:
        print(f"{len(wins)} {eg_paths.MEMORY_LLM} caption calls, {len(files)} frames in all")
        return
    part = a.out + ".partial"
    done = {r["start"]: r for r in map(json.loads, open(part))} if os.path.exists(part) else {}
    from utils import query_multimodal   # needs OPENAI_API_KEY
    eg_paths.watch_api("captions", eg_paths.MEMORY_LLM)

    def cap(w):
        s, e = w
        paths = [os.path.join(a.frames_dir, f) for f in files[s:e]]
        out = query_multimodal(SYSTEM, PROMPT.format(n=e - s, sec=e - s), paths, eg_paths.MEMORY_LLM)
        return {"start": s, "end": e, "caption": out.content, "tokens": (out.usage_metadata or {}).get("total_tokens")}

    todo = [w for w in wins if w[0] not in done]
    # each call in a copy of this context (a new thread starts without the API watch); a finished caption is
    # saved at once, and on any failure the windows not yet sent are dropped, so nothing is paid for and lost
    ex = ThreadPoolExecutor(a.workers)
    futs = [ex.submit(contextvars.copy_context().run, cap, w) for w in todo]
    try:
        with open(part, "a") as f:
            for fut in as_completed(futs):
                r = fut.result()
                done[r["start"]] = r
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"[{r['start']:>5}-{r['end']:>5}] {r['caption'][:100]!r}", flush=True)
    except BaseException:
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown()
    rows = [done[w[0]] for w in wins]
    assert all(r["caption"].strip() for r in rows), "empty caption"
    json.dump(rows, open(a.out + ".tmp", "w"), indent=1)
    os.replace(a.out + ".tmp", a.out)
    os.remove(part)
    print(f"{len(rows)} captions -> {a.out}")


if __name__ == "__main__":
    main()
