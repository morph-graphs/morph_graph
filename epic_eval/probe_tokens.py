"""Measure Gemini video tokens per media_resolution / fps, and project the whole bank.

Uploads a prepared ``full.mp4`` once, calls count_tokens for each setting, and
scales tokens/second to every input the run would send (one full video per
video + one clip per streaming cutoff). Use it to pick ``model.media_resolution``
and ``model.fps`` before the main run -- and on the longest video, since that is
the one that can overflow the context window.

    python -m epic_eval.probe_tokens --run_dir <run> --video P09-20240622-150155
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .backends import gemini
from .common import (group_by_cutoff, load_run_config, media_done_marker, media_path,
                     questions_for_run, source_video)
from .prep_media import probe_duration

LEVELS = [None, "MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--fps", type=float, nargs="+", help="fps values to try (default: config fps)")
    p.add_argument("--context_limit", type=int, default=1_048_576)
    a = p.parse_args()

    cfg = load_run_config(a.run_dir)
    media_dir = Path(cfg["resolved"]["media_dir"])
    manifest = json.loads(media_done_marker(media_dir, a.video).read_text())
    dur = manifest["files"]["full.mp4"]["duration_s"]

    # seconds of video the whole bank would send, per group
    rows = questions_for_run(cfg)
    total_s, longest_s = 0.0, 0.0
    for v in sorted({r["video"] for r in rows}):
        full_s = probe_duration(source_video(cfg, v))
        for cutoff in group_by_cutoff([r for r in rows if r["video"] == v]):
            d = full_s if cutoff is None else min(cutoff, full_s)
            total_s += d
            longest_s = max(longest_s, d)

    client = gemini.make_client(cfg["model"])
    f = gemini.upload(client, cfg["model"], media_path(media_dir, a.video, None))
    try:
        print(f"{a.video}: {dur:.1f}s; bank sends {total_s / 3600:.2f} h of video over all groups "
              f"(longest single input {longest_s / 60:.1f} min)\n")
        print(f"{'media_resolution':26s} {'fps':>4s} {'tokens':>9s} {'tok/s':>7s} {'bank Mtok':>9s} {'longest':>9s}")
        for fps in a.fps or [cfg["model"]["fps"]]:
            for level in LEVELS:
                m = copy.deepcopy(cfg["model"])
                m["fps"], m["media_resolution"] = fps, level
                try:
                    n = gemini.count_video_tokens(client, m, f)
                except Exception as e:
                    print(f"{str(level):26s} {fps:4.1f}  error: {e}")
                    continue
                rate = n / dur
                flag = "  OVER" if rate * longest_s > a.context_limit else ""
                print(f"{str(level):26s} {fps:4.1f} {n:9d} {rate:7.1f} {rate * total_s / 1e6:9.1f} "
                      f"{rate * longest_s / 1e3:8.0f}k{flag}")
    finally:
        gemini.close_context(client, {"file": f})


if __name__ == "__main__":
    main()
