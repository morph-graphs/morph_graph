#!/usr/bin/env python3
"""One HD-EPIC video -> EGAgent's 1 fps frames (CPU, ffmpeg; no EGAgent import beyond the shared names).

    prepare_epic.py --video <HD-EPIC mp4> --frames_dir <work>/frames/<V>

The frames are cut as upstream's prepare_datasources/sample_videos_1fps.py cuts them (`ffmpeg -vf fps=1`, the
video's own resolution) and named HHMMSS00.jpg, EgoLife's naming, which is what the agent's frame-time
formatter reads. Frame N (1-based) is named for second N: ffmpeg takes it from the source around N - 1 s, so
a frame named for a time <= t_ask never shows anything after t_ask. Skips a finished folder (_done.json).
"""
import argparse
import json
import os
import shutil
import subprocess

from eg_paths import hhmmss


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    done = os.path.join(a.frames_dir, "_done.json")
    if os.path.exists(done):
        print(f"frames done: {json.load(open(done))}")
        return
    tmp = a.frames_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(a.frames_dir, ignore_errors=True)
    os.makedirs(tmp)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", str(a.threads), "-i", a.video,
                    "-vf", "fps=1", os.path.join(tmp, "%06d.jpg")], check=True)
    n = 0
    for f in sorted(os.listdir(tmp)):
        n = int(f[:-4])
        os.rename(os.path.join(tmp, f), os.path.join(tmp, hhmmss(n) + "00.jpg"))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.video],
                               check=True, capture_output=True, text=True).stdout.strip())
    info = {"frames": n, "duration": dur, "video": os.path.basename(a.video)}
    json.dump(info, open(os.path.join(tmp, "_done.json"), "w"))
    os.rename(tmp, a.frames_dir)
    print(f"{n} frames at 1 fps ({dur:.1f} s) -> {a.frames_dir}")


if __name__ == "__main__":
    main()
