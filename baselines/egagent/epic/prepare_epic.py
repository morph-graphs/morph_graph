#!/usr/bin/env python3
"""One HD-EPIC video -> EGAgent's 1 fps frames (CPU, ffmpeg; no EGAgent import beyond the shared names).

    prepare_epic.py --video <HD-EPIC mp4> --frames_dir <work>/frames/<V> [--px 0]

The frames are cut as upstream's prepare_datasources/sample_videos_1fps.py cuts them (`ffmpeg -vf fps=1`, the
video's own resolution) and named HHMMSS00.jpg, EgoLife's naming, which is what the agent's frame-time
formatter reads. Frame N (1-based) is named for second N: ffmpeg takes it from the source around N - 1 s, so
a frame named for a time <= t_ask never shows anything after t_ask. --px N scales the frames to N pixels wide
(0, the default: the video's own size, as published; smaller frames cost fewer API tokens but are a different
memory). Skips a finished folder (_done.json), and refuses one cut at another size.
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
    ap.add_argument("--px", type=int, default=0)
    a = ap.parse_args()
    done = os.path.join(a.frames_dir, "_done.json")
    if os.path.exists(done):
        info = json.load(open(done))
        assert info.get("px", 0) == a.px, f"{a.frames_dir} holds frames at px={info.get('px', 0)}, asked {a.px}: use another work folder"
        print(f"frames done: {info}")
        return
    tmp = a.frames_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(a.frames_dir, ignore_errors=True)
    os.makedirs(tmp)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", str(a.threads), "-i", a.video,
                    "-vf", "fps=1" + (f",scale={a.px}:-2" if a.px else ""), os.path.join(tmp, "%06d.jpg")], check=True)
    n = 0
    for f in sorted(os.listdir(tmp)):
        n = int(f[:-4])
        os.rename(os.path.join(tmp, f), os.path.join(tmp, hhmmss(n) + "00.jpg"))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.video],
                               check=True, capture_output=True, text=True).stdout.strip())
    info = {"frames": n, "duration": dur, "video": os.path.basename(a.video), "px": a.px}
    json.dump(info, open(os.path.join(tmp, "_done.json"), "w"))
    os.rename(tmp, a.frames_dir)
    print(f"{n} frames at 1 fps ({dur:.1f} s) -> {a.frames_dir}")


if __name__ == "__main__":
    main()
