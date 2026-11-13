#!/usr/bin/env python3
"""Cut one HD-EPIC video into the EgoLife layout Ego-R1's visual tools read (api/visual_tools/egor1_*/utils.py):

    <clip_root>/<video>/A1_JAKE/DAY1/DAY1_A1_JAKE_<HHMMSS>00.mp4     30 s clips, with the audio track

The whole video is DAY1 starting at 00:00:00, seen by one wearer; upstream addresses a wearer as "A1" and looks
in the folder A1_JAKE, so that name is kept (it never reaches a model). Unlike MERIT's prep the audio stays:
upstream uploads the stitched clips to Gemini with their sound. Skips a finished video (_clips.txt).

    python prepare_clips.py --video <mp4> --subject <video id> --clip_root <dir>
"""
import argparse
import glob
import math
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "merit", "epic"))
from prepare_epic import duration, hhmmss   # noqa: E402  (MERIT's helpers: ffprobe duration, HHMMSS)

IDENT = "A1_JAKE"


def day_dir(clip_root, subject):
    return os.path.join(clip_root, subject, IDENT, "DAY1")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--clip_root", required=True)
    ap.add_argument("--clip_sec", type=int, default=30)
    ap.add_argument("--size", type=int, default=704, help="clip height and width (HD-EPIC is square)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    d = day_dir(a.clip_root, a.subject)
    done = os.path.join(d, "_clips.txt")
    if os.path.exists(done):
        print(f"{a.subject}: {len(open(done).read().split())} clips already in {d}")
        return
    n = math.ceil(duration(a.video) / a.clip_sec)
    tmp = os.path.join(d, "_seg")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-threads", str(a.threads), "-i", a.video, "-map", "0:v:0", "-map", "0:a:0?",
         "-vf", f"scale={a.size}:{a.size}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
         "-force_key_frames", f"expr:gte(t,n_forced*{a.clip_sec})",
         "-f", "segment", "-segment_time", str(a.clip_sec), "-reset_timestamps", "1",
         os.path.join(tmp, "seg_%04d.mp4")], check=True)
    segs = sorted(glob.glob(os.path.join(tmp, "seg_*.mp4")))
    # the container can run a little past the last video frame (see MERIT's prepare_epic.make_clips)
    assert max(1, n - 1) <= len(segs) <= n + 1, f"expected about {n} segments, got {len(segs)}"
    names = [f"DAY1_{IDENT}_{hhmmss(i * a.clip_sec)}00.mp4" for i in range(len(segs))]
    for s, name in zip(segs, names):
        os.replace(s, os.path.join(d, name))
    os.rmdir(tmp)
    open(done, "w").write("".join(x + "\n" for x in names))
    print(f"{a.subject}: {len(names)} clips of {a.clip_sec} s in {d}")


if __name__ == "__main__":
    main()
