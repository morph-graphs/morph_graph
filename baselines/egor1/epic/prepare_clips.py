#!/usr/bin/env python3
"""Cut one HD-EPIC video into the EgoLife layout Ego-R1's visual tools read (api/visual_tools/egor1_*/utils.py):

    <clip_root>/<video>/A1_JAKE/DAY1/DAY1_A1_JAKE_<HHMMSS>00.mp4     30 s clips, with the audio track

The whole video is DAY1 starting at 00:00:00, seen by one wearer; upstream addresses a wearer as "A1" and looks
in the folder A1_JAKE, so that name is kept (it never reaches a model). Unlike MERIT's prep the audio stays:
upstream uploads the stitched clips to Gemini with their sound. Skips a finished video (_clips.txt); an interrupted one
is resumed.

    python prepare_clips.py --video <mp4> --subject <video id> --clip_root <dir>
"""
import argparse
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

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
    os.makedirs(d, exist_ok=True)
    names = [f"DAY1_{IDENT}_{hhmmss(i * a.clip_sec)}00.mp4" for i in range(n)]

    def cut(i):
        # one seek per clip, so clip i holds exactly [30 i, 30 i + 30) s. Cutting with the segment muxer and forced
        # key frames gave clips of 26-36 s that ran up to 8 s late with ffmpeg 7.1 (measured 2026-11-13), and the
        # streaming check in tools.py relies on a clip starting where its name says.
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-threads", "2", "-ss", str(i * a.clip_sec), "-i", a.video, "-t", str(a.clip_sec),
             "-map", "0:v:0", "-map", "0:a:0?", "-vf", f"scale={a.size}:{a.size}", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "23", "-c:a", "aac", "-f", "mp4", os.path.join(d, names[i] + ".part")], check=True)
        os.replace(os.path.join(d, names[i] + ".part"), os.path.join(d, names[i]))
    with ThreadPoolExecutor(max(1, a.threads // 2)) as ex:
        list(ex.map(cut, [i for i in range(n) if not os.path.exists(os.path.join(d, names[i]))]))

    def video_seconds(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of",
                              "csv=p=0", path], capture_output=True, text=True).stdout.strip()
        return float(out) if out and out != "N/A" else 0.0
    lens = [video_seconds(os.path.join(d, x)) for x in names]
    if lens[-1] < 0.1:      # the container can run a little past the last video frame: no such clip
        os.remove(os.path.join(d, names.pop())); lens.pop()
    bad = [(x, l) for x, l in zip(names[:-1], lens[:-1]) if abs(l - a.clip_sec) > 0.2] + \
        [(names[-1], lens[-1])] * (lens[-1] > a.clip_sec + 0.2)
    assert not bad, f"clips that are not {a.clip_sec} s long: {bad[:5]}"
    open(done, "w").write("".join(x + "\n" for x in names))
    print(f"{a.subject}: {len(names)} clips of {a.clip_sec} s in {d}")


if __name__ == "__main__":
    main()
