#!/usr/bin/env python3
"""Cut one HD-EPIC video into EgoLife-style 30 s clips WITH audio (EgoGPT is an audio-visual captioner), and
write the clip list.

  <clip_root>/<subject>/DAY1/DAY1_<subject>_<HHMMSSFF>.mp4   30 s clips, short side 448 px, 16 kHz mono audio
  <data_dir>/<subject>_30sec_skeleton.json                    the clip list, without captions

MERIT's clips cannot be reused: they are cut without audio.

The clock. EgoRAG works in EgoLife's day-and-clock times (DAYn, HHMMSSFF), and its code only works for clock
times of 8 digits: it parses the hour as the first two digits of the integer and asks the LLM for `[\\d{8}]`
time stamps. So the whole video is DAY1 and starts at 10:00:00 (CLOCK0): video second s is clock(s). Every
file of this baseline is in that clock; run_qa.py maps t_ask in and the retrieved times back out.
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "merit", "epic"))
from prepare_epic import duration, hhmmss   # noqa: E402

CLOCK0 = 10 * 3600   # video second 0 is 10:00:00 on DAY1


def clock(sec):
    """Video second -> EgoLife clock string HHMMSSFF."""
    return hhmmss(int(sec) + CLOCK0) + "00"


def video_sec(hhmmssff):
    """EgoLife clock (string or int HHMMSSFF) -> video second."""
    s = str(hhmmssff).zfill(8)
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) - CLOCK0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--clip_root", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--clip_sec", type=int, default=30)
    p.add_argument("--size", type=int, default=448, help="short side of the clips")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()

    has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                "-of", "csv=p=0", a.video], check=True, capture_output=True, text=True).stdout.strip()
    assert has_audio, f"{a.video} has no audio stream: EgoGPT's loader needs one"
    dur = duration(a.video)
    n = math.ceil(dur / a.clip_sec)
    clip_dir = os.path.join(a.clip_root, a.subject, "DAY1")
    os.makedirs(clip_dir, exist_ok=True)
    done = os.path.join(clip_dir, f"_clips_{a.subject}.txt")
    if os.path.exists(done):
        names = [l.strip() for l in open(done) if l.strip()]
    else:
        tmp = os.path.join(clip_dir, "_seg")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        s = a.size
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-threads", str(a.threads), "-i", a.video,
             "-vf", f"scale='if(gt(iw,ih),-2,{s})':'if(gt(iw,ih),{s},-2)'",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-ar", "16000", "-ac", "1",
             "-force_key_frames", f"expr:gte(t,n_forced*{a.clip_sec})",
             "-f", "segment", "-segment_time", str(a.clip_sec), "-reset_timestamps", "1",
             os.path.join(tmp, "seg_%04d.mp4")], check=True)
        segs = sorted(glob.glob(os.path.join(tmp, "seg_*.mp4")))
        # the container can run a few seconds past the last video frame (as in MERIT's prep): one clip fewer
        assert max(1, n - 1) <= len(segs) <= n, f"expected {n} segments, got {len(segs)}"
        names = [f"DAY1_{a.subject}_{clock(i * a.clip_sec)}.mp4" for i in range(len(segs))]
        for seg, name in zip(segs, names):
            os.replace(seg, os.path.join(clip_dir, name))
        os.rmdir(tmp)
        open(done, "w").write("".join(x + "\n" for x in names))
    skeleton = []
    for i, name in enumerate(names):
        start, end = i * a.clip_sec, min((i + 1) * a.clip_sec, math.ceil(dur))
        skeleton.append({"start_time": clock(start), "end_time": clock(end), "date": "DAY1",
                         "video_path": f"{a.subject}/DAY1/{name}", "t_start": start, "t_end": end})
    os.makedirs(a.data_dir, exist_ok=True)
    json.dump(skeleton, open(os.path.join(a.data_dir, f"{a.subject}_30sec_skeleton.json"), "w"), indent=2)
    print(f"video {dur:.1f}s -> {len(names)} clips with audio in {clip_dir}; clock {skeleton[0]['start_time']}"
          f" to {skeleton[-1]['end_time']}")


if __name__ == "__main__":
    main()
