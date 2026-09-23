"""Encode the model inputs for one video: the full recording plus one [0, t_ask] clip per streaming cutoff.

Streaming clips are a physical re-encoded cut (``-t t_ask``), not an API-side
offset, so "the model saw nothing after t_ask" is checkable with ffprobe and the
same clip can be reused by local models. Every output is validated by probing its
duration; ``_done.json`` is written only when all of a video's files pass.

    python -m epic_eval.prep_media --run_dir <run> --video P09-20240621-093545
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import (group_by_cutoff, load_run_config, media_done_marker, media_path,
                     questions_for_run, source_video)


def probe_duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def has_audio(path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return bool(out.stdout.strip())


def last_frame_pts(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return max(float(x) for x in out.stdout.split() if x.strip() not in ("", "N/A"))


def expected_ok(path, target_s: float, media: dict) -> tuple[bool, str]:
    """A file is usable if it opens, has the right length, and has no audio unless asked for."""
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return False, "missing"
    try:
        dur = probe_duration(path)
        audio = has_audio(path)
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        return False, f"unreadable: {e}"
    tol = 1.5 / float(media["encode_fps"] or 30.0)
    if dur > target_s + tol:
        return False, f"too long: {dur:.3f}s > {target_s:.3f}s (+{tol:.3f}) -- would leak"
    if dur < target_s - max(tol, 1.0):
        return False, f"truncated: {dur:.3f}s < {target_s:.3f}s"
    if last_frame_pts(path) >= target_s:
        return False, f"last frame at {last_frame_pts(path):.3f}s >= {target_s:.3f}s -- would leak"
    if audio != bool(media["audio"]):
        return False, f"audio={audio}, config wants {media['audio']}"
    return True, f"{dur:.3f}s"


def encode(src, dst, media: dict, duration_s: float | None = None, threads: int = 0) -> None:
    side = int(media["max_side"])
    vf = [f"scale='if(gte(iw,ih),min({side},iw),-2)':'if(gte(iw,ih),-2,min({side},ih))'"]
    if media.get("encode_fps"):
        vf.append(f"fps={media['encode_fps']}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", str(threads)]
    if duration_s is not None:
        # Input-side -t: source frames at or after t_ask are never decoded, so no output
        # frame can carry them. (An output-side cut of an fps-resampled video can keep a
        # frame whose pixels come from up to half a frame interval after t_ask.)
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-i", str(src)]
    cmd += ["-vf", ",".join(vf), "-c:v", "libx264", "-preset", media.get("preset", "veryfast"),
            "-crf", str(media["crf"]), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-threads", str(threads)]
    cmd += ["-c:a", "aac", "-b:a", "64k"] if media["audio"] else ["-an"]
    tmp = Path(dst).with_suffix(".tmp.mp4")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd + [str(tmp)], check=True)
    os.replace(tmp, dst)  # atomic: a preempted encode never leaves a half file under the real name


def prep_video(cfg: dict, video: str, force: bool = False, workers: int = 1) -> dict:
    media, media_dir = cfg["media"], Path(cfg["resolved"]["media_dir"])
    rows = [r for r in questions_for_run(cfg) if r["video"] == video]
    if not rows:
        raise ValueError(f"no questions for {video}")
    src = source_video(cfg, video)
    full_s = probe_duration(src)
    cutoffs = [c for c in group_by_cutoff(rows) if c is not None]
    # Longest first, so the long encodes start early and the pool finishes evenly.
    targets = sorted([(None, full_s)] + [(c, min(c, full_s)) for c in cutoffs], key=lambda t: -t[1])
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    threads = max(1, cpus // max(1, workers))

    def one(cutoff, target):
        dst = media_path(media_dir, video, cutoff)
        ok, why = expected_ok(dst, target, media)
        if ok and not force:
            print(f"  keep {dst.name} ({why})")
        else:
            t0 = time.time()
            encode(src, dst, media, cutoff, threads)  # clips come from the source, never from full.mp4
            ok, why = expected_ok(dst, target, media)
            print(f"  wrote {dst.name} in {time.time() - t0:.0f}s ({why})")
            if not ok:
                raise RuntimeError(f"{dst} failed validation: {why}")
        return dst.name, {"cutoff_s": cutoff, "duration_s": probe_duration(dst), "bytes": dst.stat().st_size}

    print(f"  {len(targets)} files, {workers} parallel encodes x {threads} threads")
    with ThreadPoolExecutor(max(1, workers)) as pool:
        files = dict(pool.map(lambda t: one(*t), targets))
    marker = media_done_marker(media_dir, video)
    marker.write_text(json.dumps({"video": video, "source": str(src), "source_duration_s": full_s,
                                  "media": media, "files": files}, indent=2))
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--force", action="store_true", help="re-encode even valid files")
    a = p.parse_args()
    cfg = load_run_config(a.run_dir)
    print(f"prep {a.video} -> {cfg['resolved']['media_dir']}")
    t0 = time.time()
    files = prep_video(cfg, a.video, a.force, int(cfg.get("runtime", {}).get("prep_workers", 1)))
    print(f"prep wall time {time.time() - t0:.0f}s")
    print(f"done: {len(files)} files")


if __name__ == "__main__":
    main()
