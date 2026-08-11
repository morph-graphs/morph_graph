"""Shared video frame extraction for providers without native video upload.

Both Claude and GPT require frame-by-frame image input.  This module
provides a common ``extract_frames`` → ``frames_to_base64`` pipeline so
the per-provider annotators stay thin.
"""

from __future__ import annotations

import base64
import math
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class FrameInfo:
    """A single extracted frame with its filesystem path and timestamp."""

    path: Path
    timestamp_sec: float


def extract_frames(
    video_path: str | Path,
    *,
    fps: float = 1.0,
    max_frames: int = 100,
    max_width: int = 2000,
    max_height: int = 2000,
    duration_sec: float | None = None,
) -> tuple[list[FrameInfo], Path]:
    """Extract JPEG frames from *video_path* into a temp directory.

    Parameters
    ----------
    video_path:
        Path to the source video.
    fps:
        Desired frame sampling rate.  If the resulting frame count exceeds
        *max_frames*, frames are uniformly sub-sampled.
    max_frames:
        Hard upper bound on returned frames (Claude caps at 100 images per
        request; GPT is limited mainly by tokens).
    max_width, max_height:
        Frames are scaled down so neither dimension exceeds these limits
        (Claude requires ≤ 2000 px per dimension in multi-image requests).
    duration_sec:
        If already known, pass the video duration to avoid a redundant
        ffprobe call.  When *None* a fresh probe is executed.

    Returns
    -------
    (frames, tmp_dir)
        *frames* is ordered by timestamp.  The caller **must** clean up
        *tmp_dir* (e.g. ``shutil.rmtree``) when finished.
    """
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    if duration_sec is None:
        from .gemini import video_duration_sec

        duration_sec = video_duration_sec(video_path)

    # Total frames at the requested fps.
    total_frames = max(1, int(math.floor(duration_sec * fps)))

    # Decide the effective extraction fps: if total_frames > max_frames we
    # lower the ffmpeg fps so we extract at most max_frames directly.
    if total_frames > max_frames:
        effective_fps = max_frames / duration_sec
    else:
        effective_fps = fps

    tmp_dir = Path(tempfile.mkdtemp(prefix="frame_extract_"))
    try:
        # Scale filter: fit within max_width × max_height, only downscale.
        vf = (
            f"fps={effective_fps},"
            f"scale='min({max_width},iw)':min'({max_height},ih)'"
            f":force_original_aspect_ratio=decrease"
        )
        # Simpler, more portable filter string:
        vf = (
            f"fps={effective_fps},"
            f"scale=w='min({max_width},iw)':h='min({max_height},ih)'"
            f":force_original_aspect_ratio=decrease"
        )

        output_pattern = str(tmp_dir / "frame_%06d.jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vf",
                vf,
                "-q:v",
                "2",  # high-quality JPEG
                "-y",
                output_pattern,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Collect and sort the extracted frames.
        frame_paths = sorted(tmp_dir.glob("frame_*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"ffmpeg extracted 0 frames from {video_path}")

        # Assign timestamps: frame_i corresponds to i / effective_fps seconds.
        frames: list[FrameInfo] = []
        for idx, fp in enumerate(frame_paths):
            ts = idx / effective_fps
            frames.append(FrameInfo(path=fp, timestamp_sec=round(ts, 3)))

        # Safety: if we still exceed max_frames (shouldn't happen with fps
        # adjustment, but guard against rounding), uniformly subsample.
        if len(frames) > max_frames:
            indices = [
                int(round(i * (len(frames) - 1) / (max_frames - 1)))
                for i in range(max_frames)
            ]
            frames = [frames[i] for i in indices]

        return frames, tmp_dir

    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def frames_to_base64(frames: Sequence[FrameInfo]) -> list[tuple[str, float]]:
    """Read extracted frames and return ``(base64_str, timestamp)`` pairs."""
    result: list[tuple[str, float]] = []
    for frame in frames:
        data = frame.path.read_bytes()
        b64 = base64.standard_b64encode(data).decode("ascii")
        result.append((b64, frame.timestamp_sec))
    return result


def build_frame_preamble(
    n_frames: int,
    fps: float,
    duration_sec: float,
    timestamps: Sequence[float],
    *,
    subsampled: bool = False,
) -> str:
    """Generate a text preamble that maps frame indices to timestamps.

    Prepended to the event-segmentation prompt so the model knows which
    temporal position each image corresponds to.
    """
    lines = [
        f"You are shown {n_frames} frames sampled from a {duration_sec:.1f}-second video.",
    ]
    if subsampled:
        lines.append(
            f"Frames were uniformly sub-sampled from a higher rate to fit "
            f"the {n_frames}-image limit. Some moments are therefore not shown."
        )
    lines.append("")
    lines.append("Frame timestamps (seconds):")
    for idx, ts in enumerate(timestamps, start=1):
        lines.append(f"  Frame {idx}: {ts:.1f}s")
    lines.append("")
    lines.append(
        "Use these timestamps to assign start_sec / end_sec to events. "
        "An event's interval should cover the frames where it is visible."
    )
    lines.append("")
    return "\n".join(lines)
