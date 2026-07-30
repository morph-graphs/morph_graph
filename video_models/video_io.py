"""Video decoding + frame sampling (torchcodec only; never decord)."""

from __future__ import annotations

import torch
from ezcolorlog import root_logger as logger

import ctypes
import os

# NOTE: temporary hard cap on sampled frames. Vision-token count (and memory)
# grows linearly with frames, so until chunked encoding lands we clamp any
# fps-based sample to this many frames. FIXME: replace with chunked/streaming
# encoding so long videos keep full temporal coverage.
MAX_SAMPLED_FRAMES = 1024


def _ensure_ffmpeg_libs() -> None:
    """Preload FFmpeg C shared libraries if not already in system search path."""
    ffmpeg_dir = os.environ.get(
        "MORPH_GRAPH_FFMPEG_DIR", "/opt/local/stow/ffmpeg-7.1/lib"
    )
    if not os.path.exists(ffmpeg_dir):
        ffmpeg_dir = "/opt/local/stow/ffmpeg-6.1/lib"
    if not os.path.exists(ffmpeg_dir):
        return

    libs = [
        ["libavutil.so.59", "libavutil.so.58", "libavutil.so"],
        ["libswresample.so.5", "libswresample.so.4", "libswresample.so"],
        ["libavcodec.so.61", "libavcodec.so.60", "libavcodec.so"],
        ["libavformat.so.61", "libavformat.so.60", "libavformat.so"],
        ["libswscale.so.8", "libswscale.so.7", "libswscale.so"],
        ["libavfilter.so.10", "libavfilter.so.9", "libavfilter.so"],
        ["libavdevice.so.61", "libavdevice.so.60", "libavdevice.so"],
    ]

    for names in libs:
        for name in names:
            full_path = os.path.join(ffmpeg_dir, name)
            if os.path.exists(full_path):
                try:
                    ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                    break
                except Exception:
                    pass


def load_decoder(video_path: str):
    # Imported lazily so the package stays importable where torchcodec's native
    # extension isn't loadable. torchcodec only; never decord.
    _ensure_ffmpeg_libs()
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(video_path)



def sample_frame_indices(
    decoder,
    fps: float,
    max_frames: int | None = None,
) -> list[int]:
    """Indices for uniform sampling at target `fps`, capped at `max_frames`."""
    meta = decoder.metadata
    total = meta.num_frames
    src_fps = meta.average_fps
    if total <= 0:
        raise ValueError("Video has no frames")
    if fps <= 0:
        raise ValueError("fps must be positive")

    # Fall back to every frame when source fps is unknown.
    if src_fps is None or src_fps <= 0:
        n = total
    else:
        n = max(1, min(int(round(total / src_fps * fps)), total))
    if max_frames is not None:
        n = min(n, max_frames)
    # Temporary global cap (see MAX_SAMPLED_FRAMES note above).
    n = min(n, MAX_SAMPLED_FRAMES)

    if n == 1:
        return [0]
    return torch.linspace(0, total - 1, n).round().long().tolist()


def decode_frames(
    video_path: str,
    fps: float = 1.0,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Decode a video and return `(frames, indices)`.

    `frames` is a uint8 tensor `(T, C, H, W)` at the sampled indices.
    """
    decoder = load_decoder(video_path)
    meta = decoder.metadata

    # Duration: prefer the container value, else derive from frame count / fps.
    duration = getattr(meta, "duration_seconds", None)
    if (duration is None or duration <= 0) and meta.average_fps and meta.average_fps > 0:
        duration = meta.num_frames / meta.average_fps
    dur_str = f"{duration:.1f}s" if duration else "unknown"
    logger.info(
        f"Loaded {video_path}: duration={dur_str} "
        f"num_frames={meta.num_frames} src_fps={meta.average_fps}"
    )

    indices = sample_frame_indices(decoder, fps, max_frames)
    logger.info(
        f"Sampled {len(indices)} frames at fps={fps} from {meta.num_frames} total"
    )
    batch = decoder.get_frames_at(indices)
    frames = batch.data if hasattr(batch, "data") else batch
    if frames.dtype != torch.uint8:
        frames = frames.clamp(0, 255).to(torch.uint8)
    return frames, indices
