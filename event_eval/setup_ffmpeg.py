"""Fetch static ffmpeg/ffprobe binaries and link them into the active venv.

``event_eval/frames.py`` and the annotators shell out to plain ``ffmpeg`` /
``ffprobe``.  The ``static-ffmpeg`` package ships the binaries but under its
own names, so this one-time step downloads them (first call) and symlinks
them into the environment's ``bin`` so they are on PATH whenever the venv is
active.

Usage (inside the venv):
    python -m event_eval.setup_ffmpeg
"""

from __future__ import annotations

import sys
from pathlib import Path

import static_ffmpeg.run


def main() -> None:
    ffmpeg, ffprobe = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    bin_dir = Path(sys.prefix) / ("Scripts" if sys.platform == "win32" else "bin")
    for src, name in ((ffmpeg, "ffmpeg"), (ffprobe, "ffprobe")):
        dest = bin_dir / name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src)
        print(f"{dest} -> {src}")
    print("Done. ffmpeg/ffprobe are on PATH while this venv is active.")


if __name__ == "__main__":
    main()
