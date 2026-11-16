#!/usr/bin/env python3
"""The session table of a kitchen's stitched video (<K>-stitched), cut at a question's ask time, for the prompt.

A stitched video joins one kitchen's recording sessions in day order, each after a 4 s date card (epic-morph
v3/stitched, Vatsal). Its questions (epic-morph bank `stitched`) are asked at t_ask, mid-video; the date cards are
mostly missed by frame sampling and a memory's timestamps are video time only, so every method that answers them
gets the sessions that started before t_ask in its prompt (Vatsal's protocol, 2026-11-16): session, its date and
start time, and the video time it starts at, then the question's own video time. Later sessions are not listed.

The tables are stitched/sessions/<K>_sessions.json of the results dataset; the same files are committed in
epic-morph (outputs/v3/stitched_handoff/hf/stitched/sessions), which every qa task has ($EPIC_MORPH_DIR), so no
download is needed. STITCHED_SESSIONS_DIR overrides the folder.

    python baselines/common/stitched_sessions.py P05-stitched 8095.8 [--day]   # print the context for t = 8095.8 s
"""
import argparse
import datetime
import json
import os

REL = os.path.join("outputs", "v3", "stitched_handoff", "hf", "stitched", "sessions")


def is_stitched(video):
    return str(video).endswith("-stitched")


def sessions(video, epic=None):
    """[{video, start, video_at_s, ...}] of the stitched video's sessions, in stream order."""
    d = os.environ.get("STITCHED_SESSIONS_DIR") or os.path.join(epic or os.environ["EPIC_MORPH_DIR"], REL)
    return json.load(open(os.path.join(d, f"{video.split('-')[0]}_sessions.json")))["sessions"]


def clock(sec, day=False):
    """Video time: H:MM:SS, or MERIT's / WorldMM's memory timestamp form DAY1 HH:MM:SS (day=True)."""
    s = int(sec)
    return f"DAY1 {s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}" if day else f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def context(video, t, day=False, epic=None):
    """The prompt text: the sessions started before t (video seconds) and the question's video time t (floored,
    as the memory methods ask it); "" for a video that is not stitched."""
    if not is_stitched(video):
        return ""
    ss = [(i, s) for i, s in enumerate(sessions(video, epic), 1) if s["video_at_s"] < t]
    lines = [f"This video is one kitchen's recording sessions joined in time order, each after a 4-second date card. "
             f"The sessions recorded up to the question (session, when it was recorded, where it starts in the video):"]
    for i, s in ss:
        w = datetime.datetime.strptime(s["start"], "%Y-%m-%d %H:%M:%S")
        lines.append(f"S{i}: {w:%A %d %B %Y, %H:%M:%S}, starts at video time {clock(s['video_at_s'], day)}")
    lines.append(f"The question is asked at video time {clock(t, day)}, during session S{ss[-1][0]}; "
                 f"nothing after that moment has happened yet.")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("t", type=float)
    ap.add_argument("--day", action="store_true")
    a = ap.parse_args()
    print(context(a.video, a.t, a.day))
