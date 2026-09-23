"""Shared helpers for the EPIC-Morph MCQ evaluation: config, questions, media paths, prompts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

LETTERS = "ABCDEFGH"
RUN_CONFIG = "reproducibility_config.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def config_hash(cfg: dict, n: int = 8) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:n]


def load_run_config(run_dir) -> dict:
    return load_json(Path(run_dir) / RUN_CONFIG)


# ---------------------------------------------------------------- questions

def make_qid(video: str, question: str, t_ask) -> str:
    # The same question text recurs within a video at different ask times
    # (36 cases in the P09 bank), so t_ask is part of the identity.
    key = f"{question}|{'' if t_ask is None else f'{float(t_ask):.3f}'}"
    return f"{video}__{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def load_questions(questions_dir, pattern: str) -> list[dict]:
    """Flatten every ``questions`` list into rows the eval consumes.

    Options come from ``gemini_mcq`` (always 4, answer at ``answer_index``);
    the generator's own ``options`` field is ignored.
    """
    rows = []
    for path in sorted(Path(questions_dir).glob(pattern)):
        bank = load_json(path)
        for q in bank["questions"]:
            mcq = q["gemini_mcq"]
            streaming = bool(q["streaming"])
            t_ask = float(q["t_ask"]) if streaming else None
            if streaming and t_ask is None:
                raise ValueError(f"streaming question without t_ask in {path}: {q['q']!r}")
            rows.append({
                "qid": make_qid(q["video"], q["q"], t_ask),
                "video": q["video"],
                "question": q["q"],
                "options": [str(o) for o in mcq["options"]],
                "answer_index": int(mcq["answer_index"]),
                "answer": str(q["answer"]),
                "streaming": streaming,
                "t_ask": t_ask,
                "form": q["form"],
                "hops": q.get("hops"),
                "caveat": mcq.get("caveat"),
                "binary": str(q["answer"]).strip().lower() in ("yes", "no"),
                "options_model": mcq.get("model"),
                "guessable": q.get("guessable"),
                "majority_baseline": q.get("majority_baseline"),
                "source_file": path.name,
            })
    qids = [r["qid"] for r in rows]
    if len(qids) != len(set(qids)):
        dup = sorted({q for q in qids if qids.count(q) > 1})
        raise ValueError(f"duplicate qids (same video, question and t_ask): {dup[:5]}")
    for r in rows:
        if not 0 <= r["answer_index"] < len(r["options"]):
            raise ValueError(f"{r['qid']}: answer_index {r['answer_index']} out of range")
    return rows


def questions_for_run(cfg: dict) -> list[dict]:
    return load_questions(cfg["resolved"]["questions_dir"], cfg["questions"]["glob"])


def group_by_cutoff(rows: list[dict]) -> dict:
    """{None: full-video questions, t_ask: streaming questions asked at t_ask}.

    One group = one video input = one upload and (maybe) one cache.
    """
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["t_ask"], []).append(r)
    return dict(sorted(groups.items(), key=lambda kv: (kv[0] is not None, kv[0] or 0.0)))


# ---------------------------------------------------------------- media

def media_hash(cfg: dict) -> str:
    return config_hash({"media": cfg["media"], "video_dir": cfg["paths"]["video_dir"]})


def source_video(cfg: dict, video: str) -> Path:
    return Path(cfg["paths"]["video_dir"]) / video.split("-")[0] / f"{video}.mp4"


def media_path(media_dir, video: str, cutoff) -> Path:
    name = "full.mp4" if cutoff is None else f"upto_{float(cutoff):09.3f}s.mp4"
    return Path(media_dir) / video / name


def media_done_marker(media_dir, video: str) -> Path:
    return Path(media_dir) / video / "_done.json"


# ---------------------------------------------------------------- prompts

SYSTEM_VIDEO = (
    "You are watching an egocentric video recorded by a head-mounted camera. "
    "The camera wearer is the person asking the question: 'I', 'me' and 'my' refer "
    "to the camera wearer. {timing} Answer the multiple-choice question using what "
    "is shown in the video, then report only the letter of the single best option."
)
TIMING_STREAMING = (
    "The video ends at the moment the question is asked, so it shows everything "
    "that has happened so far and nothing after."
)
TIMING_FULL = "The question is asked after the whole recording has ended."
SYSTEM_TEXT_ONLY = (
    "You will be asked a multiple-choice question about an egocentric video recorded "
    "by a head-mounted camera; 'I', 'me' and 'my' refer to the camera wearer. The video "
    "is not available to you. Pick the single most likely option from the question "
    "and options alone, then report only its letter."
)


def system_instruction(streaming: bool, video_enabled: bool) -> str:
    if not video_enabled:
        return SYSTEM_TEXT_ONLY
    return SYSTEM_VIDEO.format(timing=TIMING_STREAMING if streaming else TIMING_FULL)


def build_prompt(row: dict) -> str:
    lines = [f"Question: {row['question']}", "", "Options:"]
    lines += [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(row["options"])]
    lines += ["", 'Respond with JSON: {"answer": "<letter>"}.']
    return "\n".join(lines)


def parse_letter(text: str | None, n_options: int) -> str | None:
    """Letter from a model reply: JSON {"answer": "B"} first, then a bare letter. None if neither."""
    if not text:
        return None
    valid = {chr(ord("A") + i) for i in range(n_options)}
    try:
        ans = str(json.loads(text).get("answer", "")).strip().upper()
        if ans in valid:
            return ans
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r'"answer"\s*:\s*"([A-Z])"', text) or re.fullmatch(r"\s*\(?([A-Z])\)?[.\s]*", text)
    return m.group(1) if m and m.group(1) in valid else None
