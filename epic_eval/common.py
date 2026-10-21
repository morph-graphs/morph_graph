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


BANKS = Path(__file__).with_name("banks.json")


def resolve_bank(questions: dict) -> tuple[dict, str | None]:
    """(questions section, bank name) for a config's ``questions``.

    ``{"bank": "<name>"}`` is looked up in banks.json and becomes exactly the
    ``{repo_url, commit, glob}`` an explicit config would hold, so run hashes do not
    depend on which spelling a config uses. An explicit section passes through.
    """
    if "bank" not in questions:
        return questions, None
    name = questions["bank"]
    if set(questions) != {"bank"}:
        raise ValueError(f"questions: give either a bank name or repo_url/commit/glob, not both: {sorted(questions)}")
    banks = load_json(BANKS)
    if name not in banks:
        raise ValueError(f"unknown question bank {name!r}; known: {', '.join(banks)}")
    b = banks[name]
    return {"repo_url": b["repo_url"], "commit": b["commit"], "glob": b["glob"]}, name


def bank_files_at(repo, commit: str, pattern: str) -> list[str]:
    """Files in ``repo`` at ``commit`` matching ``pattern`` the way Path.glob would (no **)."""
    import fnmatch
    import subprocess
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit], cwd=repo,
                         capture_output=True, text=True, check=True).stdout.split()
    want = pattern.split("/")
    return sorted(f for f in out if len(f.split("/")) == len(want)
                  and all(fnmatch.fnmatchcase(p, w) for p, w in zip(f.split("/"), want)))


def load_questions_at(repo, commit: str, pattern: str) -> list[dict]:
    """load_questions, read straight from git at ``commit`` (works on a bare mirror or any clone)."""
    import subprocess
    banks = [(Path(f), json.loads(subprocess.run(["git", "show", f"{commit}:{f}"], cwd=repo, capture_output=True,
                                                   text=True, check=True).stdout))
             for f in bank_files_at(repo, commit, pattern)]
    return _rows(banks)


def load_questions(questions_dir, pattern: str) -> list[dict]:
    """Flatten every ``questions`` list into rows the eval consumes.

    Options come from ``gemini_mcq`` (always 4, answer at ``answer_index``);
    the generator's own ``options`` field is ignored.
    """
    return _rows((path, load_json(path)) for path in sorted(Path(questions_dir).glob(pattern)))


def _rows(banks) -> list[dict]:
    rows = []
    for path, bank in banks:
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
                # V2 banks: evidence = {events, clips: [{event_id, clip: [t0, t1]}]}; V1 evidence is a list
                "evidence_clips": [[float(t) for t in c["clip"]] for c in q["evidence"]["clips"]]
                                  if isinstance(q.get("evidence"), dict) else [],
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


def group_by_qid(rows: list[dict]) -> dict:
    """{qid: [row]}: evidence mode, where every question has its own video input."""
    return {r["qid"]: [r] for r in sorted(rows, key=lambda r: (r["t_ask"] or 0.0, r["qid"]))}


def group_questions(rows: list[dict], mode: str) -> dict:
    return group_by_qid(rows) if mode == "evidence" else group_by_cutoff(rows)


def evidence_spans(row: dict) -> list[list[float]]:
    """The time spans an evidence-mode question is shown: its clips, sorted, merged, ending by t_ask.

    Overlapping clips are merged so no moment is shown twice. A clip is cut at t_ask
    (dropped if it starts there or later): some banks list evidence events of the
    object after the ask time, which would show the model the future.
    """
    end = row["t_ask"] if row["t_ask"] is not None else float("inf")
    spans = []
    for t0, t1 in sorted((max(0.0, a), min(b, end)) for a, b in row["evidence_clips"]):
        if t1 <= t0:
            continue
        if spans and t0 <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], t1)
        else:
            spans.append([t0, t1])
    if not spans:
        raise ValueError(f"{row['qid']}: no evidence clips before t_ask ({row['evidence_clips']}, t_ask={row['t_ask']})")
    return [[round(a, 3), round(b, 3)] for a, b in spans]


# ---------------------------------------------------------------- media

def media_mode(cfg: dict) -> str:
    """Either "cutoff" (full video / [0, t_ask]) or "evidence" (only the question's evidence clips)."""
    return cfg["media"].get("mode", "cutoff")


def media_hash(cfg: dict) -> str:
    # "mode" is a media key, so evidence media get their own folder; cutoff configs
    # without the key keep the hash (and folder) they always had
    return config_hash({"media": cfg["media"], "video_dir": cfg["paths"]["video_dir"]})


def source_video(cfg: dict, video: str) -> Path:
    return Path(cfg["paths"]["video_dir"]) / video.split("-")[0] / f"{video}.mp4"


def media_path(media_dir, video: str, key, mode: str = "cutoff") -> Path:
    """key is the group key: a cutoff (None = full video), or a qid in evidence mode."""
    if mode == "evidence":
        return Path(media_dir) / video / f"evidence_{key}.mp4"
    name = "full.mp4" if key is None else f"upto_{float(key):09.3f}s.mp4"
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
SYSTEM_EVIDENCE = (
    "You are watching excerpts from an egocentric video recorded by a head-mounted camera. "
    "The camera wearer is the person asking the question: 'I', 'me' and 'my' refer to the "
    "camera wearer. You are shown only a few short excerpts from the recording, joined "
    "together in chronological order. Answer the multiple-choice question using what is "
    "shown in these excerpts, then report only the letter of the single best option."
)
SYSTEM_TEXT_ONLY = (
    "You will be asked a multiple-choice question about an egocentric video recorded "
    "by a head-mounted camera; 'I', 'me' and 'my' refer to the camera wearer. The video "
    "is not available to you. Pick the single most likely option from the question "
    "and options alone, then report only its letter."
)


def system_instruction(streaming: bool, video_enabled: bool, mode: str = "cutoff") -> str:
    if not video_enabled:
        return SYSTEM_TEXT_ONLY
    if mode == "evidence":
        return SYSTEM_EVIDENCE
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
