"""Tests for the things that decide whether the numbers mean anything.

    module load ffmpeg/7.1   # for the media test; it is skipped without ffprobe
    python -m epic_eval.test_epic_eval            (or: pytest epic_eval/test_epic_eval.py)

EPIC_MORPH_DIR points at an epic-morph checkout (default: the local one).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

from . import backends
from .common import (LETTERS, build_prompt, group_by_cutoff, load_questions, media_path,
                     system_instruction)

EPIC = Path(os.environ.get("EPIC_MORPH_DIR", "/fs/cfar-projects/mem_open_vocab/morph_graphs/anubhav/epic-morph"))
GLOB = "outputs/questions/*_questions.json"
CFG = json.loads((Path(__file__).parent / "configs" / "gemini-3.8-flash-high.json").read_text())
SHORT_VIDEO = "P09-20240622-194642"  # 6.3 min, 13 questions incl. 2 streaming


def bank():
    return load_questions(EPIC, GLOB)


def test_bank_shape():
    rows = bank()
    assert len(rows) == 449, len(rows)
    assert sum(r["streaming"] for r in rows) == 222
    assert all(r["t_ask"] is not None for r in rows if r["streaming"])
    assert all(r["t_ask"] is None for r in rows if not r["streaming"])
    assert all(len(r["options"]) == 4 for r in rows)
    assert len({r["qid"] for r in rows}) == len(rows)
    # the letter we score against is the bank's own answer text
    assert all(r["options"][r["answer_index"]].strip().lower() == r["answer"].strip().lower() for r in rows)


def test_grouping():
    rows = bank()
    n_groups = 0
    for v in {r["video"] for r in rows}:
        g = group_by_cutoff([r for r in rows if r["video"] == v])
        n_groups += len(g)
        for cutoff, members in g.items():
            assert all(m["t_ask"] == cutoff for m in members)
            assert all(m["streaming"] == (cutoff is not None) for m in members)
    assert n_groups == 140 + 12, n_groups  # 140 distinct streaming cutoffs + 1 full group per video


def test_prompt():
    r = bank()[0]
    p = build_prompt(r)
    for i, opt in enumerate(r["options"]):
        assert f"{LETTERS[i]}. {opt}" in p
    assert r["question"] in p
    assert "ends at the moment" in system_instruction(True, True)
    assert "after the whole recording" in system_instruction(False, True)
    assert "not available" in system_instruction(True, False)


def test_parse_letter():
    from .backends.gemini import parse_letter
    assert parse_letter('{"answer": "C"}', 4) == "C"
    assert parse_letter('{"answer": "c"}', 4) == "C"
    assert parse_letter('garbage "answer": "B" more', 4) == "B"
    assert parse_letter("D", 4) == "D"
    assert parse_letter('{"answer": "E"}', 4) is None
    assert parse_letter("", 4) is None


def make_run(tmp: Path, media_enabled: bool, media_dir=None) -> Path:
    cfg = json.loads(json.dumps(CFG))
    cfg["media"]["enabled"] = media_enabled
    cfg["model"]["backend"] = "fake"
    cfg["code"]["commit"] = "test"
    cfg["resolved"] = {"run_dir": str(tmp), "code_dir": str(Path(__file__).parents[1]),
                       "questions_dir": str(EPIC), "media_dir": str(media_dir or tmp / "media")}
    (tmp / "reproducibility_config.json").write_text(json.dumps(cfg))
    return tmp


def install_fake_backend(seen: list):
    """Answers correctly iff the question is streaming; records every context it opens."""
    fake = types.ModuleType("epic_eval.backends.fake")
    fake.make_client = lambda mcfg: None
    fake.open_context = lambda client, mcfg, path, si, n: (seen.append((path, si, n)) or
                                                            {"path": path, "si": si})
    fake.close_context = lambda client, ctx: None

    def ask(client, mcfg, ctx, prompt, n_options):
        streaming = "ends at the moment" in ctx["si"]
        return {"pred_letter": "?", "raw_text": "", "usage": {"total_token_count": 1},
                "streaming": streaming, "cache_used": False}
    fake.ask = ask
    sys.modules["epic_eval.backends.fake"] = fake
    backends.BACKENDS["fake"] = "epic_eval.backends.fake"


def test_run_eval_groups_resume_and_score():
    from . import run_eval, score
    seen = []
    install_fake_backend(seen)
    video = "P09-20240621-153208"  # 24 questions, mixed streaming/full
    rows = [r for r in bank() if r["video"] == video]
    gt = {r["qid"]: LETTERS[r["answer_index"]] for r in rows}
    wrong = {q: LETTERS[(LETTERS.index(l) + 1) % 4] for q, l in gt.items()}
    stream = {r["qid"] for r in rows if r["streaming"]}

    orig_ask = sys.modules["epic_eval.backends.fake"].ask
    def ask(client, mcfg, ctx, prompt, n_options):
        out = orig_ask(client, mcfg, ctx, prompt, n_options)
        qid = next(r["qid"] for r in rows if r["question"] in prompt and r["t_ask"] == ctx["cutoff"])
        out["pred_letter"] = gt[qid] if qid in stream else wrong[qid]
        return out
    sys.modules["epic_eval.backends.fake"].ask = ask
    sys.modules["epic_eval.backends.fake"].open_context = lambda client, mcfg, path, si, n: (
        seen.append((path, si, n)) or {"path": path, "si": si,
                                        "cutoff": None if path.name == "full.mp4" else
                                        float(path.name[5:-5])})

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        run = make_run(tmp, True)
        vdir = tmp / "media" / video
        vdir.mkdir(parents=True)
        (vdir / "_done.json").write_text(json.dumps({"files": {}}))  # media presence is prep's job, not this test's

        ok, fail = run_eval.run_video(run, video, limit=5)
        assert (ok, fail) == (5, 0)
        ok, fail = run_eval.run_video(run, video)
        assert (ok, fail) == (len(rows) - 5, 0), (ok, fail)
        assert run_eval.run_video(run, video) == (0, 0)  # resume: nothing left

        # every streaming group got its own [0, t_ask] clip, full questions got full.mp4
        for path, si, n in seen:
            if path.name == "full.mp4":
                assert "after the whole recording" in si
            else:
                assert "ends at the moment" in si
                assert path == media_path(tmp / "media", video, float(path.name[5:-5]))
        cutoffs_seen = {p.name for p, _, _ in seen}
        want = {media_path(tmp / "media", video, c).name for c in group_by_cutoff(rows)}
        assert cutoffs_seen == want

        res = score.score_run(run)
        s = res["splits"]
        assert s["streaming"]["acc_answered"] == 1.0 and s["full_video"]["acc_answered"] == 0.0
        n_stream = sum(r["streaming"] for r in rows)
        assert abs(s["all"]["acc"] - n_stream / 449) < 1e-9  # missing questions count as wrong
        assert s["all"]["answered"] == len(rows) and s["all"]["n"] == 449
        assert len(res["missing_qids"]) == 449 - len(rows)


def test_text_only_sends_no_video():
    from . import run_eval
    seen = []
    install_fake_backend(seen)
    fake = sys.modules["epic_eval.backends.fake"]
    fake.ask = lambda client, mcfg, ctx, prompt, n: {"pred_letter": "A", "raw_text": "", "usage": {},
                                                     "cache_used": False}
    video = "P09-20240622-194642"
    with tempfile.TemporaryDirectory() as d:
        run = make_run(Path(d), False)
        cfg = json.loads((run / "reproducibility_config.json").read_text())
        cfg["resolved"]["media_dir"] = None
        (run / "reproducibility_config.json").write_text(json.dumps(cfg))
        ok, fail = run_eval.run_video(run, video)
        n = len([r for r in bank() if r["video"] == video])
        assert (ok, fail) == (n, 0)
        assert all(path is None and "not available" in si for path, si, _ in seen)


def test_media_trim_no_leak():
    if not shutil.which("ffprobe"):
        print("  (skipped: no ffprobe on PATH -- module load ffmpeg/7.1)")
        return
    from .prep_media import has_audio, prep_video, probe_duration
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        run = make_run(tmp, True)
        cfg = json.loads((run / "reproducibility_config.json").read_text())
        files = prep_video(cfg, SHORT_VIDEO)
        rows = [r for r in bank() if r["video"] == SHORT_VIDEO]
        tol = 1.5 / cfg["media"]["encode_fps"]
        for cutoff in group_by_cutoff(rows):
            p = media_path(tmp / "media", SHORT_VIDEO, cutoff)
            dur = probe_duration(p)
            assert files[p.name]["duration_s"] == dur
            assert not has_audio(p)
            if cutoff is not None:
                assert cutoff - 1.0 <= dur <= cutoff + tol, (p.name, dur, cutoff)
                print(f"  {p.name}: t_ask={cutoff:.3f}s clip={dur:.3f}s")
            else:
                print(f"  full.mp4: {dur:.3f}s, {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
