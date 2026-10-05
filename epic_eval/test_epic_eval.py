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
from .common import (LETTERS, build_prompt, group_by_cutoff, group_questions, load_questions, media_hash,
                     media_path, system_instruction)

EPIC = Path(os.environ.get("EPIC_MORPH_DIR", "/fs/cfar-projects/mem_open_vocab/morph_graphs/anubhav/epic-morph"))
GLOB = "outputs/questions/*_questions.json"
CFG = json.loads((Path(__file__).parent / "configs" / "gemini-3.8-flash-high.json").read_text())
SHORT_VIDEO = "P09-20240622-194642"  # 6.3 min, 13 questions incl. 2 streaming


def bank():
    return load_questions(EPIC, GLOB)


def _raw_questions():
    return [q for f in sorted((EPIC / "outputs" / "questions").glob("*_questions.json"))
            for q in json.loads(f.read_text())["questions"]]


def test_bank_shape():
    """Invariants, not counts: the bank is expected to change between commits."""
    rows, raw = bank(), _raw_questions()
    assert len(rows) == len(raw) > 0
    assert sum(r["streaming"] for r in rows) == sum(bool(q["streaming"]) for q in raw)
    assert all(r["t_ask"] is not None for r in rows if r["streaming"])
    assert all(r["t_ask"] is None for r in rows if not r["streaming"])
    assert all(2 <= len(r["options"]) <= 4 for r in rows)  # 2 for yes/no and event_order from epic-morph a240b88
    assert len({r["qid"] for r in rows}) == len(rows)
    # the option we score as correct is the bank's answer, verbatim or as "<yes|no> - <reason>"
    for r in rows:
        opt, ans = r["options"][r["answer_index"]].strip().lower(), r["answer"].strip().lower()
        assert opt == ans or (r["binary"] and opt.startswith(ans + " - ")), (r["qid"], ans, opt)
    print(f"  {len(rows)} questions, {sum(r['streaming'] for r in rows)} streaming, "
          f"{sum(r['binary'] for r in rows)} yes/no")


def test_grouping():
    rows, raw = bank(), _raw_questions()
    n_groups = 0
    for v in {r["video"] for r in rows}:
        g = group_by_cutoff([r for r in rows if r["video"] == v])
        n_groups += len(g)
        for cutoff, members in g.items():
            assert all(m["t_ask"] == cutoff for m in members)
            assert all(m["streaming"] == (cutoff is not None) for m in members)
    # one context per distinct (video, t_ask) plus one full-video context per video with full questions
    want = len({(q["video"], float(q["t_ask"])) for q in raw if q["streaming"]}) + \
        len({q["video"] for q in raw if not q["streaming"]})
    assert n_groups == want, (n_groups, want)
    print(f"  {n_groups} contexts")


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
    from .common import parse_letter
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


def test_concurrency_is_bounded_and_complete():
    """Parallel groups/requests: every question answered once, never more in flight than allowed."""
    import threading
    import time
    from . import run_eval
    seen = []
    install_fake_backend(seen)
    fake = sys.modules["epic_eval.backends.fake"]
    state = {"now": 0, "peak": 0, "open": 0, "peak_open": 0}
    lock = threading.Lock()

    def open_context(client, mcfg, path, si, n):
        with lock:
            state["open"] += 1
            state["peak_open"] = max(state["peak_open"], state["open"])
        return {"path": path, "si": si}

    def close_context(client, ctx):
        with lock:
            state["open"] -= 1

    def ask(client, mcfg, ctx, prompt, n):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return {"pred_letter": "A", "raw_text": "", "usage": {}, "cache_used": False}
    fake.open_context, fake.close_context, fake.ask = open_context, close_context, ask
    video = "P09-20240622-150155"  # 91 questions, the largest video
    with tempfile.TemporaryDirectory() as d:
        run = make_run(Path(d), False)
        cfg = json.loads((run / "reproducibility_config.json").read_text())
        cfg["resolved"]["media_dir"] = None
        cfg["runtime"] = {"max_concurrent_groups": 3, "max_concurrent_requests": 8}
        (run / "reproducibility_config.json").write_text(json.dumps(cfg))
        ok, fail = run_eval.run_video(run, video)
        n = len([r for r in bank() if r["video"] == video])
        rows = run_eval.read_jsonl(run / "answers" / f"{video}.jsonl")
        assert (ok, fail) == (n, 0) and len(rows) == n and len({r["qid"] for r in rows}) == n
        assert 1 < state["peak"] <= 8, state
        assert 1 < state["peak_open"] <= 3, state
        assert state["open"] == 0  # every context closed
        print(f"  peak in-flight requests {state['peak']}/8, open contexts {state['peak_open']}/3")


def test_one_bad_reply_does_not_close_context_early():
    import threading
    import time
    from . import run_eval
    seen = []
    install_fake_backend(seen)
    fake = sys.modules["epic_eval.backends.fake"]
    state = {"closed": set(), "late": 0, "calls": 0}
    lock = threading.Lock()
    fake.open_context = lambda client, mcfg, path, si, n: {"id": path.name, "si": si}
    fake.close_context = lambda client, ctx: state["closed"].add(ctx["id"])

    def ask(client, mcfg, ctx, prompt, n):
        with lock:
            state["calls"] += 1
            first = state["calls"] == 1
        if first:
            return None  # malformed: row construction will raise
        time.sleep(0.05)
        if ctx["id"] in state["closed"]:
            state["late"] += 1
        return {"pred_letter": "A", "raw_text": "", "usage": {}, "cache_used": False}
    fake.ask = ask
    video = "P09-20240622-194642"  # 11 full-video questions share one context
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        run = make_run(tmp, True)
        (tmp / "media" / video).mkdir(parents=True)
        (tmp / "media" / video / "_done.json").write_text(json.dumps({"files": {}}))
        ok, fail = run_eval.run_video(run, video)
        n = len([r for r in bank() if r["video"] == video])
        assert (ok, fail) == (n - 1, 1), (ok, fail)
        assert state["late"] == 0  # nobody used a context after it was closed
        errs = run_eval.read_jsonl(tmp / "errors" / f"{video}.jsonl")
        assert len(errs) == 1 and errs[0]["stage"] == "ask"


def test_truncated_replies_are_not_retried():
    from . import run_eval, score
    seen = []
    install_fake_backend(seen)
    fake = sys.modules["epic_eval.backends.fake"]
    calls = []

    def ask(client, mcfg, ctx, prompt, n):
        calls.append(prompt)
        if len(calls) == 1:  # thinking ate the output budget: no answer text
            return {"pred_letter": None, "raw_text": "", "finish_reason": "MAX_TOKENS", "usage": {},
                    "cache_used": False}
        return {"pred_letter": "A", "raw_text": "", "finish_reason": "STOP", "usage": {}, "cache_used": False}
    fake.ask = ask
    video = "P09-20240622-194642"
    n = len([r for r in bank() if r["video"] == video])
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        run = make_run(tmp, False)
        cfg = json.loads((run / "reproducibility_config.json").read_text())
        cfg["resolved"]["media_dir"] = None
        (run / "reproducibility_config.json").write_text(json.dumps(cfg))
        assert run_eval.run_video(run, video) == (n - 1, 1)
        assert run_eval.run_video(run, video) == (0, 0) and len(calls) == n  # not re-asked
        res = score.score_run(run)
        assert len(res["truncated_qids"]) == 1 and len(res["missing_qids"]) == 449 - n + 1
        cfg["runtime"] = {"retry_truncated": True}
        (run / "reproducibility_config.json").write_text(json.dumps(cfg))
        assert run_eval.run_video(run, video) == (1, 0)  # opt-in retry
        assert score.score_run(run)["truncated_qids"] == []


def test_explorer_joins_runs_and_keeps_verdicts():
    """Two runs over one video: the explorer's per-run right/wrong must equal each run's answers."""
    from . import explorer, run_eval
    seen = []
    install_fake_backend(seen)
    fake = sys.modules["epic_eval.backends.fake"]
    video = "P09-20240622-194642"
    rows = [r for r in bank() if r["video"] == video]
    with tempfile.TemporaryDirectory() as d:
        runs = []
        for k, pick in enumerate(("A", "B")):
            fake.ask = lambda client, mcfg, ctx, prompt, n, pick=pick: {
                "pred_letter": pick, "raw_text": "", "usage": {"thoughts_token_count": 7},
                "cache_used": False, "thoughts": "because " * 400}
            (Path(d) / f"run{k}").mkdir()
            run = make_run(Path(d) / f"run{k}", False)
            cfg = json.loads((run / "reproducibility_config.json").read_text())
            cfg["resolved"]["media_dir"] = None
            (run / "reproducibility_config.json").write_text(json.dumps(cfg))
            assert run_eval.run_video(run, video) == (len(rows), 0)
            runs.append(run)
        data = explorer.build(runs, ["first", "second"], max_thought=100,
                              verdicts={rows[0]["qid"]: {"verdict": "bank answer wrong", "note": "x"}})
        # every bank question is listed (unanswered ones included); each run answered this video only
        assert len(data["runs"]) == 2 and len(data["questions"]) == len(bank())
        assert [r["answered"] for r in data["runs"]] == [len(rows), len(rows)]
        by = {q["qid"]: q for q in data["questions"]}
        for r in rows:
            q = by[r["qid"]]
            for k, pick in enumerate("AB"):
                got = q["per"][str(k)]
                assert got["pick"] == LETTERS.index(pick)
                assert got["ok"] == (LETTERS[r["answer_index"]] == pick)
                assert len(got["thoughts"]) <= 102
            assert q["opts_by_run"] == {}          # same bank in both runs
        html = explorer.render(data)
        assert "<title>EPIC-Morph Run Explorer</title>" in html and "/*DATA*/" not in html
        blob = html.split("const DATA = ", 1)[1].split(";\nconst RUNS", 1)[0]
        assert json.loads(blob.replace("<\\/", "</"))["verdicts"][rows[0]["qid"]]["verdict"] == "bank answer wrong"
        answered = [q for q in data["questions"] if q["per"]]
        print(f"  {len(answered)} questions x 2 runs joined; A right on "
              f"{sum(q['per']['0']['ok'] for q in answered)}, B right on {sum(q['per']['1']['ok'] for q in answered)}")


def _max_luma(path) -> int:
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return max(raw) if raw else -1


def test_clip_has_no_pixels_after_t_ask():
    """Synthetic source: black before T, white from T on. Any white pixel in the clip is a leak."""
    if not shutil.which("ffmpeg"):
        print("  (skipped: no ffmpeg on PATH -- module load ffmpeg/7.1)")
        return
    import subprocess
    from .prep_media import encode, expected_ok
    T = 10.2  # between two 2-fps output ticks, where resampling could pull a later frame in
    media = dict(CFG["media"], max_side=64)
    with tempfile.TemporaryDirectory() as d:
        src, clip, full, old = (Path(d) / n for n in ("src.mp4", "clip.mp4", "full.mp4", "old.mp4"))
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=black:s=64x64:r=30:d={T}",
                        "-f", "lavfi", "-i", "color=white:s=64x64:r=30:d=3",
                        "-filter_complex", "[0][1]concat=n=2:v=1", "-c:v", "libx264", "-qp", "0", str(src)],
                       check=True)
        encode(src, clip, media, T)
        assert _max_luma(clip) < 40, _max_luma(clip)  # all black: nothing at or after T
        assert expected_ok(clip, T, media)[0], expected_ok(clip, T, media)
        encode(src, full, media)  # the previous method: cut the fps-resampled full video
        encode(full, old, media, T)
        print(f"  new cut max luma {_max_luma(clip)}; old cut-from-full max luma {_max_luma(old)}")


def test_media_trim_no_leak():
    if not shutil.which("ffprobe"):
        print("  (skipped: no ffprobe on PATH -- module load ffmpeg/7.1)")
        return
    from .prep_media import has_audio, last_frame_pts, prep_video, probe_duration
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        run = make_run(tmp, True)
        cfg = json.loads((run / "reproducibility_config.json").read_text())
        files = prep_video(cfg, SHORT_VIDEO, workers=3)  # parallel path
        rows = [r for r in bank() if r["video"] == SHORT_VIDEO]
        tol = 1.5 / cfg["media"]["encode_fps"]
        for cutoff in group_by_cutoff(rows):
            p = media_path(tmp / "media", SHORT_VIDEO, cutoff)
            dur = probe_duration(p)
            assert files[p.name]["duration_s"] == dur
            assert not has_audio(p)
            if cutoff is not None:
                assert cutoff - 1.0 <= dur <= cutoff + tol, (p.name, dur, cutoff)
                assert last_frame_pts(p) < cutoff
                print(f"  {p.name}: t_ask={cutoff:.3f}s clip={dur:.3f}s")
            else:
                print(f"  full.mp4: {dur:.3f}s, {p.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------- evidence mode

EV_VIDEO = "P00-20990101-000000"


def make_evidence_run(tmp: Path) -> tuple[Path, dict]:
    """A 12 s source, black inside [2, 4) and [6, 9) and white elsewhere, and a two-question bank.

    q1 has two clips inside the black spans, listed out of order; q2 has one clip that
    runs from black into white past its t_ask (9.0), so only [8.3, 9.0] may be shown.
    Any white pixel in an evidence file is footage from outside its clips.
    """
    import subprocess
    src = tmp / "videos" / "P00" / f"{EV_VIDEO}.mp4"
    src.parent.mkdir(parents=True)
    segs = [("white", 2), ("black", 2), ("white", 2), ("black", 3), ("white", 3)]
    cmd = ["ffmpeg", "-v", "error"]
    for color, d in segs:
        cmd += ["-f", "lavfi", "-i", f"color={color}:s=64x64:r=30:d={d}"]
    cmd += ["-filter_complex", "".join(f"[{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1",
            "-c:v", "libx264", "-qp", "0", str(src)]
    subprocess.run(cmd, check=True)

    def q(text, t_ask, clips):
        return {"q": text, "video": EV_VIDEO, "form": "test", "answer": "a", "streaming": True, "t_ask": t_ask,
                "evidence": {"events": [], "clips": [{"event_id": f"EV{i}", "clip": c} for i, c in enumerate(clips)]},
                "gemini_mcq": {"options": ["a", "b", "c", "d"], "answer_index": 0}}
    bank_dir = tmp / "bank"
    bank_dir.mkdir()
    (bank_dir / f"{EV_VIDEO}_questions.json").write_text(json.dumps({"questions": [
        q("first?", 11.0, [[6.3, 7.9], [2.2, 3.7]]), q("second?", 9.0, [[8.3, 10.5]])]}))
    cfg = json.loads(json.dumps(CFG))
    cfg["media"].update(mode="evidence", max_side=64)
    cfg["paths"]["video_dir"] = str(tmp / "videos")
    cfg["questions"]["glob"] = "*_questions.json"
    cfg["model"]["backend"] = "fake"
    cfg["resolved"] = {"run_dir": str(tmp), "code_dir": str(Path(__file__).parents[1]),
                       "questions_dir": str(bank_dir), "media_dir": str(tmp / "media")}
    (tmp / "reproducibility_config.json").write_text(json.dumps(cfg))
    return tmp, cfg


def _n_frames(path) -> int:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def test_evidence_file_holds_only_the_clips():
    if not shutil.which("ffprobe"):
        print("  (skipped: no ffprobe on PATH -- module load ffmpeg/7.1)")
        return
    from .prep_media import has_audio, prep_video, probe_duration
    from .run_eval import run_video
    with tempfile.TemporaryDirectory() as d:
        run, cfg = make_evidence_run(Path(d))
        rows = load_questions(cfg["resolved"]["questions_dir"], cfg["questions"]["glob"])
        groups = group_questions(rows, "evidence")
        assert sorted(groups) == sorted(r["qid"] for r in rows) and all(len(g) == 1 for g in groups.values())

        files = prep_video(cfg, EV_VIDEO, workers=2)
        fps = cfg["media"]["encode_fps"]
        want = {"first?": [[2.2, 3.7], [6.3, 7.9]], "second?": [[8.3, 9.0]]}
        for r in rows:
            p = media_path(run / "media", EV_VIDEO, r["qid"], "evidence")
            spans = files[p.name]["clips"]
            total = sum(b - a for a, b in spans)
            dur, n = probe_duration(p), _n_frames(p)
            assert spans == want[r["question"]], spans  # sorted, and cut at t_ask
            assert abs(dur - total) <= 0.3, (p.name, dur, total)
            assert abs(n - total * fps) <= 1, (p.name, n, total * fps)
            assert _max_luma(p) < 40, _max_luma(p)  # all black: no frame from outside the clips
            assert not has_audio(p)
            print(f"  {r['question']} clips={spans} -> {dur:.2f}s, {n} frames, max luma {_max_luma(p)}")

        # the model is handed exactly that question's evidence file, with the evidence instruction
        seen = []
        install_fake_backend(seen)
        fake = sys.modules["epic_eval.backends.fake"]
        fake.ask = lambda client, mcfg, ctx, prompt, n: {"pred_letter": "A", "raw_text": "", "usage": {},
                                                         "cache_used": False}
        assert run_video(run, EV_VIDEO) == (2, 0)
        assert sorted(p.name for p, _, _ in seen) == sorted(f"evidence_{r['qid']}.mp4" for r in rows)
        assert all(si == system_instruction(True, True, "evidence") and n == 1 for _, si, n in seen)


def test_evidence_mode_instruction_and_media_folder():
    si = system_instruction(True, True, "evidence").lower()
    for word in ("cutoff", "t_ask", "ends at", "moment the question", "whole recording", "after", "omitted"):
        assert word not in si, word
    assert "excerpts" in si and "chronological" in si
    ev = json.loads(json.dumps(CFG))
    ev["media"]["mode"] = "evidence"
    assert media_hash(ev) != media_hash(CFG)  # evidence files never land in the cutoff media folder


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
