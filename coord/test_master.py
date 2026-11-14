"""master.desired: qa tasks only where the bank has questions and the dataset lacks answers; they wait on a listed build."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import master as M  # noqa: E402


def test_qa_tasks(tmp_path, monkeypatch):
    d = tmp_path / "outputs/v3/banks/tb"
    d.mkdir(parents=True)
    for v, n in [("P01-20240101-000001", 2), ("P01-20240101-000002", 3), ("P01-20240101-000003", 0), ("P01-20240101-000004", 1)]:
        (d / f"{v}_questions.json").write_text(json.dumps({"questions": [{"key": str(i)} for i in range(n)]}))
    (tmp_path / "v3").mkdir()
    (tmp_path / "v3/export_bank.py").write_text("")   # the real one regenerates the files above
    order = tmp_path / "order.txt"
    order.write_text("P01-20240101-000002\n")
    monkeypatch.setenv("EPIC_MORPH_DIR", str(tmp_path))
    monkeypatch.setattr(M, "BUILD_ORDER", str(order))
    monkeypatch.setattr(M, "EXPERIMENTS", [("merit", "m8", "build", ["p"], None), ("merit", "m8", "qa", ["p"], "tb")])
    have = {("merit", "m8", "P01-20240101-000001"), ("merit", "m8", "P01-20240101-000004")}
    files = {"merit/m8/P01-20240101-000004/answers_tb.json.gz"}
    tasks = {t["id"]: t for t in M.desired(have, files)}
    # 000001: built, unanswered -> qa, no wait; 000002: build listed -> qa waits on it; 000003: no questions;
    # 000004: answered already
    assert sorted(tasks) == ["merit-m8-build-P01-20240101-000002", "merit-m8-qa-tb-P01-20240101-000001",
                             "merit-m8-qa-tb-P01-20240101-000002"]
    assert tasks["merit-m8-qa-tb-P01-20240101-000001"]["deps"] == []
    assert tasks["merit-m8-qa-tb-P01-20240101-000002"]["deps"] == ["merit-m8-build-P01-20240101-000002"]
    assert tasks["merit-m8-qa-tb-P01-20240101-000002"]["bank"] == "tb"
    assert [t["priority"] for t in M.desired(have, files)] == [0, 1, 2]
    monkeypatch.delenv("EPIC_MORPH_DIR")
    try:   # without the bank, publishing would drop the qa tasks from the board: stop instead
        M.desired(have, files)
        assert False, "desired() ran without EPIC_MORPH_DIR"
    except SystemExit:
        pass


def test_built_from():
    """A memory counts as built only with runs.jsonl and its baseline's last memory file (a prep-only folder does not)."""
    v = "P01-20240101-000001"
    files = {f"merit/m8/{v}/runs.jsonl", f"merit/m8/{v}/{v}_30sec_4_key.json",
             f"magic/m8/{v}/runs.jsonl", f"magic/m8/{v}/graph.pkl",
             f"magic/m27/{v}/runs.jsonl",                                    # prep or half a build: not built
             f"worldmm/m8/{v}/runs.jsonl", f"worldmm/m8/{v}/{v}_visual.pkl",
             f"other/m8/{v}/runs.jsonl"}
    have, _ = M.built_from(files)
    assert have == {("merit", "m8", v), ("magic", "m8", v), ("worldmm", "m8", v)}


def test_experiments():
    """Every baseline on the board has build + qa per arm, qa on the build's pools; arm00 (aarch64) runs MERIT only."""
    ex = {(b, arm, stage): (pools, bank) for b, arm, stage, pools, bank in M.EXPERIMENTS}
    for b in M.MEMORY_DONE:
        for arm in ("m8", "m27"):
            assert ex[b, arm, "build"] == (M.POOLS[b, arm], None)
            assert ex[b, arm, "qa"] == (M.POOLS[b, arm], "t2_agree2")
            assert ("arm00" in M.POOLS[b, arm]) == (b == "merit" and arm == "m27")


def test_build_waits_on_base(monkeypatch, tmp_path):
    """MAGIC's build waits for MERIT's build of the same video and arm while that is listed (it reuses its captions)."""
    order = tmp_path / "order.txt"
    order.write_text("P01-20240101-000001\nP01-20240101-000002\n")
    monkeypatch.setattr(M, "BUILD_ORDER", str(order))
    monkeypatch.setattr(M, "EXPERIMENTS", [("merit", "m8", "build", ["p"], None), ("magic", "m8", "build", ["p"], None)])
    tasks = {t["id"]: t for t in M.desired({("merit", "m8", "P01-20240101-000001")}, set())}
    assert tasks["magic-m8-build-P01-20240101-000001"]["deps"] == []   # MERIT's memory exists
    assert tasks["magic-m8-build-P01-20240101-000002"]["deps"] == ["merit-m8-build-P01-20240101-000002"]


def test_relay_claims_hold_but_are_not_done(tmp_path, monkeypatch):
    """worker --outbox: a finished task waiting for the relay is held from every pool, ranks below done and above
    running, and the upload leaves the folder (with .upload.json) in the outbox instead of the dataset."""
    import jobs
    live = dict(status="running", lease_until=9e12)
    held = jobs.taken({"a": {"claims": {"x": {"status": "relay"}}}, "b": {"claims": {"x": live, "y": live}},
                       "c": {"claims": {"x": {"status": "failed"}}}}, 1)
    assert {k: (p, c["status"]) for k, (p, c) in held.items()} == {"x": ("a", "relay"), "y": ("b", "running")}
    held = jobs.taken({"b": {"claims": {"x": live}}, "a": {"claims": {"x": {"status": "done"}}},
                       "z": {"claims": {"x": {"status": "relay"}}}}, 1)
    assert held["x"][0] == "a"
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baselines", "common"))
    import hf_results
    v, src = "P01-20240203-123350", tmp_path / "res"
    src.mkdir()
    for f in (f"{v}_graph.pkl", "answers_t2.json.gz", "runs.jsonl", "x.partial"):
        (src / f).write_text(f)
    monkeypatch.setattr(hf_results, "OUTBOX", str(tmp_path / "outbox"))
    monkeypatch.setenv("MERIT_OUTBOX", "magic-m27-qa-t2-" + v)
    monkeypatch.setattr(sys, "argv", ["hf_results.py", "upload", "--baseline", "magic", "--memory", "{video}_graph.pkl",
                                      "--arm", "m27", "--video", v, "--dir", str(src)])
    hf_results.main()
    out = tmp_path / "outbox" / ("magic-m27-qa-t2-" + v)
    assert sorted(os.listdir(out)) == sorted([".upload.json", f"{v}_graph.pkl", "answers_t2.json.gz", "runs.jsonl"])
    assert json.load(open(out / ".upload.json")) == dict(baseline="magic", arm="m27", video=v, memory=["{video}_graph.pkl"])
