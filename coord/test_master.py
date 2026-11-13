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
