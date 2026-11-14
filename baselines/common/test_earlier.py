"""A question answered under one bank is not asked again under another bank of the same video; an edited one is.

    python -m pytest baselines/common/test_earlier.py
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "merit", "epic"), os.path.join(HERE, "..", "magic", "epic")]
import collect_answers  # noqa: E402
import hf_results  # noqa: E402
import run_qa  # noqa: E402  (MAGIC's)
from earlier import reuse  # noqa: E402


def gz(path, rows):
    with gzip.open(path, "wt") as f:
        json.dump(rows, f)
    return str(path)


def merit_q(key, question, t="00100000", gold="A"):
    return {"ID": key, "key": key, "question": question, "choice_a": "sink", "choice_b": "table",
            "answer": gold, "query_time": {"date": "DAY1", "time": t}}


def merit_ans(q, response="A"):   # MERIT's answer row: choices as a dict, its own positional or key ID
    return {"ID": q["ID"], "question": q["question"], "choices": {"A": q["choice_a"], "B": q["choice_b"]},
            "answer": q["answer"], "query_time": q["query_time"], "response": response}


def test_merit_reuses_other_bank(tmp_path, monkeypatch, capsys):
    old = [merit_q("k1", "Where is the bowl?"), merit_q("k2", "Where is the cup?"), merit_q("k3", "Where is the pan?"),
           merit_q("k4", "Where is the lid?")]
    prev = gz(tmp_path / "answers_A.json.gz", [merit_ans(q) for q in old])
    new = [merit_q("k1", "Where is the bowl?"),                    # unchanged -> reused
           merit_q("k2", "Where is the red cup?"),                 # edited wording -> asked again
           merit_q("k3", "Where is the pan?", t="00200000"),       # asked later in the video -> asked again
           merit_q("k4", "Where is the lid?", gold="B"),           # other gold letter -> asked again
           merit_q("k5", "Where is the knife?")]                   # new -> asked
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps(new))
    out = tmp_path / "answers_B.json.gz"
    monkeypatch.setattr(sys, "argv", ["collect_answers.py", "--qa_file", str(qa), "--runs", str(tmp_path / "none/*.json"),
                                      "--prev", str(out), prev, "--out", str(out)])
    collect_answers.main()
    assert capsys.readouterr().out.strip().splitlines()[-1] == "k2 k3 k4 k5"
    assert [r["key"] for r in json.load(gzip.open(out, "rt"))] == ["k1"]


def magic_q(key, qhash, t="00100000", gold="A"):
    return {"key": key, "ID": "1", "qhash": qhash, "answer": gold, "query_time": {"date": "DAY1", "time": t}}


def test_magic_reuses_other_bank(tmp_path):
    rows = [magic_q("k1", "h1"), magic_q("k2", "h2x"), magic_q("k3", "h3", t="00200000"), magic_q("k4", "h4")]
    ans = lambda k, h, mh="M": {"key": k, "ID": "9", "qhash": h, "answer": "A", "query_time": run_qa.until(magic_q(k, h)),
                                "response": "A", "version": run_qa.VERSION, "memory_hash": mh}
    prev = gz(tmp_path / "answers_A.json.gz", [ans("k1", "h1"), ans("k2", "h2"), ans("k3", "h3"), ans("k4", "h4", mh="other")])
    got = run_qa.done_rows(rows, [str(tmp_path / "answers_B.json.gz"), prev], "M")
    # k2 edited, k3 asked later, k4 answered from another memory: all asked again
    assert sorted(got) == ["k1"] and got["k1"]["ID"] == "1"


def test_reuse_first_valid_wins(tmp_path):
    a = gz(tmp_path / "a.json.gz", [{"key": "k", "response": "Error", "x": 1}, {"key": "j", "response": "B", "x": 2}])
    b = gz(tmp_path / "b.json.gz", [{"key": "k", "response": "A", "x": 1}, {"key": "j", "response": "C", "x": 2}])
    got = reuse({"k": 1, "j": 2}, [a, b, str(tmp_path / "missing.json.gz")], lambda r: r["x"])
    assert got["k"]["response"] == "A" and got["j"]["response"] == "B"


def test_earlier_answers_lists_banks_and_variant(tmp_path, monkeypatch):
    epic = tmp_path / "epic"
    (epic / "v3").mkdir(parents=True)
    (epic / "v3" / "export_bank.py").write_text('BANKS = {"t2_agree2": 0, "t3_agree2": 0}\n')
    d = tmp_path / "out"
    d.mkdir()
    for n in ("answers_t2_agree2.json.gz", "answers_t2_agree2_ctx.json.gz", "answers_t3_agree2.json.gz", "answers_x.json.gz"):
        (d / n).write_text("[]")

    class Offline:
        def list_repo_tree(self, *a, **k):
            raise OSError("offline")
    monkeypatch.setattr(hf_results, "HfApi", Offline)
    a = type("A", (), dict(epic=str(epic), dir=str(d), suffix="", video="V"))
    assert [os.path.basename(p) for p in hf_results.earlier_answers("merit/m8/V", a)] == \
        ["answers_t2_agree2.json.gz", "answers_t3_agree2.json.gz"]   # not the _ctx variant, not an unknown bank
    a.suffix = "_ctx"
    assert [os.path.basename(p) for p in hf_results.earlier_answers("merit/m8/V", a)] == ["answers_t2_agree2_ctx.json.gz"]


def test_file_hash_matches_git(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello\n")
    assert hf_results.file_hash(str(p), git=True) == "ce013625030ba8dba906f756967f9e9ca394464a"   # git hash-object
