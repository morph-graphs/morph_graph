"""run_qa: questions with 2 or 3 options (the M2 probes) get a prompt naming only their letters, and a reply naming a
letter they lack is kept as a wrong answer; a four-option question is asked and parsed exactly as before.

    python baselines/egor1/epic/test_run_qa.py      (or pytest)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_qa as R  # noqa: E402


def test_four_options_as_before():
    four = {"A": "sink", "B": "table", "C": "hob", "D": "fridge"}
    assert R.answer_prompt("ABCD") is R.ANSWER_PROMPT and "in A|B|C|D format" in R.ANSWER_PROMPT
    assert [L for L in "ABCDEZ" if R.kept(L, four)] == list("ABCD")   # what `resp in choices` kept before
    assert not R.kept("", four)


def test_fewer_options():
    three, two = {"A": "x", "B": "y", "C": "z"}, {"A": "x", "B": "y"}
    p3, p2 = R.answer_prompt("ABC"), R.answer_prompt("AB")
    assert "in A|B|C format" in p3 and "A|B|C|D" not in p3 and "in A|B format" in p2
    assert p3.replace("A|B|C format", "A|B|C|D format") == R.ANSWER_PROMPT   # nothing else changes
    assert R.kept("B", three) and R.kept("D", three) and R.kept("C", two) and R.kept("D", two)   # a missing letter: kept, wrong
    assert not R.kept("", three) and not R.kept("E", three)   # no letter, or past D: not kept, as for four


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
