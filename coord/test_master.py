"""master.desired: qa tasks only where the bank has questions and the dataset lacks answers; they wait on a listed build.
Paid-API arms: needs_env, pool max only, the pilot gate, answer arms waiting on their memory arm's build.

    python coord/test_master.py      (or pytest coord/test_master.py)
"""
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
    monkeypatch.setattr(M, "REPORTED_BANKS", [])   # builds: the build order only
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
    """Every baseline on the board has build + qa of every bank per arm, qa on the build's pools; arm00 (aarch64)
    runs MERIT only."""
    ex = {(b, arm, stage, bank): pools for b, arm, stage, pools, bank in M.EXPERIMENTS}
    assert M.BANKS == M.REPORTED_BANKS == ["batch10_agree2", "t2_agree2", "t3_agree2", "t4_agree2"]
    for b in ("merit", "magic", "worldmm"):
        for arm in ("m8", "m27"):
            assert ex[b, arm, "build", None] == M.POOLS[b, arm]
            for bank in M.BANKS:
                assert ex[b, arm, "qa", bank] == M.POOLS[b, arm]
            assert ("arm00" in M.POOLS[b, arm]) == (b == "merit" and arm == "m27")
    # the plain video model: qa only (no build line), every bank; the 8B on the m8 pools, the 27B on the m27 pools
    # without arm00, its unsized N = 768 only where an H200 is; no key
    direct = {(arm, stage, bank) for b, arm, stage, bank in ex if b == "direct"}
    assert direct == {(arm, "qa", bank) for arm in M.DIRECT for bank in M.BANKS} and len(direct) == 24
    for arm, pools in M.DIRECT.items():
        assert pools == (M.M8 if arm.startswith("qwen3vl-8b") else ["nexus", "vatsal"] if arm == "qwen38-27b-f768" else M.M27)
        assert all(ex["direct", arm, "qa", bank] == pools for bank in M.BANKS) and not M.needs("direct", arm, "qa")


def test_build_waits_on_base(monkeypatch, tmp_path):
    """MAGIC's build waits for MERIT's build of the same video and arm while that is listed (it reuses its captions)."""
    order = tmp_path / "order.txt"
    order.write_text("P01-20240101-000001\nP01-20240101-000002\n")
    monkeypatch.setattr(M, "BUILD_ORDER", str(order))
    monkeypatch.setattr(M, "EXPERIMENTS", [("merit", "m8", "build", ["p"], None), ("magic", "m8", "build", ["p"], None)])
    monkeypatch.setattr(M, "REPORTED_BANKS", [])
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


B10 = ["P05-20240425-201444", "P01-20240101-000010"]                      # stand-ins for the two banks' videos
T2 = ["P01-20240101-000020", "P01-20240101-000021"]
T3 = ["P01-20240101-000040"]
T4 = ["P01-20240101-000050"]
REST = ["P01-20240101-000030"]


def board(monkeypatch, tmp_path, experiments=None):
    """The real EXPERIMENTS (or a given list) over a small world: 2 batch10 videos, 2 tranche-2 videos, 1 tranche-3
    video, 1 other."""
    order = tmp_path / "order.txt"
    order.write_text("\n".join(REST[:1] + T2 + T3 + T4 + REST[1:]) + "\n")   # as build_order.txt: tranches 2, 3, 4 in it, batch10 not
    monkeypatch.setattr(M, "BUILD_ORDER", str(order))
    monkeypatch.setattr(M, "bank_videos", lambda bank: {"batch10_agree2": B10, "t2_agree2": T2, "t3_agree2": T3, "t4_agree2": T4}[bank])
    if experiments is not None:
        monkeypatch.setattr(M, "EXPERIMENTS", experiments)


def test_needs_env_and_pools(monkeypatch, tmp_path):
    """Every task of a paid-API arm names its key and is for pool max only, and --only-pools never widens that."""
    board(monkeypatch, tmp_path)
    tasks = M.desired(set(), set())
    assert {(t["baseline"], t["arm"]) for t in tasks if t.get("needs_env")} == {k[:2] for k in M.NEEDS_ENV}
    for t in tasks:
        need = M.needs(t["baseline"], t["arm"], t["stage"])
        assert t.get("needs_env") == need
        if need:
            assert t["pools"] == ["max"]
            if (t["baseline"], t["arm"]) not in {(b, arm) for b, arm, *_ in M.PAID}:
                assert need == (["GEMINI_API_KEY"] if t["arm"].startswith("gemini") else ["OPENAI_API_KEY"])
    by = {(t["baseline"], t["arm"]): t for t in tasks}
    # MERIT's Qwen answer arms call no API: no needs_env, the GPU pools of m8 / m27
    assert by["merit", "g5m-q8"]["pools"] == M.M8 and by["merit", "g5m-q27"]["pools"] == M.POOLS["merit", "m27"]
    assert set(M.NEEDS_ENV) >= {("api", c) for c in M.API_CONFIGS} | {
        ("worldmm", "g5m"), ("worldmm", "g5m-q8"), ("worldmm", "g5m-g5"), ("merit", "g5m"), ("merit", "g5m-g5")}
    own = lambda t: t.get("needs_env") or (t["baseline"], t["stage"]) == ("hippomm", "build")   # keyless, but Max's
    M.restrict(tasks, ["a-test-pool"])
    assert all(t["pools"] == ([] if own(t) else ["a-test-pool"]) for t in tasks)
    M.restrict(tasks := M.desired(set(), set()), ["max", "a-test-pool"])
    assert all(t["pools"] == (["max"] if own(t) else ["max", "a-test-pool"]) for t in tasks)


def test_pilot_gate(monkeypatch, tmp_path):
    """While PILOT is non-empty, the only listed tasks of the reported arms are PILOT's; the rest are untouched."""
    board(monkeypatch, tmp_path)
    everything = M.desired(set(), set())
    assert M.PILOT == ["worldmm-g5m-build-P05-20240425-201444"]
    listed, withheld = M.gate(everything)
    assert [t["id"] for t in listed if M.reported(t)] == M.PILOT
    assert [t for t in listed if not M.reported(t)] == [t for t in everything if not M.reported(t)]
    assert withheld == sum(M.reported(t) for t in everything) - 1 > 0
    assert not any(d for t in listed if M.reported(t) for d in t["deps"])   # the pilot waits on nothing
    # MERIT's key-less answer arms are withheld too: their memory's build is, and they cannot build it themselves
    assert not any(t["arm"] in ("g5m-q8", "g5m-q27") for t in listed)
    monkeypatch.setattr(M, "PILOT", [])
    assert M.gate(everything) == (everything, 0)


def test_memory_arm(monkeypatch, tmp_path):
    """g5m is built for every video, batch10's first; an answer arm's qa waits on g5m's build, not its own arm's."""
    board(monkeypatch, tmp_path)
    v, w = B10
    have = {("worldmm", "g5m", w)}                                         # one memory already in the dataset
    files = {f"merit/g5m-g5/{v}/answers_batch10_agree2.json.gz"}           # one video already answered
    tasks = M.desired(have, files)
    by = {t["id"]: t for t in tasks}
    for b in ("worldmm", "merit"):
        built = [t["video"] for t in tasks if (t["baseline"], t["arm"], t["stage"]) == (b, "g5m", "build")]
        assert built == [x for x in B10 + T2 + T3 + T4 + REST if (b, "g5m", x) not in have]   # all videos, once, in that order
    g5m = lambda t: t["arm"].startswith("g5m")
    assert not any(t["stage"] == "build" and t["arm"] != "g5m" for t in tasks if g5m(t))   # answer arms never build
    assert by[f"worldmm-g5m-g5-qa-batch10_agree2-{v}"]["deps"] == [f"worldmm-g5m-build-{v}"]
    assert by[f"worldmm-g5m-q8-qa-batch10_agree2-{w}"]["deps"] == []       # that memory exists
    assert by[f"merit-g5m-q27-qa-t2_agree2-{T2[0]}"]["deps"] == [f"merit-g5m-build-{T2[0]}"]
    assert f"merit-g5m-g5-qa-batch10_agree2-{v}" not in by and f"merit-g5m-g5-qa-batch10_agree2-{w}" in by
    assert not any(t["bank"] == "t2_agree2" and t["video"] in B10 for t in tasks)
    # order: the API tasks, then the reported arms by video group (a batch10 answer before any tranche-2 build), then the rest
    kind = [0 if t["baseline"] == "api" else 1 if M.reported(t) else 2 for t in tasks]
    assert kind == sorted(kind) and [t["priority"] for t in tasks] == list(range(len(tasks)))
    rep = [t for t in tasks if g5m(t)]
    group = [0 if t["video"] in B10 else 1 if t["video"] in T2 else 2 if t["video"] in T3 else 3 if t["video"] in T4 else 4 for t in rep]
    assert group == sorted(group)
    assert all(by[d]["priority"] < t["priority"] for t in tasks for d in t["deps"])
    # a memory folder counts as built with its baseline's last memory file; an answer arm's folder never does
    done, _ = M.built_from({f"worldmm/g5m/{v}/runs.jsonl", f"worldmm/g5m/{v}/{v}_visual.pkl", f"merit/g5m/{v}/runs.jsonl",
                            f"merit/g5m/{v}/{v}_30sec_4_key.json", f"merit/g5m-q8/{v}/runs.jsonl",
                            f"merit/g5m-q8/{v}/{v}_memory_ref.json", f"merit/g5m-q8/{v}/answers_t2_agree2.json.gz"})
    assert done == {("worldmm", "g5m", v), ("merit", "g5m", v)}


def test_paid_rows(monkeypatch, tmp_path):
    """The other paid rows: batch10 only (builds too), pool max only, each stage's own keys, shared memories, after the
    g5m lines and before the local-model lines, and all of them behind the pilot gate."""
    board(monkeypatch, tmp_path)
    v, w = B10
    have = {("egagent", "gpt41", w)}                                       # one EGAgent memory already in the dataset
    files = {f"egor1/gemini25pro/{v}/answers_batch10_agree2.json.gz"}      # one Ego-R1 video already answered
    tasks = M.desired(have, files)
    by = {t["id"]: t for t in tasks}
    rows = {(b, arm) for b, arm, *_ in M.PAID}
    paid = [t for t in tasks if (t["baseline"], t["arm"]) in rows]
    assert rows == {("egor1", "gemini25pro"), ("egagent", "gpt41"), ("egagent", "gemini25pro"), ("egorag", "egogpt_gpt4o"),
                    ("hippomm", "q25vl7b-gpt4o"), ("magic", "oss120"), ("magic", "oss120-flash"), ("magic", "oss120-q27")}
    assert {(t["baseline"], t["arm"]) for t in paid} == rows
    # batch10 only: no other video is built or answered, no other bank; pool max only, with or without a key
    assert {t["video"] for t in paid} == set(B10) and {t["bank"] for t in paid} == {None, "batch10_agree2"}
    assert all(t["pools"] == ["max"] for t in paid)
    count = lambda b, arm, stage: sum((t["baseline"], t["arm"], t["stage"]) == (b, arm, stage) for t in paid)
    assert count("egor1", "gemini25pro", "build") == 2 and count("egor1", "gemini25pro", "qa") == 1   # v is answered
    assert count("egagent", "gpt41", "build") == 1 and count("egagent", "gemini25pro", "build") == 0  # w is built
    assert count("magic", "oss120", "build") == 2 and count("magic", "oss120-flash", "build") == 0
    need = lambda tid: by[tid].get("needs_env")
    assert need(f"egor1-gemini25pro-build-{v}") == ["GEMINI_API_KEY", "OPENAI_API_KEY"] == need(f"egor1-gemini25pro-qa-batch10_agree2-{w}")
    assert need(f"egagent-gpt41-build-{v}") == ["OPENAI_API_KEY"] == need(f"egagent-gpt41-qa-batch10_agree2-{v}")
    assert need(f"egagent-gemini25pro-qa-batch10_agree2-{v}") == ["GEMINI_API_KEY", "OPENAI_API_KEY"]
    assert need(f"egorag-egogpt_gpt4o-build-{v}") == ["OPENAI_API_KEY", "DEEPSEEK_API_KEY"]
    assert need(f"egorag-egogpt_gpt4o-qa-batch10_agree2-{v}") == ["OPENAI_API_KEY"]
    assert need(f"hippomm-q25vl7b-gpt4o-build-{v}") is None and "needs_env" not in by[f"hippomm-q25vl7b-gpt4o-build-{v}"]
    assert need(f"hippomm-q25vl7b-gpt4o-qa-batch10_agree2-{v}") == ["OPENAI_API_KEY"]
    assert all(need(f"magic-{arm}-{x}-{v}") == ["OPENROUTER_API_KEY"]
               for arm, x in (("oss120", "build"), ("oss120-flash", "qa-batch10_agree2"), ("oss120-q27", "qa-batch10_agree2")))
    # shared memories: the answer arm waits on the memory arm's build while it is listed, and says where the memory is
    assert by[f"egagent-gemini25pro-qa-batch10_agree2-{v}"]["deps"] == [f"egagent-gpt41-build-{v}"]
    assert by[f"egagent-gemini25pro-qa-batch10_agree2-{w}"]["deps"] == []          # that memory exists
    assert by[f"egagent-gemini25pro-qa-batch10_agree2-{v}"]["memory_arm"] == "gpt41"
    assert by[f"magic-oss120-q27-qa-batch10_agree2-{v}"]["deps"] == [f"magic-oss120-build-{v}"]
    assert by[f"magic-oss120-build-{v}"]["deps"] == []                             # no MERIT oss120 captions to wait for
    assert by[f"hippomm-q25vl7b-gpt4o-qa-batch10_agree2-{v}"]["deps"] == [f"hippomm-q25vl7b-gpt4o-build-{v}"]
    # order: api, the g5m arms, the paid rows, the local-model arms
    kind = [0 if t["baseline"] == "api" else 1 if t["arm"].startswith("g5m") else 2 if t in paid else 3 for t in tasks]
    assert kind == sorted(kind) and set(kind) == {0, 1, 2, 3}
    assert all(by[d]["priority"] < t["priority"] for t in paid for d in t["deps"])
    # the gate withholds every one of them, and --only-pools moves none of them (the keyless HippoMM build either)
    assert all(M.reported(t) for t in paid) and M.GATED == rows | {(b, arm) for b, arm, *_ in M.REPORTED}
    listed, withheld = M.gate(tasks)
    assert not [t for t in listed if t in paid] and withheld == sum(M.reported(t) for t in tasks) - 1
    M.restrict(tasks, ["a-test-pool"])
    assert all(t["pools"] == [] for t in paid)
    monkeypatch.setattr(M, "PILOT", [])
    assert len(M.gate(tasks)[0]) == len(tasks)
    # built: each baseline's last memory file, next to runs.jsonl
    done, _ = M.built_from({f"{b}/{arm}/{v}/{f}" for b, arm, last in (
        ("egor1", "gemini25pro", f"{v}_1day.json"), ("egagent", "gpt41", f"{v}_graph.json"),
        ("egorag", "egogpt_gpt4o", f"{v}_l3_day.json"), ("hippomm", "q25vl7b-gpt4o", f"{v}_memory.json"),
        ("magic", "oss120", "graph.pkl")) for f in ("runs.jsonl", last)} | {f"egorag/egogpt_gpt4o/{w}/runs.jsonl",
                                                                           f"egorag/egogpt_gpt4o/{w}/{w}_30sec.json"})
    assert done == {("egor1", "gemini25pro", v), ("egagent", "gpt41", v), ("egorag", "egogpt_gpt4o", v),
                    ("hippomm", "q25vl7b-gpt4o", v), ("magic", "oss120", v)}


def desired_before(have, files):
    """master.desired as it was before the reported arms (origin/main 4c57eb0), for the comparison below, except that
    builds now cover every video: each bank's (batch10's, missing from build_order.txt, first), then the build order."""
    order = list(dict.fromkeys([v for bank in M.BANKS for v in M.bank_videos(bank)]
                               + [v.strip() for v in open(M.BUILD_ORDER) if v.strip()]))
    tasks = []
    for b, arm, stage, pools, bank in M.EXPERIMENTS:
        if stage == "build":
            for v in order:
                if (b, arm, v) not in have:
                    base = f"{M.BUILD_ON.get(b)}-{arm}-build-{v}"
                    tasks.append(dict(id=f"{b}-{arm}-build-{v}", baseline=b, arm=arm, stage="build", video=v, bank=None,
                                      pools=pools, deps=[base] if any(t["id"] == base for t in tasks) else []))
            continue
        for v in M.bank_videos(bank):
            if f"{b}/{arm}/{v}/answers_{bank}.json.gz" in files:
                continue
            build = f"{b}-{arm}-build-{v}"
            deps = [build] if any(t["id"] == build for t in tasks) else []
            tasks.append(dict(id=f"{b}-{arm}-{stage}-{bank}-{v}", baseline=b, arm=arm, stage=stage, video=v, bank=bank,
                              pools=pools, deps=deps))
    for i, t in enumerate(tasks):
        t["priority"] = i
    return tasks


def test_existing_tasks_unchanged(monkeypatch, tmp_path):
    """Without the reported arms the board is main's, field for field (api tasks gain needs_env only); with them,
    the merit / magic / worldmm / api tasks are the same ones in the same order."""
    have = {("merit", "m8", T2[0]), ("worldmm", "m27", REST[0])}
    files = {f"magic/m8/{T2[1]}/answers_t2_agree2.json.gz", f"api/gpt-5-f1024/{B10[0]}/answers_batch10_agree2.json.gz"}
    full = list(M.EXPERIMENTS)
    board(monkeypatch, tmp_path, [e for e in full if e not in M.REPORTED + M.PAID])
    assert len(M.EXPERIMENTS) == len(full) - len(M.REPORTED) - len(M.PAID) == 57   # 6 arms x (build + 4 banks) + 3 api + 6 direct x 4 banks
    old = desired_before(have, files)
    strip = lambda ts: [{k: v for k, v in t.items() if k != "needs_env"} for t in ts]
    new = M.desired(have, files)
    assert {t["baseline"] for t in old} == {"api", "direct", "merit", "magic", "worldmm"} and strip(new) == old
    assert [t.get("needs_env") for t in new] == [M.needs(t["baseline"], t["arm"], t["stage"]) for t in old]
    assert M.gate(new) == (new, 0)                                          # the gate holds none of them back
    # direct: a task per arm, bank and video unless answered, waiting on nothing, after api and before the memory arms
    direct = [t for t in M.desired(have, files | {f"direct/qwen3vl-8b-f32/{T2[0]}/answers_t2_agree2.json.gz"}) if t["baseline"] == "direct"]
    assert len(direct) == 6 * len(B10 + T2 + T3 + T4) - 1 and not any(t["deps"] or t.get("needs_env") for t in direct)
    assert f"direct-qwen3vl-8b-f32-qa-t2_agree2-{T2[0]}" not in {t["id"] for t in direct}
    kind = [0 if t["baseline"] == "api" else 1 if t["baseline"] == "direct" else 2 for t in new]
    assert kind == sorted(kind)
    monkeypatch.setattr(M, "EXPERIMENTS", full)
    noprio = lambda ts: [{k: v for k, v in t.items() if k != "priority"} for t in ts]
    assert noprio(strip(t for t in M.desired(have, files) if not M.reported(t))) == noprio(old)
    listed, _ = M.gate(M.desired(have, files))                             # as published now: main's tasks + the pilot
    assert noprio(strip(t for t in listed if t["id"] not in M.PILOT)) == noprio(old)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
