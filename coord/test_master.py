"""master.desired: qa tasks only where the bank has questions and the dataset lacks answers, or holds stale ones; they
wait on a listed build.
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
             f"m3agent/m3/{v}/runs.jsonl", f"m3agent/m3/{v}/{v}_m3graph.pkl", f"m3agent/m3/{v}/{v}_memories.json",
             f"rekv/m8/{v}/runs.jsonl", f"rekv/m8/{v}/memory.json",          # ReKV stores no memory: nothing to build
             f"other/m8/{v}/runs.jsonl"}
    have, _ = M.built_from(files)
    assert have == {("merit", "m8", v), ("magic", "m8", v), ("worldmm", "m8", v), ("m3agent", "m3", v)}


urgent = lambda t: (t["baseline"], t["arm"], t["stage"]) in {(b, arm, s) for b, arm, s, *_ in M.URGENT}   # right after the API tasks


def test_experiments():
    """Every baseline on the board has build + qa of every bank per arm, qa on the build's pools; arm00 (aarch64)
    runs MERIT only."""
    ex = {(b, arm, stage, bank): pools for b, arm, stage, pools, bank in BASE if not arm.endswith("-evclips")}
    # the evidence-clips ablation: 12 rows, qa of every bank, no build; paid rows keyed, Qwen3.5-Flash also nexus/zaratan
    ev = [(b, arm, stage, pools, bank) for b, arm, stage, pools, bank in BASE if arm.endswith("-evclips")]
    assert len(ev) == 12 * len(M.BANKS) and {e[2] for e in ev} == {"qa"}
    import worker
    assert all(worker.VALID["arm"].fullmatch(arm) for _, arm, *_ in ev)
    assert {(b, arm) for b, arm, *_ in ev if M.needs(b, arm, "qa")} == {("api", c) for b, c, _ in M.EVCLIPS if b == "api"}
    assert M.key_pools(M.needs("api", "qwen3.5-flash-f384-evclips", "qa")) + M.ALSO_POOLS["api", "qwen3.5-flash-f384-evclips"] == \
        M.KEY_POOLS + ["nexus", "zaratan"]
    # the api rows: three configs (gpt-5.5-f384 among them), every bank
    assert {(arm, stage, bank) for b, arm, stage, bank in ex if b == "api"} == {
        (c, "qa", bank) for bank in M.BANKS for c in ("gemini-3.1-flash-lite-high", "gemini-3.6-flash-high", "gpt-5.5-f384")}
    assert M.BANKS == M.REPORTED_BANKS == ["batch10_agree2_end", "t2_agree2_end", "t3_agree2_end", "t4_agree2_end", "t5_agree2_end",
                                           "rescue_agree2_end", "extra_agree2_end"]
    for b in ("merit", "magic", "worldmm"):
        for arm in ("m8", "m27"):
            assert ex[b, arm, "build", None] == (["pulkit"] if (b, arm) == ("magic", "m27") else   # Pulkit first
                                                 M.POOLS[b, arm] + (M.LEFTOVER_POOLS if b == "magic" else []))   # URGENT
            for bank in M.BANKS:
                assert ex[b, arm, "qa", bank] == M.POOLS[b, arm]
            assert ("arm00" in M.POOLS[b, arm]) == (b == "merit" and arm == "m27")
    # the plain video model: qa only (no build line), every bank; the 8B on the m8 pools, the 27B on the m27 pools
    # without arm00; Qwen3-VL-8B and Qwen3.8-27B at N = 768 only; no key
    assert {a for a in M.DIRECT if a.startswith(("qwen3vl-8b", "qwen38-27b"))} == {"qwen3vl-8b-f768", "qwen38-27b-f768"}
    direct = {(arm, stage, bank) for b, arm, stage, bank in ex if b == "direct"}
    assert direct == {(arm, "qa", bank) for arm in M.DIRECT for bank in M.BANKS} and len(direct) == len(M.DIRECT) * len(M.BANKS)
    for arm, pools in M.DIRECT.items():
        assert pools == (M.M8 if arm.startswith(("qwen3vl-8b", "qwen35-9b", "molmo2-8b")) else M.M27)
        assert all(ex["direct", arm, "qa", bank] == pools for bank in M.BANKS) and not M.needs("direct", arm, "qa")
    # ReKV: qa only, as the plain video model, every bank, on the m8 pools. M3-Agent: a build on the m8 pools, its qa
    # (a 32.8B control model) on the m27 pools; its paid arm m3api is not listed. Neither needs a key or is gated
    lines = lambda x: {(arm, stage, bank): pools for (b, arm, stage, bank), pools in ex.items() if b == x}
    assert lines("rekv") == {("m8", "qa", bank): M.M8 for bank in M.BANKS} and "rekv" not in M.MEMORY_DONE
    assert lines("m3agent") == {("m3", "build", None): M.M8} | {("m3", "qa", bank): M.M27 for bank in M.BANKS}
    assert not any(M.needs(b, arm, stage) or (b, arm) in M.GATED for b, arm, stage, _ in ex if b in ("rekv", "m3agent"))
    # VST: qa only, every bank, on the m8 pools; its thoughts are made in qa per cutoff, so no build and no key
    assert lines("vst") == {("vst7b", "qa", bank): M.M8 for bank in M.BANKS} and "vst" not in M.MEMORY_DONE
    assert not M.needs("vst", "vst7b", "qa") and ("vst", "vst7b") not in M.GATED and ("vst", "vst7b") not in M.MEMORY_ARM
    import worker   # a baseline or arm the worker does not know would make every task of it malformed there
    assert all(worker.VALID["baseline"].fullmatch(b) and worker.VALID["arm"].fullmatch(arm) for b, arm, *_ in ex)


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
T5 = ["P01-20240101-000060"]
REST = ["P01-20240101-000030"]
BANK_VIDEOS = {"batch10_agree2_end": B10, "t2_agree2_end": T2, "t3_agree2_end": T3, "t4_agree2_end": T4, "t5_agree2_end": T5,
               "rescue_agree2_end": T2[:1], "extra_agree2_end": T3}   # the last two: second banks of videos that have one


PRIORITY, QA_FROZEN, HELD, FIRST = set(M.PRIORITY), set(M.QA_FROZEN), set(M.HELD), list(M.FIRST)
# the board's lines with and without the stitched videos' lines: the tests of the normal videos' tasks use BASE
# (test_stitched_builds and test_stitched_qa check that adding the stitched lines changes none of those tasks); BUILDS:
# with the stitched builds, without their answers
FULL = list(M.EXPERIMENTS)
ALL = [e for e in FULL if e[4] != M.PROBES]   # (the M2 probe lines: test_probes)
api = lambda t: t["baseline"] == "api" and t["arm"] in M.API_CONFIGS   # the API tasks that lead (not the evidence-clips configs)
BASE = [e for e in ALL if e[4] not in ((M.STITCHED,), M.STITCHED_EOV)]
BUILDS = [e for e in ALL if e[4] != M.STITCHED_EOV]


def board(monkeypatch, tmp_path, experiments=None):
    """The real EXPERIMENTS (or a given list) over a small world: 2 batch10 videos, 2 tranche-2 videos, 1 video each of
    tranches 3, 4 and 5, 1 other; the rescue and extra banks ask videos of tranches 2 and 3 again."""
    order = tmp_path / "order.txt"
    order.write_text("\n".join(REST[:1] + T2 + T3 + T4 + T5 + REST[1:]) + "\n")   # as build_order.txt: the tranches in it, batch10 not
    monkeypatch.setattr(M, "BUILD_ORDER", str(order))
    monkeypatch.setattr(M, "bank_videos", lambda bank: BANK_VIDEOS[bank])
    monkeypatch.setattr(M, "PRIORITY", set())   # the tests below are of the whole list; test_paid_rows puts both back
    monkeypatch.setattr(M, "QA_FROZEN", set())
    monkeypatch.setattr(M, "HELD", set())
    monkeypatch.setattr(M, "FIRST", [])   # nor of what leads it: test_first_and_rank puts it back
    monkeypatch.setattr(M, "EXPERIMENTS", BASE if experiments is None else experiments)


ROUTER = {("merit", "g5m"), ("worldmm", "g5m"), ("worldmm", "g5m-q8"), ("magic", "oss120"), ("magic", "oss120-flash"),
          ("magic", "oss120-q27")}   # the arms an OpenRouter key alone runs: also for pool nexus


def test_needs_env_and_pools(monkeypatch, tmp_path):
    """Every task of a paid-API arm names its key and is for pool max only, and --only-pools never widens that."""
    board(monkeypatch, tmp_path)
    tasks = M.desired(set(), set())
    assert {(t["baseline"], t["arm"]) for t in tasks if t.get("needs_env")} == {k[:2] for k in M.NEEDS_ENV}
    for t in tasks:
        need = M.needs(t["baseline"], t["arm"], t["stage"])
        assert t.get("needs_env") == need
        if need:
            # every keyed task is Max's and Pulkit's (since 2026-11-14 no longer also our Nexus worker's for the
            # OpenRouter-only arms)
            router = (t["baseline"], t["arm"]) in ROUTER
            assert M.KEY_POOLS == ["max", "pulkit"] and t["pools"] == M.KEY_POOLS + M.ALSO_POOLS.get((t["baseline"], t["arm"]), []), t
            if (t["baseline"], t["arm"]) not in {(b, arm) for b, arm, *_ in M.PAID}:
                assert need == (["GEMINI_API_KEY"] if t["arm"].startswith("gemini") else ["OPENAI_API_KEY|OPENROUTER_API_KEY"]
                                if router else ["OPENAI_API_KEY"])
    by = {(t["baseline"], t["arm"]): t for t in tasks}
    # MERIT's Qwen answer arms call no API: no needs_env, the GPU pools of m8 / m27
    assert by["merit", "g5m-q8"]["pools"] == M.M8 and by["merit", "g5m-q27"]["pools"] == M.POOLS["merit", "m27"]
    assert set(M.NEEDS_ENV) >= {("api", c) for c in M.API_CONFIGS} | {
        ("worldmm", "g5m"), ("worldmm", "g5m-q8"), ("worldmm", "g5m-g51"), ("merit", "g5m"), ("merit", "g5m-g51")}
    own = lambda t: t.get("needs_env") or (t["baseline"], t["stage"]) == ("hippomm", "build")   # keyless, but Max's
    M.restrict(tasks, ["a-test-pool"])
    assert all(t["pools"] == ([] if own(t) else ["a-test-pool"]) for t in tasks)
    M.restrict(tasks := M.desired(set(), set()), ["max", "a-test-pool"])
    assert all(t["pools"] == (["max"] if own(t) else ["max", "a-test-pool"]) for t in tasks)
    M.restrict(tasks := M.desired(set(), set()), ["nexus"])   # the only keyed tasks that reach nexus: MAGIC's OpenRouter memory and Flash row
    assert M.ALSO_POOLS == {("magic", "oss120"): ["nexus"], ("magic", "oss120-flash"): ["nexus"]}
    assert {(t["baseline"], t["arm"]) for t in tasks if t.get("needs_env") and t["pools"] == ["nexus"]} == set(M.ALSO_POOLS)
    assert {t["baseline"] for t in tasks if t["arm"] == "g5m-g51"} == {"merit", "worldmm"}
    assert all(t["pools"] == [] and t["needs_env"] == ["OPENAI_API_KEY"] for t in tasks if t["arm"] == "g5m-g51")
    assert all(t["pools"] in ([], ["nexus"]) for t in tasks if t.get("needs_env"))


def test_pilot_gate(monkeypatch, tmp_path):
    """While PILOT is non-empty, the only listed tasks of the reported arms are PILOT's; the rest are untouched."""
    board(monkeypatch, tmp_path)
    everything = M.desired(set(), set())
    # as published: the gate is open and nothing is held. A HELD line (here WorldMM's g5m builds, as while they ran off
    # the board) is withheld, and the qa tasks that read those memories stay listed and keep waiting on them
    assert M.PILOT == [] and M.HELD == set() and M.gate(everything) == (everything, 0)
    monkeypatch.setattr(M, "HELD", {("worldmm", "g5m", "build")})
    listed, withheld = M.gate(everything)
    held = [t for t in everything if (t["baseline"], t["arm"], t["stage"]) in M.HELD]
    assert listed == [t for t in everything if t not in held] and withheld == len(held) > 0
    assert all(t["deps"] == [f"worldmm-g5m-build-{t['video']}"] for t in listed if (t["baseline"], t["arm"]) == ("worldmm", "g5m-q8"))
    monkeypatch.setattr(M, "HELD", set())
    monkeypatch.setattr(M, "PILOT", ["worldmm-g5m-build-P05-20240425-201444"])
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
    files = {f"merit/g5m-g51/{v}/answers_batch10_agree2_end.json.gz"}           # one video already answered
    tasks = M.desired(have, files)
    by = {t["id"]: t for t in tasks}
    for b in ("worldmm", "merit"):
        built = [t["video"] for t in tasks if (t["baseline"], t["arm"], t["stage"]) == (b, "g5m", "build")]
        assert built == [x for x in B10 + T2 + T3 + T4 + T5 + REST if (b, "g5m", x) not in have]   # all videos, once, in that order
    g5m = lambda t: t["arm"].startswith("g5m")
    assert not any(t["stage"] == "build" and t["arm"] != "g5m" for t in tasks if g5m(t))   # answer arms never build
    assert by[f"worldmm-g5m-g51-qa-batch10_agree2_end-{v}"]["deps"] == [f"worldmm-g5m-build-{v}"]
    assert by[f"worldmm-g5m-q8-qa-batch10_agree2_end-{w}"]["deps"] == []       # that memory exists
    assert by[f"merit-g5m-q27-qa-t2_agree2_end-{T2[0]}"]["deps"] == [f"merit-g5m-build-{T2[0]}"]
    assert f"merit-g5m-g51-qa-batch10_agree2_end-{v}" not in by and f"merit-g5m-g51-qa-batch10_agree2_end-{w}" in by
    assert not any(t["bank"] == "t2_agree2_end" and t["video"] in B10 for t in tasks)
    # order: the API tasks, then the reported arms by video group (a batch10 answer before any tranche-2 build), then the rest
    kind = [0 if api(t) else 1 if urgent(t) else 2 if M.reported(t) else 3 for t in tasks]
    assert kind == sorted(kind) and [t["priority"] for t in tasks] == list(range(len(tasks)))
    rep = [t for t in tasks if g5m(t)]
    group = [0 if t["video"] in B10 else 1 if t["video"] in T2 else 2 if t["video"] in T3 else 3 if t["video"] in T4
             else 4 if t["video"] in T5 else 5 for t in rep]
    assert group == sorted(group)
    assert all(by[d]["priority"] < t["priority"] for t in tasks for d in t["deps"])
    # a memory folder counts as built with its baseline's last memory file; an answer arm's folder never does
    done, _ = M.built_from({f"worldmm/g5m/{v}/runs.jsonl", f"worldmm/g5m/{v}/{v}_visual.pkl", f"merit/g5m/{v}/runs.jsonl",
                            f"merit/g5m/{v}/{v}_30sec_4_key.json", f"merit/g5m-q8/{v}/runs.jsonl",
                            f"merit/g5m-q8/{v}/{v}_memory_ref.json", f"merit/g5m-q8/{v}/answers_t2_agree2_end.json.gz"})
    assert done == {("worldmm", "g5m", v), ("merit", "g5m", v)}


def test_paid_rows(monkeypatch, tmp_path):
    """The other paid rows, with the arms Max's and Pulkit's gateway can run: every bank, builds for the banks' videos
    only, pool max / pulkit only, each stage's own keys, shared memories, after the g5m lines and before the
    local-model lines, by video group, and all of them behind the pilot gate."""
    board(monkeypatch, tmp_path)
    v, w = B10
    have = {("egagent", "gpt41", w)}                                       # one EGAgent memory already in the dataset
    files = {f"egor1/gemini36flash-gpt41/{v}/answers_batch10_agree2_end.json.gz"}   # one Ego-R1 video already answered
    tasks = M.desired(have, files)
    by = {t["id"]: t for t in tasks}
    rows = {(b, arm) for b, arm, *_ in M.PAID}
    paid = [t for t in tasks if (t["baseline"], t["arm"]) in rows]
    assert rows == {("egor1", "gemini36flash"), ("egor1", "gemini36flash-gpt41"), ("egagent", "gpt41"),
                    ("egagent", "gemini36flash"), ("egorag", "egogpt_gpt41"), ("hippomm", "q25vl7b-gpt4o"),
                    ("hippomm", "q25vl7b-gpt41"), ("magic", "oss120"), ("magic", "oss120-flash"), ("magic", "oss120-q27")}
    assert {(t["baseline"], t["arm"]) for t in paid} == rows
    # the lines: per row one build naming every bank, and a qa per answer arm and bank; the published arms nobody can
    # run (no GPT-4o, no Gemini 2.5 Pro) are off the board
    lines = {(b, arm, stage, bank) for b, arm, stage, _, bank in M.PAID}
    assert M.PAID_BANKS == M.BANKS and len(lines) == len(M.PAID) == 5 + 6 * len(M.BANKS)
    assert {(b, arm) for b, arm, stage, _ in lines if stage == "build"} == {
        ("egor1", "gemini36flash"), ("egagent", "gpt41"), ("egorag", "egogpt_gpt41"), ("hippomm", "q25vl7b-gpt4o"), ("magic", "oss120")}
    assert all(bank == tuple(M.BANKS) for *_, stage, bank in lines if stage == "build")
    assert {(b, arm, bank) for b, arm, stage, bank in lines if stage == "qa"} == {(b, arm, bank) for bank in M.BANKS for b, arm in (
        ("egor1", "gemini36flash-gpt41"), ("egagent", "gemini36flash"), ("egorag", "egogpt_gpt41"),   # no GPT-4.1 EGAgent row
        ("hippomm", "q25vl7b-gpt41"), ("magic", "oss120-flash"), ("magic", "oss120-q27"))}
    assert not any(arm in ("gemini25pro", "egogpt_gpt4o") for _, arm, *_ in M.EXPERIMENTS) and not any(
        "gemini25pro" in k or "egogpt_gpt4o" in k for k in list(M.NEEDS_ENV) + list(M.MEMORY_ARM))
    assert ("hippomm", "q25vl7b-gpt4o", "qa") not in {l[:3] for l in lines}
    # every bank is answered; builds only for the videos that have questions (REST has none), each once; pool max /
    # pulkit only, with or without a key
    asked = list(dict.fromkeys(x for vs in BANK_VIDEOS.values() for x in vs))
    assert {t["video"] for t in paid} == set(asked) and {t["bank"] for t in paid} == {None, *M.BANKS}
    assert all(t["pools"] == M.KEY_POOLS + M.ALSO_POOLS.get((t["baseline"], t["arm"]), []) for t in paid)
    count = lambda b, arm, stage: sum((t["baseline"], t["arm"], t["stage"]) == (b, arm, stage) for t in paid)
    pairs = sum(len(vs) for vs in BANK_VIDEOS.values())                    # (bank, video) with questions
    assert count("egor1", "gemini36flash", "build") == len(asked) and count("egor1", "gemini36flash-gpt41", "qa") == pairs - 1   # v is answered
    assert count("egagent", "gpt41", "build") == len(asked) - 1 and count("egagent", "gemini36flash", "build") == 0   # w is built
    assert count("egagent", "gpt41", "qa") == 0 and count("egagent", "gemini36flash", "qa") == pairs
    assert count("egorag", "egogpt_gpt41", "build") == len(asked) and count("egorag", "egogpt_gpt41", "qa") == pairs
    assert count("hippomm", "q25vl7b-gpt4o", "build") == len(asked) and count("hippomm", "q25vl7b-gpt4o", "qa") == 0
    assert count("hippomm", "q25vl7b-gpt41", "qa") == pairs and count("hippomm", "q25vl7b-gpt41", "build") == 0
    assert count("magic", "oss120", "build") == len(asked) and count("magic", "oss120-flash", "build") == 0
    assert count("magic", "oss120-flash", "qa") == count("magic", "oss120-q27", "qa") == pairs
    need = lambda tid: by[tid].get("needs_env")
    assert need(f"egor1-gemini36flash-build-{v}") == ["GEMINI_API_KEY", "OPENAI_API_KEY"] == need(f"egor1-gemini36flash-gpt41-qa-batch10_agree2_end-{w}")
    assert need(f"egagent-gpt41-build-{v}") == ["OPENAI_API_KEY"]
    assert need(f"egagent-gemini36flash-qa-batch10_agree2_end-{v}") == ["GEMINI_API_KEY"]
    assert need(f"egorag-egogpt_gpt41-build-{v}") == ["OPENAI_API_KEY", "DEEPSEEK_API_KEY"]
    assert need(f"egorag-egogpt_gpt41-qa-batch10_agree2_end-{v}") == ["OPENAI_API_KEY"]
    assert need(f"hippomm-q25vl7b-gpt4o-build-{v}") is None and "needs_env" not in by[f"hippomm-q25vl7b-gpt4o-build-{v}"]
    assert need(f"hippomm-q25vl7b-gpt41-qa-t2_agree2_end-{T2[0]}") == ["OPENAI_API_KEY"]
    assert all(need(f"magic-{arm}-{x}-{T3[0]}") == ["OPENROUTER_API_KEY"]
               for arm, x in (("oss120", "build"), ("oss120-flash", "qa-extra_agree2_end"), ("oss120-q27", "qa-t3_agree2_end")))
    # shared memories: the answer arm waits on the memory arm's build while it is listed, and says where the memory is
    assert by[f"egagent-gemini36flash-qa-batch10_agree2_end-{v}"]["deps"] == [f"egagent-gpt41-build-{v}"]
    assert by[f"egagent-gemini36flash-qa-batch10_agree2_end-{w}"]["deps"] == []          # that memory exists
    assert by[f"egagent-gemini36flash-qa-batch10_agree2_end-{v}"]["memory_arm"] == "gpt41"
    assert by[f"egor1-gemini36flash-gpt41-qa-t4_agree2_end-{T4[0]}"]["deps"] == [f"egor1-gemini36flash-build-{T4[0]}"]
    assert by[f"egor1-gemini36flash-gpt41-qa-t4_agree2_end-{T4[0]}"]["memory_arm"] == "gemini36flash"
    assert by[f"magic-oss120-q27-qa-batch10_agree2_end-{v}"]["deps"] == [f"magic-oss120-build-{v}"]
    assert by[f"magic-oss120-build-{v}"]["deps"] == []                             # no MERIT oss120 captions to wait for
    assert by[f"hippomm-q25vl7b-gpt41-qa-rescue_agree2_end-{T2[0]}"]["deps"] == [f"hippomm-q25vl7b-gpt4o-build-{T2[0]}"]
    assert by[f"hippomm-q25vl7b-gpt41-qa-rescue_agree2_end-{T2[0]}"]["memory_arm"] == "q25vl7b-gpt4o"
    assert "memory_arm" not in by[f"egorag-egogpt_gpt41-qa-batch10_agree2_end-{v}"]
    # order: api, the g5m arms, the paid rows, the local-model arms; the paid rows by video group (every row's batch10
    # memories and answers before any tranche-2 build), a second bank's questions with their video
    kind = [0 if api(t) else 1 if urgent(t) else 2 if t["arm"].startswith("g5m") else 3 if t in paid else 4 for t in tasks]
    assert kind == sorted(kind) and set(kind) == {0, 1, 2, 3, 4}
    group = [next(i for i, vs in enumerate(BANK_VIDEOS.values()) if t["video"] in vs) for t in paid]
    assert group == sorted(group) and set(group) == {0, 1, 2, 3, 4}
    assert all(by[d]["priority"] < t["priority"] for t in paid for d in t["deps"])
    # a non-empty PILOT withholds every one of them, and --only-pools moves none of them (the keyless HippoMM build either)
    assert all(M.reported(t) for t in paid) and M.GATED == rows | {(b, arm) for b, arm, *_ in M.REPORTED}
    monkeypatch.setattr(M, "HELD", set())
    monkeypatch.setattr(M, "PILOT", ["worldmm-g5m-build-P05-20240425-201444"])
    listed, withheld = M.gate(tasks)
    assert not [t for t in listed if t in paid] and withheld == sum(M.reported(t) for t in tasks) - 1
    M.restrict(tasks, ["a-test-pool"])
    assert all(t["pools"] == [] for t in paid)
    monkeypatch.setattr(M, "PILOT", [])
    assert len(M.gate(tasks)[0]) == len(tasks)
    monkeypatch.setattr(M, "PRIORITY", PRIORITY)   # as published: only the priority arms' tasks while there are any
    listed, withheld = M.gate(tasks)
    assert listed and withheld == len(tasks) - len(listed) > 0
    on_board = {(b, arm) for b, arm, *_ in M.EXPERIMENTS}   # (VST's line comes with its image)
    prio = lambda t: (t["baseline"], t["arm"]) in PRIORITY or (t["baseline"], t["arm"], t["stage"]) in PRIORITY
    arms = {k for k in PRIORITY if len(k) == 2}                           # (an entry with a stage: that stage only)
    assert {k for k in PRIORITY if len(k) == 3} == {("magic", "m27", "build")}
    assert {(t["baseline"], t["arm"]) for t in listed} == arms & on_board | {("magic", "m27")} and arms - on_board <= {("vst", "vst7b")}
    assert all(t["stage"] == "build" for t in listed if (t["baseline"], t["arm"]) == ("magic", "m27"))
    assert {t["stage"] for t in listed if (t["baseline"], t["arm"]) == ("merit", "m27")} == {"build", "qa"}   # MERIT 27B: the whole arm
    assert not [t for t in listed if (t["arm"] in ("m8", "m27", "oss120-q27", "q25vl7b-gpt41") or t["arm"].startswith(("qwen35", "molmo2")))
                and (t["baseline"], t["arm"], t["stage"]) != ("magic", "m27", "build") and (t["baseline"], t["arm"]) != ("merit", "m27")]
    assert M.gate([t for t in tasks if not prio(t)])[1] == 0   # the priority rows done: everything else is listed
    # answering frozen for four baselines: none of their qa tasks is listed, with or without PRIORITY; their builds are
    assert QA_FROZEN == set()                                          # as published: nothing is frozen
    QA4 = {"magic", "worldmm", "egagent", "m3agent"}                   # the rule, with the four it was made for
    monkeypatch.setattr(M, "QA_FROZEN", QA4)
    frozen = lambda t: t["stage"] == "qa" and t["baseline"] in QA4
    listed, withheld = M.gate(tasks)
    assert not any(map(frozen, listed)) and withheld == len(tasks) - len(listed)
    assert {(t["baseline"], t["arm"]) for t in listed if t["baseline"] in QA4} == {
        ("magic", "oss120"), ("magic", "m27"), ("worldmm", "g5m"), ("egagent", "gpt41"), ("m3agent", "m3")}
    assert all(t["stage"] == "build" for t in listed if t["baseline"] in QA4)
    monkeypatch.setattr(M, "PRIORITY", set())
    listed, withheld = M.gate(tasks)
    assert listed == [t for t in tasks if not frozen(t)] and withheld == sum(map(frozen, tasks)) > 0
    monkeypatch.setattr(M, "QA_FROZEN", set())
    monkeypatch.setattr(M, "PRIORITY", set())
    # nothing is held as published (the GPT-5 rows run on stand-ins); a held line of a priority arm is not listed, and
    # the other rows do not wait for it
    assert HELD == set()
    lines = {("api", "gpt-5.5-f384", "qa"), ("merit", "g5m-g51", "qa")}
    monkeypatch.setattr(M, "HELD", lines)
    held = lambda t: (t["baseline"], t["arm"], t["stage"]) in lines
    listed, withheld = M.gate(tasks)
    assert withheld == sum(map(held, tasks)) > 0 and not any(map(held, listed))
    monkeypatch.setattr(M, "PRIORITY", PRIORITY)
    rest = [t for t in tasks if not prio(t)]
    assert M.gate(rest + [t for t in tasks if held(t)])[0] == rest      # only held tasks left of the priority arms: the rest is listed
    monkeypatch.setattr(M, "PRIORITY", set())
    monkeypatch.setattr(M, "HELD", set())
    # built: each baseline's last memory file, next to runs.jsonl
    done, _ = M.built_from({f"{b}/{arm}/{v}/{f}" for b, arm, last in (
        ("egor1", "gemini36flash", f"{v}_1day.json"), ("egagent", "gpt41", f"{v}_graph.json"),
        ("egorag", "egogpt_gpt41", f"{v}_l3_day.json"), ("hippomm", "q25vl7b-gpt4o", f"{v}_memory.json"),
        ("magic", "oss120", "graph.pkl")) for f in ("runs.jsonl", last)} | {f"egorag/egogpt_gpt41/{w}/runs.jsonl",
                                                                           f"egorag/egogpt_gpt41/{w}/{w}_30sec.json"})
    assert done == {("egor1", "gemini36flash", v), ("egagent", "gpt41", v), ("egorag", "egogpt_gpt41", v),
                    ("hippomm", "q25vl7b-gpt4o", v), ("magic", "oss120", v)}


def test_first_and_rank(monkeypatch, tmp_path):
    """FIRST: MAGIC-Video's tasks lead the list, then MERIT's with Qwen3.8-27B, both ahead of the API tasks. rank(): each
    pool's queue, the free tasks it is listed for in the list's order; a task no pool can take now says why."""
    board(monkeypatch, tmp_path)
    monkeypatch.setattr(M, "FIRST", FIRST)
    assert FIRST == [("magic",), ("merit", "m27")] and ("merit", "m27") in PRIORITY and ("merit", "m8") not in PRIORITY
    v, w = B10
    have = {("merit", "m27", x) for x in B10 + T2 + T3 + T4 + T5 + REST if x != v}   # one MERIT 27B memory to build
    tasks = M.desired(have, set())
    by = {t["id"]: t for t in tasks}
    kind = lambda t: 0 if t["baseline"] == "magic" else 1 if (t["baseline"], t["arm"]) == ("merit", "m27") else 2 if api(t) else 3
    assert [kind(t) for t in tasks] == sorted(kind(t) for t in tasks) and {kind(t) for t in tasks} == {0, 1, 2, 3}
    assert [t["priority"] for t in tasks] == list(range(len(tasks))) and M.lead(tasks[-1]) == len(FIRST) + 7
    # inside MAGIC-Video's group: its memory builds (8B, 27B), the GPT-OSS-120B row, then its 27B and 8B answers
    magic = [("oss120", "") if t["arm"].startswith("oss120") else (t["arm"], t["stage"]) for t in tasks if t["baseline"] == "magic"]
    inside = [("m8", "build"), ("m27", "build"), ("oss120", ""), ("m27", "qa"), ("m8", "qa")]
    assert sorted(magic, key=inside.index) == magic and set(magic) == set(inside)
    # MERIT 27B: the memory still to build before the answers, and the answer that waits for it; a task never before its deps
    merit = [t for t in tasks if kind(t) == 1]
    assert merit[0]["id"] == f"merit-m27-build-{v}" and all(t["stage"] == "qa" for t in merit[1:])
    assert by[f"merit-m27-qa-batch10_agree2_end-{v}"]["deps"] == [f"merit-m27-build-{v}"] and not by[f"merit-m27-qa-batch10_agree2_end-{w}"]["deps"]
    assert all(by[d]["priority"] < t["priority"] for t in tasks for d in t["deps"])
    # as published: the priority rows only, in the same order (MAGIC-Video's non-priority answers are withheld with the rest)
    monkeypatch.setattr(M, "PRIORITY", PRIORITY)
    listed, _ = M.gate(tasks)
    assert [kind(t) for t in listed] == sorted(kind(t) for t in listed) and {kind(t) for t in listed} == {0, 1, 2, 3}
    assert {(t["arm"], t["stage"]) for t in listed if t["baseline"] == "magic"} == {
        ("m27", "build"), ("oss120", "build"), ("oss120-flash", "qa")}
    assert {t["bank"] for t in listed if (t["baseline"], t["arm"], t["stage"]) == ("merit", "m27", "qa")} == set(M.BANKS)
    # rank: a per-pool queue over the free tasks. a: held by a live claim; b: failed twice; c: a build, done, so the
    # answers that waited for it are free; e: failed once on pulkit, which does not retry it, max may
    live = {"status": "running", "lease_until": 2}
    a, b, c = listed[0]["id"], f"magic-oss120-flash-qa-batch10_agree2_end-{w}", f"magic-oss120-build-{v}"
    e = f"api-gpt-5.5-f384-qa-batch10_agree2_end-{v}"
    assert a == f"magic-m27-build-{v}" and by[a]["pools"] == ["pulkit"] and by[e]["pools"] == M.KEY_POOLS
    pools = {"pulkit": {"claims": {a: live, b: {"status": "failed"}, e: {"status": "failed"}}},
             "max": {"claims": {b: {"status": "failed"}, c: {"status": "done"}}}}
    before = json.dumps([{k: x for k, x in t.items()} for t in listed])
    free = M.rank(listed, pools, 1)
    assert json.dumps([{k: x for k, x in t.items() if k not in ("rank", "waiting")} for t in listed]) == before   # nothing else changes
    for p in ("pulkit", "max", "nexus", "wriva"):
        mine = [t["rank"][p] for t in listed if p in t["rank"]]
        assert mine == list(range(1, len(mine) + 1)) and len(mine) == free[p] > 0   # 1, 2, 3, ... down the list
    assert all(set(t["rank"]) <= set(t["pools"]) and ("waiting" in t) != bool(t["rank"]) for t in listed)
    assert (by[a]["waiting"], by[b]["waiting"], by[c]["waiting"]) == ("claimed", "failed", "claimed")
    # pulkit's first seven are MAGIC-Video's other 27B builds (its alone); c is done, so v's answers are free
    assert by[f"magic-oss120-build-{w}"]["rank"] == {"max": 1, "pulkit": 8, "nexus": 1}
    assert by[f"magic-oss120-flash-qa-batch10_agree2_end-{v}"]["rank"] == {"max": 2, "pulkit": 9, "nexus": 2}
    assert set(by[e]["rank"]) == {"max"} and "waiting" not in by[e]
    assert by[f"merit-m27-qa-batch10_agree2_end-{v}"]["waiting"] == "deps" and by[f"merit-m27-qa-batch10_agree2_end-{w}"]["rank"]
    waits = lambda t: t["id"] in (a, b, c) or any(d != c for d in t["deps"])
    assert all(("waiting" in t) == waits(t) for t in listed)
    # the same task stands differently for two pools: far down the list, and one pool's next
    far = next(t for t in listed if t["rank"].get("wriva") == 1)
    assert far["priority"] > 0 and far["rank"]["wriva"] < far["rank"]["max"] < far["priority"] + 1
    # the lapsed lease of a is free again; a pool file with no claims ranks like no file
    assert M.rank(listed, pools, 3)["pulkit"] == free["pulkit"] + 1 and "waiting" not in by[a] and by[a]["rank"] == {"pulkit": 1}
    M.restrict(listed, [])
    M.rank(listed, {})
    assert all(t["pools"] == [] and t["rank"] == {} and t["waiting"] == ("deps" if t["deps"] else "no pool") for t in listed)


def test_stitched_builds(monkeypatch, tmp_path):
    """The stitched videos: a build of each, for every line that builds, on that line's pools among STITCHED_POOLS and
    with its keys; no qa; the canary's (P05) ahead of everything, the others after the URGENT lines; MAGIC-Video's
    waits for MERIT's; a memory already in the dataset is not built again; not offered to the leftover pools. The normal
    videos' tasks are the same, in the same order, with or without them."""
    board(monkeypatch, tmp_path, BUILDS)
    stitched = lambda t: t["video"] in M.STITCHED_VIDEOS
    lines = {(b, arm) for b, arm, stage, *_ in BASE if stage == "build"}
    assert BUILDS[:len(BASE)] == BASE and len(BUILDS) == len(BASE) + len(lines)   # one stitched line per build line, appended
    have = {("merit", "m8", "P01-stitched")}
    tasks = M.desired(have, set())
    st = [t for t in tasks if stitched(t)]
    assert len(M.STITCHED_VIDEOS) == 9 and M.CANARY == "P05-stitched" and all(t["stage"] == "build" and t["bank"] is None for t in st)
    assert {(t["baseline"], t["arm"]) for t in st} == lines and len(st) == 9 * len(lines) - 1
    assert "merit-m8-build-P01-stitched" not in {t["id"] for t in st}                      # built already
    normal = {(t["baseline"], t["arm"]): t for t in tasks if t["stage"] == "build" and not stitched(t)}
    for t in st:
        n = normal[t["baseline"], t["arm"]]
        assert t["pools"] == [p for p in n["pools"] if p in M.STITCHED_POOLS] and t["pools"]
        assert (t.get("needs_env"), t.get("memory_arm")) == (n.get("needs_env"), n.get("memory_arm"))
        assert not {"zaratan", "wriva", "arm00", "vatsal"} & set(t["pools"])
    assert {p for t in st for p in t["pools"]} == {"max", "pulkit", "nexus"}
    by = {t["id"]: t for t in tasks}
    assert by["magic-m27-build-P02-stitched"]["deps"] == ["merit-m27-build-P02-stitched"]
    assert by["magic-m8-build-P01-stitched"]["deps"] == []                                  # MERIT's memory is there
    assert all(by[d]["priority"] < t["priority"] for t in tasks for d in t["deps"])
    # order (FIRST is empty here): the canary's builds lead, then the API tasks, the URGENT lines, the other stitched
    # builds, the rest
    group = [-2 if t["video"] == M.CANARY else 1 if api(t) else 2 if urgent(t) and not stitched(t)
             else 3 if stitched(t) else 5 for t in tasks]
    assert group == sorted(group) and set(group) == {-2, 1, 2, 3, 5} and [M.lead(t) for t in tasks] == group
    # the normal videos' tasks are those of the board without the stitched lines, in the same order, field for field
    # (only the priority numbers move)
    strip = lambda ts: [{k: v for k, v in t.items() if k != "priority"} for t in ts]
    monkeypatch.setattr(M, "EXPERIMENTS", BASE)
    assert strip(t for t in tasks if not stitched(t)) == strip(M.desired(have, set()))
    # as published: the priority arms' stitched builds are listed; the rest wait with their arms; the listed normal tasks
    # keep their order, and a pool that takes no stitched build ranks them as before
    monkeypatch.setattr(M, "PRIORITY", PRIORITY)
    monkeypatch.setattr(M, "FIRST", FIRST)
    pools = {"max": {"claims": {}}, "pulkit": {"claims": {}}}
    without, _ = M.gate(M.desired(set(), set()))
    M.rank(without, pools)
    monkeypatch.setattr(M, "EXPERIMENTS", BUILDS)
    listed, _ = M.gate(M.desired(set(), set()))
    M.rank(listed, pools)
    assert [t["id"] for t in listed if not stitched(t)] == [t["id"] for t in without]
    for p in ("wriva", "zaratan", "arm00"):
        assert [t["rank"].get(p) for t in listed if not stitched(t)] == [t["rank"].get(p) for t in without]
    assert {(t["baseline"], t["arm"]) for t in listed if stitched(t)} == {
        k for k in lines if k in PRIORITY or k + ("build",) in PRIORITY}
    canary = [i for i, t in enumerate(listed) if t["video"] == M.CANARY]
    assert canary == list(range(len(canary)))                                                  # ahead of the FIRST groups
    for p in ("max", "pulkit"):   # the next a pool claims: its canary builds (those not waiting on another), 1, 2, ...
        r = [t["rank"][p] for t in listed[:len(canary)] if p in t["rank"]]
        assert r == list(range(1, len(r) + 1)) and r
    # a pool that says it lacks a video does not rank its tasks; with every pool so, they wait for "no video"
    lack = {p: {"claims": {}, "no_video": list(M.STITCHED_VIDEOS)} for p in ("max", "pulkit")}
    M.rank(listed, lack)
    assert all(t["waiting"] == "no video" for t in listed if stitched(t) and set(t["pools"]) <= {"max", "pulkit"} and not t["deps"])
    assert all(set(t["rank"]) <= {"nexus"} for t in listed if stitched(t))
    # a failure on one pool shows as "failed", not hidden behind another pool's missing video
    x = next(t for t in listed if stitched(t) and t["pools"] == ["max", "pulkit"] and not t["deps"])
    lack["max"]["claims"] = {x["id"]: {"status": "failed"}}
    lack["max"]["no_video"] = []
    M.rank(listed, lack)
    assert x["waiting"] == "failed"
    # never offered to the leftover pools, however long they wait
    before = {t["id"]: list(t["pools"]) for t in listed if stitched(t)}
    n = M.leftovers(listed, [dict(id=t["id"], listed_t=1) for t in listed], 10 ** 6)
    assert n and all(t["pools"] == before[t["id"]] for t in listed if stitched(t))

def test_stitched_qa(monkeypatch, tmp_path):
    """The stitched end-of-video bank: a qa task per video for every arm that answers the normal banks from a memory
    (not MAGIC-Video and EgoRAG yet, nor the plain video models, API rows, ReKV, VST), on the stitched pools, waiting on
    the video's stitched memory build and never listed without it; the canary's after its builds, the others after
    theirs; the normal videos' tasks unchanged, in the same order."""
    board(monkeypatch, tmp_path, ALL)
    monkeypatch.setattr(M, "bank_videos", lambda bank: M.STITCHED_VIDEOS if bank == M.STITCHED_EOV else BANK_VIDEOS[bank])
    stitched = lambda t: t["video"] in M.STITCHED_VIDEOS
    qa_arms = {(b, arm) for b, arm, stage, _, bank in BASE if stage == "qa" and b in M.MEMORY_DONE}
    eov = {(b, arm) for b, arm, stage, _, bank in ALL if bank == M.STITCHED_EOV}
    assert eov == {k for k in qa_arms if k[0] not in ("magic", "egorag")} and M.STITCHED_QA_LATER == {"magic", "egorag"}
    assert not eov & {(b, a) for b, a, *_ in BASE if b in ("direct", "api", "rekv", "vst")}
    have = {("merit", "m27", "P01-stitched")}                          # one stitched memory in the dataset
    tasks = M.desired(have, set())
    by = {t["id"]: t for t in tasks}
    qa = [t for t in tasks if stitched(t) and t["stage"] == "qa"]
    assert len(qa) == 9 * len(eov) and all(t["bank"] == M.STITCHED_EOV for t in qa)
    for t in qa:
        mem = M.MEMORY_ARM.get((t["baseline"], t["arm"]), t["arm"])
        build = f"{t['baseline']}-{mem}-build-{t['video']}"
        assert t["deps"] == ([] if (t["baseline"], mem, t["video"]) in have else [build]), t["id"]
        assert t["deps"] == [] or by[build]["priority"] < t["priority"]
        assert t["pools"] and set(t["pools"]) <= set(M.STITCHED_POOLS)
        assert t.get("needs_env") == M.needs(t["baseline"], t["arm"], "qa")
    # never listed without its memory: with no stitched build line, only the built video's answers are
    monkeypatch.setattr(M, "EXPERIMENTS", [e for e in ALL if e[4] != (M.STITCHED,)])
    alone = [t for t in M.desired(have, set()) if stitched(t)]
    assert {(t["baseline"], t["arm"], t["video"]) for t in alone} == {("merit", "m27", "P01-stitched")} and not alone[0]["deps"]
    # order: canary builds, canary answers, ..., the other stitched builds, their answers, the rest
    lead = [M.lead(t) for t in tasks]
    assert lead == sorted(lead) and {M.lead(t) for t in qa} == {-1, 4}
    assert all(M.lead(t) == -1 for t in qa if t["video"] == M.CANARY)
    strip = lambda ts: [{k: v for k, v in t.items() if k != "priority"} for t in ts]
    monkeypatch.setattr(M, "EXPERIMENTS", BASE)
    assert strip(t for t in tasks if not stitched(t)) == strip(M.desired(have, set()))
    # as published: the priority arms' answers are listed (waiting on their builds), the rest withheld with their arms
    monkeypatch.setattr(M, "PRIORITY", PRIORITY)
    monkeypatch.setattr(M, "FIRST", FIRST)
    monkeypatch.setattr(M, "EXPERIMENTS", ALL)
    listed, _ = M.gate(M.desired(set(), set()))
    M.rank(listed, {})
    lq = [t for t in listed if stitched(t) and t["stage"] == "qa"]
    assert lq and {(t["baseline"], t["arm"]) for t in lq} == {k for k in eov if k in PRIORITY}
    assert all(t["waiting"] == "deps" for t in lq)
    assert [M.lead(t) for t in listed] == sorted(M.lead(t) for t in listed)


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


def test_probes(monkeypatch, tmp_path):
    """The M2 probe bank: a qa line for every memory arm that answers the normal banks (none for the plain video
    models, API rows, ReKV, VST), on that arm's pools with its keys, from the memory of the normal video (waiting on its
    build when listed, never listed without one); listed after the memory builds and ahead of the other answers; the
    normal tasks unchanged; each arm behind its own PRIORITY gate."""
    probe_lines = [e for e in FULL if e[4] == M.PROBES]
    base = [e for e in BASE if e[4] != M.STITCHED]   # (Vatsal's stitched bank: its own tests)
    board(monkeypatch, tmp_path, base + probe_lines)
    monkeypatch.setattr(M, "bank_videos", lambda bank: B10 + T2 if bank == M.PROBES else BANK_VIDEOS[bank])
    arms = {(b, arm) for b, arm, stage, _, bank in base if stage == "qa" and bank in M.BANKS and b in M.MEMORY_DONE}
    # every memory arm but the frozen ones (none of their probes listed: hippomm's only arm, MAGIC's m8, m27, oss120-q27,
    # WorldMM's m8, m27), each PROBES_FIRST arm among them
    assert M.PROBES_FROZEN <= arms and set(M.PROBES_FIRST) <= arms - M.PROBES_FROZEN
    assert {(b, arm) for b, arm, *_ in probe_lines} == arms - M.PROBES_FROZEN and len(probe_lines) == len(arms - M.PROBES_FROZEN)
    assert not {b for b, *_ in probe_lines} & {"direct", "api", "rekv", "vst", "hippomm"}
    assert {b for b, *_ in probe_lines} == {"merit", "magic", "worldmm", "egagent", "egorag", "egor1", "m3agent"}
    assert {(b, arm) for b, arm, *_ in probe_lines} - set(M.PROBES_FIRST) == {("magic", "oss120-flash"), ("merit", "g5m-g51")}
    arms -= M.PROBES_FROZEN
    assert M.PROBES in M.NOT_IN_ALL and M.STITCHED_EOV in M.NOT_IN_ALL
    have = {(b, M.MEMORY_ARM.get((b, arm), arm), v) for b, arm in arms for v in B10}   # batch10's memories are built
    tasks = M.desired(have, set())
    by = {t["id"]: t for t in tasks}
    pr = [t for t in tasks if t["bank"] == M.PROBES]
    assert len(pr) == len(arms) * len(B10 + T2) and all(t["stage"] == "qa" for t in pr)
    for t in pr:
        mem = M.MEMORY_ARM.get((t["baseline"], t["arm"]), t["arm"])
        build = f"{t['baseline']}-{mem}-build-{t['video']}"
        assert t["deps"] == ([] if t["video"] in B10 else [build]) and all(by[d]["priority"] < t["priority"] for d in t["deps"])
        normal = next(x for x in tasks if (x["baseline"], x["arm"], x["stage"], x["video"]) == (t["baseline"], t["arm"], "qa", t["video"])
                      and x["bank"] != M.PROBES)
        assert (t["pools"], t.get("needs_env"), t.get("memory_arm")) == (normal["pools"], normal.get("needs_env"), normal.get("memory_arm"))
    # never listed without the memory: with no build lines, only the built videos' probes
    monkeypatch.setattr(M, "EXPERIMENTS", [e for e in base + probe_lines if e[2] != "build"])
    assert {t["video"] for t in M.desired(have, set()) if t["bank"] == M.PROBES} == set(B10)
    # order: every memory build and the evidence-clips rows, then the probes, then the other answers; normal tasks as before
    lead = [M.lead(t) for t in tasks]
    assert lead == sorted(lead) and all(len(M.FIRST) + 6 <= M.lead(t) < len(M.FIRST) + 7 or M.lead(t) == -3 for t in pr)
    # PROBES_FIRST's arms lead the probes, in their order, then the others
    rank = lambda t: M.PROBES_FIRST.index((t["baseline"], t["arm"])) if (t["baseline"], t["arm"]) in M.PROBES_FIRST else len(M.PROBES_FIRST)
    order = lambda t: (M.lead(t) != -3, rank(t))   # PROBES_NOW's free tasks first, then PROBES_FIRST's order
    assert [order(t) for t in pr] == sorted(order(t) for t in pr) and {rank(t) for t in pr} == set(range(len(M.PROBES_FIRST) + 1))
    assert max(M.lead(t) for t in tasks if t["stage"] == "build") < len(M.FIRST) + 6   # after every memory build and the
    assert all(M.lead(t) < len(M.FIRST) + 6 for t in tasks if t["arm"].endswith("-evclips"))   # evidence-clips rows, ahead of
    assert {M.lead(t) for t in tasks if t["stage"] == "qa" and t not in pr and not api(t)  # the other answers
            and not t["arm"].endswith("-evclips")} == {len(M.FIRST) + 7}
    strip = lambda ts: [{k: v for k, v in t.items() if k != "priority"} for t in ts]
    monkeypatch.setattr(M, "EXPERIMENTS", base)
    assert strip(t for t in tasks if t["bank"] != M.PROBES) == strip(M.desired(have, set()))
    # as published (the real PRIORITY and FIRST): an arm's probes are listed with its normal rows (a priority arm, or a
    # local one that needs no key), the paid others withheld
    monkeypatch.setattr(M, "PRIORITY", PRIORITY)
    monkeypatch.setattr(M, "FIRST", FIRST)
    monkeypatch.setattr(M, "EXPERIMENTS", base + probe_lines)
    listed, _ = M.gate(M.desired(have, set()))
    assert {(t["baseline"], t["arm"]) for t in listed if t["bank"] == M.PROBES} == {
        k for k in arms if k in PRIORITY or k + ("qa",) in PRIORITY or not M.needs(*k, "qa")}   # (the gate lets every keyless task by)
    assert [M.lead(t) for t in listed] == sorted(M.lead(t) for t in listed)

    # PROBES_NOW: those arms' probe tasks lead every other task, the stitched ones included
    now = dict(baseline="worldmm", arm="g5m-g51", stage="qa", video="P01-20240202-110250", bank=M.PROBES)
    assert ("worldmm", "g5m-g51") in M.PROBES_NOW and M.lead(now) < M.lead(dict(now, video=M.CANARY, bank=None, stage="build"))
    assert M.lead(dict(now, bank="t3_agree2_end")) > M.lead(now) and M.lead(dict(now, arm="g5m-q8")) > M.lead(now)

def test_existing_tasks_unchanged(monkeypatch, tmp_path):
    """Without the reported arms the board is main's, field for field (api tasks gain needs_env only); with them,
    the merit / magic / worldmm / api tasks are the same ones in the same order."""
    have = {("merit", "m8", T2[0]), ("worldmm", "m27", REST[0])}
    files = {f"magic/m8/{T2[1]}/answers_t2_agree2_end.json.gz", f"api/gpt-5.5-f384/{B10[0]}/answers_batch10_agree2_end.json.gz"}
    full = [e for e in BASE if not e[1].endswith("-evclips")]   # the evidence-clips lines (EVCLIPS) came after main's
    board(monkeypatch, tmp_path, [e for e in full if e not in M.REPORTED + M.PAID])
    assert len(M.EXPERIMENTS) == len(full) - len(M.REPORTED) - len(M.PAID) == 6 * (1 + len(M.BANKS)) + 3 * len(M.BANKS) + len(M.DIRECT) * len(M.BANKS) + 2 * len(M.BANKS) + 1 + len(M.BANKS)   # 6 arms x (build + banks) + 3 api x banks + the direct arms x banks + rekv's and vst's banks + m3agent's build + banks
    old = desired_before(have, files)
    strip = lambda ts: [{k: v for k, v in t.items() if k != "needs_env"} for t in ts]
    new = M.desired(have, files)
    assert {t["baseline"] for t in old} == {"api", "direct", "rekv", "vst", "merit", "magic", "worldmm", "m3agent"} and strip(new) == old
    assert [t.get("needs_env") for t in new] == [M.needs(t["baseline"], t["arm"], t["stage"]) for t in old]
    assert M.gate(new) == (new, 0)                                          # the gate holds none of them back
    # direct: a task per arm, bank and video unless answered, waiting on nothing, after api and before the memory arms
    direct = [t for t in M.desired(have, files | {f"direct/qwen3vl-8b-f768/{T2[0]}/answers_t2_agree2_end.json.gz"}) if t["baseline"] == "direct"]
    assert len(direct) == len(M.DIRECT) * sum(len(v) for v in BANK_VIDEOS.values()) - 1 and not any(t["deps"] or t.get("needs_env") for t in direct)
    assert f"direct-qwen3vl-8b-f768-qa-t2_agree2_end-{T2[0]}" not in {t["id"] for t in direct}
    kind = [0 if api(t) else 1 if urgent(t) else 2 if t["baseline"] == "direct" else 3 if t["baseline"] == "rekv"
            else 4 if t["baseline"] == "vst" else 6 if t["baseline"] == "m3agent" else 5 for t in new]
    assert kind == sorted(kind) and set(kind) == {0, 1, 2, 3, 4, 5, 6}
    # ReKV: as direct, a task per bank and video, waiting on nothing; M3-Agent: a build per video, and qa waits on it
    by = {t["id"]: t for t in new}
    rekv = [t for t in new if t["baseline"] == "rekv"]
    assert len(rekv) == sum(len(v) for v in BANK_VIDEOS.values()) and not any(t["deps"] or t["stage"] != "qa" for t in rekv)
    # VST: the same shape, a task per bank and video (vst-vst7b-qa-<bank>-<video>), done when its answers are in
    vst = [t for t in new if t["baseline"] == "vst"]
    assert len(vst) == len(rekv) and not any(t["deps"] or t.get("needs_env") or t["stage"] != "qa" or t["pools"] != M.M8 for t in vst)
    assert by[f"vst-vst7b-qa-t2_agree2_end-{T2[0]}"]["arm"] == "vst7b"
    assert f"vst-vst7b-qa-t2_agree2_end-{T2[0]}" not in {t["id"] for t in M.desired(have, files | {f"vst/vst7b/{T2[0]}/answers_t2_agree2_end.json.gz"})}
    assert sum(t["baseline"] == "m3agent" and t["stage"] == "build" for t in new) == len(B10 + T2 + T3 + T4 + T5 + REST)
    assert by[f"m3agent-m3-qa-t2_agree2_end-{T2[0]}"]["deps"] == [f"m3agent-m3-build-{T2[0]}"]
    assert by[f"m3agent-m3-build-{T2[0]}"]["pools"] == M.M8 and by[f"m3agent-m3-qa-t2_agree2_end-{T2[0]}"]["pools"] == M.M27
    # M3-Agent's qa is never listed without its memory: with no build line to wait on, only the built videos' qa is
    was = M.EXPERIMENTS
    monkeypatch.setattr(M, "EXPERIMENTS", [e for e in was if e[:3] != ("m3agent", "m3", "build")])
    only = [t for t in M.desired(have | {("m3agent", "m3", T2[0])}, files) if t["baseline"] == "m3agent"]
    assert only and {t["video"] for t in only} == {T2[0]} and not any(t["deps"] or t["stage"] != "qa" for t in only)
    assert M.NEEDS_MEMORY == {"m3agent"} and not [t for t in M.desired(have, files) if t["baseline"] == "m3agent"]
    monkeypatch.setattr(M, "EXPERIMENTS", was)
    monkeypatch.setattr(M, "EXPERIMENTS", full)
    noprio = lambda ts: [{k: v for k, v in t.items() if k != "priority"} for t in ts]
    assert noprio(strip(t for t in M.desired(have, files) if not M.reported(t))) == noprio(old)
    listed, _ = M.gate(M.desired(have, files))                             # as published now: main's tasks + the gated arms
    assert noprio(strip(t for t in listed if not M.reported(t))) == noprio(old)


def bank_q(key, t, gold=1):
    return {"key": key, "q": f"Where is the {key}?", "form": "f", "streaming": True, "t_ask": t,
            "gemini_mcq": {"options": ["sink", "table"], "answer_index": gold}}


def merit_rows(tmp_path, v, qs, end):
    """MERIT's answer rows for these bank questions (prepare_epic.convert_questions: asked at floor(t_ask), or at the
    video's end for a t_ask in its last second), as collect_answers keeps them."""
    from prepare_epic import convert_questions
    f = tmp_path / "q.json"
    f.write_text(json.dumps({"questions": qs}))
    return [{k: x for k, x in r.items() if not k.startswith("choice_")} | {
        "choices": {"A": r["choice_a"], "B": r["choice_b"]}, "response": "ANSWER: B\nREASONING: ..."}
        for r in convert_questions(str(f), v, end)]


def test_stale_answers(tmp_path, monkeypatch):
    """A qa task is listed when the answers file is missing or stale (a question moved, an Error, a file that does not
    parse), under a -redo- id when the file exists; current answers list nothing, also batch10's from before keys,
    asked at ceil(t_ask)."""
    V = [f"P01-20240101-00010{i}" for i in range(6)]
    qs = [bank_q("k1", 100.4), bank_q("k2", 1734.9)]   # k2: the end-of-video t_ask of a moved question
    d = tmp_path / "outputs/v3/banks/tb"
    d.mkdir(parents=True)
    for v in V:
        (d / f"{v}_questions.json").write_text(json.dumps({"questions": qs}))
    (tmp_path / "v3").mkdir()
    (tmp_path / "v3/export_bank.py").write_text("")
    (tmp_path / "order.txt").write_text("")
    monkeypatch.setenv("EPIC_MORPH_DIR", str(tmp_path))
    monkeypatch.setattr(M, "BUILD_ORDER", str(tmp_path / "order.txt"))
    monkeypatch.setattr(M, "EXPERIMENTS", [("merit", "m8", "qa", ["p"], "tb")])
    monkeypatch.setattr(M, "REPORTED_BANKS", [])
    M.exported.cache_clear()
    rows = {V[0]: merit_rows(tmp_path, V[0], qs, 1735),                          # current: k2 asked at the end (1735 s)
            V[1]: merit_rows(tmp_path, V[1], [qs[0], bank_q("k2", 900.0)], 1800),  # k2 answered at its old t_ask
            V[3]: merit_rows(tmp_path, V[3], qs, 1800)}
    rows[V[3]][0]["response"] = "Error"                                           # k1 failed
    old = merit_rows(tmp_path, V[5], qs, 1800)                                    # batch10 before keys: positional ID,
    for i, r in enumerate(old, 1):                                                # no key or qhash, asked at ceil(t_ask)
        r.update(ID=str(i), query_time={"date": "DAY1", "time": ["00014100", "00285500"][i - 1]})
        del r["key"], r["qhash"]
    rows[V[5]] = old
    path = lambda v: f"merit/m8/{v}/answers_tb.json.gz"
    answers = {path(v): M.summary(r) for v, r in rows.items()} | {path(V[4]): None}   # V[4]: does not parse
    files = set(answers)                                                           # V[2]: no answers file
    have = {("merit", "m8", v) for v in V}
    tasks = {t["video"]: t for t in M.desired(have, files, answers)}
    assert sorted(tasks) == [V[1], V[2], V[3], V[4]]
    assert tasks[V[2]]["id"] == f"merit-m8-qa-tb-{V[2]}" and "stale" not in tasks[V[2]]
    assert [tasks[v]["stale"] for v in (V[1], V[3], V[4])] == [1, 1, 2]
    redo = tasks[V[1]]["id"]
    assert redo.startswith(f"merit-m8-qa-tb-{V[1]}-redo-") and redo == M.desired(have, files, answers)[0]["id"]
    # the hash follows the bank: the same questions keep the id (a done redo stays done), a changed bank gets a new one
    assert M.redo_id("x", qs) == M.redo_id("x", list(reversed(qs))) != M.redo_id("x", [qs[0], bank_q("k2", 1730.0)])
    assert M.unanswered([bank_q("k1", 100.4, gold=0)], answers[path(V[0])]) == ["k1"]   # another gold letter
    import worker
    assert all(worker.valid(t) for t in tasks.values())                            # the worker runs the -redo- id
    # not while the first run is still going (or its results wait for the relay): its upload may be current
    assert V[1] not in {t["video"] for t in M.desired(have, files, answers, busy={f"merit-m8-qa-tb-{V[1]}"})}
    # without summaries the files are not read: the board as before
    assert sorted(t["video"] for t in M.desired(have, files)) == [V[2]]


def test_end_bank_reuses_base(tmp_path, monkeypatch):
    """An _end bank's questions are its base bank's: answers in answers_<base> count, alone or with the _end file's. A
    video they cover is not listed; one they leave a question of is a plain task while no _end file exists, a redo
    once one does; read_answers reads both files."""
    V = [f"P01-20240101-00020{i}" for i in range(5)]
    qs = [bank_q("k1", 1734.9), bank_q("k2", 1734.9)]   # both asked at the end
    d = tmp_path / "outputs/v3/banks/tb_end"
    d.mkdir(parents=True)
    for v in V:
        (d / f"{v}_questions.json").write_text(json.dumps({"questions": qs}))
    (tmp_path / "v3").mkdir()
    (tmp_path / "v3/export_bank.py").write_text("")
    (tmp_path / "order.txt").write_text("")
    monkeypatch.setenv("EPIC_MORPH_DIR", str(tmp_path))
    monkeypatch.setattr(M, "BUILD_ORDER", str(tmp_path / "order.txt"))
    monkeypatch.setattr(M, "EXPERIMENTS", [("merit", "m8", "qa", ["p"], "tb_end")])
    monkeypatch.setattr(M, "REPORTED_BANKS", [])
    M.exported.cache_clear()
    assert M.base_bank("tb_end") == "tb" and M.base_bank("tb") is None and M.BANKS[1] == "t2_agree2_end"
    row = lambda q: [q["key"], *M.asked(q)[1:], False]
    base, end = (lambda v: f"merit/m8/{v}/answers_tb.json.gz"), (lambda v: f"merit/m8/{v}/answers_tb_end.json.gz")
    answers = {base(V[0]): [row(q) for q in qs],                                    # the base file covers both
               base(V[1]): [row(qs[0])],                                            # k1 only, no _end file
               base(V[2]): [row(qs[0])], end(V[2]): [row(qs[1])],                   # together they cover both
               end(V[3]): [row(qs[0])]}                                             # the _end file lacks k2
    files, have = set(answers), {("merit", "m8", v) for v in V}                     # V[4]: no file at all
    tasks = {t["video"]: t for t in M.desired(have, files, answers)}
    assert sorted(tasks) == [V[1], V[3], V[4]]
    assert tasks[V[1]]["id"] == f"merit-m8-qa-tb_end-{V[1]}" and "stale" not in tasks[V[1]]
    assert tasks[V[3]]["id"].startswith(f"merit-m8-qa-tb_end-{V[3]}-redo-") and tasks[V[3]]["stale"] == 1
    # a running plain task stays listed (claimed), whatever the base file holds; a running redo is not (its upload
    # may be current)
    busy = {f"merit-m8-qa-tb_end-{v}" for v in V}
    assert sorted(t["video"] for t in M.desired(have, files, answers, busy)) == [V[0], V[1], V[4]]
    monkeypatch.setattr(M, "hf_hub_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")))
    monkeypatch.setattr(M, "ANSWERS_CACHE", str(tmp_path / "cache"))
    os.makedirs(tmp_path / "cache")
    for p in files:
        (tmp_path / "cache" / f"{p.replace('/', '_')}.json").write_text(json.dumps(answers[p]))
    assert set(M.read_answers("c", {p: p.replace("/", "_") for p in files})) == files


def test_rebuilt_memory(tmp_path, monkeypatch):
    """A video whose memory was rebuilt (REBUILT): no qa task while the dataset holds the old memory or a task from
    before the rebuild still runs; then its tasks carry +mem<tag>, on every bank, and the answers in a folder that
    still names the old memory do not count (base bank's included). Once a pool uploaded answers made from the new
    memory (another memory_ref) the files are read as any other's. A worker from before this change refuses the ids."""
    V = [f"P01-20240101-00030{i}" for i in range(6)]
    qs = [bank_q("k1", 1734.9), bank_q("k2", 1734.9)]
    for bank in ("tb_end", "pb"):
        d = tmp_path / "outputs/v3/banks" / bank
        d.mkdir(parents=True)
        for v in V:
            (d / f"{v}_questions.json").write_text(json.dumps({"questions": qs}))
    (tmp_path / "v3").mkdir()
    (tmp_path / "v3/export_bank.py").write_text("")
    (tmp_path / "order.txt").write_text("")
    monkeypatch.setenv("EPIC_MORPH_DIR", str(tmp_path))
    monkeypatch.setattr(M, "BUILD_ORDER", str(tmp_path / "order.txt"))
    monkeypatch.setattr(M, "EXPERIMENTS", [("merit", "g5m-q8", "qa", ["p"], "tb_end"), ("merit", "g5m-q8", "qa", ["p"], "pb")])
    monkeypatch.setattr(M, "REPORTED_BANKS", [])
    monkeypatch.setattr(M, "REBUILT", {("merit", "g5m"): ("t1", {v: ["oldmem", "oldref"] for v in V if v != V[4]})})
    M.exported.cache_clear()
    row = lambda q: [q["key"], *M.asked(q)[1:], False]
    d = lambda v, f: f"merit/g5m-q8/{v}/{f}"
    files, answers = {}, {}
    for i, v in enumerate(V):   # V[0]: the old memory still in the dataset; the others: the rebuilt one (V[4]: never rebuilt)
        files[f"merit/g5m/{v}/{v}_30sec_4_key.json"] = "oldmem" if i == 0 else "newmem"
        files[d(v, f"{v}_memory_ref.json")] = "newref" if i in (2, 3) else "oldref"   # V[2], V[3]: answered again already
        answers[d(v, "answers_tb.json.gz")] = [row(q) for q in qs]                  # the base bank's file covers both
    answers[d(V[3], "answers_tb.json.gz")] = [row(qs[0])]
    answers[d(V[3], "answers_tb_end.json.gz")] = []                                 # V[3]: k2 unanswered in both
    answers[d(V[1], "answers_pb.json.gz")] = [row(q) for q in qs]                   # V[1]: pb answered, from the old memory
    files |= {p: "h" for p in answers}
    have = {("merit", "g5m", v) for v in V}
    ids = lambda **k: sorted(t["id"] for t in M.desired(have, files, answers, **k))
    plain = lambda bank, v: f"merit-g5m-q8-qa-{bank}-{v}"
    listed = ids()
    redo = [t for t in listed if "-redo-" in t]
    assert listed == sorted([plain("pb", V[1]) + "+memt1", plain("tb_end", V[1]) + "+memt1",       # old answers ignored
                             plain("pb", V[2]) + "+memt1",                                         # tb_end current, pb missing
                             plain("pb", V[3]) + "+memt1", *redo,                                  # a redo of the +mem task
                             plain("pb", V[4]),                                                    # not rebuilt: as ever
                             plain("pb", V[5]) + "+memt1", plain("tb_end", V[5]) + "+memt1"])
    assert redo == [M.redo_id(plain("tb_end", V[3]) + "+memt1", qs)]
    tasks = {t["id"]: t for t in M.desired(have, files, answers)}
    assert tasks[redo[0]]["stale"] == 1 and "stale" not in tasks[plain("tb_end", V[1]) + "+memt1"]
    # a task of the video and arm from before the rebuild still runs (any bank): nothing of it is listed meanwhile
    assert not [t for t in ids(busy={plain("pb", V[5])}) if V[5] in t]
    # a running +mem task stays listed (claimed), as a plain one does; its redo is not listed beside a running one
    assert plain("tb_end", V[1]) + "+memt1" in ids(busy={plain("tb_end", V[1]) + "+memt1"})
    assert not [t for t in ids(busy={plain("tb_end", V[3]) + "+memt1"}) if "-redo-" in t]
    import re
    import worker
    assert all(worker.valid(t) for t in tasks.values())
    before = re.compile(r"^[A-Za-z0-9_.-]+$")   # worker.VALID["id"] before this change: such a checkout leaves the tasks
    assert [t for t in tasks if not before.fullmatch(t)] == [t for t in tasks if "+mem" in t]
    # the real setting: MERIT's gpt-5-mini memory, 82 videos, each with the old memory's hash and its answer arms' old reference
    real = M.rebuilt("rebuilt_merit_g5m.txt")
    assert len(real) == 82 and all(len(x) == 2 and len(x[0]) == 40 for x in real.values())
    assert sum(x[1] != "-" for x in real.values()) == 59


def test_pools_of_the_lines_listed_2026_11_19(monkeypatch, tmp_path):
    """MERIT's answers from the gpt-5-mini memory: the Qwen arms (no key) on Max's, Pulkit's, wriva's pools, Zaratan and
    Nexus; GPT-5.1's on the key pools. The stitched lines: MAGIC-Video's and EGAgent's builds and MAGIC-Video's answers
    with their keys on the key pools and ours that have an OpenRouter key; MERIT m27's answers on Zaratan too; EGAgent's
    answers on the key pools and Max's CPU pool, each waiting on its build; the end-of-video bank frozen; never the
    leftover pools."""
    (tmp_path / "order.txt").write_text("")
    monkeypatch.setattr(M, "BUILD_ORDER", str(tmp_path / "order.txt"))
    monkeypatch.setattr(M, "REPORTED_BANKS", [])
    monkeypatch.setattr(M, "bank_videos", lambda bank: M.STITCHED_VIDEOS if bank == M.STITCHED else ["P01-20240101-000001"])
    monkeypatch.setattr(M, "EXPERIMENTS", [e for e in M.EXPERIMENTS if e[4] in ((M.STITCHED,), M.STITCHED, M.STITCHED_EOV)
                                           or (e[:2] in M.MEMORY_ARM and e[0] == "merit" and e[4] in (M.BANKS[0], M.PROBES))])
    have = {(b, arm, v) for b, arm in (("merit", "m27"), ("worldmm", "g5m"), ("merit", "g5m")) for v in M.STITCHED_VIDEOS + ["P01-20240101-000001"]}
    have |= {("magic", "oss120", v) for v in M.STITCHED_VIDEOS if v[:3] not in ("P03", "P04", "P06")}
    tasks = M.desired(have, {})
    pools = {}
    for t in tasks:
        assert pools.setdefault((t["baseline"], t["arm"], t["stage"], t["bank"]), t["pools"]) == t["pools"]
    key, own = ["max", "pulkit"], ["nexus", "zaratan"]
    assert pools == {
        ("merit", "g5m-q8", "qa", M.BANKS[0]): M.M8 + ["nexus"], ("merit", "g5m-q8", "qa", M.PROBES): M.M8 + ["nexus"],
        ("merit", "g5m-q27", "qa", M.BANKS[0]): M.M27 + ["arm00", "nexus"], ("merit", "g5m-q27", "qa", M.PROBES): M.M27 + ["arm00", "nexus"],
        ("merit", "g5m-g51", "qa", M.BANKS[0]): key, ("merit", "g5m-g51", "qa", M.PROBES): key,
        ("magic", "oss120", "build", None): key + own, ("magic", "oss120-flash", "qa", M.STITCHED): key + own,
        ("egagent", "gpt41", "build", None): key + ["nexus"],
        ("egagent", "gemini36flash", "qa", M.STITCHED): key + ["max-cpu"],
        ("merit", "m27", "qa", M.STITCHED): ["zaratan", "max", "pulkit", "nexus"],
        ("worldmm", "g5m-g51", "qa", M.STITCHED): key, ("api", M.STITCHED_API, "qa", M.STITCHED): key + ["max-cpu"]}
    assert all({"nexus", "zaratan"} <= set(p) for (b, arm, *_), p in pools.items() if not M.needs(b, arm, "qa") and b == "merit")
    by = {t["id"]: t for t in tasks}
    st = [t for t in tasks if t["video"] in M.STITCHED_VIDEOS]
    assert sorted(t["video"][:3] for t in st if (t["baseline"], t["stage"]) == ("magic", "build")) == ["P03", "P04", "P06"]
    assert sum((t["baseline"], t["stage"]) == ("egagent", "build") for t in st) == 9 and not [t for t in st if t["bank"] == M.STITCHED_EOV]
    assert by["magic-oss120-flash-qa-stitched-P03-stitched"]["deps"] == ["magic-oss120-build-P03-stitched"]
    assert by["magic-oss120-flash-qa-stitched-P01-stitched"]["deps"] == []
    assert by["magic-oss120-build-P03-stitched"]["needs_env"] == ["OPENROUTER_API_KEY"]
    assert by["egagent-gpt41-build-P01-stitched"]["needs_env"] == ["OPENAI_API_KEY|OPENROUTER_API_KEY"]
    # EGAgent's stitched answers: nine, with Gemini's key, each waiting on its video's GPT-4.1 memory build
    eg = [t for t in st if (t["baseline"], t["arm"], t["stage"]) == ("egagent", "gemini36flash", "qa")]
    assert M.STITCHED_ANSWERS == set(M.STITCHED_ANSWER_LINES) and sorted(t["video"] for t in eg) == M.STITCHED_VIDEOS
    assert all(t["bank"] == M.STITCHED and t["needs_env"] == ["GEMINI_API_KEY"] and t["memory_arm"] == "gpt41"
               and t["deps"] == [f"egagent-gpt41-build-{t['video']}"] and {"max", "max-cpu"} <= set(t["pools"]) for t in eg)
    assert all(by[t["deps"][0]]["priority"] < t["priority"] for t in eg)
    # with the memories in the dataset no build is listed and the answers wait on nothing
    assert not [t for t in M.desired(have | {("egagent", "gpt41", v) for v in M.STITCHED_VIDEOS}, {})
                if t["baseline"] == "egagent" and (t["stage"] == "build" or t["deps"])]
    listed, withheld = M.gate(tasks)   # every stitched line is a priority row; they lead the list, builds first
    assert [t for t in st if t not in listed] == [] and [M.lead(t) for t in listed] == sorted(M.lead(t) for t in listed)
    assert {M.lead(t) for t in st} == {-2, -1}
    before = {t["id"]: list(t["pools"]) for t in st}
    M.leftovers(listed, [dict(id=t["id"], listed_t=1) for t in listed], 10 ** 6)
    assert all(t["pools"] == before[t["id"]] for t in st)


def test_end_of_video_ask_time(tmp_path):
    """A question asked at the video's end counts as answered by a row asked at the baseline's own end: the last whole
    clip before the bank's end, when the file runs a few seconds past its last frame (P01-20240202-171220: clips to
    510 s, the bank asks at 512.5). Not by an answer from before the question moved to the end, nor for a question
    asked mid-video, nor when another row of the file was asked later."""
    end = lambda k: bank_q(k, 512.5) | {"ask_at": "end"}
    qs = [end("k1"), end("k2")]
    merit = M.summary(merit_rows(tmp_path, "P01-20240101-000401", qs, 510))        # MERIT asks at its clip list's end
    assert [r[3] for r in merit] == [510, 510] and M.unanswered(qs, merit) == []
    row = lambda q, t: [q["key"], *M.asked(q)[1:3], t, False]
    assert M.unanswered(qs, [row(qs[0], 507), row(qs[1], 510)]) == ["k1"]            # 507: its t_ask before the move
    assert M.unanswered(qs, [row(qs[0], 510), row(qs[1], 513)]) == ["k1"]            # the baseline's end here is 513
    assert M.unanswered(qs, [row(qs[0], 480), row(qs[1], 480)]) == ["k1", "k2"]      # a clip or more before the end
    moved = [q | {"t_ask_orig": 510.4} for q in qs]                                  # the old t_ask fell on the boundary:
    assert M.unanswered(moved, [row(qs[0], 510), row(qs[1], 510)]) == ["k1", "k2"]   # such a row may be from before the move
    assert M.unanswered([q | {"t_ask_orig": 507.5} for q in qs], merit) == []
    assert M.unanswered(qs, [row(qs[0], 512), row(qs[1], 513)]) == []                # within a second, as before
    mid = [bank_q("k1", 512.5) | {"ask_at": "t"}, bank_q("k2", 512.5)]               # asked at t_ask, not "the end"
    assert M.unanswered(mid, [row(mid[0], 510), row(mid[1], 510)]) == ["k1", "k2"]
    assert M.redo_id("x", qs) == M.redo_id("x", [bank_q("k1", 512.5), bank_q("k2", 512.5)])   # ids do not move


def test_done_redo_is_not_listed_again():
    """A redo task a pool has marked done leaves the list (it ran with this state of the bank; no pool can take a done
    id); one that never ran, one that runs or failed, and a plain task with a done claim stay."""
    t = lambda i, **k: dict(id=i, **k)
    tasks = [t("a-redo-1", stale=2), t("b-redo-1", stale=1), t("c-redo-1", stale=1), t("d-redo-1", stale=1), t("e"), t("f-redo-2", stale=1)]
    held = {"a-redo-1": ("p", {"status": "done"}), "b-redo-1": ("p", {"status": "running"}), "e": ("p", {"status": "done"}),
            "f-redo-1": ("p", {"status": "done"}), "f": ("p", {"status": "done"})}   # f: the bank changed since (another hash)
    keep, out = M.not_again(tasks, held)
    assert out == ["a-redo-1"] and [x["id"] for x in keep] == ["b-redo-1", "c-redo-1", "d-redo-1", "e", "f-redo-2"]


def test_worldmm_m27_answers_not_on_zaratan(monkeypatch, tmp_path):
    """NOT_POOLS: WorldMM m27's answers are not for Zaratan (its own copies of those memories); its other lines are."""
    (tmp_path / "order.txt").write_text("")
    monkeypatch.setattr(M, "BUILD_ORDER", str(tmp_path / "order.txt"))
    monkeypatch.setattr(M, "REPORTED_BANKS", [])
    monkeypatch.setattr(M, "bank_videos", lambda bank: ["P01-20240101-000001"])
    monkeypatch.setattr(M, "EXPERIMENTS", [e for e in M.EXPERIMENTS if e[0] in ("worldmm", "merit") and e[1] in ("m8", "m27")
                                           and e[4] in (None, M.BANKS[0])])
    pools = {(t["baseline"], t["arm"], t["stage"]): t["pools"] for t in M.desired(set(), {})}
    assert pools["worldmm", "m27", "qa"] == ["wriva", "max", "pulkit", "nexus"]
    assert all("zaratan" in p for k, p in pools.items() if k != ("worldmm", "m27", "qa") and k != ("merit", "m8", "qa"))
    assert M.NOT_POOLS == {("worldmm", "m27", "qa"): ["zaratan"]}


def test_trace_stale():
    """For a TRACED baseline an answer row without a trace leaves its question unanswered (listed again); other
    baselines, and summaries from before the trace flag, compare as before."""
    q = [bank_q("k1", 100.4)]
    row = {"key": "k1", "qhash": M.asked(q[0])[1], "answer": "B", "response": "B", "t_ask": 100.4}
    assert M.unanswered(q, M.summary([row]), trace=True) == ["k1"]
    assert M.unanswered(q, M.summary([row | {"trace": []}]), trace=True) == []
    assert M.unanswered(q, M.summary([row])) == [] and M.unanswered(q, [["k1", None, "B", 100]]) == []
    assert "magic" in M.TRACED and "merit" not in M.TRACED


def test_ask_time_formats():
    """Each baseline's record of when a question was asked, in seconds: MERIT's query_time, MAGIC / WorldMM's int,
    the API's and the paid rows' t_ask; none for a row that records none."""
    assert M.ask_time({"query_time": {"date": "DAY1", "time": "00285400"}}) == 1734
    assert M.ask_time({"query_time": 100285400}) == 1734
    assert M.ask_time({"t_ask": 1734.9, "cutoff": 1734.9}) == 1734.9
    assert M.ask_time({"t_ask": 1734.9, "query_time": {"date": "DAY1", "time": "10285400"}}) == 1734.9   # EgoRAG's clock
    assert M.ask_time({"response": "A"}) is None


def test_read_answers_downloads_changed_only(tmp_path, monkeypatch):
    """Summaries are cached by content hash: a second cycle downloads nothing, a changed file downloads again, two
    identical files download once, and a file that does not parse summarises as None."""
    import gzip
    p1, p2, p3 = "merit/m8/V1/answers_tb.json.gz", "merit/m8/V2/answers_tb.json.gz", "merit/m27/V1/answers_tb.json.gz"
    body = {p1: gzip.compress(json.dumps([{"key": "k1", "qhash": "h", "answer": "B", "response": "B", "t_ask": 1.5}]).encode()),
            p2: b"not gzip"}
    body[p3] = body[p1]
    got = []

    def download(repo, path, repo_type, revision, local_dir):
        got.append(path)
        f = os.path.join(local_dir, path)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "wb").write(body[path])
        return f
    monkeypatch.setattr(M, "hf_hub_download", download)
    monkeypatch.setattr(M, "ANSWERS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(M, "EXPERIMENTS", [("merit", "m8", "qa", ["p"], "tb")])
    files = {p1: "sha1", p2: "sha2", p3: "sha1", "merit/m8/V1/runs.jsonl": "sha3", "merit/m8/V1/answers_tb_ctx.json.gz": "sha4"}
    first = M.read_answers("c0", files)
    assert len(got) == 2 and p2 in got and first == {p1: [["k1", "h", "B", 1.5, False]], p2: None, p3: [["k1", "h", "B", 1.5, False]]}
    got.clear()
    assert M.read_answers("c1", files) == first and got == []
    files[p1] = "sha5"
    M.read_answers("c2", files)
    assert got == [p1]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))



def test_now_arms_lead_the_list():
    """NOW's arms (MERIT's GPT-5.1 re-runs) come before the stitched tasks, PROBES_NOW and the other arms' probes."""
    mk = lambda b, arm, stage, bank, video="P01-20240202-110250", **kw: dict(baseline=b, arm=arm, stage=stage, bank=bank, video=video, **kw)
    g51_bench, g51_probe = mk("merit", "g5m-g51", "qa", M.BANKS[0]), mk("merit", "g5m-g51", "qa", M.PROBES)
    others = [mk("egagent", "gpt41", "build", None, "P05-stitched"), mk("worldmm", "g5m-g51", "qa", M.PROBES),
              mk("merit", "g5m-q8", "qa", M.PROBES), mk("merit", "g5m-q27", "qa", M.BANKS[0]), mk("magic", "m8", "build", None)]
    assert M.lead(g51_bench) < M.lead(g51_probe) < min(map(M.lead, others))
    assert M.lead(dict(g51_bench, deps=["merit-g5m-build-x"])) > M.lead(g51_probe)   # one waiting on a build keeps its place
