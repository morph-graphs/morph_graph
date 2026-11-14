#!/usr/bin/env bash
# The official arms (g5m, g5m-g5, g5m-q8) end to end on a slice of one real video, with no API call: the OpenAI
# client talks to epic/mock_openai.py (OPENAI_BASE_URL), everything else is the real pipeline (frames, embedding
# models, HippoRAG, the strict-streaming patches, run_worldmm.sh's resume and runs.jsonl). One GPU.
#
#   PY=<client python> EPIC_MORPH_DIR=<epic-morph> bash baselines/worldmm/test_official.sh <scratch dir> [video]
#
#   SLICE_S    seconds of the video to caption (default 420: the test video's first questions are asked at 290
#              and 297 s, so captions and frames after t_ask exist and must not be used)
#   QUESTIONS  how many of the video's earliest batch10_agree2 questions to answer (default 3)
#   Q8         1 (default) also answers with g5m-q8, the respond model being the real local Qwen3-VL-8B
#              (needs a GPU with >= 40 GB); 0 skips it
#   GATEWAY    openai (default): OPENAI_API_KEY, calls on /v1/responses with WorldMM's snapshots and no `provider`
#              field (an OPENROUTER_API_KEY that is also set is ignored). openrouter: only OPENROUTER_API_KEY is
#              set and the stand-in answers as OpenRouter under /openrouter/api/v1: openai/gpt-5-mini and
#              openai/gpt-5, pinned to provider OpenAI without fallbacks, every image with detail auto
# Prints the calls per pipeline step and their sizes, the runs.jsonl usage, and checks that no caption, triple or
# frame that ends after a question's t_ask reached a retriever or respond call. Exits non-zero when a check fails.
set -euo pipefail
T=$(mkdir -p "${1:?scratch dir}" && cd "$1" && pwd); V=${2:-P05-20240425-201444}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REAL=$(cd "${EPIC_MORPH_DIR:?}" && pwd)
PY=${PY:?the client python}; SLICE_S=${SLICE_S:-420}; NQ=${QUESTIONS:-3}
export WORLDMM_WORK=$T/work WORLDMM_OUT=$T/results WORLDMM_FETCH=0 HDEPIC_VIDEOS=${HDEPIC_VIDEOS:-$REAL/data/HD-EPIC/videos}
LOG=$T/mock_calls.jsonl; : > "$LOG"
PORT=$("$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$PY" "$M/epic/mock_openai.py" --port "$PORT" --log "$LOG" & MOCK=$!
trap 'kill $MOCK 2>/dev/null' EXIT
# both stand-in URLs are always set; the keys decide the gateway
export OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1 OPENROUTER_BASE_URL=http://127.0.0.1:$PORT/openrouter/api/v1
KEY=sk-mock-not-a-real-key; GATEWAY=${GATEWAY:-openai}; unset OPENAI_API_KEY
case $GATEWAY in
    openai) export OPENAI_API_KEY=$KEY OPENROUTER_API_KEY=sk-or-mock-must-not-be-used ;;
    openrouter) export OPENROUTER_API_KEY=$KEY ;;
    *) echo "GATEWAY must be openai or openrouter"; exit 1 ;;
esac
sleep 3
# a bank of the video's NQ earliest questions, in a stand-in epic-morph folder (run_worldmm.sh reads banks there)
export EPIC_MORPH_DIR=$T/epic; mkdir -p "$EPIC_MORPH_DIR/outputs/v3/banks/mock"
"$PY" - "$REAL/outputs/v3/banks/batch10_agree2/${V}_questions.json" "$EPIC_MORPH_DIR/outputs/v3/banks/mock/${V}_questions.json" "$NQ" <<'PYEOF'
import json, sys
bank = json.load(open(sys.argv[1]))
bank["questions"] = sorted((q for q in bank["questions"] if q.get("streaming") and "t_ask" in q), key=lambda q: q["t_ask"])[:int(sys.argv[3])]
json.dump(bank, open(sys.argv[2], "w"))
print("questions asked at", [q["t_ask"] for q in bank["questions"]])
PYEOF
step() { echo "{\"kind\": \"#step\", \"name\": \"$1\"}" >> "$LOG"; echo; echo "##### $1"; }
step prep; bash "$M/run_worldmm.sh" prep "$V" g5m
SKEL=$WORLDMM_WORK/skeleton/$V/10sec_skeleton.json   # the slice: only the first SLICE_S seconds get a memory
"$PY" -c 'import json, sys; s = [x for x in json.load(open(sys.argv[1])) if x["end_seconds"] <= int(sys.argv[2])]; json.dump(s, open(sys.argv[1], "w")); print(len(s), "segments kept")' "$SKEL" "$SLICE_S"
step build;        bash "$M/run_worldmm.sh" build "$V" g5m
step qa-g5m-g5;    bash "$M/run_worldmm.sh" qa "$V" g5m-g5 mock
step blind-g5m-g5; bash "$M/run_worldmm.sh" blind "$V" g5m-g5 mock
if [ "${Q8:-1}" = 1 ]; then step qa-g5m-q8; bash "$M/run_worldmm.sh" qa "$V" g5m-q8 mock; fi
step rerun;        bash "$M/run_worldmm.sh" build "$V" g5m-g5; bash "$M/run_worldmm.sh" qa "$V" g5m-g5 mock
echo; echo "##### checks"
if grep -rIl "$KEY" "$T" >/dev/null; then echo "FAIL: the key is in a file under $T:"; grep -rIl "$KEY" "$T"; exit 1; fi
echo "the key is in no file under $T"
"$PY" - "$T" "$V" "$GATEWAY" "$PORT" <<'PYEOF'
import collections, glob, gzip, json, math, re, sys
T, V, GATEWAY, PORT = sys.argv[1:]
MARK = re.compile(r"CAP<(\d+)-(\d+)>|tm(\d+)x(\d+)")
RANGE = re.compile(r"DAY1 (\d\d):(\d\d):(\d\d) - DAY1 (\d\d):(\d\d):(\d\d)$")   # a visual search by time range
fail = []
# 1. calls per step and kind, as the mock saw them
step, calls = None, collections.defaultdict(list)
log = [json.loads(l) for l in open(f"{T}/mock_calls.jsonl")]
for r in log:
    if r["kind"] == "#step":
        step = r["name"]
    else:
        calls[(step, r["kind"], r["model"])].append(r)
print(f"{'step':13} {'call':22} {'model':24} {'calls':>5} {'text tok':>9} {'images':>6} {'image tok':>9} {'out tok':>7}   (means per call)")
for (s, k, m), rs in calls.items():
    n = len(rs)
    print(f"{s:13} {k:22} {m:24} {n:5} {sum(r['text_tokens'] for r in rs) / n:9.0f} {sum(r['images'] for r in rs) / n:6.1f} "
          f"{sum(r['image_tokens'] for r in rs) / n:9.0f} {sum(r['output_tokens'] for r in rs) / n:7.0f}")
if any(s == "rerun" for s, _, _ in calls):
    fail.append("the rerun made API calls: finished work was not skipped")
if any(r.get("auth") != "bearer" for r in log if r["kind"] != "#step"):
    fail.append("a call came without the key header")
# 1b. the gateway: every call on its path, with its model ids, provider pin and image detail
gw = {"openai": ("/v1/responses", {"gpt-5-mini-2025-08-07", "gpt-5-2025-08-07"}, None, [None]),
        "openrouter": ("/openrouter/api/v1/responses", {"openai/gpt-5-mini", "openai/gpt-5"},
                       {"only": ["openai"], "allow_fallbacks": False}, ["auto"])}[GATEWAY]
real = [r for r in log if r["kind"] != "#step"]
bad = [r for r in real if r["path"] != gw[0] or r["model"] not in gw[1] or r["provider"] != gw[2]
       or (r["images"] and r["detail"] != gw[3])]
print(f"gateway {GATEWAY}: {len(real)} calls, paths {sorted({r['path'] for r in real})}, models {sorted({r['model'] for r in real})}, "
      f"provider {sorted({json.dumps(r['provider']) for r in real})}, image detail {sorted({str(r['detail']) for r in real if r['images']})}")
if bad or not real:
    fail.append(f"{len(bad)} calls did not go the {GATEWAY} way, e.g. {bad[:1]}")
# 2. runs.jsonl: usage per step / role and the model ids the API answered with
for f in sorted(glob.glob(f"{T}/results/*/{V}/runs.jsonl")):
    for line in open(f):
        r = json.loads(line)
        print(f.split("/")[-3], "runs.jsonl:", json.dumps({k: r.get(k) for k in ("stage", "model", "respond_model", "openai")}))
build = [json.loads(l) for l in open(f"{T}/results/g5m/{V}/runs.jsonl")]
used = next(r["openai"] for r in build if r["stage"] == "build")
n_log = sum(len(rs) for (s, _, _), rs in calls.items() if s == "build")
n_use = sum(m["calls"] for st in used.values() for m in st.values())
print(f"build: {n_log} calls at the mock, {n_use} in runs.jsonl")
if n_log != n_use:
    fail.append("runs.jsonl's call count differs from the mock's")
urows = [m for st in used.values() for m in st.values()]
gok = all(m["endpoint"].rstrip("/") == f"http://127.0.0.1:{PORT}" + gw[0][:-len("/responses")] for m in urows)
if GATEWAY == "openrouter":
    gok = gok and all(m.get("provider") == "OpenAI" and abs(m["cost"] - 0.001 * m["calls"]) < 1e-6 for m in urows)
else:
    gok = gok and not any("provider" in m or "cost" in m for m in urows)
if not gok:
    fail.append("runs.jsonl's endpoint, provider or cost is not the gateway's")
# 3. streaming: nothing that ends after t_ask in the rows (both arms) or in any call of the all-OpenAI arm
caps = json.load(open(f"{T}/results/g5m/{V}/{V}_10sec.json"))
end = lambda c: int(c["end_time"][:2]) * 3600 + int(c["end_time"][2:4]) * 60 + int(c["end_time"][4:6])
t_of, ranged = {}, {}
for arm in ("g5m-g5", "g5m-q8"):
    files = glob.glob(f"{T}/results/{arm}/{V}/answers_mock.json.gz")
    if not files:
        continue
    rows = json.load(gzip.open(files[0], "rt"))
    for r in rows:
        t = math.floor(r["t_ask"])
        t_of[r["question"]] = t
        future = sum(end(c) > t for c in caps)
        seen = [int(b or d) for h in r["round_history"] for _, b, _, d in MARK.findall(h["retrieved_content"])]
        # a visual search by time range is recorded with the range that was asked, not the frames returned (the fork
        # cuts those at the indexed time): its frames are counted at the answer call below instead
        spans = [s[1] for x in r["retrieved"] for s in x["spans"] if not RANGE.match(x["query"])]
        if arm == "g5m-g5":
            ranged[r["question"]] = [x["query"] for x in r["retrieved"] if x["memory"] == "visual"]
        want = sum(end(c) <= t for c in caps)
        ok = max(seen + spans, default=0) <= t and r["indexed"]["10sec"] == want and r["indexed"]["visual"] == want
        print(f"{arm} ID {r['ID']} t_ask {r['t_ask']}: indexed 10sec {r['indexed']['10sec']} visual {r['indexed']['visual']} of {len(caps)} "
              f"({future} captions end later); latest second in retrieved text {max(seen, default=None)}, in retrieved spans "
              f"{max(spans, default=None)}; searches {[(h['memory_type'], h.get('retrieved_clips') or len(h['retrieved_content'])) for h in r['round_history']]}"
              f" -> {'ok' if ok else 'LEAK'}")
        if not ok:
            fail.append(f"{arm} ID {r['ID']}: something after t_ask was indexed or retrieved")
    print(arm, "row:", json.dumps({k: (v if k != "round_history" else "[...]") for k, v in rows[0].items()})[:900])
step, t, worst = None, None, collections.Counter()
for r in log:
    if r["kind"] == "#step":
        step = r["name"]
        continue
    if step != "qa-g5m-g5":
        continue
    if r["query"]:
        t = t_of[r["query"]]
    worst[r["kind"]] += 1
    if t is None:
        fail.append(f"qa-g5m-g5: a {r['kind']} call before any question")
    elif r["max_end"] is not None and r["max_end"] > t:
        fail.append(f"qa-g5m-g5: a {r['kind']} call carried second {r['max_end']} for a question asked at {t}")
    if r["kind"] == "answer":
        # frames: 3 clips x 11 for a visual search by text; by time range at most 11 per 10 s up to t_ask
        allowed = 0
        for query in ranged[r["query"]]:
            m = RANGE.match(query)
            if m:
                a, b = (int(m[i]) * 3600 + int(m[i + 1]) * 60 + int(m[i + 2]) for i in (1, 4))
                allowed += max(0, math.ceil((min(b, t) - a) / 10) + 1) * 11
            else:
                allowed += 33
        if r["images"] > allowed:
            fail.append(f"qa-g5m-g5: the answer call for t_ask {t} carried {r['images']} frames, more than the {allowed} that exist by then")
        print(f"  (at most {allowed} frames exist by t_ask for its visual searches {ranged[r['query']]})")
        print(f"answer call for t_ask {t}: {r['images']} frames {r['image_size']}, {r['text_tokens']} text + {r['image_tokens']} image tokens, latest second in its text {r['max_end']}")
print("qa-g5m-g5 calls checked against t_ask:", dict(worst))
print("FAILED:\n  " + "\n  ".join(fail) if fail else "ALL CHECKS PASSED")
sys.exit(1 if fail else 0)
PYEOF
