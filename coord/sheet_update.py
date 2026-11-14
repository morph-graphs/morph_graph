#!/usr/bin/env python3
"""Keep one tab of the results Google Sheet in step with the results dataset. A plain script (cron), no LLM.

Reads every answers_<bank>[_<tag>].json.gz, blind_<bank>[_<tag>].json and runs.jsonl of the Hugging Face dataset
(<baseline>/<arm>/<video>/; cached by blob hash, so a run with nothing new downloads nothing), scores each
(baseline, arm, bank, tag) against the CURRENT bank (generated here with epic-morph v3/export_bank.py) and rewrites
the tab "Auto: EPIC-Morph": a header block, then three rows per (baseline, arm, bank, tag), one per `asked`: "end"
(the bank's questions asked at the video's end, ask_at "end"), "streaming" (asked mid-video at t_ask, ask_at "t"),
and "both" (all of them, under the row_key the run had before the split). See coord/SHEET.md.

Scoring. An answer counts only while its question's wording, options and gold letter are those of the current bank
(prepare_epic.match_answers: by key, or by content for answer files from before keys); a failed answer (empty or
"Error") or one that no longer matches is missing. acc_all_pct = right / questions in the bank (missing = wrong);
acc_answered_pct, the 95% interval and the per-leaf columns are over answered questions. Each `asked` row counts
only its own questions (question and answered counts, accuracy, interval, floor, chance, leaves): the answer files
are matched against the whole bank, then cut to them. Streaming answers are provisional (column `provisional`): until
streaming memories exist every method answers them from its full-video memory with retrieval cut at t_ask, and
they will be redone.

The sheet. This script writes to ONE tab, which it creates itself with a sentinel in A1. Every write goes through
send(), which refuses anything but a values PUT into that tab or the request that creates it; put() reads A1 back
before each write and stops when the sentinel is not there. It has no code path that names another tab, inserts or
deletes rows, or deletes a sheet. On failure it writes one ERROR line into B2 (the table of the last good run
stays) and exits 1.

    EPIC_MORPH_DIR=<epic-morph checkout> HF_TOKEN=... python coord/sheet_update.py --dry-run
    EPIC_MORPH_DIR=<epic-morph checkout> HF_TOKEN=... python coord/sheet_update.py --cache <dir>

--json <file> also writes that scoring pass as JSON for the job board's Accuracy tab (coord/collect_live.py reads
it): the table's rows under their column names, and per row the counts behind its leaf columns, by leaf, video,
hops and question form. With --dry-run it is written too (the sheet is still not contacted).
"""
import argparse
import datetime
import glob
import gzip
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.parse import quote, unquote
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "baselines", "merit", "epic"))
sys.path.insert(0, os.path.join(ROOT, "coord"))
from prepare_epic import convert_questions, match_answers  # noqa: E402
import master  # noqa: E402

RESULTS = os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results")
API = "https://sheets.googleapis.com/v4/spreadsheets/1huSbWg8HCiWnaRdX4I3imscqA90vrZ86xB2zAIVbfZs"
TAB = "Auto: EPIC-Morph"                    # the one tab this script writes (--tab replaces it, for tests)
PREFIX = "Auto: "                           # a tab name without it is refused before anything is sent
SENTINEL = "AUTO (coord/sheet_update.py)"   # start of A1 of a tab this script created
GRID = (2000, 80)                           # rows, columns the tab is created with; a larger table is an error
# every bank the job board answers (coord/master.py EXPERIMENTS), in its order: a new bank there is a new set of rows
# here, and the "all" rows (ALL) are over the questions of every one of them
BANKS = list(dict.fromkeys(e[4] for e in master.EXPERIMENTS if e[4]))
ALL = "all"
# a question's ask_at in the bank -> its rows' `asked`; BOTH: every question (the row_key without a suffix)
ASKED = {"end": "end", "t": "streaming"}
BOTH = "both"
ASKED_ROWS = [*ASKED.values(), BOTH]   # a run's rows, in table order
# answers that will be redone: streaming ones come from the full-video memory cut at t_ask until streaming memories
# exist (empty it then)
PROVISIONAL = {"streaming"}
# (baseline, arm) -> Method, Memory Model, Reasoning Model, N. Frames: the rows of the hand tab "Epic Morph Results".
# An arm not listed here is shown as "<baseline>/<arm>".
METHODS = {
    ("merit", "m8"): ("MERIT", "Qwen3-VL-8B", "Qwen3-VL-8B", ""),
    ("merit", "m27"): ("MERIT", "Qwen3.8-27B", "Qwen3.8-27B", ""),
    ("worldmm", "m8"): ("WorldMM", "Qwen3-VL-8B", "Qwen3-VL-8B", ""),
    ("worldmm", "m27"): ("WorldMM", "Qwen3.8-27B", "Qwen3.8-27B", ""),
    ("magic", "m8"): ("MAGIC-Video", "Qwen3-VL-8B", "Qwen3-VL-8B", ""),
    ("magic", "m27"): ("MAGIC-Video", "Qwen3.8-27B", "Qwen3.8-27B", ""),
    ("api", "gemini-3.1-flash-lite-high"): ("Gemini 3.1-Flash-Lite", "--", "Gemini 3.1-Flash-Lite", "video, 1 fps"),
    ("api", "gemini-3.6-flash-high"): ("Gemini 3.6-Flash", "--", "Gemini 3.6-Flash", "video, 1 fps"),
    ("api", "gemini-3.8-flash-high"): ("Gemini 3.8-Flash (bank verifier: score inflated)", "--", "Gemini 3.8-Flash", "video, 1 fps"),
    ("api", "gpt-5-f1024"): ("GPT-5", "--", "GPT-5", 1024),
}
COLS = ["row_key", "method", "memory_model", "reasoning_model", "n_frames", "baseline", "arm", "bank", "tag",
        "videos_done", "videos_started", "videos_in_bank", "q_answered", "q_in_bank", "n_right",
        "acc_all_pct", "acc_answered_pct", "ci95_lo_pct", "ci95_hi_pct", "ci_method",
        "blind_videos", "blind_answered", "blind_right", "blind_acc_all_pct", "blind_acc_answered_pct",
        "chance_pct", "answers_dropped", "note", "gpus", "last_updated_utc",
        "asked", "provisional"]   # then one leaf_<code>_pct per leaf; new columns go last, so none before them moves
KEEP = ("ID", "key", "question", "choices", "choice_a", "choice_b", "choice_c", "choice_d", "answer", "evaluate",
        "pred", "parsed_answer")   # what scoring reads of an answer row (the rest, ~2 MB a video, is not cached)


# ---------------------------------------------------------------------------------------------- dataset and banks
def parse(fname):
    """answers_<bank>[_<tag>].json[.gz] or blind_<bank>[_<tag>].json[.gz] -> (kind, bank, tag); None otherwise."""
    stem = fname[:-3] if fname.endswith(".gz") else fname
    for kind in ("answers", "blind"):
        if stem.startswith(kind + "_") and stem.endswith(".json"):
            rest = stem[len(kind) + 1:-5]
            for b in sorted(BANKS, key=len, reverse=True):
                if rest == b or rest.startswith(b + "_"):
                    return kind, b, rest[len(b) + 1:]
    return None


def fetch(info, cache):
    """{path: content} of the dataset's answers / blind files (slim rows) and runs.jsonl (lines), one cache file
    per blob hash: only files that are new or changed since the last run are downloaded."""
    from huggingface_hub import hf_hub_download
    os.makedirs(cache, exist_ok=True)
    out, keep, new = {}, set(), 0
    for s in info.siblings:
        p = s.rfilename.split("/")
        if len(p) != 4 or not (p[3] == "runs.jsonl" or parse(p[3])):
            continue
        assert s.blob_id, f"no blob id for {s.rfilename}"
        c = os.path.join(cache, f"v2_{s.blob_id}.json")
        keep.add(os.path.basename(c))
        if not os.path.exists(c):
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                f = hf_hub_download(RESULTS, s.rfilename, repo_type="dataset", revision=info.sha, local_dir=tmp)
                if p[3] == "runs.jsonl":
                    data = []
                    for x in open(f):
                        try:
                            data.append(json.loads(x))
                        except ValueError:
                            pass
                    data = [x for x in data if isinstance(x, dict)]
                else:
                    rows = json.load(gzip.open(f, "rt") if f.endswith(".gz") else open(f))
                    rows = rows if isinstance(rows, list) else list(rows.values())
                    data = [{**{k: r[k] for k in KEEP if k in r}, "response": str(r.get("response") or r.get("pred") or r.get("parsed_answer") or "").strip()[:8]}
                            for r in rows if isinstance(r, dict)]
                json.dump(data, open(os.path.join(tmp, "x"), "w"))
                os.replace(os.path.join(tmp, "x"), c)
            new += 1
        out[s.rfilename] = json.load(open(c))
    for f in os.listdir(cache):   # blobs the dataset no longer holds
        if f.startswith(("v1_", "v2_")) and f not in keep:
            os.remove(os.path.join(cache, f))
    print(f"{len(out)} result files, {new} downloaded", flush=True)
    return out


def load_bank(epic, bank):
    """{video: question rows} of the bank as the epic-morph checkout defines it now (regenerated, as coord/master.py
    does: banks are generated, not committed). Rows are prepare_epic's (ID = key, answer = gold letter), plus leaves
    and asked."""
    subprocess.run([sys.executable, "v3/export_bank.py", bank], cwd=epic, check=True, stdout=subprocess.DEVNULL)
    out = {}
    for f in sorted(glob.glob(os.path.join(epic, "outputs", "v3", "banks", bank, "*_questions.json"))):
        qa = convert_questions(f, "x", 0)
        for row, q in zip(qa, json.load(open(f))["questions"]):
            row["leaves"] = list(q.get("taxonomy") or [])
            assert q.get("ask_at") in ASKED, f"{f}: question {row['ID']} has ask_at {q.get('ask_at')!r}, not one of {list(ASKED)}"
            row["asked"] = ASKED[q["ask_at"]]
        if qa:
            out[os.path.basename(f)[:-len("_questions.json")]] = qa
    assert out, f"bank {bank}: no questions under {epic}"
    return out


# ------------------------------------------------------------------------------------------------------- scoring
def score(qa_by_video, rows_by_video):
    """Score one set of answer files against the bank. Per video with a file: {question ID: right?} for the answers
    that still match their question; `dropped` counts every other row (failed, stale, duplicate, not in the bank)."""
    per, dropped = {}, 0
    for v, rows in rows_by_video.items():
        # a keyed row is matched by its key (WorldMM keeps the position in ID); unkeyed rows by ID, then content
        rows = [{**r, "ID": r.get("key") or r.get("ID", "")} for r in rows if r.get("question")]
        got = match_answers(qa_by_video[v], rows) if v in qa_by_video else {}
        per[v] = {i: (str(r["evaluate"]) == "True" if "evaluate" in r
                      else (r.get("pred") or r.get("parsed_answer")) == r.get("answer")) for i, r in got.items()}
        dropped += len(rows_by_video[v]) - len(got)
    return {v: ok for v, ok in per.items() if v in qa_by_video}, dropped


def only(per, qa):
    """score()'s {video: {ID: right?}} cut to the questions of qa (a part of the bank it was scored against)."""
    ids = {v: {str(q["ID"]) for q in rows} for v, rows in qa.items()}
    return {v: {i: ok for i, ok in x.items() if i in ids[v]} for v, x in per.items() if v in ids}


def ci95(per_video):
    """95% interval of right / answered. per_video: [(right, answered)], answered > 0. Percentile bootstrap over
    videos (2,000 resamples, seed 0: the same input always gives the same interval); Wilson with one video."""
    k, n = sum(a for a, _ in per_video), sum(b for _, b in per_video)
    if len(per_video) == 1:
        z, p = 1.959964, k / n
        mid, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return mid - half, mid + half, "Wilson (1 video)"
    rng = random.Random(0)
    stats = []
    for _ in range(2000):
        pick = rng.choices(per_video, k=len(per_video))
        stats.append(sum(a for a, _ in pick) / sum(b for _, b in pick))
    stats.sort()
    return stats[50], stats[1949], f"bootstrap over {len(per_video)} videos"


def pct(k, n):
    return round(100 * k / n, 2) if n else ""


DIMS = ("leaf", "video", "hops", "form")   # what --json breaks a row down by; a question can carry several leaves


def cells(qa, per):
    """{(dimension, category): [answered, right, in bank]} of one row: every question of the bank counted under its
    taxonomy leaves, its video, its hops and its form (qa: {video: bank rows}, per: score()'s {video: {ID: right?}})."""
    out = {}
    for v, rows in qa.items():
        for q in rows:
            ok = per.get(v, {}).get(str(q["ID"]))   # None: no counted answer
            for dim, cats in zip(DIMS, (q["leaves"] or ["(none)"], [v], [q["meta"].get("hops")], [q["meta"].get("form")])):
                for c in cats:
                    x = out.setdefault((dim, "(none)" if c is None else str(c)), [0, 0, 0])
                    x[0] += ok is not None
                    x[1] += bool(ok)
                    x[2] += 1
    return out


def build(a, now):
    """Everything read-only: returns the tab's rows (header block, column names, table) and the same pass as the
    --json object {meta, runs, breakdown}."""
    from huggingface_hub import HfApi
    epic = a.epic_morph or os.environ.get("EPIC_MORPH_DIR")
    if not epic:
        raise RuntimeError("set EPIC_MORPH_DIR (or --epic-morph) to an epic-morph checkout")
    git = lambda *x: subprocess.run(["git", "-C", epic, *x], check=True, capture_output=True, text=True).stdout.strip()
    banks = {b: load_bank(epic, b) for b in BANKS}
    info = HfApi().dataset_info(RESULTS, files_metadata=True)
    files = fetch(info, a.cache)

    sets, runs, skipped = {}, {}, 0   # (baseline, arm, bank, kind, tag) -> {video: rows}; (baseline, arm, video) -> lines
    for path, data in files.items():
        b, arm, v, f = path.split("/")
        if f == "runs.jsonl":
            runs[b, arm, v] = data
        else:
            kind, bank, tag = parse(f)
            sets.setdefault((b, arm, bank, kind, tag), {})[v] = data
    # the "all" rows: each (baseline, arm, kind, tag) over the questions of every bank, videos keyed <bank>/<video>
    # (unanswered = wrong, as in every row, so a method that has answered one bank shows partial)
    banks[ALL] = {f"{bank}/{v}": qa for bank in BANKS for v, qa in banks[bank].items()}
    for (b, arm, bank, kind, tag), by_video in list(sets.items()):
        sets.setdefault((b, arm, ALL, kind, tag), {}).update({f"{bank}/{v}": d for v, d in by_video.items()})
    for s in info.siblings:   # answers of a bank not in BANKS: said in the header, not silently left out
        p = s.rfilename.split("/")
        skipped += len(p) == 4 and p[3].startswith(("answers_", "blind_")) and not parse(p[3])

    leaves = sorted({x for qa in banks.values() for rows in qa.values() for r in rows for x in r["leaves"]})
    tax = json.load(open(os.path.join(epic, "v3", "taxonomy.json")))["leaves"]
    table, breakdown = [], []
    for ((b, arm, bank, kind, tag), by_video), asked in itertools.product(sets.items(), ASKED_ROWS):
        if kind == "blind" and not tag:
            continue   # the arm's no-memory floor: a column of its rows, not a row
        # a tagged blind file is the model answering from a given context (MERIT_CONTEXT): its own row
        tag = f"blind-{tag}" if kind == "blind" else tag
        # the row's questions: those asked that way (BOTH: all); a video with none of them is not in the row
        qa = {v: x for v, rows in banks[bank].items() if (x := [q for q in rows if asked in (BOTH, q["asked"])])}
        if not qa:
            continue
        n_bank = sum(len(x) for x in qa.values())
        # matched against the whole bank, then cut to the row's questions: answers_dropped is the files' own count,
        # the same on the three rows (an answer to a question of the other kind is not dropped)
        per, dropped = score(banks[bank], by_video)
        per = only(per, qa)
        answered, right = sum(len(x) for x in per.values()), sum(sum(x.values()) for x in per.values())
        done = sum(len(per[v]) == len(qa[v]) for v in per)
        pv = [(sum(per[v].values()), len(per[v])) for v in sorted(per) if per[v]]   # sorted: the resampling is by position
        lo, hi, how = ci95(pv) if pv else ("", "", "")
        floor = only(score(banks[bank], sets.get((b, arm, bank, "blind", ""), {}))[0], qa)
        f_ans, f_right = sum(len(x) for x in floor.values()), sum(sum(x.values()) for x in floor.values())
        # runs.jsonl lines of these videos: the memory's stages (no bank named) and this bank's
        # (an "all" row: each video's own bank's, from its key <bank>/<video>)
        own = lambda v: v.split("/")[0] if bank == ALL else bank
        lines = [x for v in by_video for x in runs.get((b, arm, v.split("/")[-1]), [])
                 if not any(o in f"{x.get('stage', '')} {x.get('bank', '')}".split() for o in BANKS if o != own(v))]
        notes = list(dict.fromkeys(str(x["note"]) for x in lines if x.get("note")))
        if done < len(qa):
            notes.insert(0, f"partial: {done}/{len(qa)} videos done" + (f", {len(per) - done} more started" if len(per) > done else ""))
        f_done = sum(len(floor[v]) == len(qa[v]) for v in floor)
        if floor and f_done < len(qa):
            notes.append(f"blind floor from {f_done}/{len(qa)} videos")
        stray = [v for v in by_video if v not in banks[bank]]
        if stray:
            notes.append(f"{len(stray)} answer files of videos not in the bank")
        gpus = sorted({str(g).replace("NVIDIA ", "") for x in lines for g in (x.get("gpus") or [])})
        times = []
        for x in lines:
            try:
                times.append(datetime.datetime.fromisoformat(x["finished"]).astimezone(datetime.timezone.utc))
            except (KeyError, TypeError, ValueError):
                pass
        key = f"{b}/{arm}/{bank}" + (f"/{tag}" if tag else "") + ("" if asked == BOTH else f"/{asked}")
        by = cells(qa, per)
        # over answered questions carrying the leaf (a question can carry several)
        leaf = [pct(by[x][1], by[x][0]) if x in by else "" for x in (("leaf", code) for code in leaves)]
        # the counts behind them, for --json; an "all" row's videos are its banks' rows', so not repeated
        breakdown += [dict(run=key, bank=bank, asked=asked, dimension=dim, category=cat,
                           label=tax[cat]["name"] if dim == "leaf" and cat in tax else cat,
                           n_answered=n, n_right=k, n_in_bank=m)
                      for (dim, cat), (n, k, m) in sorted(by.items(), key=lambda kv: (DIMS.index(kv[0][0]), kv[0][1]))
                      if not (bank == ALL and dim == "video")]
        table.append([key, *METHODS.get((b, arm), (f"{b}/{arm}", "", "", "")),
                      b, arm, bank, tag, done, len(per), len(qa), answered, n_bank, right,
                      pct(right, n_bank), pct(right, answered), *(round(100 * x, 2) if x != "" else "" for x in (lo, hi)), how,
                      f_done if floor else "", f_ans if floor else "", f_right if floor else "",
                      pct(f_right, n_bank) if floor else "", pct(f_right, f_ans) if floor else "",
                      round(100 * sum(1 / q["meta"]["n_options"] for x in qa.values() for q in x) / n_bank, 2),
                      dropped, "; ".join(notes), ", ".join(gpus),
                      max(times).strftime("%Y-%m-%d %H:%M") if times else "",
                      asked, any(q["asked"] in PROVISIONAL for x in qa.values() for q in x), *leaf])
    table.sort(key=lambda r: ((BANKS + [ALL]).index(r[7]), r[5], r[6], r[8], ASKED_ROWS.index(r[COLS.index("asked")])))
    head = [
        [f"{SENTINEL}: written by coord/sheet_update.py, do not edit. Rewritten on every run; look rows up by row_key."],
        ["status", "OK"],
        ["last run (UTC)", now.strftime("%Y-%m-%d %H:%M:%S")],
        ["last run (US Eastern)", now.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")],
        ["results dataset revision", f"{RESULTS}@{info.sha} (last modified {info.last_modified:%Y-%m-%d %H:%M} UTC)"
         + (f"; {skipped} answer files of banks not in {BANKS} left out" if skipped else "")],
        ["epic-morph commit of the banks", git("log", "-1", "--format=%H (%cI)")],
        ["rows", len(table)],
        ["conventions", "Percent values (36.29 means 36.29%). acc_all_pct = n_right / q_in_bank (unanswered = wrong). "
         "acc_answered_pct, ci95 and leaf_* are over answered questions; ci95: percentile bootstrap over videos "
         "(2,000 resamples, seed 0), Wilson when one video. An answer counts only while its question's wording, options "
         "and gold letter match the current bank; other rows are in answers_dropped. blind_*: the arm's no-memory floor "
         "(blind_<bank>.json), same rules. chance_pct: mean of 1 / number of options. videos_done: every question answered. "
         "asked: each run has an end row (questions asked at the video's end; row_key ends /end), a streaming row "
         "(asked mid-video at t_ask; /streaming) and a both row (all its questions; the row_key without a suffix), each "
         "counting only its own questions; answers_dropped is the run's, the same on the three. provisional = TRUE: the "
         "row holds streaming answers, which until streaming memories exist come from the full-video memory with "
         "retrieval cut at t_ask and will be redone."],
        ["taxonomy leaves", "; ".join(f"{c} = {tax[c]['name']}" if c in tax else c for c in leaves)],
        COLS + [f"leaf_{c}_pct" for c in leaves],
    ]
    # --json: the table's own rows under their column names (an empty cell is null; the leaf columns are in
    # breakdown as counts), so the board shows the sheet's numbers and nothing is scored twice
    runs = [{k: (None if v == "" else v) for k, v in zip(COLS, r)} for r in table]
    for r in runs:
        r["partial"] = r["videos_done"] < r["videos_in_bank"]
    meta = dict(written_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"), dataset=RESULTS, dataset_revision=info.sha,
                dataset_last_modified_utc=f"{info.last_modified:%Y-%m-%dT%H:%M:%SZ}",
                bank_commit=git("log", "-1", "--format=%H"), banks=BANKS + [ALL], asked=ASKED_ROWS, dimensions=list(DIMS),
                intervals="runs: ci95_lo_pct / ci95_hi_pct as in the sheet (ci_method); breakdown: counts only, "
                          "the reader computes its interval (the board page: Wilson)")
    return head + table, dict(meta=meta, runs=runs, breakdown=breakdown)


# ------------------------------------------------------------------------------------------------------ the sheet
def a1(cells=""):
    return "'" + TAB.replace("'", "''") + "'!" + cells


def send(s, method, url, **kw):
    """Every HTTP call to the sheet. Anything but a GET must be a values PUT whose range is in the owned tab, or the
    one request that creates that tab (addSheet + its A1 sentinel, atomic); everything else is refused here."""
    if method != "GET":
        body = kw.get("json") or {}
        if method == "POST" and url == API + ":batchUpdate":
            r = body.get("requests", [])
            ok = (len(r) == 2 and list(r[0]) == ["addSheet"] and list(r[1]) == ["updateCells"]
                  and r[0]["addSheet"]["properties"]["title"] == TAB
                  and r[1]["updateCells"]["start"] == {"sheetId": r[0]["addSheet"]["properties"]["sheetId"], "rowIndex": 0, "columnIndex": 0})
        else:
            ok = method == "PUT" and unquote(url).startswith(f"{API}/values/{a1()}") and str(body.get("range")).startswith(a1())
        assert ok and TAB.startswith(PREFIX), f"refusing a write outside the owned tab {TAB!r}: {method} {unquote(url)}"
    for attempt in range(4):
        r = s.request(method, url, timeout=60, **kw)
        if r.status_code not in (429, 500, 502, 503) or attempt == 3:
            break
        time.sleep(5 * 2 ** attempt)
    if r.status_code != 200:
        raise RuntimeError(f"Sheets {method} {unquote(url)[len(API):][:80]} -> HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def open_tab(s):
    """Create the tab when the sheet has none of that name, with the sentinel in A1 in the same atomic request.
    Returns its grid size (rows, columns)."""
    props = {x["properties"]["title"]: x["properties"] for x in send(s, "GET", API, params={"fields": "sheets.properties"})["sheets"]}
    if TAB in props:
        g = props[TAB]["gridProperties"]
        return g["rowCount"], g["columnCount"]
    sid = max(p["sheetId"] for p in props.values()) + 1
    send(s, "POST", API + ":batchUpdate", json={"requests": [
        {"addSheet": {"properties": {"title": TAB, "sheetId": sid, "index": len(props),
                                     "gridProperties": {"rowCount": GRID[0], "columnCount": GRID[1], "frozenRowCount": 10}}}},
        {"updateCells": {"start": {"sheetId": sid, "rowIndex": 0, "columnIndex": 0}, "fields": "userEnteredValue",
                         "rows": [{"values": [{"userEnteredValue": {"stringValue": SENTINEL}}]}]}}]})
    print(f"created tab {TAB!r}", flush=True)
    return GRID


def put(s, cells, values):
    """The only write of values: into the owned tab, after reading its A1 back. RAW: numbers stay numbers and no
    text (a note from runs.jsonl) is ever taken for a formula."""
    top = send(s, "GET", f"{API}/values/{quote(a1('A1'), safe='')}").get("values") or [[""]]
    if not str((top[0] or [""])[0]).startswith(SENTINEL):
        raise SystemExit(f"tab {TAB!r} exists but was not created by this script (A1 lacks {SENTINEL!r}): not writing")
    rng = a1(cells)
    out = send(s, "PUT", f"{API}/values/{quote(rng, safe='')}", params={"valueInputOption": "RAW"},
               json={"range": rng, "majorDimension": "ROWS", "values": values})
    print(f"wrote {out.get('updatedRange')}: {out.get('updatedCells')} cells", flush=True)


def col(n):
    return (col((n - 1) // 26) if n > 26 else "") + chr(65 + (n - 1) % 26)


def main():
    global TAB
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print the tab's rows, write nothing to the sheet (it is not contacted)")
    ap.add_argument("--json", help="also write the scores to this file as JSON (runs and their breakdown), for the job board; with --dry-run too")
    ap.add_argument("--tab", default=TAB, help="tests only: another tab of this script's (name starting with 'Auto: ')")
    ap.add_argument("--epic-morph", help="epic-morph checkout the banks are generated in (default $EPIC_MORPH_DIR)")
    ap.add_argument("--cache", default=os.path.join(ROOT, "coord", ".sheet_cache"), help="slim copies of the result files, by blob hash")
    ap.add_argument("--sa-key", default=os.path.expanduser("~/.config/gsheet_sa.json"), help="service account key file")
    a = ap.parse_args()
    if not a.tab.startswith(PREFIX):
        raise SystemExit(f"--tab must start with {PREFIX!r}: this script never writes to a hand-maintained tab")
    TAB = a.tab
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        (rows, acc), err = build(a, now), None
    except Exception as e:   # anything: the tab must say that it is stale
        traceback.print_exc()
        rows, err = [], f"ERROR {now:%Y-%m-%d %H:%M} UTC: {type(e).__name__}: {' '.join(str(e).split())[:300]}"
    if a.json and not err:   # one line, replaced whole; after a failed run the last good file stays (its written_utc ages)
        try:
            with open(a.json + ".tmp", "w") as f:
                json.dump(acc, f)
            os.replace(a.json + ".tmp", a.json)
            print(f"wrote {a.json}: {len(acc['runs'])} runs, {len(acc['breakdown'])} breakdown rows", file=sys.stderr, flush=True)
        except OSError:   # the sheet is still written: the board shows its file as stale
            traceback.print_exc()
    if a.dry_run:
        print(err or "\n".join("\t".join(str(x) for x in r) for r in rows))
        sys.exit(1 if err else 0)

    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.service_account import Credentials
    s = AuthorizedSession(Credentials.from_service_account_file(a.sa_key, scopes=["https://www.googleapis.com/auth/spreadsheets"]))
    grid = open_tab(s)
    width = max((len(r) for r in rows), default=0)
    if not err and (len(rows) > grid[0] or width > grid[1]):
        err = f"ERROR {now:%Y-%m-%d %H:%M} UTC: the table ({len(rows)} x {width}) does not fit the tab's grid {grid}"
    if err:
        put(s, "A2:B2", [["status", err + " | the table below is from the last successful run"]])
        print(err, file=sys.stderr)
        sys.exit(1)
    # one write of the whole rectangle, padded with empty cells over whatever a longer or wider earlier table left
    old = send(s, "GET", f"{API}/values/{quote(a1('A1:' + col(grid[1]) + str(grid[0])), safe='')}").get("values", [])
    n, w = max(len(old), len(rows)), max([width] + [len(r) for r in old])
    put(s, f"A1:{col(w)}{n}", [list(r) + [""] * (w - len(r)) for r in rows] + [[""] * w] * (n - len(rows)))
    print(f"{len(rows) - 10} rows", flush=True)


if __name__ == "__main__":
    main()
