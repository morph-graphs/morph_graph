#!/usr/bin/env python3
"""The stitched videos' answers files -> Vatsal's per-card predictions, stitched/eov_results/<system>/<K>.jsonl.

Vatsal's hand-off asks each system's predictions on his stitched bank (epic-morph `stitched`, 630 questions on the 9
<K>-stitched videos) as one jsonl per kitchen: the card key, the chosen option index (into the card's
gemini_mcq.options), the rationale when the system gives one; plus a note of the system's memory or frame budget
(note.json). This reads the baselines' own answers files, <baseline>/<arm>/<K>-stitched/answers_<bank>.json.gz (from
--src, a folder in the results dataset's layout, or fetched one file each from the dataset with --hf), checks each row
against the bank (key, wording and options), and writes --out/<system>/<K>.jsonl. No model is called, nothing is uploaded.

    python baselines/common/stitched_results.py --hf --out <dir> [--systems merit/m27 worldmm/g5m-g51 \
        api/gemini-3.6-flash-high-stitched-f2048] [--bank stitched]    # needs EPIC_MORPH_DIR (the bank)
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys

LETTERS = "ABCD"
RESULTS = os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results")
SYSTEMS = ["merit/m27", "worldmm/g5m-g51", "api/gemini-3.6-flash-high-stitched-f2048"]
# what each system answers from (README of each baseline; budgets as run on the board)
NOTES = {
    "merit": "MERIT: one memory per kitchen over the whole stitched video (30 s clips, captions and retrieval keys by the "
             "arm's model; arm m27: Qwen3.8-27B writes and answers). Each card is answered at its t_ask (Query Time DAY1 "
             "HH:MM:SS, floored) from the clips that end by then: up to 5 search rounds of top-5 clips with +/-2 "
             "neighbours, 6 frames per clip in round 1. The kitchen's sessions started by t_ask are in the question text.",
    "worldmm": "WorldMM: one memory per kitchen over the whole stitched video (10 s / 30 s / 3 min / 10 min captions, "
               "semantic snapshots, visual clip embeddings; arm g5m-*: written by gpt-5-mini). Cards are answered in t_ask "
               "order on one index that only grows: each sees the captions, the latest semantic snapshot and the visual "
               "clips that end by its t_ask (floored); top-k episodic 3, semantic 10, visual 3, up to 5 rounds. Arm "
               "g5m-g51 answers with GPT-5.1 (stand-in for GPT-5). The kitchen's sessions started by t_ask precede the "
               "question.",
    "api": "Long context, no memory: the stitched video cut at t_ask, sampled uniformly to the config's frame budget from a "
           "1 fps grid (frames strictly before t_ask), one request per card, the kitchen's sessions started by t_ask "
           "before the question. Per-card frames are in the rows (n_frames, frame_budget).",
}


def bank(epic, name):
    d = os.path.join(epic, "outputs", "v3", "banks", name)
    if not os.path.isdir(d):   # `stitched` is committed in epic-morph; another bank is generated
        subprocess.run([sys.executable, "v3/export_bank.py", name], cwd=epic, check=True, stdout=subprocess.DEVNULL)
    return {f[:-len("_questions.json")]: json.load(open(os.path.join(d, f)))["questions"]
            for f in sorted(os.listdir(d)) if f.endswith("_questions.json")}


def letter(r):
    """The letter a row chose: WorldMM's pred, MERIT's parsed_answer, else the response's leading letter (API rows'
    response is the letter)."""
    for x in (r.get("pred"), r.get("parsed_answer"), r.get("response")):
        m = re.match(r"\s*\(?([A-D])\b", str(x or ""))
        if m:
            return m.group(1)
    return None


def rationale(r):
    return r.get("reasoning") or r.get("thoughts") or (r.get("response") if len(str(r.get("response") or "")) > 3 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", help="a folder in the results dataset's layout (<baseline>/<arm>/<video>/answers_<bank>.json.gz)")
    ap.add_argument("--hf", action="store_true", help="fetch the answers files from the results dataset instead")
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, help="<baseline>/<arm>")
    ap.add_argument("--bank", default="stitched")
    ap.add_argument("--epic", default=os.environ.get("EPIC_MORPH_DIR"))
    a = ap.parse_args()
    if not (a.src or a.hf) or not a.epic:
        ap.error("give --src or --hf, and --epic (or EPIC_MORPH_DIR)")
    qs = bank(a.epic, a.bank)
    for system in a.systems:
        b, arm = system.split("/")
        name = arm if b == "api" else f"{b}-{arm}"
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        done, budgets = {}, {}
        for v, questions in qs.items():
            rel = f"{b}/{arm}/{v}/answers_{a.bank}.json.gz"
            if a.hf:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.errors import EntryNotFoundError
                try:
                    path = hf_hub_download(RESULTS, rel, repo_type="dataset")
                except EntryNotFoundError:
                    path = None
            else:
                path = os.path.join(a.src, rel)
            if not path or not os.path.exists(path):
                print(f"{name} {v}: no answers file ({rel})")
                continue
            rows = {r.get("key"): r for r in json.load(gzip.open(path, "rt"))}
            out = []
            for q in questions:
                r = rows.get(q["key"])
                if r is None:
                    continue
                opts = [str(o) for o in q["gemini_mcq"]["options"]]
                assert [r["choices"][k] for k in sorted(r["choices"])] == opts, f"{rel} {q['key']}: options differ from the bank"
                assert r["question"].endswith(q["q"]), f"{rel} {q['key']}: wording differs from the bank"
                x = letter(r)
                idx = LETTERS.index(x) if x and LETTERS.index(x) < len(opts) else None
                out.append({"key": q["key"], "pred_index": idx, "rationale": rationale(r),
                            "correct": idx == q["gemini_mcq"]["answer_index"], "t_ask": q["t_ask"],
                            **({k: r[k] for k in ("n_frames", "frame_budget") if k in r})})
                if "frame_budget" in r:
                    budgets.setdefault(v, set()).add(r["frame_budget"])
            k = v.split("-")[0]
            with open(os.path.join(a.out, name, f"{k}.jsonl"), "w") as f:
                f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
            done[k] = {"cards": len(questions), "answered": len(out), "right": sum(o["correct"] for o in out)}
            print(f"{name} {k}: {len(out)}/{len(questions)} cards, {done[k]['right']} right")
        note = {"system": name, "baseline": b, "arm": arm, "bank": a.bank, "memory": NOTES.get(b),
                "frame_budget": {k.split("-")[0]: sorted(x) for k, x in budgets.items()} or None, "kitchens": done}
        json.dump(note, open(os.path.join(a.out, name, "note.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
