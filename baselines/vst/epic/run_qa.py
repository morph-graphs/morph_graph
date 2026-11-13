#!/usr/bin/env python3
"""Answer one video's questions of an EPIC-Morph bank with VST (Video Streaming Thinking), streaming.

VST's inference is its lmms-eval model class (VST/eval/lmms-eval/lmms_eval/models/simple/
qwen2_5_vl_stream_think.py). This script uses that class for the model, the video loader and the frame budget,
and repeats its generate_until loop outside the harness, one cutoff at a time:
  - a question asked at t_ask gets the video cut at t_ask (the loader's video_end: frames with pts <= t_ask),
    as VST's streaming benchmarks hand it videos cut at the question time;
  - the cut video is sampled at 2 fps up to --max_frames (above that, uniformly over the whole cut), resized
    to the token budget and split into equal segments (2 / 3 / 5 / 5 for <= 0.5 / < 4 / < 30 / >= 30 min with
    the default --think_times);
  - the model writes a streaming thought for every segment but the last, each time reading the earlier
    thoughts as text; then it answers from all thoughts + the last segment's frames + the question.
The thoughts depend on the cutoff but not on the question: they are cached per cutoff in <out>/thoughts/ and
shared by every question (of any bank) asked at that time.

--blind: the class's text-only path (system prompt, empty thought memory, the question): the no-video floor.

<out>/<video>_settings.json holds everything the thoughts and answers depend on; it is written on first use
and a run with other settings refuses to write into the same folder. --settings_only just writes it.

Writes --answers (.json.gz): one row per question in bank order, keyed by `key`. Resumes from it: a row counts
as done when it has a valid answer from this VERSION, these settings (memory_hash) and the same question
(qhash). Exit code 2 when some questions are still unanswered (an error, e.g. CUDA out of memory): run again.

    python run_qa.py --video <V> --video_file <V>.mp4 --bank_file <V>_questions.json --model <VST-7B> \
        --out <results folder> --answers <results folder>/answers_<bank>.json.gz
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
import re
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "VST", "eval", "lmms-eval"), os.path.join(HERE, "..", "..", "merit", "epic")]

VERSION = "2"   # bump when a change alters answers (prompt, loop, parsing)
POST_PROMPT = "Answer with the option's letter from the given choices directly."   # VST's OVO-Bench / StreamingBench tasks
THINK_PROMPT = "[System]\nYou are a Streaming Video Analyst.\n"                     # generate_until's STREAM_THINK_PROMPT
FAILED = ("", "Error")


def settings(a):
    return {"version": VERSION, "model": a.model_name, "think_times": a.think_times, "vid_tokens": a.vid_tokens,
            "max_frames": a.max_frames, "attn": a.attn, "post_prompt": POST_PROMPT, "max_new_tokens": 16}


def check_settings(a):
    """Write <out>/<video>_settings.json, or stop when the folder was made with other settings."""
    s = json.dumps(settings(a), indent=1, sort_keys=True) + "\n"
    path = os.path.join(a.out, f"{a.video}_settings.json")
    if os.path.exists(path):
        if open(path).read() != s:
            sys.exit(f"{path} was made with other settings:\n{open(path).read()}now:\n{s}use another results folder")
    else:
        os.makedirs(a.out, exist_ok=True)
        open(path, "w").write(s)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def question_text(row):
    """VST's ovobench_doc_to_text: question, lettered options one per line, the post prompt."""
    opts = "".join(f"{k}. {row[f'choice_{k.lower()}']}\n" for k in "ABCD" if f"choice_{k.lower()}" in row)
    return row["question"] + "\n" + opts + POST_PROMPT


def pick(response, row):
    """The option letter in the model's reply: the first standalone letter among the question's options. An
    "A" followed by a lowercase word is the article ("A person picks ..."), not the option, unless that word
    starts option A's own text or is "is" / "because". A reply with no letter is unparsed (pred None, counted
    wrong, the raw response kept)."""
    letters = "".join(k for k in "ABCD" if f"choice_{k.lower()}" in row)
    s = response.strip()
    for m in re.finditer(rf"\b([{letters}])\b", s):
        rest = s[m.end():]
        if m.group(1) == "A" and re.match(r" +[a-z]", rest) and not rest.lstrip().lower().startswith(
                (str(row["choice_a"]).lower(), "is ", "because ")):
            continue
        return m.group(1)
    return None


def generate(m, msgs, videos, max_new_tokens):
    """One model.generate as in generate_until (greedy); the new text."""
    text = m.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = {"videos": videos} if videos else {}
    inputs = m.processor(text=text, padding=True, return_tensors="pt", **kw).to("cuda")
    out = m.model.generate(**inputs, eos_token_id=m.tokenizer.eos_token_id, pad_token_id=m.tokenizer.pad_token_id,
                           do_sample=False, temperature=None, top_p=None, num_beams=1,
                           max_new_tokens=max_new_tokens, use_cache=True)
    n = inputs.input_ids.shape[1]
    return m.processor.batch_decode(out[:, n:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], n


def segments(m, mod, video_file, t_ask, vr):
    """The cut video as generate_until's _preprocess_batch prepares it: [(time text, frames)], end time text."""
    ele = {"video": video_file}
    if t_ask is not None:
        ele["video_end"] = t_ask
    clip, sample_fps, pts = mod._read_video_decord_plus(ele, return_pts=True, strict_fps=True, drop_last=False,
                                                        max_num_frames=m.max_num_frames, vr=vr)
    assert t_ask is None or pts[-1] <= t_ask, (pts[-1], t_ask)   # the streaming rule
    minutes = (pts[-1] - pts[0]) / 60.0
    T = m.stream_think_times
    times = T[0] if minutes <= 0.5 else T[1] if minutes < 4 else T[2] if minutes < 30 else T[3]
    clip = mod._spatial_resize_video(clip, m.max_stream_vid_tokens * (28 * 28) * times, min_pixels=m.min_pixels)
    total = len(clip)
    seconds = total / sample_fps   # upstream's clock: shorter than the real cut when the budget dropped frames
    seg_len, seg_sec = max(int(total / times), 1), seconds / times
    segs = []
    for j in range(times):
        lo = j * seg_len
        hi = total if j == times - 1 else min((j + 1) * seg_len, total)
        if lo >= hi:
            continue
        segs.append((f"Time={j * seg_sec:.1f}-{(j + 1) * seg_sec:.1f}s ", clip[lo:hi]))
    info = {"frames_read": len(pts), "frames": total, "size": list(clip.shape[2:]), "last_pts": round(float(pts[-1]), 3),
            "segments": len(segs), "clock_s": round(seconds, 1)}
    return segs, f"Time={seconds:.1f}s ", info


def video_msg(seg):
    return {"role": "user", "content": [{"type": "text", "text": seg[0]}, {"type": "video", "video": seg[1]}]}


def think(m, segs):
    """generate_until's stream-think loop: a thought per segment but the last, each reading the earlier ones."""
    memory, system = [THINK_PROMPT], {"role": "system", "content": m.system_prompt}
    for seg in segs[:-1]:
        prev = {"role": "previous text", "content": m.get_textual_memory(memory)}
        thought, _ = generate(m, [system, prev, video_msg(seg)], [seg[1]], 5000)
        memory.append(seg[0] + thought + "\n")
    return memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video_file")
    ap.add_argument("--bank_file")
    ap.add_argument("--out", required=True, help="results folder of this video: settings, thoughts/, answers")
    ap.add_argument("--answers", help="answers_<bank>.json.gz (blind: blind_<bank>.json.gz)")
    ap.add_argument("--model", default="Catalan258/VST-7B", help="Hugging Face id or a local copy")
    ap.add_argument("--model_name", default="Catalan258/VST-7B", help="the model's name in settings and rows")
    # the defaults are upstream's OVO-Bench (streaming) run: eval/scripts/ovobench.sh
    ap.add_argument("--think_times", default="2-3-5-5", help="segments for <= 0.5 / < 4 / < 30 / >= 30 min of video")
    ap.add_argument("--vid_tokens", type=int, default=8192, help="visual tokens per segment")
    ap.add_argument("--max_frames", type=int, default=384)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--settings_only", action="store_true")
    ap.add_argument("--limit", type=int, help="smoke test: only the first N cutoffs' questions; --answers then holds only those "
                                              "(run_vst.sh gives it its own file)")
    a = ap.parse_args()
    mh = check_settings(a)
    if a.settings_only:
        return

    from prepare_epic import convert_questions
    rows = convert_questions(a.bank_file, a.video, 0)   # letters, key, qhash as for every baseline (its query time is unused)
    cut = lambda r: r["meta"]["t_ask"] if r["meta"].get("streaming") and r["meta"].get("t_ask") is not None else None   # None: whole video
    cuts = sorted({cut(r) for r in rows}, key=lambda t: float("inf") if t is None else t)
    if a.limit:
        cuts = cuts[:a.limit]
        rows = [r for r in rows if cut(r) in cuts]

    done = {}
    if os.path.exists(a.answers):
        qh = {r["key"]: r["qhash"] for r in rows}   # a question edited under the same key is redone
        done = {r["key"]: r for r in json.load(gzip.open(a.answers, "rt")) if r.get("version") == VERSION
                and r.get("memory_hash") == mh and r.get("qhash") == qh.get(r["key"])
                and (r.get("response") or "").strip() not in FAILED}
    todo = [r for r in rows if r["key"] not in done]
    print(f"{a.video}: {len(rows)} questions at {len(cuts)} cutoffs, {len(done)} done, {len(todo)} to answer"
          f"{' (blind)' if a.blind else ''} (settings {mh})", flush=True)

    def save():
        tmp = a.answers + ".tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump([done[r["key"]] for r in rows if r["key"] in done], f, indent=1, ensure_ascii=False)
        os.replace(tmp, a.answers)

    if todo:
        import torch
        torch.cuda.init()   # before decord is imported: with decord loaded first, CUDA start-up segfaulted (A5000 nodes)
        import decord
        import lmms_eval.models.simple.qwen2_5_vl_stream_think as mod
        from lmms_eval.models.model_utils.reasoning_model_utils import parse_reasoning_model_answer
        torch.manual_seed(0)
        m = mod.Qwen2_5_VL_Stream_Think(pretrained=a.model, attn_implementation=a.attn, device_map="cuda:0",
                                        stream_think_times=a.think_times, max_stream_vid_tokens=a.vid_tokens,
                                        max_num_frames=a.max_frames)
        system = {"role": "system", "content": m.system_prompt}
        vr = None if a.blind else decord.VideoReader(a.video_file, num_threads=2)
        os.makedirs(os.path.join(a.out, "thoughts"), exist_ok=True)

    segs = head = vids = None
    for c in cuts if todo else []:
        qs = [r for r in todo if cut(r) == c]
        if not qs:
            continue
        info, t0 = {}, time.time()
        segs = head = vids = None   # drop the last cutoff's frames before decoding the next (several GB each)
        gc.collect()
        try:
            if a.blind:
                head = [system, {"role": "previous text", "content": THINK_PROMPT}]
                end, vids = "", None
            else:
                segs, end, info = segments(m, mod, a.video_file, c, vr)
                info["load_s"] = round(time.time() - t0, 1)
                tf = os.path.join(a.out, "thoughts", f"{'end' if c is None else f'{c:.3f}'}.json")
                old = json.load(open(tf)) if os.path.exists(tf) else {}
                if old.get("settings") == mh:   # thoughts made with other settings are redone
                    memory = old["memory"]
                else:
                    t1 = time.time()
                    memory = think(m, segs)
                    info["think_s"] = round(time.time() - t1, 1)
                    with open(tf + ".tmp", "w") as f:
                        json.dump({"video": a.video, "t_ask": c, "settings": mh, **info, "memory": memory}, f,
                                  indent=1, ensure_ascii=False)
                    os.replace(tf + ".tmp", tf)
                head = [system, {"role": "previous text", "content": m.get_textual_memory(memory)}, video_msg(segs[-1])]
                vids = [segs[-1][1]]
        except Exception as e:   # the cutoff's questions stay unanswered; the next run retries
            print(f"ERROR at cutoff {c}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        print(f"cutoff {c}: {info}", flush=True)
        for r in qs:
            t1 = time.time()
            try:
                raw, n_in = generate(m, head + [{"role": "user", "content": end + question_text(r)}], vids, 16)
                resp = parse_reasoning_model_answer(raw).strip() or "Error"
            except Exception as e:
                print(f"ERROR on {r['key']}: {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache()
                resp, n_in = "Error", 0
            pred = pick(resp, r)
            row = {"key": r["key"], "qhash": r["qhash"], "ID": r["ID"], "pos": r["pos"], "type": r["type"],
                   "question": r["question"], "choices": {k: r[f"choice_{k.lower()}"] for k in "ABCD" if f"choice_{k.lower()}" in r},
                   "answer": r["answer"], "response": resp, "pred": pred, "evaluate": pred == r["answer"],
                   "t_ask": r["meta"]["t_ask"], "cutoff": c, "input_tokens": n_in, **{k: info[k] for k in ("frames", "segments", "last_pts") if k in info},
                   "sec": round(time.time() - t1, 1), "version": VERSION, "memory_hash": mh, "model": a.model_name}
            if resp not in FAILED:
                done[r["key"]] = row
            print(f"{r['key']} @ {c}: {resp!r} gold {r['answer']} -> {row['evaluate']} ({row['sec']} s, {n_in} tokens in)", flush=True)
        save()

    if todo:
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
              f"peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB", flush=True)
    if done and not os.path.exists(a.answers):
        save()
    right = sum(bool(done[r["key"]]["evaluate"]) for r in rows if r["key"] in done)
    missing = [r["key"] for r in rows if r["key"] not in done]
    print(f"# {len(rows) - len(missing)}/{len(rows)} answered, {right} right, {len(missing)} missing -> {a.answers}", flush=True)
    sys.exit(2 if missing else 0)


if __name__ == "__main__":
    main()
