#!/usr/bin/env python3
"""Direct baseline: a plain video model, no memory. For each question the model gets frames sampled from the
video before the question's ask time, then the question, and answers.

It is MERIT's blind control (baselines/merit/epic/blind_qa.py) with the frames as the Context: the same model
loader (MERIT's Qwen wrapper), system prompt, query text, greedy decoding, answer parser and scorer, and the
same answer rows, so one scorer reads MERIT, blind and direct. What differs is only what sits under "Context:".

Frames: the 1 fps, 448 px frames MERIT's prep cut. ffmpeg's fps filter gives frame k the last picture before
k + 0.5 s (checked against exact pictures: t00100.jpg is closest to 100.5 s), so a question asked at t_ask
may use frames 0..floor(t_ask - 0.5) and nothing after t_ask is shown. --frames N takes N of them, evenly spaced from the first to the
last (which is always included); when fewer than N exist it takes all, i.e. 1 fps. So N is both "uniform N
frames" and "1 fps up to a cap of N". They go in as one video with their true time stamps (the model's own
video input: 2 frames per token group, 196 tokens per 2 frames at 448 px).

    python epic/direct_qa.py --model qwen3vl-8b --frames 128 --bank_file <V>_questions.json --video <V> \
        --frame_root <work>/frames_448 --skeleton <work>/prep/captions/<V>/<V>_30sec_skeleton.json \
        --output answers_<bank>.json.gz
"""
import argparse
import copy
import glob
import gzip
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MERIT = os.path.join(HERE, "..", "..", "merit")
sys.path[:0] = [os.path.join(MERIT, "epic"), os.path.join(MERIT, "MERIT", "src"), os.path.join(MERIT, "MERIT", "eval")]
from prepare_epic import convert_questions, match_answers  # noqa: E402

PX = 448   # the side of prepare_epic's frames, shown to the model as they are
# models MERIT's wrapper does not know, loaded by load_hf: name -> Hugging Face id
HF = {"qwen35-9b": "Qwen/Qwen3.5-9B", "qwen35-35b-a3b": "Qwen/Qwen3.5-35B-A3B", "molmo2-8b": "allenai/Molmo2-8B"}
FINAL = "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."


def load_hf(hf_id):
    """Model and processor by Hugging Face id, set up as MERIT's wrapper sets up its Qwen models (local snapshot
    when cached, bf16, flash-attention when installed, one GPU or spread over those visible, seed 42). Molmo2
    brings its own code (trust_remote_code); its card loads it as float32 (dtype auto), which needs 46.7 GB at
    128 frames, so it is loaded in bf16 like the others."""
    import torch
    from types import SimpleNamespace
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor
    try:
        path = snapshot_download(hf_id, local_files_only=True)
    except Exception:
        path = hf_id
    kw = {"device_map": "auto" if torch.cuda.device_count() > 1 else "cuda:0", "dtype": torch.bfloat16}
    if "Molmo" in hf_id:
        kw["trust_remote_code"] = True
    else:
        try:
            import flash_attn  # noqa: F401
            kw["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kw["attn_implementation"] = "sdpa"
    torch.manual_seed(42)
    model = AutoModelForImageTextToText.from_pretrained(path, **kw).eval()
    processor = AutoProcessor.from_pretrained(path, **({"trust_remote_code": True} if "Molmo" in hf_id else {}))
    print(f"loaded {hf_id}: {kw}, devices {sorted(set(getattr(model, 'hf_device_map', {'': model.device}).values()), key=str)}", flush=True)
    return SimpleNamespace(model=model, processor=processor, model_name=hf_id)


def qwen_inputs(llm, system, query, frames, stamps):
    """Qwen3-VL / Qwen3.5 / Qwen3.8: MeritMemory._build_qa_content with the video where the retrieved captions
    and frames would be. Returns the processor's inputs and the number of video tokens."""
    from transformers.video_utils import VideoMetadata
    content = ([{"type": "text", "text": query + "\n\nContext:\n"}] + ([{"type": "video"}] if frames else []) +
               [{"type": "text", "text": FINAL}])
    thinking = {"enable_thinking": False} if ("Qwen3.5" in llm.model_name or "Qwen3.8" in llm.model_name) else {}   # instruct mode, as MERIT
    text = llm.processor.apply_chat_template(copy.deepcopy(system) + [{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True, **thinking)
    video = {}
    if frames:
        frames = frames + frames[-1:] * (len(frames) % 2)   # an even count: the model joins frames in pairs (the processor
        stamps = stamps + stamps[-1:] * (len(stamps) % 2)   # would repeat the last one too, but only after resizing for the odd count)
        # total pixels pinned (min = max), so the processor keeps every frame at 448 px (as video_models/qwen3_vl.py);
        # index 2 * time at 2 fps gives each pair the mean of its two times as its stamp
        budget = PX ** 2 * len(frames)
        video = dict(videos=[frames], do_sample_frames=False, size={"shortest_edge": budget, "longest_edge": budget},
                     video_metadata=[VideoMetadata(total_num_frames=int(2 * stamps[-1]) + 1, fps=2.0, frames_indices=[int(2 * t) for t in stamps])])
    inputs = llm.processor(text=[text], padding=True, return_tensors="pt", **video)
    n_video = int((inputs["input_ids"] == llm.processor.video_token_id).sum())
    if frames:   # every frame kept at 448 px: (448 / 32)^2 = 196 tokens per 2 frames
        assert n_video == len(frames) // 2 * (PX // 32) ** 2, (n_video, len(frames))
    return inputs, n_video


def molmo_inputs(llm, system, query, frames, stamps):
    """Molmo2, as its model card and processor work: no system role, so MERIT's system text opens the user turn;
    the chat template puts the video before all text; the processor resizes every frame to 378 px (81 tokens
    a frame) and writes each frame's time before it."""
    import numpy as np
    from transformers.video_utils import VideoMetadata
    text = system[0]["content"].rstrip() + "\n\n" + query + "\n\nContext:\n" + FINAL
    content = [{"type": "text", "text": text}] + ([{"type": "video"}] if frames else [])
    text = llm.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    video = {}
    if frames:
        video = dict(videos=[np.stack([np.asarray(f) for f in frames])], do_sample_frames=False,
                     video_metadata=[VideoMetadata(total_num_frames=int(2 * stamps[-1]) + 1, fps=2.0, frames_indices=[int(2 * t) for t in stamps])])
    inputs = llm.processor(text=[text], padding=True, return_tensors="pt", **video)
    n_video = int((inputs["input_ids"] == llm.model.config.image_patch_id).sum())
    assert n_video == len(frames) * 81, (n_video, len(frames))
    return inputs, n_video


def pick_frames(t_ask, n_have, n):
    """Seconds of the frames shown for a question asked at t_ask: at most n, evenly spaced over
    0..floor(t_ask - 0.5), both ends included; n_have is the number of 1 fps frames the video has."""
    last = min(n_have - 1, math.floor(t_ask - 0.5))
    if last < 0:
        return []
    if n == 1:
        return [last]
    return sorted({round(i * last / (n - 1)) for i in range(n)})


def frame_paths(frame_root, video):
    """{second: jpg} of prepare_epic's frames (<frame_root>/<video>/<clip>/tNNNNN.jpg)."""
    return {int(os.path.basename(p)[1:-4]): p for p in glob.glob(os.path.join(frame_root, video, "*", "t*.jpg"))}


def load(path):
    return json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="qwen3vl-8b", help="a name of MERIT's Qwen wrapper (qwen3vl-8b, qwen38-27b) or one of: " + ", ".join(HF))
    p.add_argument("--frames", type=int, required=True, help="frame budget N per question")
    p.add_argument("--video", required=True)
    p.add_argument("--bank_file", required=True, help="epic-morph outputs/v3/banks/<bank>/<video>_questions.json")
    p.add_argument("--frame_root", required=True)
    p.add_argument("--skeleton", required=True, help="prepare_epic's clip list (gives the video's end, as for MERIT)")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0, help="answer only this many of the unanswered questions, spread over the bank (smoke test)")
    args = p.parse_args()

    end = json.load(open(args.skeleton))[-1]["end_time"]            # HHMMSS00, as run_merit.sh's qa_file
    rows = convert_questions(args.bank_file, args.video, int(end[:2]) * 3600 + int(end[2:4]) * 60 + int(end[4:6]))
    paths = frame_paths(args.frame_root, args.video)
    assert paths and sorted(paths) == list(range(len(paths))), f"frames of {args.video} missing or not 0..n-1 in {args.frame_root}"
    setting = {"model": args.model, "frames": args.frames, "frame_px": PX}
    # an earlier answer is reused only for the same question (key, wording, options, gold letter) and setting
    done = {}
    if os.path.exists(args.output):
        done = match_answers(rows, [r for r in load(args.output) if r.get("setting") == setting])

    def save():   # every answer so far, in bank order
        out = [done[str(r["ID"])] for r in rows if str(r["ID"]) in done]
        tmp = args.output + ".tmp"
        with (gzip.open(tmp, "wt") if args.output.endswith(".gz") else open(tmp, "w")) as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, args.output)
        return out

    todo = [str(r["ID"]) for r in rows if str(r["ID"]) not in done]
    if args.limit:   # the rest stay unanswered; answers already in the file are kept either way
        todo = todo[::max(1, len(todo) // args.limit)][:args.limit]
    llm, shown = None, False
    for row in rows:
        if str(row["ID"]) not in todo:
            continue
        if llm is None:   # loaded only when something is left to answer
            import torch
            from PIL import Image
            from evaluate_egolife import evaluate_prediction, parse_reasoning_answer, query_time_int
            from merit.llm.prompt_template_manager import PromptTemplateManager
            from merit.types import transform_timestamp
            if args.model in HF:
                llm = load_hf(HF[args.model])
            else:
                from merit.llm import LLMModel
                llm = LLMModel(model_name=args.model).model      # MERIT's loader: .model, .processor, greedy, seed 42
            qa_prompt = PromptTemplateManager().render("final_qa_egolife")
            build = molmo_inputs if "Molmo" in llm.model_name else qwen_inputs
        t0 = time.time()
        choices = {k: row[f"choice_{k.lower()}"] for k in "ABCD" if f"choice_{k.lower()}" in row}
        until = query_time_int(row)
        secs = pick_frames(row["meta"]["t_ask"], len(paths), args.frames)
        assert all(s + 0.5 <= row["meta"]["t_ask"] for s in secs), (row["ID"], row["meta"]["t_ask"], secs[-1])
        query = (f"Query: {row['question']}\nQuery Time: {transform_timestamp(str(until))}"
                 f"\nChoices: " + " ".join(f"({k}) {v}" for k, v in sorted(choices.items())))
        # frame k is the picture at k + 0.5 s: that is the time the model is told
        inputs, n_video = build(llm, qa_prompt, query, [Image.open(paths[s]).convert("RGB") for s in secs], [s + 0.5 for s in secs])
        n_in = inputs["input_ids"].shape[1]
        # the model's context (Molmo2: 36,864 tokens, i.e. at most 384 frames): the prompt must fit with room to answer
        cfg = llm.model.config
        room = getattr(getattr(cfg, "text_config", cfg), "max_position_embeddings", 10 ** 9) - n_in
        assert room >= 256, f"{n_in} prompt tokens with {len(secs)} frames: over the model's context ({room + n_in} tokens)"
        if secs and not shown:   # once per run: the start and end of the prompt as the model reads it (time stamps, wording)
            ids = inputs["input_ids"][0]
            seen = llm.processor.tokenizer.decode(ids[(ids != getattr(llm.processor, "video_token_id", -1)) & (ids != getattr(llm.model.config, "image_patch_id", -1))])
            print(f"prompt with {len(secs)} frames ({secs[:2]} .. {secs[-2:]} s): {seen[:400]!r} ... {seen[-500:]!r}", flush=True)
            shown = True
        try:
            torch.manual_seed(42)
            with torch.no_grad():
                gen = llm.model.generate(**inputs.to(llm.model.device), do_sample=False, max_new_tokens=min(2048, room))
            response = llm.processor.tokenizer.decode(gen[0, n_in:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        except torch.OutOfMemoryError as e:   # "Error" rows are retried on the next run, as MERIT's
            print(f"{row['ID']}: {str(e)[:200]}", flush=True)
            response = "Error"
        del inputs
        torch.cuda.empty_cache()
        parsed, reasoning = parse_reasoning_answer(response)
        if not parsed:   # Molmo2 writes the letter alone on the first line, without "ANSWER:"
            m = re.match(r"\(?([A-D])\)?[.:]?\s*(\n|$)", response)
            parsed = m.group(1) if m and m.group(1) in choices else ""
        res = done[str(row["ID"])] = {
            "ID": row["ID"], "key": row.get("key"), "qhash": row.get("qhash"), "pos": row.get("pos"), "type": row.get("type", ""),
            "question": row["question"], "choices": choices, "answer": row["answer"], "response": response,
            "parsed_answer": parsed, "reasoning": reasoning,
            "evaluate": evaluate_prediction(parsed or response, row["answer"], choices),
            "query_time": row["query_time"], "query_time_int": until, "round_history": [],
            # direct only: what the model was shown
            "t_ask": row["meta"]["t_ask"], "frame_seconds": secs, "n_frames": len(secs), "video_tokens": n_video,
            "prompt_tokens": n_in, "seconds": round(time.time() - t0, 1), "setting": setting,
        }
        print(f"{row['ID']:>12} t_ask {row['meta']['t_ask']:7.1f} frames {len(secs):4d} (last {secs[-1] if secs else '-'} s) "
              f"tokens {n_in:6d} gold {row['answer']} pred {parsed or '?'} {'ok' if res['evaluate'] else '--'} "
              f"{res['seconds']} s", flush=True)
        save()
    out = save()
    bad = [r["ID"] for r in out if r["response"] == "Error"]
    if llm is not None:
        print("peak GPU memory (GB per GPU):", [round(torch.cuda.max_memory_allocated(i) / 2 ** 30, 1) for i in range(torch.cuda.device_count())])
    k = sum(r["evaluate"] for r in out)
    print(f"direct {args.model} f{args.frames}: accuracy {k}/{len(out)} = {k / max(1, len(out)):.4f}; "
          f"{len(bad)} failed, {len(rows) - len(out)} of the bank's {len(rows)} not asked")
    sys.exit(2 if bad else 0)


if __name__ == "__main__":
    main()
