#!/usr/bin/env python3
"""Answer one video's questions of an EPIC-Morph bank with the Ego-R1 agent, from a built memory.

The loop is Ego-R1-Agent/eval/infer_bench_summ.py (the README's "with a summary model" run), on the EgoLife
setting the agent was trained for: the system prompt is utils/constants.py prompt + format_prompt, the question
is "<question> <timestamp>DAY1_HHMMSSFF</timestamp>" plus the lettered options, the agent (Ego-R1-Agent-3B,
temperature 0.6, top-p 0.95, up to 1024 new tokens per turn, MAX_TURNS 7 as in infer_bench_summ.sh) thinks and
calls one tool per turn, and GPT-4.1 turns the trace into the final letter (after the agent's own <answer>, or
at the last turn). Upstream serves the agent with vLLM; here it is the same weights in transformers (fp16).

Streaming: a question is asked at floor(t_ask) of its video (MERIT's conversion); the tools only see the video
and the memory before it (tools.py).

Writes --answers (.json.gz): one row per question in bank order, keyed by the bank's `key`, with the chosen
option, correctness and the whole trace (each turn's output, tool call and returned information). Resumes from
it: a row is done when it has an answer from this VERSION, this memory and the same question (qhash). A question
with a failed API call is not saved; exit code 2 says some are left (run again). Exit code 5: an API key is
missing, nothing was called.

--first_turn runs only what needs no paid API: the agent's first turn on every question, the parsed tool call
and the streaming check on it, written to <answers minus .json.gz>_firstturn.json. It needs no memory.

    python run_qa.py --video <V> --arm gemini25pro --bank_file <V>_questions.json --memory <dir> --clips <clip_root>/<V> \
        --answers answers_<bank>.json.gz --cache_dir <dir>
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "merit", "epic")]
import apis                                                    # noqa: E402
import tools as T                                              # noqa: E402
from build_memory import MEMORY, jload, secs                   # noqa: E402
from prepare_epic import convert_questions, duration           # noqa: E402  (MERIT's)

VERSION = "1"   # bump when a change alters answers (prompts, settings, the cutoff view)
MAX_TURNS, TEMPERATURE, TOP_P, MAX_NEW = 7, 0.6, 0.95, 1024
# infer_bench_summ.py's client_prompt (its 4-option form; the script defines none for EgoLife)
ANSWER_PROMPT = ("You are given some information and a chain-of-thought reasoning process, with actions made and "
                 "observations from the environment. Given these information, try to answer the MCQ question in A|B|C|D "
                 "format. You should only answer one option. For example, if the answer to this question is A, you should "
                 "only return ```<answer>A</answer>```.")


def memory_hash(mem, video):
    h = hashlib.sha1()
    for m in MEMORY:
        h.update(open(os.path.join(mem, m.format(video=video)), "rb").read())
    return h.hexdigest()[:16]


class Agent:
    def __init__(self, model):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="cuda:0").eval()

    def __call__(self, history, seed):
        text = self.tok.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        self.torch.manual_seed(seed)
        with self.torch.inference_mode():   # vLLM's SamplingParams defaults: no top-k, no repetition penalty
            out = self.model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                                      top_k=0, repetition_penalty=1.0)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def final_answer(history_text, question):
    """GPT-4.1 reads the trace and returns '<answer>X</answer>' (three tries, as upstream; None when all fail,
    where upstream would answer A)."""
    messages = [{"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": f"The chat history is as follows:\n{history_text}\n"},
                {"role": "user", "content": f"Question: {question}"}]
    errors = []
    for _ in range(3):
        try:
            out = apis.gpt41("answer", messages, max_tokens=800)
            if "<answer>" in out and "</answer>" in out:
                return out
        except Exception as e:
            errors.append(f"answer: {type(e).__name__}: {e}")
    T.API_ERRORS.extend(errors)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--arm", required=True, choices=["gemini25pro", "llava"])
    ap.add_argument("--bank_file", required=True, help="epic-morph outputs/v3/banks/<bank>/<video>_questions.json")
    ap.add_argument("--memory", help="the memory folder (build_memory.py --out)")
    ap.add_argument("--clips", required=True, help="<clip_root>/<video> (prepare_clips.py)")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--first_turn", action="store_true", help="no paid API: only the agent's first turn per question")
    ap.add_argument("--video_file", help="--first_turn: the mp4, for the video's length when there is no memory yet")
    ap.add_argument("--video_end", help="--first_turn: the video's length in seconds, instead of --video_file (no ffprobe)")
    ap.add_argument("--limit", type=int, default=0, help="--first_turn: only the first N questions")
    ap.add_argument("--attempt", type=int, default=0, help="added to the sampling seed: a retry pass draws a new trace")
    a = ap.parse_args()
    a.answers, a.cache_dir, a.clips, a.bank_file = map(os.path.abspath, (a.answers, a.cache_dir, a.clips, a.bank_file))
    a.memory = os.path.abspath(a.memory) if a.memory else None      # the tools run in another folder (tools._quiet)
    V = a.video
    if not a.first_turn:
        apis.need("rag", "vlm", "answer", *(["video_llm"] if a.arm != "llava" else []))
        apis.USAGE = os.path.join(a.memory, "api_usage.jsonl")
        end = secs(jload(os.path.join(a.memory, f"{V}_30sec.json"))[-1]["end_time"])
        mh = memory_hash(a.memory, V)
    else:
        end, mh = int(float(a.video_end or duration(a.video_file))), None
    process = T.upstream("Ego-R1-Agent", "utils.process")      # parsing of tool calls and answers; pulls in constants
    system = process.constants.prompt + process.constants.format_prompt

    rows = convert_questions(a.bank_file, V, end)
    done = {}
    if not a.first_turn and os.path.exists(a.answers):
        done = {r["key"]: r for r in json.load(gzip.open(a.answers, "rt"))
                if r.get("version") == VERSION and r.get("memory_hash") == mh and r.get("response") in set("ABCD")}
        done = {r["key"]: done[r["key"]] for r in rows if r["key"] in done and done[r["key"]]["qhash"] == r["qhash"]}
    todo = [r for r in rows if r["key"] not in done][:a.limit or None]
    print(f"{V}: {len(rows)} questions, {len(done)} done, {len(todo)} to answer", flush=True)
    agent = Agent(os.environ.get("EGOR1_AGENT") or apis.MODELS["agent"]) if todo else None
    os.makedirs(a.cache_dir, exist_ok=True)

    def save():
        with gzip.open(a.answers + ".tmp", "wt") as f:
            json.dump([done[r["key"]] for r in rows if r["key"] in done], f, indent=1)
        os.replace(a.answers + ".tmp", a.answers)

    first = []
    for r in todo:
        cut = secs(r["query_time"]["time"])
        choices = {L: r[f"choice_{L.lower()}"] for L in "ABCD" if r.get(f"choice_{L.lower()}")}
        question = (f"{r['question']} <timestamp>DAY1_{r['query_time']['time']}</timestamp>\n"
                    + "\n".join(f"{L}. {c}" for L, c in choices.items()))
        logs = None if a.first_turn else T.cutoff_logs(a.memory, V, cut, end)
        tools = None if a.first_turn else T.Tools(a.arm, a.clips, logs, cut, a.cache_dir)
        del T.API_ERRORS[:]
        seed = int(hashlib.sha1(r["key"].encode()).hexdigest()[:8], 16) + 1000 * a.attempt
        history, turns, out, t0 = [{"role": "system", "content": system}], [], None, time.time()
        user = f"You should answer the question in {MAX_TURNS} turns. {question}"
        for turn in range(MAX_TURNS):
            history.append({"role": "user", "content": user})
            if turn == MAX_TURNS - 1:      # out of turns: GPT-4.1 answers from the whole trace
                out = final_answer(str(history), question)
                break
            msg = agent(history, seed + turn)
            history.append({"role": "assistant", "content": msg})
            if "<answer>" in msg and "</answer>" in msg:   # GPT-4.1 puts the agent's answer in the format
                turns.append({"output": msg})
                out = final_answer(str(history[-1]), question)
                break
            call = process.parse_tool_call(process.get_query(msg))
            if a.first_turn:
                checked, err = T.check(call, cut)
                first.append({"key": r["key"], "question": question, "cutoff": cut, "output": msg, "tool_call": call,
                              "after_streaming_check": err or checked, "seconds": round(time.time() - t0, 1)})
                break
            info = tools.call(call) if call else ""
            turns.append({"output": msg, "tool_call": call, "information": info})
            user = f"<information>{info}</information>"
        if a.first_turn:
            print(f"{r['key']} @ {cut}s: {json.dumps(first[-1]['tool_call'])[:300]}", flush=True)
            continue
        m = re.search(r"\s*([A-Z])\s*", process.parse_answer(out.replace("\n", " ")) or "") if out else None
        resp = m.group(1) if m else ""        # the first capital letter, as process.compute_score reads it
        if T.API_ERRORS or resp not in choices:
            print(f"{r['key']}: not answered ({T.API_ERRORS or repr(out)})", flush=True)
            continue
        done[r["key"]] = {"key": r["key"], "qhash": r["qhash"], "ID": r["ID"], "type": r["type"], "question": r["question"],
                          "choices": choices, "answer": r["answer"], "response": resp, "evaluate": resp == r["answer"],
                          "t_ask": r["meta"]["t_ask"], "cutoff": cut, "turns": turns, "final": out,
                          "seconds": round(time.time() - t0, 1), "version": VERSION, "memory_hash": mh}
        print(f"{r['key']} @ {cut}s: {resp} gold {r['answer']} ({len(turns)} turns, {done[r['key']]['seconds']}s)", flush=True)
        save()
    if a.first_turn:
        p = a.answers.replace(".json.gz", "") + "_firstturn.json"
        json.dump(first, open(p, "w"), indent=1)
        n = sum(bool(x["tool_call"]) for x in first)
        print(f"# first turn: {len(first)} questions, {n} valid tool calls -> {p}")
        return
    save()
    missing = [r["key"] for r in rows if r["key"] not in done]
    right = sum(done[k]["evaluate"] for k in done)
    print(f"# {len(done)}/{len(rows)} answered, {right} right, {len(missing)} missing -> {a.answers}", flush=True)
    sys.exit(2 if missing else 0)


if __name__ == "__main__":
    main()
