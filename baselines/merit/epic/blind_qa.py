#!/usr/bin/env python3
"""Blind control for MERIT: the same respond model, system prompt, query text and answer
parser as MERIT's final QA step, but with an empty Context (no captions, no frames, no
retrieval). Whatever this scores is what the model gets from the question and options alone.
Writes a *_final.json in MERIT's format so epic/report.py and epic/prior_check.py read it.
"""
import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "MERIT", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "MERIT", "eval"))
from evaluate_egolife import evaluate_prediction, parse_reasoning_answer, query_time_int  # noqa: E402
from merit.llm import LLMModel  # noqa: E402
from merit.llm.prompt_template_manager import PromptTemplateManager  # noqa: E402
from merit.types import transform_timestamp  # noqa: E402
from prepare_epic import match_answers  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa_file", required=True)
    p.add_argument("--model", default="qwen3vl-8b")
    p.add_argument("--output", required=True)
    p.add_argument("--context_file", help="JSON {ID: context text}: answer from this given context instead of an empty one "
                                          "(e.g. the captions of the clips holding the evidence, with no search)")
    args = p.parse_args()

    rows = json.load(open(args.qa_file))
    ctx = json.load(open(args.context_file)) if args.context_file else None
    if ctx is not None:   # a question missing from the file would silently be answered with no context
        missing = [r["ID"] for r in rows if str(r["ID"]) not in ctx]
        assert not missing, f"{args.context_file} has no entry for {len(missing)} questions, e.g. {missing[:3]}"
    # an earlier answer is reused only for the same question (key, wording, options, gold letter)
    done = match_answers(rows, json.load(open(args.output))) if os.path.exists(args.output) else {}
    llm = LLMModel(model_name=args.model)
    qa_prompt = PromptTemplateManager().render("final_qa_egolife")

    out = []
    for row in rows:
        if str(row["ID"]) in done:
            out.append(done[str(row["ID"])])
            continue
        choices = {k: row[f"choice_{k.lower()}"] for k in "ABCD" if f"choice_{k.lower()}" in row}
        until = query_time_int(row)
        # Built exactly as MeritMemory.answer / _build_qa_content, with nothing retrieved.
        query = (f"Query: {row['question']}\nQuery Time: {transform_timestamp(str(until))}"
                 f"\nChoices: " + " ".join(f"({k}) {v}" for k, v in sorted(choices.items())))
        content = [{"type": "text", "text": query + "\n\nContext:\n" + (ctx[str(row["ID"])] if ctx else "")},
                   {"type": "text", "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."}]
        msgs = copy.deepcopy(qa_prompt) + [{"role": "user", "content": content}]
        response = llm.generate(msgs)
        parsed, reasoning = parse_reasoning_answer(response)
        out.append({
            "ID": row["ID"], "key": row.get("key"), "qhash": row.get("qhash"), "pos": row.get("pos"), "type": row.get("type", ""), "question": row["question"],
            "choices": choices, "answer": row["answer"], "response": response,
            "parsed_answer": parsed, "reasoning": reasoning,
            "evaluate": evaluate_prediction(parsed or response, row["answer"], choices),
            "query_time": row["query_time"], "query_time_int": until, "round_history": [],
        })
        print(f"{row['ID']:>12} gold {row['answer']} pred {parsed or '?'} {'ok' if out[-1]['evaluate'] else '--'}", flush=True)
        json.dump(out, open(args.output, "w"), indent=1)
    json.dump(out, open(args.output, "w"), indent=1)
    k = sum(r["evaluate"] for r in out)
    print(f"blind accuracy {k}/{len(out)} = {k / len(out):.4f}")


if __name__ == "__main__":
    main()
