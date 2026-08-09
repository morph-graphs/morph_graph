"""
Split compound questions from the EgoLongQA dataset.

Two-pass approach:
  Pass 1 (Python, zero cost): Detect compound questions via semicolons in
          the answer field. Split answers and MCQ options programmatically.
          Handles N-way splits (2, 3, or more sub-questions).
  Pass 2 (LLM, ~$0.01):  Send ONLY the compound question *texts* (plus the
          number of sub-questions N) to Gemini Flash and ask it to return N
          independent question strings.

Usage:
  python scripts/split_compound_questions_v2.py [--dry-run]

  --dry-run   Run only the Python pass; skip the LLM call entirely.
              Useful for inspecting what will be split before spending credits.
"""

import os
import re
import json
import time
import argparse
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-3.6-flash"  # cheaper flash model
URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}"
    f":generateContent?key={API_KEY}"
)

IN_FILE = Path(
    "/fs/cfar-projects/mem_open_vocab/datasets/wearable-ai/egolongqa/"
    "wearable_ai_2026_egolongqa_val_700.jsonl"
)
OUT_FILE = Path("wearable_ai_2026_egolongqa_val_split_v2.jsonl")

# ---------------------------------------------------------------------------
# Pass 1 helpers – pure Python, zero API cost
# ---------------------------------------------------------------------------

def count_parts(row: dict) -> int:
    """Return the number of sub-questions in a row (1 = single question).

    Uses the number of semicolons in the answer as the signal.
    """
    return row.get("answer", "").count(";") + 1


def is_compound(row: dict) -> bool:
    """A question is compound iff its answer contains at least one semicolon."""
    return count_parts(row) >= 2


def parse_options(opts_str: str) -> dict:
    """Parse 'A. foo B. bar C. baz D. qux' → {'A': 'foo', 'B': 'bar', …}"""
    parts = re.split(r"(?:^|\s)([A-D])\.\s", opts_str.strip())
    result = {}
    for i in range(1, len(parts), 2):
        letter = parts[i]
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        result[letter] = text
    return result


def format_options(opts_dict: dict) -> str:
    """{'A': 'foo', 'B': 'bar', …} → 'A. foo B. bar …'"""
    return " ".join(f"{k}. {v}" for k, v in sorted(opts_dict.items()))


def split_answer_n(answer: str, n: int) -> list[str]:
    """Split an answer into exactly n parts on semicolons.

    E.g. split_answer_n("X; Y; Z", 3) → ["X", "Y", "Z"]
    If fewer semicolons than expected, the last part gets the remainder.
    """
    parts = answer.split(";", n - 1)
    return [p.strip() for p in parts]


def split_options_n(opts_str: str, n: int) -> list[str]:
    """Split compound MCQ options into n independent option sets.

    Each original option like 'A. part1; part2; part3' is split on semicolons
    to yield n separate option dicts, then each is reformatted.
    """
    parsed = parse_options(opts_str)
    # Initialize n option dicts
    opts_list = [{} for _ in range(n)]
    for letter, text in parsed.items():
        parts = text.split(";", n - 1)
        for k in range(n):
            if k < len(parts):
                opts_list[k][letter] = parts[k].strip()
            else:
                # Fallback: if fewer semicolons than expected, use empty string
                opts_list[k][letter] = ""
    return [format_options(od) for od in opts_list]


# ---------------------------------------------------------------------------
# Pass 2 – LLM call for splitting question *text* only
# ---------------------------------------------------------------------------

# Minimal prompt – asks for N-way question text splitting
SPLIT_PROMPT = """You will receive a JSON array of objects. Each object has:
- "question": a compound question that actually asks multiple separate things
- "n": the number of independent sub-questions to produce

For each input, split the question text into exactly `n` independent sub-questions.
- You MUST carry over all necessary context (entities, locations, pronouns, temporal references) so that EACH sub-question makes complete sense on its own.
- The sub-questions should correspond to the parts of the answer in order (the first sub-question corresponds to the first answer part, etc.)

Return a JSON array with one object per input, in order. Each object has a single key "questions" whose value is an array of exactly `n` strings.

Example input:  [{"question": "What color was the car, and who was driving it?", "n": 2}]
Example output: [{"questions": ["What color was the car?", "Who was driving the car?"]}]
"""

# Schema: array of objects, each with a "questions" array of strings
SPLIT_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "questions": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
        },
        "required": ["questions"],
    },
}


def split_question_texts_with_llm(
    items: list[dict], batch_size: int = 40
) -> list[list[str]]:
    """Send compound question texts to Gemini and get N-way splits.

    Args:
        items: List of {"question": str, "n": int} dicts.
        batch_size: Number of questions per API call.

    Returns:
        List of lists of sub-question strings, one list per input.
    """
    all_results = []
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    print(f"  LLM pass: {len(items)} questions in {len(batches)} batch(es)")

    for batch_idx, batch in enumerate(batches):
        payload = {
            "system_instruction": {"parts": [{"text": SPLIT_PROMPT}]},
            "contents": [{"parts": [{"text": json.dumps(batch)}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": SPLIT_SCHEMA,
                "temperature": 0.1,
            },
        }

        result = None
        for attempt in range(3):
            try:
                resp = requests.post(URL, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usageMetadata", {})
                    in_tok = usage.get("promptTokenCount", "?")
                    out_tok = usage.get("candidatesTokenCount", "?")
                    print(
                        f"  Batch {batch_idx + 1}/{len(batches)}: "
                        f"in={in_tok} tok, out={out_tok} tok"
                    )
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(text.strip())
                    if len(parsed) == len(batch):
                        result = [item["questions"] for item in parsed]
                        break
                    else:
                        print(
                            f"    ⚠ Size mismatch: got {len(parsed)}, "
                            f"expected {len(batch)}"
                        )
                elif resp.status_code == 429:
                    wait = 2 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    Error {resp.status_code}: {resp.text[:200]}")
                    time.sleep(1)
            except Exception as e:
                print(f"    Exception: {e}")
                time.sleep(1)

        if result is None:
            # Fallback: return original question duplicated n times
            print(
                f"    ✗ Batch {batch_idx + 1} failed, "
                f"using original questions as fallback"
            )
            result = [[item["question"]] * item["n"] for item in batch]

        all_results.extend(result)

    return all_results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split compound questions (two-pass, N-way)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only the Python pass; skip the LLM call.",
    )
    args = parser.parse_args()

    # Load data
    rows = []
    with open(IN_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {IN_FILE.name}")

    # ── Pass 1: Identify compound questions (Python) ──────────────────────
    compound_indices = [i for i, r in enumerate(rows) if is_compound(r)]

    # Report breakdown by number of parts
    from collections import Counter
    parts_counter = Counter(count_parts(rows[i]) for i in compound_indices)
    print(f"Pass 1: Found {len(compound_indices)} compound questions:")
    for n_parts in sorted(parts_counter):
        print(f"  {n_parts}-way splits: {parts_counter[n_parts]}")

    # ── Pass 2: Split question texts (LLM) ────────────────────────────────
    if args.dry_run:
        print("Dry-run mode: skipping LLM call. Using original question text as placeholder.")
        q_splits = [
            [rows[i]["question"]] * count_parts(rows[i])
            for i in compound_indices
        ]
    else:
        llm_items = [
            {"question": rows[i]["question"], "n": count_parts(rows[i])}
            for i in compound_indices
        ]
        q_splits = split_question_texts_with_llm(llm_items)

    # ── Assemble final output ─────────────────────────────────────────────
    # Build a lookup: original index → list of sub-question strings
    split_lookup = {}
    for j, orig_idx in enumerate(compound_indices):
        split_lookup[orig_idx] = q_splits[j]

    final_rows = []
    for i, row in enumerate(rows):
        if i not in split_lookup:
            # Single question – pass through unchanged
            final_rows.append(row)
        else:
            # Compound question – N-way split
            n = count_parts(row)
            answers = split_answer_n(row["answer"], n)
            options_list = split_options_n(row["mcq_options"], n)
            sub_questions = split_lookup[i]

            # Safety: ensure all lists are length n
            if len(sub_questions) != n:
                print(
                    f"  ⚠ Question at index {i}: expected {n} sub-questions, "
                    f"got {len(sub_questions)}. Padding/truncating."
                )
                # Pad with original question or truncate
                while len(sub_questions) < n:
                    sub_questions.append(row["question"])
                sub_questions = sub_questions[:n]

            for k in range(n):
                r_new = dict(row)
                r_new["question"] = sub_questions[k]
                r_new["answer"] = answers[k] if k < len(answers) else ""
                r_new["mcq_options"] = options_list[k] if k < len(options_list) else ""
                r_new["mcq_answer"] = row["mcq_answer"]
                final_rows.append(r_new)

    print(
        f"Done. Final count: {len(final_rows)} "
        f"(was {len(rows)}, +{len(final_rows) - len(rows)} from splits)"
    )

    with open(OUT_FILE, "w") as f:
        for r in final_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
