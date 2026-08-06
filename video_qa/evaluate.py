"""Aggregate and score resumable ``video_qa`` Slurm outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from .benchmarks import BENCHMARKS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = Path(os.environ.get("MEM_OPEN_VOCAB_DATA_DIR", _REPO_ROOT / "data"))
_EGO_GT = _DATA_DIR / "wearable-ai" / "egolongqa" / "wearable_ai_2026_egolongqa_val_700.jsonl"
_VSI_GT = _REPO_ROOT / "vsi_output" / "vsi_events_mc_poc.json"


def _load_official_evaluator():
    path = _DATA_DIR / "wearable-ai" / "starter_kit" / "run_evaluation.py"
    spec = importlib.util.spec_from_file_location("wearable_ai_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluation helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_chunk_answers(output_dir: Path) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("answers_chunk_*.json")):
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list")
        answers.extend(rows)
    if not answers:
        raise FileNotFoundError(f"No answers_chunk_*.json files in {output_dir}")
    return answers


def _vsi_golden(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "qid": row["qid"],
            "video_path": f"{str(row['object']).replace(' ', '_')}/{row['video_uid']}.mp4",
            "question": row["question"],
            "mcq_answer": row["correct_answer"],
            "category": row.get("question_type", ""),
        }
        for row in json.loads(path.read_text())
    ]


def _match_key(benchmark: str, row: dict[str, Any]) -> str | tuple[str, str]:
    if benchmark == "vsi_morph":
        return str(row["qid"])
    return (
        Path(str(row.get("video_path", ""))).stem,
        str(row.get("question", "")).strip(),
    )


def evaluate(benchmark: str, output_dir: Path, gt_path: Path) -> dict[str, Any]:
    predictions = load_chunk_answers(output_dir)
    official = _load_official_evaluator()
    golden = _vsi_golden(gt_path) if benchmark == "vsi_morph" else official.load_jsonl(str(gt_path))
    by_key = {_match_key(benchmark, row): row for row in predictions}
    ordered_predictions = []
    missing = []
    for row in golden:
        prediction = by_key.pop(_match_key(benchmark, row), None)
        if prediction is None:
            missing.append(_match_key(benchmark, row))
            ordered_predictions.append({"mcq_answer": ""})
        else:
            answer = "" if prediction.get("qa_error") else prediction.get("answer", "")
            ordered_predictions.append({"mcq_answer": str(answer)})
    result = official.evaluate_longqa(golden, ordered_predictions)
    return {
        "benchmark": benchmark,
        "num_predictions": len(predictions),
        "num_missing": len(missing),
        "num_extra": len(by_key),
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-path", type=Path)
    parser.add_argument("--write-results", type=Path)
    args = parser.parse_args()
    gt_path = args.gt_path or (_VSI_GT if args.benchmark == "vsi_morph" else _EGO_GT)
    result = evaluate(args.benchmark, args.output_dir, gt_path)
    print(
        f"{args.benchmark}: {result['correct']}/{result['total']} "
        f"({result['accuracy']:.2%}); missing={result['num_missing']}"
    )
    if args.write_results:
        args.write_results.parent.mkdir(parents=True, exist_ok=True)
        args.write_results.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
