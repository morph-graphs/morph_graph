"""Compare Gemini reference annotations (default: query-agnostic) to TSM exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .metrics import compare_annotation
from .schema import OracleAnnotation


def _mean_present(results: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(result["summary"][key])
        for result in results
        if result["summary"].get(key) is not None
    ]
    return mean(values) if values else None


def _iter_reference_json(reference_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in reference_dir.glob("*.json")
            if not path.name.endswith(".error.json")
        ),
        key=lambda path: path.stem,
    )


def build_report(
    reference_dir: str | Path,
    tsm_dir: str | Path,
    *,
    tolerances_sec: tuple[float, ...] = (2.0, 5.0, 10.0),
    reference_mode: str = "query_agnostic",
) -> dict[str, Any]:
    """Compare each reference annotation JSON under ``reference_dir`` to TSM.

    Default reference is query-agnostic Gemini (one file per ``video_id``).
    Pass ``reference_mode="query_conditioned"`` with an oracle directory for the
    older per-question evaluation.
    """
    reference_root, tsm_root = Path(reference_dir), Path(tsm_dir)
    comparisons: list[dict[str, Any]] = []
    missing_tsm: list[str] = []
    skipped_mode: list[str] = []
    for reference_path in _iter_reference_json(reference_root):
        annotation = OracleAnnotation.load(reference_path)
        if (
            reference_mode == "query_agnostic"
            and annotation.annotation_mode not in {None, "query_agnostic"}
        ):
            skipped_mode.append(str(reference_path.name))
            continue
        if (
            reference_mode == "query_conditioned"
            and annotation.annotation_mode == "query_agnostic"
        ):
            skipped_mode.append(str(reference_path.name))
            continue
        tsm_path = tsm_root / f"{annotation.video_id}.json"
        if not tsm_path.exists():
            missing_tsm.append(annotation.video_id)
            continue
        result = compare_annotation(
            annotation,
            json.loads(tsm_path.read_text()),
            tolerances_sec=tolerances_sec,
        )
        result["reference_mode"] = annotation.annotation_mode
        result["reference_json"] = str(reference_path)
        comparisons.append(result)

    boundary_f1 = {}
    for tolerance in tolerances_sec:
        key = f"{tolerance:g}s"
        scores = [
            float(result["boundary_f1"][key]["f1"]) for result in comparisons
        ]
        boundary_f1[key] = mean(scores) if scores else None
    return {
        "reference_mode": reference_mode,
        "reference_dir": str(reference_root.resolve()),
        "n_compared": len(comparisons),
        "missing_tsm_video_ids": sorted(set(missing_tsm)),
        "skipped_wrong_mode": skipped_mode,
        "summary": {
            "mean_fragmentation": _mean_present(
                comparisons, "mean_fragmentation"
            ),
            "mean_over_segmentation": _mean_present(
                comparisons, "mean_over_segmentation"
            ),
            "mean_under_segmentation": _mean_present(
                comparisons, "mean_under_segmentation"
            ),
            "mean_contamination_sec": _mean_present(
                comparisons, "mean_contamination_sec"
            ),
            "mean_contamination_ratio": _mean_present(
                comparisons, "mean_contamination_ratio"
            ),
            "mean_start_boundary_error_sec": _mean_present(
                comparisons, "mean_start_boundary_error_sec"
            ),
            "mean_end_boundary_error_sec": _mean_present(
                comparisons, "mean_end_boundary_error_sec"
            ),
            "mean_per_sample_boundary_f1": boundary_f1,
        },
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agnostic-dir",
        type=Path,
        help="Query-agnostic Gemini dir (default reference for TSM metrics)",
    )
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        help="Query-conditioned oracle dir (legacy reference; mutually exclusive)",
    )
    parser.add_argument("--tsm-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--tolerance-sec", type=float, action="append", dest="tolerances"
    )
    args = parser.parse_args()
    if bool(args.agnostic_dir) == bool(args.oracle_dir):
        raise SystemExit(
            "Pass exactly one of --agnostic-dir (preferred) or --oracle-dir"
        )
    if args.agnostic_dir:
        reference_dir = args.agnostic_dir
        reference_mode = "query_agnostic"
    else:
        reference_dir = args.oracle_dir
        reference_mode = "query_conditioned"

    report = build_report(
        reference_dir,
        args.tsm_dir,
        tolerances_sec=tuple(args.tolerances or (2.0, 5.0, 10.0)),
        reference_mode=reference_mode,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Saved {report['n_compared']} {reference_mode} comparisons to "
        f"{args.output_json}; missing_tsm={len(report['missing_tsm_video_ids'])}"
    )


if __name__ == "__main__":
    main()
