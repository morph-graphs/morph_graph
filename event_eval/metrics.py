"""Metrics for reconstructing query-conditioned oracle events with TSM spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .schema import OracleAnnotation, OracleEvent


def _span(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        value = value.get("time_range_sec", value.get("span_sec"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected [start_sec, end_sec], got {value!r}")
    start, end = float(value[0]), float(value[1])
    if start < 0 or end <= start:
        raise ValueError(f"Invalid span [{start}, {end})")
    return start, end


def _normalize_spans(values: Iterable[Any]) -> list[tuple[float, float]]:
    spans = sorted(_span(value) for value in values)
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1] - 1e-6:
            raise ValueError(f"TSM spans overlap: {previous} and {current}")
    return spans


def compare_event(
    event: OracleEvent,
    tsm_spans: Iterable[Any],
) -> dict[str, Any]:
    """Measure the minimal TSM cover of one oracle event.

    TSM normally partitions the video, so all spans intersecting the oracle
    interval form its minimal cover. ``fragmentation`` is their count (kept for
    compatibility); ``over_segmentation`` is ``max(0, count - 1)`` (extra TSM
    cuts). Contamination is cover duration outside the oracle interval.
    """
    spans = _normalize_spans(tsm_spans)
    selected = [
        (start, end)
        for start, end in spans
        if end > event.start_sec and start < event.end_sec
    ]
    if not selected:
        return {
            "event_id": event.event_id,
            "oracle_span_sec": [event.start_sec, event.end_sec],
            "tsm_cover_sec": None,
            "fragmentation": 0,
            "over_segmentation": 0,
            "covered_sec": 0.0,
            "coverage_ratio": 0.0,
            "contamination_sec": None,
            "contamination_ratio": None,
            "start_boundary_error_sec": None,
            "end_boundary_error_sec": None,
        }

    cover_start, cover_end = selected[0][0], selected[-1][1]
    covered_sec = sum(
        max(0.0, min(end, event.end_sec) - max(start, event.start_sec))
        for start, end in selected
    )
    oracle_duration = event.end_sec - event.start_sec
    cover_duration = sum(end - start for start, end in selected)
    contamination_sec = max(0.0, cover_duration - covered_sec)
    n_tsm = len(selected)
    return {
        "event_id": event.event_id,
        "oracle_span_sec": [event.start_sec, event.end_sec],
        "tsm_cover_sec": [cover_start, cover_end],
        "fragmentation": n_tsm,
        "over_segmentation": max(0, n_tsm - 1),
        "covered_sec": round(covered_sec, 6),
        "coverage_ratio": round(covered_sec / oracle_duration, 6),
        "contamination_sec": round(contamination_sec, 6),
        "contamination_ratio": round(
            contamination_sec / cover_duration if cover_duration else 0.0, 6
        ),
        "start_boundary_error_sec": round(abs(cover_start - event.start_sec), 6),
        "end_boundary_error_sec": round(abs(cover_end - event.end_sec), 6),
    }


def query_event_coverage(
    event: OracleEvent,
    retrieved_spans: Iterable[Any],
) -> dict[str, float | str]:
    """Coverage of an oracle event by retrieved (not all) TSM intervals."""
    clipped = []
    for start, end in _normalize_spans(retrieved_spans):
        overlap_start = max(start, event.start_sec)
        overlap_end = min(end, event.end_sec)
        if overlap_end > overlap_start:
            clipped.append((overlap_start, overlap_end))
    covered_sec = 0.0
    current: tuple[float, float] | None = None
    for start, end in clipped:
        if current is None:
            current = (start, end)
        elif start <= current[1]:
            current = (current[0], max(current[1], end))
        else:
            covered_sec += current[1] - current[0]
            current = (start, end)
    if current is not None:
        covered_sec += current[1] - current[0]
    duration = event.end_sec - event.start_sec
    return {
        "event_id": event.event_id,
        "covered_sec": round(covered_sec, 6),
        "coverage_ratio": round(covered_sec / duration, 6),
    }


def boundary_f1(
    predicted: Iterable[float],
    reference: Iterable[float],
    tolerance_sec: float,
) -> dict[str, float | int]:
    """One-to-one matching of sorted boundaries within an absolute tolerance."""
    if tolerance_sec < 0:
        raise ValueError("tolerance_sec must be non-negative")
    pred = sorted(float(value) for value in predicted)
    ref = sorted(float(value) for value in reference)
    i = j = matches = 0
    while i < len(pred) and j < len(ref):
        delta = pred[i] - ref[j]
        if abs(delta) <= tolerance_sec:
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    precision = matches / len(pred) if pred else (1.0 if not ref else 0.0)
    recall = matches / len(ref) if ref else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches,
        "predicted": len(pred),
        "reference": len(ref),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _internal_boundaries(
    events: Iterable[OracleEvent], duration_sec: float
) -> list[float]:
    return sorted(
        {
            boundary
            for event in events
            for boundary in (event.start_sec, event.end_sec)
            if 0 < boundary < duration_sec
        }
    )


def compare_providers(
    annotation_a: OracleAnnotation,
    annotation_b: OracleAnnotation,
    *,
    tolerances_sec: tuple[float, ...] = (2.0, 5.0, 10.0),
) -> dict[str, Any]:
    """Boundary F1 between two provider annotations on the same video."""
    duration = max(annotation_a.video_duration_sec, annotation_b.video_duration_sec)
    boundaries_a = _internal_boundaries(annotation_a.events, duration)
    boundaries_b = _internal_boundaries(annotation_b.events, duration)
    boundary_scores = {
        f"{tolerance:g}s": boundary_f1(
            boundaries_b, boundaries_a, tolerance
        )
        for tolerance in tolerances_sec
    }
    return {
        "model_a": annotation_a.model,
        "model_b": annotation_b.model,
        "n_events_a": len(annotation_a.events),
        "n_events_b": len(annotation_b.events),
        "boundary_f1": boundary_scores,
    }


def compare_annotation(
    annotation: OracleAnnotation,
    tsm: dict[str, Any],
    *,
    tolerances_sec: tuple[float, ...] = (2.0, 5.0, 10.0),
) -> dict[str, Any]:
    """Compare one validated oracle annotation with a serialized TSM result."""
    annotation.require_valid()
    spans = _normalize_spans(tsm.get("events_sec", []))
    event_results = [compare_event(event, spans) for event in annotation.events]
    covered = [result for result in event_results if result["fragmentation"] > 0]

    # Under-segmentation: extra oracle events mashed into one TSM segment.
    under_values: list[float] = []
    tsm_under: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(spans):
        n_oracle = sum(
            1
            for event in annotation.events
            if end > event.start_sec and start < event.end_sec
        )
        if n_oracle <= 0:
            continue
        under = max(0, n_oracle - 1)
        under_values.append(float(under))
        tsm_under.append(
            {
                "tsm_index": index,
                "tsm_span_sec": [start, end],
                "n_oracle_intersecting": n_oracle,
                "under_segmentation": under,
            }
        )

    # Internal oracle boundaries only. Start/end are scored by reconstruction
    # error above and should not inflate F1 through trivial video edges.
    reference_boundaries = sorted(
        {
            boundary
            for event in annotation.events
            for boundary in (event.start_sec, event.end_sec)
            if 0 < boundary < annotation.video_duration_sec
        }
    )
    predicted_boundaries = [
        float(value)
        for value in tsm.get(
            "boundaries_sec",
            [span[0] for span in spans[1:]],
        )
    ]
    boundary_scores = {
        f"{tolerance:g}s": boundary_f1(
            predicted_boundaries, reference_boundaries, tolerance
        )
        for tolerance in tolerances_sec
    }

    def average(key: str) -> float | None:
        values = [float(result[key]) for result in covered if result[key] is not None]
        return mean(values) if values else None

    return {
        "sample_id": annotation.sample_id,
        "video_id": annotation.video_id,
        "query": annotation.query,
        "n_oracle_events": len(annotation.events),
        "n_tsm_events": len(spans),
        "events": event_results,
        "tsm_under_segmentation": tsm_under,
        "summary": {
            "mean_fragmentation": average("fragmentation"),
            "mean_over_segmentation": average("over_segmentation"),
            "mean_under_segmentation": (
                mean(under_values) if under_values else None
            ),
            "mean_contamination_sec": average("contamination_sec"),
            "mean_contamination_ratio": average("contamination_ratio"),
            "mean_start_boundary_error_sec": average(
                "start_boundary_error_sec"
            ),
            "mean_end_boundary_error_sec": average("end_boundary_error_sec"),
        },
        "boundary_f1": boundary_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-json", type=Path, required=True)
    parser.add_argument("--tsm-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--tolerance-sec", type=float, action="append", dest="tolerances"
    )
    args = parser.parse_args()

    annotation = OracleAnnotation.load(args.oracle_json)
    tsm = json.loads(args.tsm_json.read_text())
    result = compare_annotation(
        annotation,
        tsm,
        tolerances_sec=tuple(args.tolerances or (2.0, 5.0, 10.0)),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved comparison to {args.output_json}")


if __name__ == "__main__":
    main()
