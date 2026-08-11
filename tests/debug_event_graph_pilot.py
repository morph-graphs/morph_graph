"""Run ``debug_event_graph`` across the event-eval EgoLongQA pilot.

Writes one viewer-compatible JSON per sample under
``outputs/event_eval/event_graph/<sample_id>.json`` (retrieved spans + predicted
answer). Load them in the existing event-eval viewer via the EventGraph path.

Example:
    python -m tests.debug_event_graph_pilot \\
        --qa \\
        --sim-threshold 0.9 \\
        --aligned-model pe_core_vit_l

Then:
    python -m event_eval.viewer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ezcolorlog import root_logger as logger

from morph_graph.event_eval.annotate import (
    DEFAULT_QUESTIONS,
    DEFAULT_VIDEO_DIR,
    _filter_samples,
    _load_egolongqa,
)
from morph_graph.tests.debug_event_graph import run_debug_event_graph
from morph_graph.video_models import get_extractor, get_video_llm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PILOT = _REPO_ROOT / "outputs" / "event_eval" / "pilot_ids.json"
_DEFAULT_OUT = _REPO_ROOT / "outputs" / "event_eval" / "event_graph"


def _load_pilot_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    ids: list[str] = []
    for entry in raw:
        if isinstance(entry, dict):
            ids.append(str(entry["sample_id"]))
        else:
            ids.append(str(entry))
    return list(dict.fromkeys(ids))


def _time_range_sec(segment: dict[str, Any], *, fps: float) -> tuple[float, float]:
    tr = segment.get("time_range_sec")
    if isinstance(tr, (list, tuple)) and len(tr) >= 2:
        return float(tr[0]), float(tr[1])
    fr = segment.get("frame_range")
    if isinstance(fr, (list, tuple)) and len(fr) >= 2 and fps > 0:
        return float(fr[0]) / fps, float(fr[1]) / fps
    return 0.0, 0.0


def _serialize_retrieved(
    segments: list[dict[str, Any]] | None,
    *,
    seed_node_ids: list[str] | None,
    fps: float,
) -> list[dict[str, Any]]:
    seeds = set(seed_node_ids or [])
    events: list[dict[str, Any]] = []
    for rank, segment in enumerate(segments or [], start=1):
        if not isinstance(segment, dict):
            continue
        start_sec, end_sec = _time_range_sec(segment, fps=fps)
        node_id = str(segment.get("node_id") or f"seg_{rank}")
        score = segment.get("retrieval_score")
        try:
            score_f = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_f = None
        events.append(
            {
                "event_id": node_id,
                "segment_idx": int(segment.get("segment_idx", rank - 1)),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "retrieval_score": score_f,
                "is_seed": node_id in seeds,
                "rank": rank,
                "frame_range": list(segment.get("frame_range") or []),
                "unit_range": list(segment.get("unit_range") or []),
            }
        )
    return events


def build_viewer_record(
    *,
    sample_id: str,
    query: str,
    fps: float,
    out: dict[str, Any],
) -> dict[str, Any]:
    """Compact JSON the event-eval viewer loads from ``event_graph_dir``."""
    return {
        "sample_id": sample_id,
        "video_id": out["video_id"],
        "video_path": out["video_path"],
        "query": query,
        "n_events": out.get("n_events"),
        "answer": out.get("answer"),
        "qa_error": out.get("qa_error"),
        "seed_node_ids": out.get("seed_node_ids") or [],
        "selected_node_ids": [
            str(seg.get("node_id"))
            for seg in (out.get("segments") or [])
            if isinstance(seg, dict) and seg.get("node_id") is not None
        ],
        "retrieved_events": _serialize_retrieved(
            out.get("segments"),
            seed_node_ids=out.get("seed_node_ids"),
            fps=fps,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-ids-json", type=Path, default=_DEFAULT_PILOT)
    ap.add_argument("--question-path", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help="Viewer event_graph_dir (one <sample_id>.json per pilot sample)",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional EventGraph working dirs (default: <out-dir>/_cache/<id>)",
    )
    ap.add_argument("--model", default="qwen3_vl_8b")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frame-batch-size", type=int, default=128)
    ap.add_argument("--tokens-per-frame", type=int, default=256)
    ap.add_argument("--feature-stage", choices=["post", "pre"], default="post")
    ap.add_argument("--kernel-width", type=int, default=6)
    ap.add_argument("--peak-height", type=float, default=1.0)
    ap.add_argument("--sim-threshold", type=float, default=0.9)
    ap.add_argument("--sim-top-k", type=int, default=None)
    ap.add_argument("--aligned-model", default="pe_core_vit_l")
    ap.add_argument("--qa", action="store_true", default=True)
    ap.add_argument("--no-qa", dest="qa", action="store_false")
    ap.add_argument("--retrieval-top-k", type=int, default=3)
    ap.add_argument("--retrieval-mode", choices=["local", "qafd"], default="local")
    ap.add_argument("--max-segments-for-qa", type=int, default=5)
    ap.add_argument("--max-qa-frames", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip samples that already have <sample_id>.json under --out-dir",
    )
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    sample_ids = _load_pilot_ids(args.pilot_ids_json)
    samples = _filter_samples(
        _load_egolongqa(args.question_path, args.video_dir),
        sample_ids=sample_ids,
    )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No pilot samples resolved")

    cache_root = args.cache_dir or (args.out_dir / "_cache")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"[pilot] {len(samples)} samples from {args.pilot_ids_json} -> {args.out_dir}"
    )

    logger.info(
        f"[load] extractor={args.model} tokens_per_frame={args.tokens_per_frame} "
        f"aligned={args.aligned_model}"
    )
    ext = get_video_llm(
        args.model,
        tokens_per_frame=args.tokens_per_frame,
        feature_stage=args.feature_stage,
    )
    aligned = get_extractor(args.aligned_model) if args.aligned_model else None

    results_path = args.out_dir / "results.jsonl"
    with results_path.open("w") as handle:
        for sample in samples:
            out_json = args.out_dir / f"{sample.sample_id}.json"
            if args.skip_existing and out_json.exists():
                logger.info(f"[skip] sample_id={sample.sample_id}")
                row = json.loads(out_json.read_text())
                handle.write(json.dumps(row, default=str) + "\n")
                handle.flush()
                continue

            logger.info(
                f"[run] sample_id={sample.sample_id} video={sample.video_path.name}"
            )
            ns = SimpleNamespace(
                video=sample.video_path,
                model=args.model,
                fps=args.fps,
                max_frames=args.max_frames,
                frame_batch_size=args.frame_batch_size,
                tokens_per_frame=args.tokens_per_frame,
                feature_stage=args.feature_stage,
                kernel_width=args.kernel_width,
                peak_height=args.peak_height,
                sim_threshold=args.sim_threshold,
                sim_top_k=args.sim_top_k,
                store_patch_tokens=True,
                working_dir=cache_root / sample.sample_id,
                keep_working_dir=False,
                aligned_model=args.aligned_model,
                query=sample.question,
                qa=args.qa,
                retrieval_mode=args.retrieval_mode,
                retrieval_top_k=args.retrieval_top_k,
                max_segments_for_qa=args.max_segments_for_qa,
                max_qa_frames=args.max_qa_frames,
                max_new_tokens=args.max_new_tokens,
                plot_graph=None,
            )
            try:
                out = run_debug_event_graph(ns, ext=ext, aligned=aligned)
                row = build_viewer_record(
                    sample_id=sample.sample_id,
                    query=sample.question,
                    fps=args.fps,
                    out=out,
                )
            except Exception as exc:  # noqa: BLE001 — keep pilot batch going
                logger.exception(f"[fail] sample_id={sample.sample_id}: {exc}")
                row = {
                    "sample_id": sample.sample_id,
                    "video_id": sample.video_path.stem,
                    "video_path": str(sample.video_path.resolve()),
                    "query": sample.question,
                    "answer": None,
                    "qa_error": f"{type(exc).__name__}: {exc}",
                    "seed_node_ids": [],
                    "selected_node_ids": [],
                    "retrieved_events": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            out_json.write_text(json.dumps(row, indent=2, default=str))
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()

    logger.info(f"[done] wrote {len(samples)} records under {args.out_dir}")
    logger.info("View with: python -m event_eval.viewer")


if __name__ == "__main__":
    main()
