"""Generate resumable event annotations (query-conditioned or agnostic)."""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .claude import DEFAULT_MODEL as CLAUDE_DEFAULT_MODEL
from .claude import ClaudeOracleAnnotator
from .gemini import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from .gemini import GeminiOracleAnnotator
from .gpt import DEFAULT_MODEL as GPT_DEFAULT_MODEL
from .gpt import GPTOracleAnnotator
from .kimi import DEFAULT_MODEL as KIMI_DEFAULT_MODEL
from .kimi import KimiOracleAnnotator
from .schema import OracleAnnotation

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = (
    _PACKAGE_ROOT
    / "data"
    / "wearable-ai"
    / "egolongqa"
    / "wearable_ai_2026_egolongqa_val_700.jsonl"
)
DEFAULT_VIDEO_DIR = _PACKAGE_ROOT / "data" / "wearable-ai" / "egolongqa"


@dataclass(frozen=True)
class AnnotationSample:
    sample_id: str
    video_path: Path
    question: str
    answer: str = ""


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "sample"


def _load_egolongqa(
    question_path: Path,
    video_dir: Path,
) -> list[AnnotationSample]:
    samples = []
    with question_path.open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            raw_video = Path(row["video_path"])
            video_path = raw_video if raw_video.is_absolute() else video_dir / raw_video
            if not video_path.exists() and not raw_video.is_absolute():
                video_path = video_dir / "val" / raw_video
            samples.append(
                AnnotationSample(
                    sample_id=str(row.get("qid", row.get("id", index))),
                    video_path=video_path.resolve(),
                    question=str(row["question"]),
                    answer=str(row.get("answer", "")),
                )
            )
    return samples


def _normalize_video_id(value: str) -> str:
    stem = Path(value).name
    if stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    return stem


def _filter_samples(
    samples: list[AnnotationSample],
    *,
    sample_ids: list[str] | None = None,
    video_ids: list[str] | None = None,
) -> list[AnnotationSample]:
    """Filter by sample ID and/or video stem; preserve first-seen order."""
    selected = samples
    if sample_ids:
        wanted = list(dict.fromkeys(str(value) for value in sample_ids))
        by_id = {sample.sample_id: sample for sample in selected}
        missing = [sample_id for sample_id in wanted if sample_id not in by_id]
        if missing:
            raise ValueError(f"Sample IDs missing from benchmark: {missing}")
        selected = [by_id[sample_id] for sample_id in wanted]
    if video_ids:
        wanted_videos = list(
            dict.fromkeys(_normalize_video_id(value) for value in video_ids)
        )
        wanted_set = set(wanted_videos)
        matched = [
            sample
            for sample in selected
            if sample.video_path.stem in wanted_set
        ]
        found = {sample.video_path.stem for sample in matched}
        missing = [video_id for video_id in wanted_videos if video_id not in found]
        if missing:
            raise ValueError(f"Video IDs missing from selection: {missing}")
        # Keep stable order by requested video list, then sample order within.
        by_video: dict[str, list[AnnotationSample]] = {}
        for sample in matched:
            by_video.setdefault(sample.video_path.stem, []).append(sample)
        selected = [
            sample
            for video_id in wanted_videos
            for sample in by_video.get(video_id, ())
        ]
    return selected


def _selected_samples(args: argparse.Namespace) -> list[AnnotationSample]:
    samples = _load_egolongqa(args.question_path, args.video_dir)
    sample_ids: list[str] = list(args.sample_ids or [])
    if args.sample_ids_json:
        manifest = json.loads(args.sample_ids_json.read_text())
        sample_ids.extend(
            str(value["sample_id"] if isinstance(value, dict) else value)
            for value in manifest
        )
    samples = _filter_samples(
        samples,
        sample_ids=sample_ids or None,
        video_ids=args.video_ids or None,
    )
    if args.start_index:
        samples = samples[args.start_index :]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    return samples


def _jobs_for_mode(
    samples: list[AnnotationSample], annotation_mode: str
) -> list[AnnotationSample]:
    """Query-agnostic jobs are unique videos; conditioned jobs keep sample IDs."""
    if annotation_mode == "query_conditioned":
        return samples
    if annotation_mode != "query_agnostic":
        raise ValueError(f"Unknown annotation_mode={annotation_mode!r}")
    by_video: dict[str, AnnotationSample] = {}
    for sample in samples:
        key = str(sample.video_path.resolve())
        if key not in by_video:
            # sample_id becomes the video stem for stable on-disk names.
            by_video[key] = AnnotationSample(
                sample_id=sample.video_path.stem,
                video_path=sample.video_path,
                question="",
                answer="",
            )
    return list(by_video.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["egolongqa"], default="egolongqa")
    parser.add_argument(
        "--mode",
        choices=["query_conditioned", "query_agnostic"],
        default="query_conditioned",
        help=(
            "query_conditioned: one oracle per question (default). "
            "query_agnostic: one segmentation per unique video."
        ),
    )
    parser.add_argument("--question-path", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        dest="sample_ids",
        help="Benchmark sample/question ID to include (repeatable).",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        dest="video_ids",
        help=(
            "Restrict to these video stems or .mp4 names (repeatable). "
            "Useful for re-running one query-agnostic video."
        ),
    )
    parser.add_argument("--sample-ids-json", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--provider",
        choices=["gemini", "kimi", "claude", "gpt"],
        default="gemini",
        help="Which annotator backend to use (default: gemini).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Defaults to the provider's default model "
            f"(gemini={GEMINI_DEFAULT_MODEL!r}, kimi={KIMI_DEFAULT_MODEL!r}, "
            f"claude={CLAUDE_DEFAULT_MODEL!r}, gpt={GPT_DEFAULT_MODEL!r})."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the Kimi/Moonshot API base URL (kimi provider only).",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Video sampling fps (gemini only)."
    )
    parser.add_argument(
        "--media-resolution",
        choices=["low", "medium"],
        default="medium",
        help="Gemini media resolution (gemini only).",
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=32768,
        help="Gemini max output tokens (raise if JSON truncates on long videos)",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay-sec", type=float, default=30.0)
    parser.add_argument("--request-delay-sec", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first annotation failure (default: continue).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = _jobs_for_mode(_selected_samples(args), args.mode)
    if not samples:
        raise ValueError("No benchmark samples selected")

    if args.provider == "gemini":
        annotator = GeminiOracleAnnotator(
            model=args.model or GEMINI_DEFAULT_MODEL,
            fps=args.fps,
            media_resolution=args.media_resolution,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            retry_delay_sec=args.retry_delay_sec,
        )
    elif args.provider == "kimi":
        kimi_kwargs = {"base_url": args.base_url} if args.base_url else {}
        annotator = KimiOracleAnnotator(
            model=args.model or KIMI_DEFAULT_MODEL,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            retry_delay_sec=args.retry_delay_sec,
            **kimi_kwargs,
        )
    elif args.provider == "claude":
        annotator = ClaudeOracleAnnotator(
            model=args.model or CLAUDE_DEFAULT_MODEL,
            fps=args.fps,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            retry_delay_sec=args.retry_delay_sec,
        )
    elif args.provider == "gpt":
        gpt_kwargs = {"base_url": args.base_url} if args.base_url else {}
        annotator = GPTOracleAnnotator(
            model=args.model or GPT_DEFAULT_MODEL,
            fps=args.fps,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            retry_delay_sec=args.retry_delay_sec,
            **gpt_kwargs,
        )
    else:
        raise ValueError(f"Unknown provider: {args.provider!r}")
    completed = skipped = failed = 0
    for index, sample in enumerate(samples, start=1):
        stem = _safe_id(sample.sample_id)
        output_path = args.output_dir / f"{stem}.json"
        error_path = args.output_dir / f"{stem}.error.json"
        if output_path.exists() and not args.overwrite:
            try:
                loaded = OracleAnnotation.load(output_path)
                if loaded.annotation_mode != args.mode:
                    raise ValueError(
                        f"existing file mode={loaded.annotation_mode!r} "
                        f"!= requested {args.mode!r}"
                    )
                print(f"[{index}/{len(samples)}] skip valid {output_path.name}")
                skipped += 1
                continue
            except (ValueError, KeyError, json.JSONDecodeError):
                print(f"[{index}/{len(samples)}] regenerate invalid {output_path.name}")

        print(
            f"[{index}/{len(samples)}] annotate mode={args.mode} "
            f"id={sample.sample_id} video={sample.video_path.name}"
        )
        try:
            annotation = annotator.annotate(
                sample_id=sample.sample_id,
                video_path=sample.video_path,
                question=sample.question,
                annotation_mode=args.mode,
            )
            if args.mode == "query_conditioned":
                annotation = annotation.with_answer_check(sample.answer or None)
            annotation.save(output_path)
            error_path.unlink(missing_ok=True)
            completed += 1
            if args.mode == "query_conditioned":
                print(
                    f"  answer_match={annotation.answer_match} "
                    f"predicted={annotation.predicted_answer!r} "
                    f"gt={annotation.gt_answer!r}"
                )
        except Exception as exc:
            failed += 1
            error = {
                "sample_id": sample.sample_id,
                "video_path": str(sample.video_path),
                "query": sample.question,
                "annotation_mode": args.mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            # GeminiAnnotationError (and any exc with these attrs) keeps the
            # last model payload for debugging hierarchy / JSON failures.
            raw_response = getattr(exc, "raw_response", None)
            parsed_events = getattr(exc, "parsed_events", None)
            if raw_response is not None:
                error["raw_response"] = raw_response
            if parsed_events is not None:
                error["parsed_events"] = parsed_events
            error_path.write_text(json.dumps(error, indent=2) + "\n")
            print(f"  failed: {exc}")
            if args.fail_fast:
                raise
        if args.request_delay_sec > 0 and index < len(samples):
            time.sleep(args.request_delay_sec)

    print(
        f"Done: mode={args.mode} completed={completed} skipped={skipped} "
        f"failed={failed} output_dir={args.output_dir}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
