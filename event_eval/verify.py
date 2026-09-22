"""Verify query-conditioned oracles by answering the MCQ from the oracle spans only.

The verifier sees the video clipped to the oracle event intervals — never the
full video and never the ground-truth answer — and picks an answer option. This
tests whether the oracle spans are sufficient and discriminative: if a retrieval
system returned exactly these intervals, is the answer recoverable? It does not
test boundary precision, role labels, or over-segmentation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .annotate import DEFAULT_QUESTIONS, _safe_id
from .gemini import DEFAULT_MODEL, video_duration_sec
from .schema import OracleAnnotation

VERIFY_PROMPT_VERSION = "verify_cropped_v2"
SUFFICIENCY_LEVELS = ("sufficient", "partial", "insufficient")
VERDICTS = ("pass", "uncertain", "uninformative", "fail")
# Each event becomes its own clip part; conditioned oracles are minimal, so a
# large count usually means the wrong directory was passed.
DEFAULT_MAX_CLIPS = 32
# Above this share of the video, the crop approximates the full video and a
# correct answer no longer evidences that the spans were well chosen.
DEFAULT_COVERAGE_THRESHOLD = 0.5

_OPTION_MARKER = re.compile(r"(?:(?<=^)|(?<=\s))([A-Z])\.\s+")


class VerificationError(RuntimeError):
    """Raised after retries are exhausted; carries the last model payload."""

    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def parse_mcq_options(raw: str) -> dict[str, str]:
    """Split ``"A. foo B. bar ..."`` into ``{"A": "foo", "B": "bar"}``.

    Only markers continuing the ascending A, B, C ... sequence are treated as
    option boundaries, so periods inside option text do not split it.
    """
    kept: list[re.Match[str]] = []
    expected = "A"
    for match in _OPTION_MARKER.finditer(raw):
        if match.group(1) == expected:
            kept.append(match)
            expected = chr(ord(expected) + 1)
    if len(kept) < 2:
        raise ValueError(f"Could not parse at least two options from {raw!r}")
    options: dict[str, str] = {}
    for index, match in enumerate(kept):
        end = kept[index + 1].start() if index + 1 < len(kept) else len(raw)
        options[match.group(1)] = raw[match.end() : end].strip()
    return options


def shuffled_options(
    options: dict[str, str], *, seed: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Re-letter options in a deterministic shuffle.

    Returns ``(presented, presented_to_original)`` so the original letter cannot
    leak through option position.
    """
    original_letters = sorted(options)
    order = list(original_letters)
    random.Random(seed).shuffle(order)
    presented: dict[str, str] = {}
    presented_to_original: dict[str, str] = {}
    for index, original in enumerate(order):
        letter = chr(ord("A") + index)
        presented[letter] = options[original]
        presented_to_original[letter] = original
    return presented, presented_to_original


def _strictly_contains(outer, inner) -> bool:
    return (
        outer.start_sec <= inner.start_sec
        and inner.end_sec <= outer.end_sec
        and (outer.start_sec < inner.start_sec or inner.end_sec < outer.end_sec)
    )


def leaf_events(annotation: OracleAnnotation) -> tuple:
    """Events that do not temporally contain another event.

    Coarse wrappers that swallow fine children inflate coverage without adding
    unique footage. Leaves keep standalone coarse beats (e.g. a 55s manual
    overview with no child) and drop umbrellas like ``[125–600]`` around a
    15s answer-bearing span.
    """
    events = annotation.events
    if not events:
        return ()
    umbrellas = {
        event.event_id
        for event in events
        if any(
            other.event_id != event.event_id and _strictly_contains(event, other)
            for other in events
        )
    }
    leaves = tuple(event for event in events if event.event_id not in umbrellas)
    return leaves or events


def _union_sec(spans: list[tuple[float, float]]) -> float:
    ordered = sorted(spans)
    if not ordered:
        return 0.0
    covered = 0.0
    current_start, current_end = None, None
    for start, end in ordered:
        if current_end is None or start > current_end:
            if current_end is not None:
                covered += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        covered += current_end - current_start
    return covered


def coverage_ratio(annotation: OracleAnnotation) -> float:
    """Union of *leaf* event intervals divided by video duration.

    A low ratio with a correct answer is the strong result: the oracle kept
    little of the video and the answer survived.
    """
    if annotation.video_duration_sec <= 0:
        return 0.0
    covered = _union_sec(
        (event.start_sec, event.end_sec) for event in leaf_events(annotation)
    )
    return min(1.0, covered / annotation.video_duration_sec)


def verification_response_schema(presented_letters: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected_option": {"type": "string", "enum": presented_letters},
            "supporting_event_ids": {"type": "array", "items": {"type": "string"}},
            "sufficiency": {"type": "string", "enum": list(SUFFICIENCY_LEVELS)},
            "missing_evidence": {"type": "string"},
        },
        "required": [
            "selected_option",
            "supporting_event_ids",
            "sufficiency",
            "missing_evidence",
        ],
        "additionalProperties": False,
    }


def build_verification_prompt(
    question: str, presented: dict[str, Any], n_clips: int
) -> str:
    lines = "\n".join(
        f"{letter}. {text}" for letter, text in sorted(presented.items())
    )
    return f"""You are given {n_clips} short video clip(s) taken from one longer
video. Each clip was labelled above with the event it came from and its original
time range. You cannot see any footage outside these clips.

Question:
{question.strip()}

Options:
{lines}

Answer the question using only the clips shown. Do not guess from general
knowledge about what such a video usually contains: if the clips do not show the
evidence, say so through `sufficiency` rather than picking a plausible option.

Return:
- `selected_option`: the letter best supported by the clips.
- `supporting_event_ids`: the event IDs whose clips justify that choice.
- `sufficiency`: `sufficient` if the clips fully establish the answer, `partial`
  if they narrow it but leave the decisive detail unseen, `insufficient` if the
  answer cannot be established from the clips.
- `missing_evidence`: what footage would be needed, or "" when sufficient.

Return no chain-of-thought."""


@dataclass(frozen=True)
class VerificationResult:
    sample_id: str
    video_id: str
    question: str
    gt_option: str
    selected_option: str
    correct: bool
    verdict: str
    sufficiency: str
    missing_evidence: str
    supporting_event_ids: tuple[str, ...]
    unknown_event_ids: tuple[str, ...]
    cited_roles: tuple[str, ...]
    cited_answer_bearing: bool
    n_events: int
    n_leaf_events: int
    coverage_ratio: float
    coverage_threshold: float
    video_duration_sec: float
    covered_sec: float
    options: dict[str, str]
    presented_to_original: dict[str, str]
    selected_presented_letter: str
    model: str
    prompt_version: str
    verified_at: str
    oracle_prompt_version: str
    raw_response: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def decide_verdict(
    *,
    correct: bool,
    sufficiency: str,
    cited: list[str],
    unknown: list[str],
    coverage: float,
    cited_answer_bearing: bool,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> str:
    """pass = right answer for a stated, checkable reason.

    ``uninformative`` outranks ``pass`` because a high-coverage crop approximates
    the full video, so recovering the answer says little about the spans. A
    correct answer justified only by ``anchor`` events is downgraded too: it
    means the cited footage does not contain what was asked for.
    """
    if not correct:
        return "fail"
    if coverage >= coverage_threshold:
        return "uninformative"
    if sufficiency == "sufficient" and cited and not unknown and cited_answer_bearing:
        return "pass"
    return "uncertain"


def _load_questions(question_path: Path) -> dict[str, dict[str, Any]]:
    """Map sample_id -> question row, using annotate.py's sample_id derivation."""
    rows: dict[str, dict[str, Any]] = {}
    with question_path.open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row.get("qid", row.get("id", index)))] = row
    return rows


class OracleVerifier:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        fps: float = 1.0,
        media_resolution: str | None = "medium",
        max_output_tokens: int = 4096,
        max_retries: int = 5,
        retry_delay_sec: float = 30.0,
        max_clips: int = DEFAULT_MAX_CLIPS,
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if media_resolution not in {None, "low", "medium"}:
            raise ValueError("video media_resolution must be low, medium, or None")
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass api_key")
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise ImportError("Install Gemini support with: pip install google-genai") from exc

        self.model = model
        self.fps = float(fps)
        self.media_resolution = media_resolution
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = int(max_retries)
        self.retry_delay_sec = float(retry_delay_sec)
        self.max_clips = int(max_clips)
        self.coverage_threshold = float(coverage_threshold)
        self._types = types
        self._errors = errors
        self._client = genai.Client(api_key=api_key)

    def _wait_until_active(self, uploaded: Any) -> Any:
        while True:
            uploaded = self._client.files.get(name=uploaded.name)
            state = getattr(uploaded.state, "name", str(uploaded.state))
            if state == "ACTIVE":
                return uploaded
            if state == "FAILED":
                raise RuntimeError(f"Gemini upload failed: {uploaded.name}")
            time.sleep(5)

    def _delete_file(self, name: str) -> None:
        try:
            self._client.files.delete(name=name)
        except Exception:
            pass

    def _clip_contents(
        self, uploaded: Any, annotation: OracleAnnotation
    ) -> list[Any]:
        """One labelled video part per event, clipped server-side via offsets."""
        mime_type = getattr(uploaded, "mime_type", None) or "video/mp4"
        contents: list[Any] = []
        events = sorted(leaf_events(annotation), key=lambda event: event.start_sec)
        for index, event in enumerate(events, start=1):
            start = max(0.0, event.start_sec)
            end = min(annotation.video_duration_sec, event.end_sec)
            if end <= start:
                continue
            contents.append(
                f"Clip {index} — event {event.event_id} "
                f"[{start:.3f}s–{end:.3f}s]:"
            )
            part = self._types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime_type)
            part.video_metadata = self._types.VideoMetadata(
                fps=self.fps,
                start_offset=f"{start:.3f}s",
                end_offset=f"{end:.3f}s",
            )
            contents.append(part)
        return contents

    def _config(self, presented_letters: list[str]) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
            "response_json_schema": verification_response_schema(presented_letters),
        }
        if self.media_resolution:
            kwargs["media_resolution"] = {
                "low": self._types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": self._types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
            }[self.media_resolution]
        return self._types.GenerateContentConfig(**kwargs)

    def verify(
        self,
        annotation: OracleAnnotation,
        *,
        question: str,
        options: dict[str, str],
        gt_option: str,
        video_path: str | Path | None = None,
    ) -> VerificationResult:
        if annotation.annotation_mode != "query_conditioned":
            raise ValueError(
                f"verification expects a query-conditioned oracle, got "
                f"{annotation.annotation_mode!r}"
            )
        leaves = leaf_events(annotation)
        if len(leaves) > self.max_clips:
            raise ValueError(
                f"{len(leaves)} leaf events exceeds max_clips="
                f"{self.max_clips}; raise --max-clips if this is intended"
            )
        path = Path(video_path or annotation.video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if annotation.video_duration_sec <= 0:
            annotation = OracleAnnotation.from_dict(
                {**annotation.to_dict(), "video_duration_sec": video_duration_sec(path)}
            )

        presented, presented_to_original = shuffled_options(
            options, seed=annotation.sample_id
        )
        presented_letters = sorted(presented)
        roles_by_id = {
            event.event_id: event.query_roles for event in annotation.events
        }
        known_ids = set(roles_by_id)

        uploaded = self._wait_until_active(self._client.files.upload(file=str(path)))
        try:
            clips = self._clip_contents(uploaded, annotation)
            n_clips = sum(1 for item in clips if not isinstance(item, str))
            contents = [
                *clips,
                build_verification_prompt(question, presented, n_clips),
            ]
            config = self._config(presented_letters)

            last_error: Exception | None = None
            last_raw_text: str | None = None
            for attempt in range(self.max_retries):
                try:
                    response = self._client.models.generate_content(
                        model=self.model, contents=contents, config=config
                    )
                    raw_text = (response.text or "").strip()
                    last_raw_text = raw_text
                    parsed = json.loads(raw_text)

                    selected_presented = str(parsed["selected_option"]).strip().upper()
                    if selected_presented not in presented_to_original:
                        raise ValueError(
                            f"model returned option {selected_presented!r} outside "
                            f"{presented_letters}"
                        )
                    selected = presented_to_original[selected_presented]
                    cited = [
                        str(value).strip()
                        for value in parsed.get("supporting_event_ids", [])
                        if str(value).strip()
                    ]
                    unknown = [value for value in cited if value not in known_ids]
                    cited_roles = tuple(
                        dict.fromkeys(
                            role
                            for value in cited
                            for role in roles_by_id.get(value, ())
                        )
                    )
                    cited_answer_bearing = "answer_bearing" in cited_roles
                    correct = selected == gt_option
                    ratio = coverage_ratio(annotation)
                    return VerificationResult(
                        sample_id=annotation.sample_id,
                        video_id=annotation.video_id,
                        question=question,
                        gt_option=gt_option,
                        selected_option=selected,
                        correct=correct,
                        verdict=decide_verdict(
                            correct=correct,
                            sufficiency=str(parsed["sufficiency"]),
                            cited=cited,
                            unknown=unknown,
                            coverage=ratio,
                            cited_answer_bearing=cited_answer_bearing,
                            coverage_threshold=self.coverage_threshold,
                        ),
                        sufficiency=str(parsed["sufficiency"]),
                        missing_evidence=str(parsed.get("missing_evidence", "")).strip(),
                        supporting_event_ids=tuple(cited),
                        unknown_event_ids=tuple(unknown),
                        cited_roles=cited_roles,
                        cited_answer_bearing=cited_answer_bearing,
                        n_events=len(annotation.events),
                        n_leaf_events=len(leaves),
                        coverage_ratio=ratio,
                        coverage_threshold=self.coverage_threshold,
                        video_duration_sec=annotation.video_duration_sec,
                        covered_sec=ratio * annotation.video_duration_sec,
                        options=dict(options),
                        presented_to_original=dict(presented_to_original),
                        selected_presented_letter=selected_presented,
                        model=self.model,
                        prompt_version=VERIFY_PROMPT_VERSION,
                        verified_at=datetime.now(timezone.utc).isoformat(),
                        oracle_prompt_version=annotation.prompt_version,
                        raw_response=raw_text,
                    )
                except Exception as exc:
                    last_error = exc
                    if (
                        isinstance(exc, self._errors.APIError)
                        and 400 <= getattr(exc, "code", 0) < 500
                        and getattr(exc, "code", 0) != 429
                    ):
                        raise
                    if attempt + 1 < self.max_retries:
                        time.sleep(self.retry_delay_sec)
            raise VerificationError(
                f"Verification failed after {self.max_retries} attempts: {last_error}",
                raw_response=last_raw_text,
            ) from last_error
        finally:
            self._delete_file(uploaded.name)


def summarize(results: list[VerificationResult]) -> dict[str, Any]:
    total = len(results)
    if not total:
        return {"n_verified": 0}
    counts = {verdict: 0 for verdict in VERDICTS}
    for result in results:
        counts[result.verdict] += 1
    correct = [result for result in results if result.correct]
    # Items whose crop stayed small enough for the answer to mean something.
    informative = [
        result
        for result in results
        if result.coverage_ratio < result.coverage_threshold
    ]
    informative_correct = [r for r in informative if r.correct]
    return {
        "n_verified": total,
        "verdicts": counts,
        "pass_rate": counts["pass"] / total,
        "accuracy": len(correct) / total,
        "n_informative": len(informative),
        "informative_rate": len(informative) / total,
        # The headline number: accuracy restricted to low-coverage crops.
        "informative_accuracy": (
            len(informative_correct) / len(informative) if informative else None
        ),
        "mean_coverage_ratio": sum(r.coverage_ratio for r in results) / total,
        "mean_coverage_ratio_correct": (
            sum(r.coverage_ratio for r in correct) / len(correct) if correct else None
        ),
        "sufficiency_counts": {
            level: sum(1 for r in results if r.sufficiency == level)
            for level in SUFFICIENCY_LEVELS
        },
        "n_correct_without_answer_bearing_citation": sum(
            1 for r in correct if not r.cited_answer_bearing
        ),
        "failed_sample_ids": sorted(r.sample_id for r in results if not r.correct),
        "uninformative_sample_ids": sorted(
            r.sample_id for r in results if r.verdict == "uninformative"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-path", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        dest="sample_ids",
        help="Restrict to these oracle sample IDs (repeatable).",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--media-resolution", choices=["low", "medium"], default="medium"
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-clips", type=int, default=DEFAULT_MAX_CLIPS)
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help=(
            "Correct answers at or above this coverage are marked "
            "'uninformative' (the crop approximates the full video)."
        ),
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay-sec", type=float, default=30.0)
    parser.add_argument("--request-delay-sec", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first verification failure (default: continue).",
    )
    return parser


def _oracle_paths(oracle_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in oracle_dir.glob("*.json")
            if not path.name.endswith(".error.json")
        ),
        key=lambda path: path.stem,
    )


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = _load_questions(args.question_path)

    paths = _oracle_paths(args.oracle_dir)
    if args.sample_ids:
        wanted = {str(value) for value in args.sample_ids}
        paths = [path for path in paths if path.stem in wanted]
    if args.max_samples is not None:
        paths = paths[: args.max_samples]
    if not paths:
        raise ValueError(f"No oracle JSON files selected under {args.oracle_dir}")

    verifier = OracleVerifier(
        model=args.model,
        fps=args.fps,
        media_resolution=args.media_resolution,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        retry_delay_sec=args.retry_delay_sec,
        max_clips=args.max_clips,
        coverage_threshold=args.coverage_threshold,
    )

    results: list[VerificationResult] = []
    skipped = failed = 0
    for index, oracle_path in enumerate(paths, start=1):
        stem = _safe_id(oracle_path.stem)
        output_path = args.output_dir / f"{stem}.json"
        error_path = args.output_dir / f"{stem}.error.json"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(paths)}] skip existing {output_path.name}")
            skipped += 1
            continue

        try:
            annotation = OracleAnnotation.load(oracle_path)
            row = questions.get(annotation.sample_id)
            if row is None:
                raise KeyError(
                    f"sample_id={annotation.sample_id!r} not in {args.question_path}"
                )
            options = parse_mcq_options(str(row["mcq_options"]))
            gt_option = str(row["mcq_answer"]).strip().upper()
            if gt_option not in options:
                raise ValueError(
                    f"mcq_answer={gt_option!r} not among options {sorted(options)}"
                )
            print(
                f"[{index}/{len(paths)}] verify id={annotation.sample_id} "
                f"events={len(annotation.events)} video={annotation.video_id}"
            )
            result = verifier.verify(
                annotation,
                question=str(row["question"]),
                options=options,
                gt_option=gt_option,
            )
            result.save(output_path)
            error_path.unlink(missing_ok=True)
            results.append(result)
            print(
                f"  verdict={result.verdict} selected={result.selected_option} "
                f"gt={result.gt_option} sufficiency={result.sufficiency} "
                f"coverage={result.coverage_ratio:.2%} "
                f"cited_answer_bearing={result.cited_answer_bearing}"
            )
        except Exception as exc:
            failed += 1
            error: dict[str, Any] = {
                "oracle_json": str(oracle_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            raw_response = getattr(exc, "raw_response", None)
            if raw_response is not None:
                error["raw_response"] = raw_response
            error_path.write_text(json.dumps(error, indent=2) + "\n")
            print(f"  failed: {exc}")
            if args.fail_fast:
                raise
        if args.request_delay_sec > 0 and index < len(paths):
            time.sleep(args.request_delay_sec)

    summary = summarize(results)
    summary["oracle_dir"] = str(args.oracle_dir.resolve())
    summary["model"] = args.model
    summary["prompt_version"] = VERIFY_PROMPT_VERSION
    summary["n_skipped"] = skipped
    summary["n_errors"] = failed
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"Done: verified={summary.get('n_verified', 0)} skipped={skipped} "
        f"errors={failed} -> {summary_path}"
    )
    if results:
        informative_accuracy = summary["informative_accuracy"]
        print(
            f"  accuracy={summary['accuracy']:.2%} pass_rate={summary['pass_rate']:.2%} "
            f"mean_coverage={summary['mean_coverage_ratio']:.2%}"
        )
        print(
            "  informative "
            f"({summary['n_informative']}/{summary['n_verified']} below "
            f"coverage {args.coverage_threshold:.0%}): "
            + (
                f"accuracy={informative_accuracy:.2%}"
                if informative_accuracy is not None
                else "n/a"
            )
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
