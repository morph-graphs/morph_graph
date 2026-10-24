"""Stdlib HTTP server for the event-eval HTML viewer.

Serves the static UI, JSON APIs over configurable oracle/TSM/report paths,
and byte-range video streaming from absolute paths recorded in oracle JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ..metrics import compare_annotation
from ..schema import OracleAnnotation
from ..verify import DEFAULT_COVERAGE_THRESHOLD, decide_verdict

_FRAME_CACHE_DIR = Path(tempfile.gettempdir()) / "event_eval_viewer_frames"
_FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".m4v"}
_DEFAULT_FRAME_WIDTH = 720
_WARM_LOCK = threading.Lock()
_WARM_JOBS: dict[str, dict[str, Any]] = {}

# morph_graph/ (parent of event_eval/)
_MORPH_GRAPH_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = Path(__file__).resolve().parent / "static"

DEFAULT_PATHS = {
    "oracle_dir": str(_MORPH_GRAPH_ROOT / "outputs" / "event_eval" / "oracles"),
    "agnostic_dir": str(
        _MORPH_GRAPH_ROOT / "outputs" / "event_eval" / "agnostic_gemini_latest"
    ),
    "tsm_dir": str(_MORPH_GRAPH_ROOT / "outputs" / "event_eval" / "tsm"),
    "event_graph_dir": str(
        _MORPH_GRAPH_ROOT / "outputs" / "event_eval" / "event_graph"
    ),
    "verification_dir": str(
        _MORPH_GRAPH_ROOT / "outputs" / "event_eval" / "verification"
    ),
    "questions_jsonl": str(
        _MORPH_GRAPH_ROOT
        / "data"
        / "wearable-ai"
        / "egolongqa"
        / "wearable_ai_2026_egolongqa_val_700.jsonl"
    ),
}


def _annotation_directory_options() -> dict[str, list[str]]:
    """Return annotation directories available under outputs/event_eval."""
    output_root = _MORPH_GRAPH_ROOT / "outputs" / "event_eval"
    if not output_root.is_dir():
        return {"oracle_dirs": [], "agnostic_dirs": []}

    directories = sorted(
        (path for path in output_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    )

    def relative_paths(prefix: str) -> list[str]:
        return [
            str(path.relative_to(_MORPH_GRAPH_ROOT))
            for path in directories
            if path.name.lower().startswith(prefix)
        ]

    return {
        "oracle_dirs": relative_paths("oracle"),
        "agnostic_dirs": relative_paths("agnostic"),
    }


def _resolve_path(raw: str | None, default_key: str) -> Path:
    text = (raw or "").strip() or DEFAULT_PATHS[default_key]
    path = Path(text).expanduser()
    if not path.is_absolute():
        # Inputs are relative to morph_graph/. Tolerate a duplicated
        # ``morph_graph/`` prefix from path pickers / @-mentions.
        parts = path.parts
        if parts and parts[0] in {"morph_graph", _MORPH_GRAPH_ROOT.name}:
            path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        path = (_MORPH_GRAPH_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _list_oracle_files(oracle_dir: Path) -> list[Path]:
    if not oracle_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in oracle_dir.glob("*.json")
            if not path.name.endswith(".error.json")
        ),
        key=lambda path: (
            int(path.stem) if path.stem.isdigit() else path.stem
        ),
    )


def _parse_rate(raw: str) -> float:
    text = (raw or "").strip()
    if not text or text == "0/0":
        return 0.0
    if "/" in text:
        num_s, den_s = text.split("/", 1)
        num, den = float(num_s), float(den_s)
        return num / den if den else 0.0
    return float(text)


def probe_video(path: Path) -> dict[str, float]:
    """Return duration_sec and fps for frame scrubbing."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or [{}]
    fmt = payload.get("format") or {}
    duration = float(fmt.get("duration") or 0.0)
    fps = _parse_rate(streams[0].get("avg_frame_rate", ""))
    if fps <= 1e-3:
        fps = _parse_rate(streams[0].get("r_frame_rate", ""))
    if fps <= 1e-3:
        fps = 1.0
    if duration <= 0:
        raise ValueError(f"Could not probe duration for {path}")
    n_frames = max(1, int(round(duration * fps)))
    return {
        "duration_sec": duration,
        "fps": fps,
        "n_frames": float(n_frames),
    }


def _video_cache_id(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _fps_tag(fps: float) -> str:
    return f"{float(fps):.4f}".rstrip("0").rstrip(".")


def _strip_dir(path: Path, fps: float, max_width: int) -> Path:
    out = (
        _FRAME_CACHE_DIR
        / _video_cache_id(path)
        / f"w{int(max_width)}_fps{_fps_tag(fps)}"
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def _frame_index(t_sec: float, fps: float) -> int:
    return max(0, int(round(max(0.0, float(t_sec)) * float(fps))))


def _scale_filter(max_width: int) -> str:
    w = max(160, int(max_width))
    return f"scale='min({w},iw)':-2"


def _count_cached_frames(strip: Path) -> int:
    return sum(1 for p in strip.glob("*.jpg") if p.stat().st_size > 0)


def extract_frame_jpeg(
    path: Path,
    t_sec: float,
    *,
    fps: float = 1.0,
    max_width: int = _DEFAULT_FRAME_WIDTH,
) -> bytes:
    """Extract one JPEG at the scrub grid (disk-cached, downscaled)."""
    fps = max(1e-3, float(fps))
    max_width = max(160, int(max_width))
    idx = _frame_index(t_sec, fps)
    t_aligned = idx / fps
    cache_path = _strip_dir(path, fps, max_width) / f"{idx:06d}.jpg"
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t_aligned:.3f}",
        "-i",
        str(path),
        "-an",
        "-sn",
        "-frames:v",
        "1",
        "-vf",
        _scale_filter(max_width),
        "-q:v",
        "5",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(
            (result.stderr or b"ffmpeg frame extract failed").decode(
                "utf-8", errors="replace"
            )[-400:]
        )
    tmp_path = cache_path.with_suffix(".partial.jpg")
    tmp_path.write_bytes(result.stdout)
    tmp_path.replace(cache_path)
    return result.stdout


def _warm_job_key(path: Path, fps: float, max_width: int) -> str:
    return f"{_video_cache_id(path)}|w{int(max_width)}|fps{_fps_tag(fps)}"


def warm_status(
    path: Path,
    fps: float,
    max_width: int = _DEFAULT_FRAME_WIDTH,
    duration_sec: float | None = None,
) -> dict[str, Any]:
    fps = max(1e-3, float(fps))
    max_width = max(160, int(max_width))
    strip = _strip_dir(path, fps, max_width)
    cached = _count_cached_frames(strip)
    if duration_sec is None:
        try:
            duration_sec = float(probe_video(path)["duration_sec"])
        except Exception:  # noqa: BLE001
            duration_sec = 0.0
    total = max(1, int(duration_sec * fps) + 1) if duration_sec > 0 else max(cached, 1)
    key = _warm_job_key(path, fps, max_width)
    with _WARM_LOCK:
        job = dict(_WARM_JOBS.get(key) or {})
    status = job.get("status")
    if status not in {"running", "error", "done"}:
        status = "done" if cached >= total else "idle"
    return {
        "status": status,
        "cached": cached,
        "total": total,
        "fps": fps,
        "max_width": max_width,
        "error": job.get("error"),
    }


def start_warm_frames(
    path: Path,
    fps: float,
    max_width: int = _DEFAULT_FRAME_WIDTH,
) -> dict[str, Any]:
    """Warm the scrub-FPS JPEG strip in a background thread (one ffmpeg pass)."""
    fps = max(1e-3, float(fps))
    max_width = max(160, int(max_width))
    meta = probe_video(path)
    duration = float(meta["duration_sec"])
    key = _warm_job_key(path, fps, max_width)
    strip = _strip_dir(path, fps, max_width)
    status = warm_status(path, fps, max_width, duration_sec=duration)
    if status["status"] == "done" or status["cached"] >= status["total"]:
        with _WARM_LOCK:
            _WARM_JOBS[key] = {"status": "done", "error": None}
        return status
    with _WARM_LOCK:
        job = _WARM_JOBS.get(key)
        if job and job.get("status") == "running":
            return warm_status(path, fps, max_width, duration_sec=duration)
        _WARM_JOBS[key] = {"status": "running", "error": None}

    def _run() -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-an",
            "-sn",
            "-vf",
            f"fps={fps},{_scale_filter(max_width)}",
            "-q:v",
            "5",
            "-start_number",
            "0",
            str(strip / "%06d.jpg"),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True)
        with _WARM_LOCK:
            if result.returncode != 0:
                _WARM_JOBS[key] = {
                    "status": "error",
                    "error": (result.stderr or b"warm failed").decode(
                        "utf-8", errors="replace"
                    )[-400:],
                }
            else:
                _WARM_JOBS[key] = {"status": "done", "error": None}

    threading.Thread(target=_run, name=f"warm-{key[:12]}", daemon=True).start()
    return warm_status(path, fps, max_width, duration_sec=duration)


# Root for resolving *relative* video paths in annotation JSONs (portable
# across clusters). Set via --video-root; None means relative paths only
# resolve if they happen to exist relative to the CWD.
VIDEO_ROOT: Path | None = None


def _apply_video_root(path: Path) -> Path:
    if not path.is_absolute() and VIDEO_ROOT is not None:
        return VIDEO_ROOT / path
    return path


def _resolve_video_path(raw: str) -> Path:
    path = _apply_video_root(Path(unquote(raw)).expanduser()).resolve()
    if path.suffix.lower() not in _VIDEO_EXTS:
        raise ValueError("Unsupported video type")
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def _load_tsm(tsm_dir: Path, video_id: str) -> dict[str, Any] | None:
    path = tsm_dir / f"{video_id}.json"
    if not path.is_file():
        return None
    return _read_json(path)


def _load_event_graph(
    event_graph_dir: Path | None, sample_id: str
) -> dict[str, Any] | None:
    """Load EventGraph pilot/retrieval export keyed by sample_id."""
    if not event_graph_dir or not sample_id or not event_graph_dir.is_dir():
        return None
    path = event_graph_dir / f"{sample_id}.json"
    if not path.is_file():
        return None
    return _read_json(path)


def _format_event_graph(
    record: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Return (retrieved_events, predicted_answer, qa_error)."""
    if record is None:
        return [], None, None
    events: list[dict[str, Any]] = []
    for index, event in enumerate(record.get("retrieved_events") or []):
        events.append(
            {
                "event_id": event.get("event_id", f"eg_{index}"),
                "segment_idx": event.get("segment_idx", index),
                "start_sec": float(event["start_sec"]),
                "end_sec": float(event["end_sec"]),
                "retrieval_score": event.get("retrieval_score"),
                "is_seed": bool(event.get("is_seed")),
                "rank": event.get("rank", index + 1),
                "frame_range": event.get("frame_range") or [],
                "unit_range": event.get("unit_range") or [],
            }
        )
    answer = record.get("answer")
    qa_error = record.get("qa_error") or record.get("error")
    return events, (None if answer is None else str(answer)), (
        None if qa_error is None else str(qa_error)
    )


def _parse_event_graph_dirs(raw: str | None) -> list[tuple[str, Path]]:
    """Parse multiline EventGraph dirs; optional ``label|path`` per line."""
    text = (raw or "").strip()
    if not text:
        path = _resolve_path(None, "event_graph_dir")
        return [(path.name, path)]

    entries: list[tuple[str, Path]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        label: str | None = None
        path_text = line
        if "|" in line:
            label, path_text = line.split("|", 1)
            label = label.strip() or None
            path_text = path_text.strip()
        path = _resolve_path(path_text, "event_graph_dir")
        entries.append((label or path.name, path))
    return entries


def _event_graph_run(
    label: str, event_graph_dir: Path, sample_id: str
) -> dict[str, Any]:
    record = _load_event_graph(event_graph_dir, sample_id)
    events, pred_answer, qa_error = _format_event_graph(record)
    return {
        "label": label,
        "path": str(event_graph_dir),
        "events": events,
        "pred_answer": pred_answer,
        "qa_error": qa_error,
        "has_event_graph": record is not None,
        "json_path": (
            str(event_graph_dir / f"{sample_id}.json") if record is not None else None
        ),
    }


def _event_graph_dirs_from_query(q: dict[str, str]) -> list[tuple[str, Path]]:
    raw = q.get("event_graph_dirs") or q.get("event_graph_dir") or ""
    return _parse_event_graph_dirs(raw)


def _load_agnostic(agnostic_dir: Path, video_id: str) -> dict[str, Any] | None:
    """Load query-agnostic annotation keyed by video_id."""
    if not video_id or not agnostic_dir.is_dir():
        return None
    path = agnostic_dir / f"{video_id}.json"
    if not path.is_file():
        return None
    data = _read_json(path)
    if data.get("annotation_mode") not in {None, "query_agnostic"}:
        # Allow older files without the field if they look agnostic.
        if data.get("query"):
            return None
    return data


def _format_agnostic_events(
    agnostic: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    if agnostic is None:
        return [], None, None
    events = []
    for index, event in enumerate(agnostic.get("events", [])):
        events.append(
            {
                "event_id": event.get("event_id", f"agnostic_{index}"),
                "start_sec": float(event["start_sec"]),
                "end_sec": float(event["end_sec"]),
                "granularity": event.get("granularity", ""),
                "description": event.get("description", ""),
                "categories": event.get("categories", []),
                "parent_event_ids": event.get("parent_event_ids", []),
            }
        )
    return events, agnostic.get("model"), str(agnostic.get("prompt_version") or "")


def _load_verification(
    verification_dir: Path | None, sample_id: str
) -> dict[str, Any] | None:
    """Load the span-sufficiency verification sidecar for one oracle sample."""
    if not verification_dir or not sample_id or not verification_dir.is_dir():
        return None
    path = verification_dir / f"{sample_id}.json"
    if not path.is_file():
        return None
    try:
        data = _read_json(path)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict) or "verdict" not in data:
        return None
    data["verification_json"] = str(path)
    return data


def _display_verdict(
    verification: dict[str, Any] | None,
    oracle_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Recompute the current verdict for display without rewriting the sidecar."""
    if not verification:
        return None
    cited = [
        str(value)
        for value in verification.get("supporting_event_ids") or []
        if str(value)
    ]
    roles_by_id = {
        str(event["event_id"]): list(event.get("query_roles") or [])
        for event in (oracle_events or [])
    }
    unknown = [event_id for event_id in cited if event_id not in roles_by_id] if roles_by_id else list(
        verification.get("unknown_event_ids") or []
    )
    cited_roles = list(
        dict.fromkeys(
            role for event_id in cited for role in roles_by_id.get(event_id, ())
        )
    )
    cited_answer_bearing = "answer_bearing" in cited_roles if roles_by_id else bool(
        verification.get("cited_answer_bearing")
    )
    threshold = float(
        verification.get("coverage_threshold") or DEFAULT_COVERAGE_THRESHOLD
    )
    verification = dict(verification)
    verification["cited_roles"] = cited_roles or list(verification.get("cited_roles") or [])
    verification["cited_answer_bearing"] = cited_answer_bearing
    verification["coverage_threshold"] = threshold
    verification["verdict"] = decide_verdict(
        correct=bool(verification.get("correct")),
        sufficiency=str(verification.get("sufficiency") or ""),
        cited=cited,
        unknown=unknown,
        coverage=float(verification.get("coverage_ratio") or 0.0),
        cited_answer_bearing=cited_answer_bearing,
        coverage_threshold=threshold,
    )
    return verification


def _load_verification_summary(
    verification_dir: Path | None,
) -> dict[str, Any] | None:
    if not verification_dir or not verification_dir.is_dir():
        return None
    path = verification_dir / "summary.json"
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except Exception:  # noqa: BLE001
        return None


def _summary_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Viewer summary from live verdicts, so old sidecars pick up current rules."""
    verified = [sample for sample in samples if sample.get("has_verification")]
    if not verified:
        return None
    counts = {"pass": 0, "uncertain": 0, "uninformative": 0, "fail": 0}
    for sample in verified:
        verdict = sample.get("verdict")
        if verdict in counts:
            counts[verdict] += 1
    total = len(verified)
    correct = [s for s in verified if s.get("verification_correct")]
    informative = [
        s
        for s in verified
        if (s.get("coverage_ratio") or 0)
        < float(s.get("coverage_threshold") or DEFAULT_COVERAGE_THRESHOLD)
    ]
    informative_correct = [s for s in informative if s.get("verification_correct")]
    mean_coverage = sum(float(s.get("coverage_ratio") or 0) for s in verified) / total
    return {
        "n_verified": total,
        "verdicts": counts,
        "accuracy": len(correct) / total,
        "pass_rate": counts["pass"] / total,
        "n_informative": len(informative),
        "informative_accuracy": (
            len(informative_correct) / len(informative) if informative else None
        ),
        "mean_coverage_ratio": mean_coverage,
    }


def _load_answer_index(questions_path: Path) -> dict[str, str]:
    """Map EgoLongQA sample_id -> ground-truth answer string."""
    if not questions_path.is_file():
        return {}
    answers: dict[str, str] = {}
    with questions_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("qid", row.get("id", index)))
            answer = row.get("answer")
            if answer is None:
                continue
            answers[sample_id] = str(answer)
    return answers


def _comparison_for_video(
    agnostic_dir: Path | None,
    video_id: str,
    tsm: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """TSM vs one agnostic reference for a single video (computed on load)."""
    if not agnostic_dir or not video_id or tsm is None:
        return None
    reference_path = agnostic_dir / f"{video_id}.json"
    if not reference_path.is_file():
        return None
    try:
        annotation = OracleAnnotation.load(reference_path)
        result = compare_annotation(annotation, tsm)
        result["reference_mode"] = annotation.annotation_mode
        result["reference_json"] = str(reference_path)
        return result
    except Exception:  # noqa: BLE001
        return None


def build_sample_list(
    oracle_dir: Path,
    tsm_dir: Path,
    questions_path: Path | None = None,
    agnostic_dir: Path | None = None,
    event_graph_dirs: list[tuple[str, Path]] | None = None,
    verification_dir: Path | None = None,
) -> list[dict[str, Any]]:
    answer_index = _load_answer_index(questions_path) if questions_path else {}
    samples: list[dict[str, Any]] = []
    for path in _list_oracle_files(oracle_dir):
        oracle = _read_json(path)
        sample_id = str(oracle.get("sample_id", path.stem))
        video_id = str(oracle.get("video_id", ""))
        tsm = _load_tsm(tsm_dir, video_id) if video_id else None
        agnostic = (
            _load_agnostic(agnostic_dir, video_id)
            if agnostic_dir and video_id
            else None
        )
        runs = [
            _event_graph_run(label, eg_dir, sample_id)
            for label, eg_dir in (event_graph_dirs or [])
        ]
        first = runs[0] if runs else None
        verification = _display_verdict(
            _load_verification(verification_dir, sample_id),
            oracle.get("events", []),
        )
        samples.append(
            {
                "sample_id": sample_id,
                "video_id": video_id,
                "query": oracle.get("query", ""),
                "answer": answer_index.get(sample_id),
                "verdict": (verification or {}).get("verdict"),
                "verification_correct": (verification or {}).get("correct"),
                "coverage_ratio": (verification or {}).get("coverage_ratio"),
                "coverage_threshold": (verification or {}).get("coverage_threshold"),
                "has_verification": verification is not None,
                "pred_answer": first.get("pred_answer") if first else None,
                "n_oracle_events": len(oracle.get("events", [])),
                "n_agnostic_events": (
                    len(agnostic.get("events", [])) if agnostic else 0
                ),
                "n_tsm_events": (
                    len(tsm.get("events_sec", [])) if tsm else 0
                ),
                "n_event_graph_events": len(first["events"]) if first else 0,
                "n_event_graph_runs": len(runs),
                "has_tsm": tsm is not None,
                "has_agnostic": agnostic is not None,
                "has_event_graph": any(run["has_event_graph"] for run in runs),
                "has_metrics": agnostic is not None and tsm is not None,
                "oracle_path": str(path),
            }
        )
    return samples


def build_sample_detail(
    sample_id: str,
    oracle_dir: Path,
    tsm_dir: Path,
    questions_path: Path | None = None,
    agnostic_dir: Path | None = None,
    event_graph_dirs: list[tuple[str, Path]] | None = None,
    verification_dir: Path | None = None,
) -> dict[str, Any]:
    oracle_path = oracle_dir / f"{sample_id}.json"
    if not oracle_path.is_file():
        # Fall back to scanning (ids that are not the filename stem).
        match = None
        for path in _list_oracle_files(oracle_dir):
            data = _read_json(path)
            if str(data.get("sample_id", path.stem)) == sample_id:
                match = data
                oracle_path = path
                break
        if match is None:
            raise FileNotFoundError(f"No oracle for sample_id={sample_id}")
        oracle = match
    else:
        oracle = _read_json(oracle_path)

    video_id = str(oracle.get("video_id", ""))
    tsm = _load_tsm(tsm_dir, video_id) if video_id else None
    agnostic = (
        _load_agnostic(agnostic_dir, video_id)
        if agnostic_dir and video_id
        else None
    )
    sid = str(oracle.get("sample_id", sample_id))
    event_graph_runs = [
        _event_graph_run(label, eg_dir, sid)
        for label, eg_dir in (event_graph_dirs or [])
    ]
    first_run = event_graph_runs[0] if event_graph_runs else None
    comparison = _comparison_for_video(agnostic_dir, video_id, tsm)
    answer_index = _load_answer_index(questions_path) if questions_path else {}
    answer = answer_index.get(sid)
    verification = _display_verdict(
        _load_verification(verification_dir, sid),
        oracle.get("events", []),
    )
    retrieved_events = first_run["events"] if first_run else []
    pred_answer = first_run.get("pred_answer") if first_run else None
    qa_error = first_run.get("qa_error") if first_run else None

    duration = float(
        oracle.get("video_duration_sec")
        or (tsm or {}).get("duration_sec")
        or 0.0
    )
    if duration <= 0 and oracle.get("events"):
        duration = max(float(e["end_sec"]) for e in oracle["events"])

    cited_ids = set((verification or {}).get("supporting_event_ids") or [])
    events = []
    for event in oracle.get("events", []):
        event_id = event["event_id"]
        events.append(
            {
                "event_id": event_id,
                "start_sec": float(event["start_sec"]),
                "end_sec": float(event["end_sec"]),
                "granularity": event.get("granularity", ""),
                "description": event.get("description", ""),
                "categories": event.get("categories", []),
                "query_roles": event.get("query_roles", []),
                "parent_event_ids": event.get("parent_event_ids", []),
                "cited": event_id in cited_ids,
            }
        )

    edges = []
    for event in events:
        for parent in event["parent_event_ids"]:
            edges.append({"source": parent, "target": event["event_id"]})

    tsm_events = []
    if tsm is not None:
        for index, span in enumerate(tsm.get("events_sec", [])):
            tsm_events.append(
                {
                    "index": index,
                    "start_sec": float(span[0]),
                    "end_sec": float(span[1]),
                }
            )

    agnostic_events, agnostic_model, agnostic_prompt = _format_agnostic_events(
        agnostic
    )

    video_path = oracle.get("video_path")
    if video_path:
        video_path = str(_apply_video_root(Path(video_path).expanduser()))

    return {
        "sample_id": str(oracle.get("sample_id", sample_id)),
        "video_id": video_id,
        "video_path": video_path,
        "video_exists": bool(video_path and Path(video_path).is_file()),
        "duration_sec": duration,
        "query": oracle.get("query", ""),
        "answer": answer,
        "pred_answer": pred_answer,
        "qa_error": qa_error,
        "verification": verification,
        "verification_summary": _load_verification_summary(verification_dir),
        # Free-form answer the annotator produced while watching the full video;
        # a rough ceiling reading next to the verifier's clips-only answer.
        "oracle_predicted_answer": oracle.get("predicted_answer") or oracle.get("answer"),
        "oracle_answer_match": oracle.get("answer_match"),
        "has_event_graph": any(run["has_event_graph"] for run in event_graph_runs),
        "event_graph_runs": event_graph_runs,
        "oracle_events": events,
        "oracle_edges": edges,
        "agnostic_events": agnostic_events,
        "agnostic_model": agnostic_model,
        "agnostic_prompt_version": agnostic_prompt,
        "tsm_events": tsm_events,
        "event_graph_events": retrieved_events,
        "comparison": comparison,
        "metrics_reference_mode": (comparison or {}).get(
            "reference_mode", "query_agnostic"
        ),
        "paths": {
            "oracle_json": str(oracle_path),
            "agnostic_json": (
                str(agnostic_dir / f"{video_id}.json")
                if agnostic_dir and video_id and agnostic is not None
                else None
            ),
            "tsm_json": str(tsm_dir / f"{video_id}.json") if video_id else None,
            "event_graph_json": first_run.get("json_path") if first_run else None,
            "event_graph_jsons": [
                run["json_path"] for run in event_graph_runs if run.get("json_path")
            ],
            "questions_jsonl": (
                str(questions_path) if questions_path and questions_path.is_file() else None
            ),
            "verification_json": (verification or {}).get("verification_json"),
        },
    }


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "EventEvalViewer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quieter default logging; still useful for debugging.
        sys_stderr = getattr(self, "_stderr", None)
        if sys_stderr is None:
            import sys

            sys_stderr = sys.stderr
        sys_stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self._send_error_json(404, f"Missing file: {path}")
            return
        data = path.read_bytes()
        ctype = content_type or mimetypes.guess_type(str(path))[0] or (
            "application/octet-stream"
        )
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # Viewer UI iterates quickly; never let the browser keep a stale shell.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _query(self) -> dict[str, str]:
        parsed = urlparse(self.path)
        raw = parse_qs(parsed.query)
        return {key: values[0] for key, values in raw.items() if values}

    def _paths_from_query(
        self,
    ) -> tuple[Path, Path, Path, list[tuple[str, Path]], Path, Path]:
        q = self._query()
        return (
            _resolve_path(q.get("oracle_dir"), "oracle_dir"),
            _resolve_path(q.get("agnostic_dir"), "agnostic_dir"),
            _resolve_path(q.get("tsm_dir"), "tsm_dir"),
            _event_graph_dirs_from_query(q),
            _resolve_path(q.get("questions_jsonl"), "questions_jsonl"),
            _resolve_path(q.get("verification_dir"), "verification_dir"),
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            self._send_file(_STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        if route.startswith("/static/"):
            rel = route[len("/static/") :]
            target = (_STATIC_DIR / rel).resolve()
            if not str(target).startswith(str(_STATIC_DIR.resolve())):
                self._send_error_json(403, "Forbidden")
                return
            self._send_file(target)
            return

        if route == "/api/defaults":
            self._send_json(
                {
                    "paths": DEFAULT_PATHS,
                    "directory_options": _annotation_directory_options(),
                    "morph_graph_root": str(_MORPH_GRAPH_ROOT),
                }
            )
            return

        if route == "/api/samples":
            (
                oracle_dir,
                agnostic_dir,
                tsm_dir,
                event_graph_dirs,
                questions_path,
                verification_dir,
            ) = self._paths_from_query()
            try:
                samples = build_sample_list(
                    oracle_dir,
                    tsm_dir,
                    questions_path,
                    agnostic_dir,
                    event_graph_dirs,
                    verification_dir,
                )
            except Exception as exc:  # noqa: BLE001
                self._send_error_json(400, str(exc))
                return
            self._send_json(
                {
                    "samples": samples,
                    "verification_summary": _summary_from_samples(samples),
                    "resolved": {
                        "oracle_dir": str(oracle_dir),
                        "agnostic_dir": str(agnostic_dir),
                        "tsm_dir": str(tsm_dir),
                        "event_graph_dirs": "\n".join(
                            f"{label}|{path}" for label, path in event_graph_dirs
                        ),
                        "questions_jsonl": str(questions_path),
                        "verification_dir": str(verification_dir),
                    },
                }
            )
            return

        if route.startswith("/api/sample/"):
            sample_id = unquote(route[len("/api/sample/") :])
            (
                oracle_dir,
                agnostic_dir,
                tsm_dir,
                event_graph_dirs,
                questions_path,
                verification_dir,
            ) = self._paths_from_query()
            try:
                detail = build_sample_detail(
                    sample_id,
                    oracle_dir,
                    tsm_dir,
                    questions_path,
                    agnostic_dir,
                    event_graph_dirs,
                    verification_dir,
                )
            except FileNotFoundError as exc:
                self._send_error_json(404, str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                self._send_error_json(400, str(exc))
                return
            self._send_json(detail)
            return

        if route == "/api/video_meta":
            q = self._query()
            raw = q.get("path", "").strip()
            if not raw:
                self._send_error_json(400, "Missing path query parameter")
                return
            try:
                path = _resolve_video_path(raw)
                meta = probe_video(path)
            except FileNotFoundError as exc:
                self._send_error_json(404, str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                self._send_error_json(400, str(exc))
                return
            self._send_json({"path": str(path), **meta})
            return

        if route == "/api/frame":
            self._send_frame()
            return

        if route == "/api/warm":
            self._start_warm()
            return

        if route == "/api/warm_status":
            self._warm_status()
            return

        if route == "/api/video":
            # Kept for debugging; the UI uses frame scrubbing instead.
            self._stream_video()
            return

        self._send_error_json(404, f"Unknown route: {route}")

    def _frame_params(self) -> tuple[Path, float, float, int]:
        q = self._query()
        raw = q.get("path", "").strip()
        if not raw:
            raise ValueError("Missing path query parameter")
        try:
            t_sec = float(q.get("t", "0"))
        except ValueError as exc:
            raise ValueError("Invalid t") from exc
        try:
            fps = float(q.get("fps", "1") or "1")
        except ValueError as exc:
            raise ValueError("Invalid fps") from exc
        try:
            max_width = int(float(q.get("w", str(_DEFAULT_FRAME_WIDTH))))
        except ValueError as exc:
            raise ValueError("Invalid w") from exc
        path = _resolve_video_path(raw)
        return path, t_sec, fps, max_width

    def _send_frame(self) -> None:
        try:
            path, t_sec, fps, max_width = self._frame_params()
            jpeg = extract_frame_jpeg(path, t_sec, fps=fps, max_width=max_width)
        except FileNotFoundError as exc:
            self._send_error_json(404, str(exc))
            return
        except ValueError as exc:
            self._send_error_json(400, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(400, str(exc))
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "private, max-age=86400")
        self.end_headers()
        self.wfile.write(jpeg)

    def _start_warm(self) -> None:
        q = self._query()
        raw = q.get("path", "").strip()
        if not raw:
            self._send_error_json(400, "Missing path query parameter")
            return
        try:
            fps = float(q.get("fps", "1") or "1")
            max_width = int(float(q.get("w", str(_DEFAULT_FRAME_WIDTH))))
            path = _resolve_video_path(raw)
            payload = start_warm_frames(path, fps=fps, max_width=max_width)
        except FileNotFoundError as exc:
            self._send_error_json(404, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(400, str(exc))
            return
        self._send_json(payload)

    def _warm_status(self) -> None:
        q = self._query()
        raw = q.get("path", "").strip()
        if not raw:
            self._send_error_json(400, "Missing path query parameter")
            return
        try:
            fps = float(q.get("fps", "1") or "1")
            max_width = int(float(q.get("w", str(_DEFAULT_FRAME_WIDTH))))
            path = _resolve_video_path(raw)
            payload = warm_status(path, fps=fps, max_width=max_width)
        except FileNotFoundError as exc:
            self._send_error_json(404, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(400, str(exc))
            return
        self._send_json(payload)

    def _stream_video(self) -> None:
        q = self._query()
        raw = q.get("path", "").strip()
        if not raw:
            self._send_error_json(400, "Missing path query parameter")
            return
        try:
            path = _resolve_video_path(raw)
        except FileNotFoundError as exc:
            self._send_error_json(404, str(exc))
            return
        except ValueError as exc:
            self._send_error_json(400, str(exc))
            return

        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"

        if range_header:
            # bytes=start-end
            units, _, rng = range_header.partition("=")
            if units.strip() != "bytes":
                self._send_error_json(400, "Invalid Range unit")
                return
            start_s, _, end_s = rng.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Directory that relative video_path values in annotation JSONs "
        "resolve against (e.g. the dir containing egolongqa/val/).",
    )
    args = parser.parse_args(argv)

    if args.video_root is not None:
        global VIDEO_ROOT
        VIDEO_ROOT = args.video_root.expanduser().resolve()

    httpd = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    lines = [
        f"Event eval viewer: http://{args.host}:{args.port}/",
        f"Defaults rooted at {_MORPH_GRAPH_ROOT}",
        f"  video_root:        {VIDEO_ROOT or '(none — relative paths resolve against CWD)'}",
        f"  oracle_dir:      {DEFAULT_PATHS['oracle_dir']}",
        f"  agnostic_dir:      {DEFAULT_PATHS['agnostic_dir']}",
        f"  tsm_dir:           {DEFAULT_PATHS['tsm_dir']}",
        f"  event_graph_dir:   {DEFAULT_PATHS['event_graph_dir']}",
        f"  verification_dir:  {DEFAULT_PATHS['verification_dir']}",
        f"  questions_jsonl:   {DEFAULT_PATHS['questions_jsonl']}",
        "  (TSM metrics computed on load; no report JSON required)",
    ]
    print("\n".join(lines), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    main()
