"""Streamlit visualizer for event segmentation on videos.

Run:
  streamlit run morph_graph/eval/event_seg_app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.lines import Line2D

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

REPO_ROOT = Path(__file__).resolve().parents[1]
MORPH_GRAPHS_ROOT = REPO_ROOT.parent
if str(MORPH_GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(MORPH_GRAPHS_ROOT))

from morph_graph.eval.eval_event_seg import (  # noqa: E402
    DEFAULT_VIDEO,
    EventSeg,
    compute_sim_matrix,
    encode_frames,
    events_from_boundaries,
    frame_to_pil,
    get_frames_with_indices,
    load_model_and_processor,
    load_video,
)

EVENT_COLORS = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
]

BROWSE_ROOTS = [
    REPO_ROOT / "data",
    REPO_ROOT,
    MORPH_GRAPHS_ROOT / "vsi_output",
    MORPH_GRAPHS_ROOT,
]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
DEFAULT_QUESTIONS_JSON = MORPH_GRAPHS_ROOT / "vsi_output" / "vsi_events_mc_poc.json"


def _available_browse_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for root in BROWSE_ROOTS:
        resolved = root.resolve()
        key = str(resolved)
        if resolved.exists() and key not in seen:
            roots.append(resolved)
            seen.add(key)
    return roots or [REPO_ROOT.resolve()]


@st.cache_data(show_spinner=False)
def _discover_videos(root: str) -> list[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    videos: list[str] = []
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(str(path.resolve()))
    return sorted(videos, key=str.lower)


def _list_directory(path: Path) -> tuple[list[Path], list[Path]]:
    if not path.is_dir():
        return [], []
    subdirs = sorted(
        [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name.lower(),
    )
    videos = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )
    return subdirs, videos


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _video_label(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _videos_in_folder(folder: Path, *, recursive: bool) -> list[Path]:
    if not folder.is_dir():
        return []
    if recursive:
        return [Path(p) for p in _discover_videos(str(folder))]
    _, videos = _list_directory(folder)
    return videos


def _normalize_questions_payload(data: object) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        out: list[dict] = []
        for uid, items in data.items():
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row.setdefault("video_uid", str(uid))
                out.append(row)
        return out
    return []


def _parse_jsonl_text(text: str) -> list[dict]:
    items: list[dict] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        if isinstance(obj, dict):
            items.append(obj)
        elif isinstance(obj, list):
            items.extend(item for item in obj if isinstance(item, dict))
    return items


def _parse_questions_text(text: str, *, prefer_jsonl: bool = False) -> list[dict]:
    stripped = text.lstrip("\ufeff").strip()
    if not stripped:
        return []
    if prefer_jsonl:
        return _parse_jsonl_text(stripped)
    try:
        return _normalize_questions_payload(json.loads(stripped))
    except json.JSONDecodeError:
        return _parse_jsonl_text(stripped)


def _parse_questions_bytes(raw: bytes, *, prefer_jsonl: bool = False) -> list[dict]:
    return _parse_questions_text(raw.decode("utf-8"), prefer_jsonl=prefer_jsonl)


@st.cache_data(show_spinner=False)
def _load_questions_file(path: str) -> list[dict]:
    question_path = Path(path).expanduser()
    if not question_path.is_file():
        return []
    prefer_jsonl = question_path.suffix.lower() == ".jsonl"
    return _parse_questions_bytes(question_path.read_bytes(), prefer_jsonl=prefer_jsonl)


def _resolve_questions(path: str, uploaded) -> list[dict]:
    if uploaded is not None:
        prefer_jsonl = Path(uploaded.name).suffix.lower() == ".jsonl"
        return _parse_questions_bytes(uploaded.getvalue(), prefer_jsonl=prefer_jsonl)
    if path.strip():
        return _load_questions_file(path.strip())
    return []


def _video_uid_from_path(path: Path) -> str:
    return path.stem


def _item_video_uid(item: dict) -> str | None:
    for key in ("video_uid", "video_id", "uid"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for key in ("video", "video_path", "path"):
        value = item.get(key)
        if value:
            return Path(str(value)).stem
    return None


def _questions_for_video(questions: list[dict], video_path: Path) -> list[dict]:
    video_uid = _video_uid_from_path(video_path)
    matched = [q for q in questions if _item_video_uid(q) == video_uid]
    return sorted(matched, key=lambda q: (q.get("qid", 0), str(q.get("question", ""))))


def _render_video_questions(video_path: Path, questions: list[dict]) -> None:
    matched = _questions_for_video(questions, video_path)
    video_uid = _video_uid_from_path(video_path)

    st.subheader("Questions")
    if not questions:
        st.caption("Add a questions JSON/JSONL path in the sidebar to show per-video questions.")
        return
    if not matched:
        st.info(f"No questions found for video `{video_uid}` in the loaded file.")
        return

    st.caption(f"{len(matched)} question(s) for `{video_uid}`")
    show_answers = st.checkbox("Show correct answers", value=False, key="show_question_answers")

    for item in matched:
        qid = item.get("qid", "?")
        question_text = str(item.get("question", "")).strip() or "(no question text)"
        title = question_text if len(question_text) <= 100 else f"{question_text[:97]}..."
        with st.expander(f"Q{qid}: {title}", expanded=len(matched) == 1):
            meta = []
            if item.get("object"):
                meta.append(f"Object: {item['object']}")
            if item.get("question_type"):
                meta.append(f"Type: {item['question_type']}")
            if meta:
                st.caption(" · ".join(meta))

            st.markdown(question_text)

            options = item.get("options")
            if isinstance(options, dict) and options:
                for key in sorted(options, key=str):
                    st.markdown(f"**{key}.** {options[key]}")

            if show_answers and item.get("correct_answer") is not None:
                st.success(f"Correct answer: **{item['correct_answer']}**")

            clues = item.get("required_clues")
            if isinstance(clues, list) and clues:
                st.markdown("**Required clues**")
                for clue in clues:
                    st.markdown(f"- {clue}")

            clue_frames = item.get("clue_frames")
            if clue_frames:
                st.caption(f"Clue frames: `{clue_frames}`")


def _sidebar_questions_input() -> list[dict]:
    st.subheader("Questions")
    default_path = str(DEFAULT_QUESTIONS_JSON) if DEFAULT_QUESTIONS_JSON.exists() else ""
    questions_path = st.text_input(
        "Questions JSON/JSONL path",
        value=default_path,
        help="JSON array, JSON object keyed by video id, or JSONL (one question per line).",
    )
    uploaded = st.file_uploader("Or upload questions file", type=["json", "jsonl"])
    try:
        questions = _resolve_questions(questions_path, uploaded)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        st.error(f"Could not parse questions file: {exc}")
        questions = []
    if questions_path.strip() and not uploaded:
        path = Path(questions_path.strip()).expanduser()
        if not path.is_file():
            st.warning("Questions file not found.")
        else:
            st.caption(f"Loaded {len(questions)} total questions.")
    elif uploaded is not None:
        st.caption(f"Loaded {len(questions)} total questions from upload.")
    return questions


def _sidebar_video_path() -> Path | None:
    st.subheader("Video")
    source = st.radio(
        "Video source",
        ["Folder", "Upload", "Manual path"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "Upload":
        uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "webm", "avi"])
        if uploaded is None:
            return None
        tmp_upload = Path(tempfile.mkstemp(suffix=Path(uploaded.name).suffix or ".mp4")[1])
        tmp_upload.write_bytes(uploaded.getvalue())
        st.caption(f"Uploaded: `{uploaded.name}`")
        return tmp_upload

    if source == "Manual path":
        manual = st.text_input("Video path", value=str(DEFAULT_VIDEO))
        return Path(manual) if manual.strip() else None

    roots = _available_browse_roots()
    root_labels = [str(r) for r in roots]
    browse_root = st.selectbox("Root folder", root_labels, key="browse_root_select")
    browse_root_path = Path(browse_root)

    folder_override = st.text_input(
        "Folder path",
        value=st.session_state.get("browse_dir", browse_root),
        help="Absolute path or navigate with the controls below",
    )
    folder_path = Path(folder_override.strip() or browse_root).expanduser()
    if not folder_path.is_absolute():
        folder_path = (browse_root_path / folder_path).resolve()

    if (
        "browse_dir" not in st.session_state
        or st.session_state.browse_dir != str(folder_path)
    ):
        st.session_state.browse_dir = str(folder_path)

    st.caption(f"Folder: `{folder_path}`")

    if folder_path.is_dir():
        subdirs, _ = _list_directory(folder_path)
        if subdirs:
            folder_choice = st.selectbox("Open subfolder", ["—"] + [d.name for d in subdirs])
            if folder_choice != "—":
                st.session_state.browse_dir = str(folder_path / folder_choice)
                st.rerun()
        parent = folder_path.parent
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("Up", use_container_width=True, disabled=folder_path == browse_root_path):
                st.session_state.browse_dir = str(
                    parent if _is_within_root(parent, browse_root_path) else browse_root_path
                )
                st.rerun()
        with nav_col2:
            if st.button("Reset to root", use_container_width=True):
                st.session_state.browse_dir = browse_root
                st.rerun()
    else:
        st.warning("Folder does not exist.")

    recursive = st.checkbox("Include subfolders", value=True)
    filter_text = st.text_input("Filter filenames", placeholder="optional")

    videos = _videos_in_folder(folder_path, recursive=recursive)
    if filter_text.strip():
        q = filter_text.strip().lower()
        videos = [v for v in videos if q in v.name.lower() or q in str(v).lower()]

    if not videos:
        st.warning("No videos found in this folder.")
        return None

    labels = [_video_label(v, folder_path if folder_path.is_dir() else browse_root_path) for v in videos]
    default_idx = 0
    if "selected_video_path" in st.session_state:
        try:
            default_idx = [str(v) for v in videos].index(st.session_state.selected_video_path)
        except ValueError:
            default_idx = 0

    picked = st.selectbox(
        f"Choose video ({len(videos)} in folder)",
        range(len(videos)),
        index=default_idx,
        format_func=lambda i: labels[i],
    )
    selected = videos[picked]
    st.session_state.selected_video_path = str(selected)
    st.caption(f"`{selected}`")
    return selected


def _event_span_times(
    start: int,
    end: int,
    frame_indices: list[int],
    video_fps: float,
) -> tuple[float, float]:
    t0 = frame_indices[start] / max(video_fps, 1e-6)
    if end < len(frame_indices):
        t1 = frame_indices[end] / max(video_fps, 1e-6)
    else:
        t1 = (frame_indices[end - 1] + 1) / max(video_fps, 1e-6)
    return t0, t1


def _event_spans(
    events: list[tuple[int, int]],
    frame_indices: list[int],
    video_fps: float,
    *,
    use_time_axis: bool,
) -> list[tuple[float, float]]:
    spans = []
    for start, end in events:
        if use_time_axis:
            spans.append(_event_span_times(start, end, frame_indices, video_fps))
        else:
            spans.append((start - 0.5, end - 0.5))
    return spans


def _event_at_position(
    position: float,
    spans: list[tuple[float, float]],
) -> int:
    for idx, (x0, x1) in enumerate(spans):
        if idx == len(spans) - 1:
            if x0 <= position <= x1:
                return idx
        elif x0 <= position < x1:
            return idx
    return max(0, len(spans) - 1)


@st.cache_resource
def cached_model(ckpt: str):
    return load_model_and_processor(ckpt)


def _sample_time_sec(frame_indices: list[int], sampled_idx: int, video_fps: float) -> float:
    native_idx = frame_indices[sampled_idx]
    return native_idx / max(video_fps, 1e-6)


def _event_time_range(
    event: tuple[int, int],
    frame_indices: list[int],
    video_fps: float,
) -> tuple[float, float]:
    start, end = event
    start_sec = _sample_time_sec(frame_indices, start, video_fps)
    end_idx = max(end - 1, start)
    end_sec = _sample_time_sec(frame_indices, end_idx, video_fps)
    return start_sec, end_sec


def _ffmpeg_clip(video_path: Path, start_sec: float, end_sec: float) -> Path | None:
    if end_sec <= start_sec:
        return None
    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-to",
            f"{end_sec:.3f}",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-an",
            str(tmp),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not tmp.exists():
        return None
    return tmp


@st.cache_data(show_spinner="Preparing video clip...")
def _cached_ffmpeg_clip_bytes(video_path: str, start_sec: float, end_sec: float) -> bytes | None:
    clip_path = _ffmpeg_clip(Path(video_path), start_sec, end_sec)
    if clip_path is None:
        return None
    return clip_path.read_bytes()


def _timeline_position_to_sec(
    position: float,
    frame_indices: list[int],
    video_fps: float,
    *,
    use_time_axis: bool,
) -> float:
    if use_time_axis:
        return position
    idx = int(round(position))
    idx = max(0, min(len(frame_indices) - 1, idx))
    return _sample_time_sec(frame_indices, idx, video_fps)


def _event_timeline_plotly(
    events: list[tuple[int, int]],
    spans: list[tuple[float, float]],
    selected_event: int,
    *,
    x_label: str,
) -> go.Figure:
    fig = go.Figure()
    for event_idx, (x0, x1) in enumerate(spans):
        width = max(x1 - x0, 1e-6)
        color = EVENT_COLORS[event_idx % len(EVENT_COLORS)]
        selected = event_idx == selected_event
        fig.add_trace(
            go.Bar(
                x=[width],
                y=[""],
                base=[x0],
                orientation="h",
                name=f"Event {event_idx + 1}",
                marker={
                    "color": color,
                    "line": {"color": "#111827" if selected else "white", "width": 3 if selected else 1},
                },
                customdata=[[event_idx]],
                hovertemplate=(
                    f"Event {event_idx + 1}<br>"
                    f"{x0:.2f} – {x1:.2f}<br>"
                    "Click to jump<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        barmode="overlay",
        height=110,
        margin={"l": 20, "r": 20, "t": 30, "b": 30},
        title="Event segmentation timeline (click a segment to jump)",
        xaxis_title=x_label,
        yaxis={"showticklabels": False},
        dragmode=False,
    )
    return fig


def _event_timeline_buttons(
    events: list[tuple[int, int]],
    spans: list[tuple[float, float]],
    selected_event: int,
) -> int | None:
    weights = [max(x1 - x0, 0.05) for x0, x1 in spans]
    cols = st.columns(weights)
    clicked: int | None = None
    for event_idx, col in enumerate(cols):
        x0, x1 = spans[event_idx]
        color = EVENT_COLORS[event_idx % len(EVENT_COLORS)]
        border = "3px solid #111827" if event_idx == selected_event else "1px solid white"
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{color};
                    border:{border};
                    border-radius:4px;
                    padding:10px 4px;
                    text-align:center;
                    font-weight:bold;
                    color:#111827;
                ">E{event_idx + 1}</div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Jump", key=f"timeline_jump_{event_idx}", use_container_width=True):
                clicked = event_idx
            st.caption(f"{x0:.1f}–{x1:.1f}")
    return clicked


def _render_interactive_timeline(
    events: list[tuple[int, int]],
    frame_indices: list[int],
    video_fps: float,
    selected_event: int,
    *,
    use_time_axis: bool,
) -> int:
    spans = _event_spans(events, frame_indices, video_fps, use_time_axis=use_time_axis)
    x_label = "Time (s)" if use_time_axis else "Sampled frame index"

    if HAS_PLOTLY:
        fig = _event_timeline_plotly(events, spans, selected_event, x_label=x_label)
        selection = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            key="event_timeline_plotly",
        )
        if selection and getattr(selection, "selection", None):
            sel = selection.selection
            points = getattr(sel, "points", None) or sel.get("points", [])
            if points:
                curve_number = points[0].get("curve_number")
                if curve_number is not None:
                    return int(curve_number)
        return selected_event

    st.caption("Install `plotly` for a clickable chart timeline. Using button strip fallback.")
    clicked = _event_timeline_buttons(events, spans, selected_event)
    return clicked if clicked is not None else selected_event


def _adjacent_figure(
    similarities: np.ndarray,
    thresholds: np.ndarray,
    boundary_frames: np.ndarray,
    events: list[tuple[int, int]],
    selected_event: int,
) -> plt.Figure:
    x = np.arange(len(similarities))
    fig, ax = plt.subplots(figsize=(12, 3.5))

    for event_idx, (start, end) in enumerate(events):
        color = "#fecaca" if event_idx == selected_event else "#e2e8f0"
        ax.axvspan(start - 0.5, end - 0.5, color=color, alpha=0.8)

    ax.plot(x, similarities, color="#2563eb", linewidth=1.5, marker="o", markersize=3, label="adjacent sim")

    valid = ~np.isnan(thresholds)
    if valid.any():
        ax.plot(
            x[valid],
            thresholds[valid],
            color="#ea580c",
            linewidth=1.5,
            linestyle="--",
            label="running threshold",
        )

    for frame in boundary_frames:
        ax.axvline(frame - 1, color="#dc2626", linewidth=1.2, alpha=0.7, linestyle=":")

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Sampled frame index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Adjacent frame similarity")
    ax.grid(True, alpha=0.25)

    handles = [
        Line2D([0], [0], color="#2563eb", linewidth=1.5, marker="o", label="adjacent sim"),
    ]
    if valid.any():
        handles.append(Line2D([0], [0], color="#ea580c", linewidth=1.5, linestyle="--", label="running threshold"))
    if len(boundary_frames) > 0:
        handles.append(Line2D([0], [0], color="#dc2626", linewidth=1.2, linestyle=":", label="event boundary"))
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


def _sim_matrix_figure(sims: np.ndarray, boundary_frames: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sims, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    for frame in boundary_frames:
        ax.axvline(frame - 0.5, color="#dc2626", linewidth=1.0, alpha=0.7, linestyle=":")
        ax.axhline(frame - 0.5, color="#dc2626", linewidth=1.0, alpha=0.7, linestyle=":")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frame index")
    ax.set_title("Frame self-similarity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Cosine similarity")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def run_analysis(
    video_path: str,
    sample_fps: float,
    strategy: str,
    percentile: float,
    threshold: float,
    ckpt: str,
    batch_size: int,
    compute_matrix: bool,
):
    path = Path(video_path).resolve()
    decoder = load_video(str(path))
    frames, frame_indices = get_frames_with_indices(decoder, sample_fps)
    video_fps = float(decoder.metadata.average_fps or sample_fps)

    model, processor = cached_model(ckpt)
    features = encode_frames(model, processor, frames, batch_size=batch_size)
    seg = EventSeg.chunk(
        features,
        strategy,
        percentile=percentile,
        threshold=threshold,
    )
    boundary_frames = seg["boundary_frames"]
    events = events_from_boundaries(boundary_frames, frames.shape[0])
    sims = compute_sim_matrix(features, batch_size=batch_size) if compute_matrix else None

    return {
        "video_path": str(path),
        "num_frames": int(frames.shape[0]),
        "frame_indices": frame_indices,
        "video_fps": video_fps,
        "similarities": seg["similarities"],
        "thresholds": seg["thresholds"],
        "boundary_frames": boundary_frames,
        "events": events,
        "strategy": strategy,
        "sim_matrix": sims,
        "frames_np": frames.numpy(),
    }


def _load_frames(result: dict):
    import torch

    return torch.from_numpy(result["frames_np"])


def main() -> None:
    st.set_page_config(page_title="Event Seg Visualizer", layout="wide")
    st.title("Event segmentation visualizer")

    if "selected_event" not in st.session_state:
        st.session_state.selected_event = 0

    with st.sidebar:
        st.header("Input")
        video_path = _sidebar_video_path()

        sample_fps = st.slider("Sample FPS", min_value=0.25, max_value=5.0, value=1.0, step=0.25)

        st.subheader("Chunking")
        strategy = st.radio("Strategy", ["percentile", "threshold"], horizontal=True)
        if strategy == "percentile":
            percentile = st.slider("Boundary percentile", min_value=1.0, max_value=50.0, value=25.0, step=1.0)
            threshold = 0.75
        else:
            percentile = 25.0
            threshold = st.slider(
                "Similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.01,
                help="Adjacent sim below this value starts a new event",
            )

        ckpt = st.text_input("Model checkpoint", value="google/siglip2-base-patch16-224")
        batch_size = st.slider("Batch size", min_value=4, max_value=64, value=32, step=4)
        compute_matrix = st.checkbox("Compute self-similarity matrix", value=False)
        use_time_axis = st.checkbox("Timeline in seconds", value=True)
        questions = _sidebar_questions_input()
        run = st.button("Run analysis", type="primary")

    if video_path is None:
        st.info("Select or search for a video in the sidebar, then click **Run analysis**.")
        return

    video_path = Path(video_path)
    _render_video_questions(video_path, questions)

    if not run and "analysis_result" not in st.session_state:
        st.info("Choose a video and click **Run analysis**.")
        if video_path.exists():
            st.video(str(video_path))
        return

    if not video_path.exists():
        st.error(f"Video not found: {video_path}")
        return

    if run:
        with st.spinner("Encoding frames and detecting boundaries..."):
            st.session_state.analysis_result = run_analysis(
                str(video_path),
                sample_fps=sample_fps,
                strategy=strategy,
                percentile=percentile,
                threshold=threshold,
                ckpt=ckpt,
                batch_size=batch_size,
                compute_matrix=compute_matrix,
            )
            st.session_state.selected_event = 0
            st.session_state.use_time_axis = use_time_axis
            st.session_state.pop("timeline_scrub", None)

    result = st.session_state.analysis_result
    use_time_axis = st.session_state.get("use_time_axis", use_time_axis)
    frames = _load_frames(result)
    events = result["events"]
    boundary_frames = result["boundary_frames"]
    frame_indices = result["frame_indices"]
    video_fps = result["video_fps"]
    spans = _event_spans(events, frame_indices, video_fps, use_time_axis=use_time_axis)

    st.success(
        f"Detected **{len(events)}** events from **{result['num_frames']}** sampled frames "
        f"(`{result.get('strategy', 'percentile')}` strategy)."
    )

    st.session_state.event_spans = spans

    def _seek_to_event(event_idx: int) -> None:
        st.session_state.selected_event = event_idx
        x0, _ = spans[event_idx]
        st.session_state.timeline_scrub = x0

    st.subheader("Event timeline")
    timeline_event = _render_interactive_timeline(
        events,
        frame_indices,
        video_fps,
        st.session_state.selected_event,
        use_time_axis=use_time_axis,
    )
    if timeline_event != st.session_state.selected_event:
        _seek_to_event(timeline_event)
        st.rerun()

    span_min = spans[0][0]
    span_max = spans[-1][1]
    current_span = spans[st.session_state.selected_event]
    scrub_default = current_span[0]
    scrub_label = "Scrub time (s)" if use_time_axis else "Scrub position"

    def _on_scrub_change() -> None:
        st.session_state.selected_event = _event_at_position(
            st.session_state.timeline_scrub,
            st.session_state.event_spans,
        )

    st.slider(
        scrub_label,
        min_value=float(span_min),
        max_value=float(span_max),
        value=float(st.session_state.get("timeline_scrub", scrub_default)),
        key="timeline_scrub",
        on_change=_on_scrub_change,
    )

    selected_event = st.session_state.selected_event
    start, end = events[selected_event]
    scrub_position = float(st.session_state.get("timeline_scrub", scrub_default))
    jump_start_sec = _timeline_position_to_sec(
        scrub_position,
        frame_indices,
        video_fps,
        use_time_axis=use_time_axis,
    )
    _, jump_end_sec = _event_time_range(events[selected_event], frame_indices, video_fps)
    jump_end_sec = max(jump_end_sec, jump_start_sec + 0.1)

    st.subheader(f"Video · Event {selected_event + 1}")
    st.caption(f"Playing {jump_start_sec:.2f}s – {jump_end_sec:.2f}s (click timeline or scrub to jump)")
    jump_bytes = _cached_ffmpeg_clip_bytes(
        result["video_path"],
        jump_start_sec,
        jump_end_sec,
    )
    if jump_bytes is not None:
        st.video(jump_bytes)
    else:
        st.warning("Could not extract video clip with ffmpeg.")

    with st.expander("Full video (from beginning)"):
        st.video(str(video_path))

    st.subheader("Adjacent similarity")
    st.pyplot(
        _adjacent_figure(
            result["similarities"],
            result["thresholds"],
            boundary_frames,
            events,
            selected_event,
        ),
        clear_figure=True,
    )

    if result["sim_matrix"] is not None:
        st.pyplot(_sim_matrix_figure(result["sim_matrix"], boundary_frames), clear_figure=True)

    st.subheader(f"Event {selected_event + 1} frames")
    t0, t1 = _event_time_range((start, end), frame_indices, video_fps)
    st.caption(f"Frames {start}–{end - 1} · {t0:.2f}s – {t1:.2f}s")

    thumb_cols = st.columns(min(end - start, 6))
    for col, frame_idx in zip(thumb_cols, range(start, end)):
        with col:
            st.image(frame_to_pil(frames, frame_idx), caption=f"frame {frame_idx}", use_container_width=True)

    if end - start > 6:
        st.caption(f"+ {end - start - 6} more sampled frames in this event")


if __name__ == "__main__":
    main()
