"""Plot VL-GraphRAG segment relation graphs (GraphML → PNG/HTML)."""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import Any

import networkx as nx

_FRAME_RE = re.compile(r"frames\s+(\d+)-(\d+)", re.IGNORECASE)

COLORS = {
    "node": "#e0f2fe",
    "node_border": "#0369a1",
    "retrieved": "#fed7aa",
    "retrieved_border": "#c2410c",
    "temporal": "#1d4ed8",
    "semantic": "#ea580c",
    "label": "#0f172a",
    "frame": "#64748b",
}


def load_graph(graphml_path: str | Path) -> nx.Graph:
    path = Path(graphml_path)
    if not path.exists():
        raise FileNotFoundError(f"GraphML not found: {path}")
    return nx.read_graphml(path)


def _format_score(value: Any) -> str:
    """Format a retrieval/similarity score (handles numpy scalars)."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _coerce_int(value: Any, default: int | str = "?") -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _segment_idx(attrs: dict) -> int:
    return int(_coerce_int(attrs.get("segment_idx", 0), 0))


def _ordered_nodes(graph: nx.Graph) -> list[tuple[str, dict]]:
    return sorted(graph.nodes(data=True), key=lambda item: _segment_idx(item[1]))


def _scene_label(attrs: dict, *, max_len: int = 18) -> str:
    label = str(attrs.get("scene_label") or "unknown")
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "\u2026"


def _frame_range(attrs: dict) -> tuple[int, int] | None:
    desc = str(attrs.get("description", ""))
    match = _FRAME_RE.search(desc)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _temporal_chain(ordered: list[tuple[str, dict]]) -> list[tuple[str, str]]:
    """Consecutive segment pairs on the timeline (ignore duplicate bidirectional edges)."""
    return [(ordered[i][0], ordered[i + 1][0]) for i in range(len(ordered) - 1)]


def _semantic_edges(graph: nx.Graph) -> list[tuple[str, str, float]]:
    seen: set[tuple[str, str]] = set()
    edges: list[tuple[str, str, float]] = []
    for src, tgt, attrs in graph.edges(data=True):
        if attrs.get("edge_type") != "semantic_sim":
            continue
        key = tuple(sorted((src, tgt)))
        if key in seen:
            continue
        seen.add(key)
        edges.append((src, tgt, float(attrs.get("weight", 1.0))))
    edges.sort(key=lambda e: -e[2])
    return edges


def _filter_semantic_edges(
    edges: list[tuple[str, str, float]],
    *,
    max_edges: int | None = 10,
    min_weight: float = 0.0,
) -> tuple[list[tuple[str, str, float]], int]:
    """Return edges to draw and the total count before filtering."""
    total = len(edges)
    kept = [(s, t, w) for s, t, w in edges if w >= min_weight]
    kept.sort(key=lambda e: -e[2])
    if max_edges is not None and len(kept) > max_edges:
        kept = kept[:max_edges]
    return kept, total


def _assign_arc_lanes(
    edges: list[tuple[str, str, float]],
    idx_of: dict[str, int],
) -> list[tuple[str, str, float, int]]:
    """Pack non-overlapping interval arcs into stacked lanes (short spans inner)."""
    indexed: list[tuple[int, int, str, str, float]] = []
    for src, tgt, weight in edges:
        a, b = (src, tgt) if idx_of[src] <= idx_of[tgt] else (tgt, src)
        indexed.append((idx_of[a], idx_of[b], a, b, weight))
    indexed.sort(key=lambda e: (e[1] - e[0], -e[4]))

    lanes: list[list[tuple[int, int]]] = []
    packed: list[tuple[str, str, float, int]] = []
    for i_start, i_end, a, b, weight in indexed:
        for lane_idx, intervals in enumerate(lanes):
            if all(i_end <= s or i_start >= e for s, e in intervals):
                intervals.append((i_start, i_end))
                packed.append((a, b, weight, lane_idx))
                break
        else:
            lanes.append([(i_start, i_end)])
            packed.append((a, b, weight, len(lanes) - 1))
    return packed


def _semantic_apex_y(
    x0: float,
    x1: float,
    lane: int,
    *,
    chord_y: float,
    lane_step: float,
) -> float:
    """Y at t=0.5 on the quadratic bezier (actual arc bottom)."""
    bulge = abs(x1 - x0) * 0.32 + lane * lane_step
    ctrl_y = chord_y - bulge
    return (chord_y + ctrl_y) / 2.0


def _semantic_arc_patch(
    x0: float,
    x1: float,
    lane: int,
    weight: float,
    *,
    chord_y: float,
    lane_step: float,
):
    """Quadratic bezier arc with apex strictly below chord_y."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    span = abs(x1 - x0)
    bulge = span * 0.32 + lane * lane_step
    ctrl_y = chord_y - bulge
    mid_x = (x0 + x1) / 2.0
    apex_y = (chord_y + ctrl_y) / 2.0
    patch = PathPatch(
        Path(
            [(x0, chord_y), (mid_x, ctrl_y), (x1, chord_y)],
            [Path.MOVETO, Path.CURVE3, Path.CURVE3],
        ),
        facecolor="none",
        edgecolor=COLORS["semantic"],
        linewidth=0.7 + 1.6 * weight,
        linestyle=(0, (5, 4)),
        alpha=0.4 + 0.5 * weight,
        zorder=1,
        clip_on=False,
    )
    return patch, mid_x, apex_y

def _timeline_positions(ordered: list[tuple[str, dict]]) -> dict[str, tuple[float, float]]:
    if len(ordered) == 1:
        return {ordered[0][0]: (0.0, 0.0)}
    span = max(len(ordered) - 1, 1)
    return {node_id: (i / span, 0.0) for i, (node_id, _) in enumerate(ordered)}


def _question_block(question: str, width: int = 96) -> str:
    q = question.strip()
    if not q:
        return ""
    wrapped = textwrap.fill(q, width=width)
    return f"Q: {wrapped}"


def _question_line_count(question: str, width: int = 96) -> int:
    block = _question_block(question, width=width)
    return max(1, block.count("\n") + 1) if block else 0


def _heading(title: str | None, question: str | None) -> str | None:
    parts = [p for p in (title, _question_block(question) if question else None) if p]
    return "\n\n".join(parts) if parts else None


def _draw_retrieval_panel(
    panel_ax,
    segments: list[dict[str, Any]],
    graph: nx.Graph,
    scores_by_id: dict[str, Any] | None = None,
) -> None:
    """Minimal side panel: rank, segment index, retrieval score."""
    node_attrs = dict(graph.nodes(data=True))
    scores_by_id = scores_by_id or {}

    lines = [f"Top-{len(segments)} retrieval", ""]
    for rank, seg in enumerate(segments, 1):
        node_id = str(seg.get("node_id") or "")
        attrs = node_attrs.get(node_id, seg)
        idx = _coerce_int(seg.get("segment_idx", attrs.get("segment_idx", "?")))
        score = seg.get("retrieval_score")
        if score is None:
            score = scores_by_id.get(node_id)
        lines.append(f"{rank}. seg {idx}   {_format_score(score)}")

    panel_ax.set_facecolor("#ffffff")
    panel_ax.axis("off")
    panel_ax.text(
        0.08,
        0.98,
        "\n".join(lines),
        transform=panel_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=COLORS["label"],
        linespacing=1.5,
        fontfamily="monospace",
    )


def _node_label_text(attrs: dict) -> str:
    """Two-line label: scene name on top, frame range below."""
    scene = _scene_label(attrs, max_len=14)
    frames = _frame_range(attrs)
    if frames:
        return f"{scene}\nf{frames[0]}\u2013{frames[1]}"
    return scene


def _retrieval_panel_lines(segments: list[dict[str, Any]]) -> int:
    return max(3, 2 + len(segments))

def plot_segment_graph(
    graph: nx.Graph,
    out_path: str | Path,
    *,
    highlight_nodes: set[str] | None = None,
    title: str | None = None,
    question: str | None = None,
    retrieved_segments: list[dict[str, Any]] | None = None,
    retrieval_scores: dict[str, Any] | None = None,
    show: bool = False,
    max_semantic_edges: int | None = 10,
    min_semantic_weight: float = 0.0,
) -> Path:
    """Render segment graph as a timeline with semantic arcs below."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib") from exc

    highlight_nodes = highlight_nodes or set()
    if retrieved_segments:
        for seg in retrieved_segments:
            node_id = seg.get("node_id")
            if node_id:
                highlight_nodes.add(str(node_id))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = _ordered_nodes(graph)
    pos = _timeline_positions(ordered)
    idx_of = {node_id: i for i, (node_id, _) in enumerate(ordered)}
    n_nodes = len(ordered)

    spine_y = 0.0
    arc_y = -0.05
    lane_step = 0.055

    all_semantic = _semantic_edges(graph)
    semantic, semantic_total = _filter_semantic_edges(
        all_semantic,
        max_edges=max_semantic_edges,
        min_weight=min_semantic_weight,
    )
    arc_lanes = _assign_arc_lanes(semantic, idx_of)
    max_apex_y = arc_y
    for a, b, _weight, lane in arc_lanes:
        apex_y = _semantic_apex_y(
            pos[a][0], pos[b][0], lane, chord_y=arc_y, lane_step=lane_step,
        )
        max_apex_y = min(max_apex_y, apex_y)

    has_question = bool(question and question.strip())
    if semantic_total > len(semantic):
        link_note = f"{len(semantic)} of {semantic_total} semantic links (top by weight)"
    else:
        link_note = f"{len(semantic)} semantic links"
    title_text = title or f"Segment graph \u2014 {n_nodes} segments, {link_note}"
    has_retrieval = bool(retrieved_segments)
    panel_w = 1.6 if has_retrieval else 0.0
    fig_w = max(12, n_nodes * 1.5) + panel_w
    extra_lane_h = max(0, max(lane for *_, lane in arc_lanes) if arc_lanes else 0) * 0.35
    panel_lines = _retrieval_panel_lines(retrieved_segments or [])

    if has_question or has_retrieval:
        q_lines = _question_line_count(question) if has_question else 0
        header_h = (0.9 + 0.22 * q_lines) if has_question else 0.0
        main_h = 5.8 + extra_lane_h + (0.08 * panel_lines if has_retrieval else 0.0)
        fig_h = header_h + main_h
        fig = plt.figure(figsize=(fig_w, fig_h))

        if has_question and has_retrieval:
            gs = fig.add_gridspec(
                2, 2,
                height_ratios=[header_h, main_h],
                width_ratios=[3.2, 1.0],
                hspace=0.32,
                wspace=0.12,
            )
            header_ax = fig.add_subplot(gs[0, :])
            ax = fig.add_subplot(gs[1, 0])
            panel_ax = fig.add_subplot(gs[1, 1])
        elif has_question:
            gs = fig.add_gridspec(2, 1, height_ratios=[header_h, main_h], hspace=0.35)
            header_ax = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1])
            panel_ax = None
        else:
            gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.12)
            header_ax = None
            ax = fig.add_subplot(gs[0, 0])
            panel_ax = fig.add_subplot(gs[0, 1])

        if header_ax is not None:
            header_ax.set_facecolor("#ffffff")
            header_ax.axis("off")
            header_ax.text(
                0.5,
                0.98,
                title_text,
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
                color=COLORS["label"],
                transform=header_ax.transAxes,
            )
            header_ax.text(
                0.5,
                0.72,
                _question_block(question),
                ha="center",
                va="top",
                fontsize=10,
                color=COLORS["label"],
                transform=header_ax.transAxes,
                bbox={"boxstyle": "round,pad=0.45", "fc": "#f1f5f9", "ec": "#cbd5e1", "lw": 1},
            )
    else:
        fig, ax = plt.subplots(figsize=(fig_w, 6.2 + extra_lane_h))
        panel_ax = None
        header_ax = None

    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")

    # Semantic arcs in stacked lanes below the spine; temporal spine drawn on top.
    for a, b, weight, lane in arc_lanes:
        patch, mid_x, apex_y = _semantic_arc_patch(
            pos[a][0], pos[b][0], lane, weight, chord_y=arc_y, lane_step=lane_step,
        )
        ax.add_patch(patch)
        ax.text(
            mid_x,
            apex_y,
            _format_score(weight),
            ha="center",
            va="top",
            fontsize=6.5,
            color=COLORS["semantic"],
            zorder=3,
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.12",
                "fc": "white",
                "ec": "#fdba74",
                "alpha": 0.95,
                "lw": 0.6,
            },
        )

    for src, tgt in _temporal_chain(ordered):
        ax.plot(
            [pos[src][0], pos[tgt][0]],
            [spine_y, spine_y],
            color=COLORS["temporal"],
            lw=4,
            solid_capstyle="round",
            zorder=2,
            clip_on=False,
        )

    # Nodes as round markers (scatter -> size in points, immune to axis aspect).
    xs = [pos[node_id][0] for node_id, _ in ordered]
    ys = [spine_y] * n_nodes
    face = [COLORS["retrieved"] if nid in highlight_nodes else COLORS["node"] for nid, _ in ordered]
    edge = [COLORS["retrieved_border"] if nid in highlight_nodes else COLORS["node_border"] for nid, _ in ordered]
    lw = [2.6 if nid in highlight_nodes else 1.8 for nid, _ in ordered]
    ax.scatter(xs, ys, s=620, c=face, edgecolors=edge, linewidths=lw, zorder=4)

    # Index inside node; scene + frames as a centered two-line label above the spine.
    label_y = spine_y + 0.028
    for node_id, attrs in ordered:
        x, _ = pos[node_id]
        retrieved = node_id in highlight_nodes
        ax.text(x, spine_y, str(_segment_idx(attrs)), ha="center", va="center", fontsize=9,
                fontweight="bold", zorder=5,
                color=COLORS["retrieved_border"] if retrieved else COLORS["node_border"])
        ax.text(
            x,
            label_y,
            _node_label_text(attrs),
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["label"],
            zorder=5,
            linespacing=0.95,
        )

    legend_handles = [
        Line2D([0], [0], color=COLORS["temporal"], lw=4, label="temporal (playback order)"),
        Line2D([0], [0], color=COLORS["semantic"], lw=2, linestyle="--", label="semantic similarity"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["node"],
               markeredgecolor=COLORS["node_border"], markersize=11, label="segment"),
    ]
    if highlight_nodes:
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=COLORS["retrieved"],
                              markeredgecolor=COLORS["retrieved_border"], markersize=11, label="retrieved"))
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=len(legend_handles), frameon=False, fontsize=9)

    if not has_question and header_ax is None:
        ax.set_title(title_text, fontsize=12, fontweight="bold", pad=14)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(max_apex_y - 0.10, spine_y + 0.26)
    ax.margins(x=0.02, y=0.05)
    ax.axis("off")

    if panel_ax is not None and retrieved_segments:
        _draw_retrieval_panel(panel_ax, retrieved_segments, graph, retrieval_scores)

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    return out_path

def write_graph_html(
    graph: nx.Graph,
    out_path: str | Path,
    *,
    highlight_nodes: set[str] | None = None,
    title: str | None = None,
    question: str | None = None,
) -> Path:
    """Interactive HTML with fixed timeline layout (requires pyvis)."""
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError("pyvis is required for HTML plots: pip install pyvis") from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = _ordered_nodes(graph)
    pos = _timeline_positions(ordered)

    net = Network(
        height="520px",
        width="100%",
        bgcolor="#fafafa",
        font_color=COLORS["label"],
        directed=False,
    )
    net.toggle_physics(False)

    highlight_nodes = highlight_nodes or set()

    for node_id, attrs in ordered:
        x, y = pos[node_id]
        frames = _frame_range(attrs)
        frame_str = f"f{frames[0]}–{frames[1]}" if frames else ""
        label = _scene_label(attrs)
        retrieved = node_id in highlight_nodes
        net.add_node(
            node_id,
            label=f"{_segment_idx(attrs)}: {label}",
            title=f"{label}\n{frame_str}\n{attrs.get('description', node_id)}",
            x=x * 900,
            y=y * 100,
            physics=False,
            color={
                "background": COLORS["retrieved"] if retrieved else COLORS["node"],
                "border": COLORS["retrieved_border"] if retrieved else COLORS["node_border"],
                "highlight": COLORS["retrieved"],
            },
            font={"size": 14, "face": "arial"},
            shape="dot",
            size=26 if retrieved else 22,
        )

    for src, tgt in _temporal_chain(ordered):
        net.add_edge(src, tgt, color=COLORS["temporal"], width=3, title="temporal")

    for src, tgt, weight in _semantic_edges(graph):
        net.add_edge(
            src,
            tgt,
            color=COLORS["semantic"],
            dashes=True,
            width=1 + 2 * weight,
            title=f"semantic sim ({weight:.3f})",
            smooth={"type": "curvedCW", "roundness": 0.35},
        )

    heading = _heading(title, question)
    if heading:
        net.heading = heading
    net.write_graph(str(out_path))
    return out_path


def plot_from_graphml(
    graphml_path: str | Path,
    out_path: str | Path | None = None,
    *,
    html: bool = False,
    highlight_nodes: set[str] | None = None,
    title: str | None = None,
    question: str | None = None,
    retrieved_segments: list[dict[str, Any]] | None = None,
    retrieval_scores: dict[str, Any] | None = None,
    show: bool = False,
    max_semantic_edges: int | None = 10,
    min_semantic_weight: float = 0.0,
) -> Path:
    graphml_path = Path(graphml_path)
    graph = load_graph(graphml_path)
    if out_path is None:
        suffix = ".html" if html else ".png"
        out_path = graphml_path.with_name(graphml_path.stem + suffix)
    if html:
        return write_graph_html(
            graph, out_path, highlight_nodes=highlight_nodes, title=title, question=question
        )
    return plot_segment_graph(
        graph,
        out_path,
        highlight_nodes=highlight_nodes,
        title=title,
        question=question,
        retrieved_segments=retrieved_segments,
        retrieval_scores=retrieval_scores,
        show=show,
        max_semantic_edges=max_semantic_edges,
        min_semantic_weight=min_semantic_weight,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a VL-GraphRAG segment relation graph")
    parser.add_argument(
        "graphml",
        nargs="?",
        default="outputs/vl_graphrag_cache_debug/Cat__0000/graph_segment_relation.graphml",
        help="Path to graph_segment_relation.graphml",
    )
    parser.add_argument("-o", "--output", default=None, help="Output PNG or HTML path")
    parser.add_argument("--html", action="store_true", help="Write interactive HTML (requires pyvis)")
    parser.add_argument("--show", action="store_true", help="Display matplotlib window")
    parser.add_argument("--highlight", nargs="*", default=[], help="Node ids to highlight")
    parser.add_argument("--title", default=None)
    parser.add_argument("--question", default=None, help="Retrieval question to show on the plot")
    parser.add_argument(
        "--max-semantic-edges",
        type=int,
        default=10,
        help="Max semantic edges to draw (top by weight; default 10)",
    )
    parser.add_argument(
        "--min-semantic-weight",
        type=float,
        default=0.0,
        help="Minimum semantic edge weight to draw",
    )
    args = parser.parse_args()

    out = plot_from_graphml(
        args.graphml,
        args.output,
        html=args.html,
        highlight_nodes=set(args.highlight),
        title=args.title,
        question=args.question,
        show=args.show,
        max_semantic_edges=args.max_semantic_edges,
        min_semantic_weight=args.min_semantic_weight,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
