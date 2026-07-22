"""Debug/smoke test for steps C+D (+E+F+G) — event graph, retrieval, QA.

Runs A -> B -> C -> D (-> E -> F -> G) end-to-end on a GPU node in the
`morph_graphs_qwen3_cu126` env:

EgoLongQA:
    python -m morph_graph.tests.debug_event_graph \
        --video morph_graph/data/wearable-ai/egolongqa/val/00cdb8ca10c069f8.mp4 \
        --tokens-per-frame 256 --fps 1.0 --sim-threshold 0.9 \
        --aligned-model pe_core_vit_l \
        --query "After I left the grassy trail that ran alongside the flooded marsh, what was the first man-made structure I encountered on the paved section of my walk?" \
        --qa \
        --working-dir outputs/debug/eg_00cdb8ca \
        --plot-graph outputs/debug/eg_00cdb8ca/graph.png

VSI-Morph:
    python -m morph_graph.tests.debug_event_graph \
        --video morph_graph/data/vsi_morph/vsi_constructed/Cat/0000.mp4 \
        --tokens-per-frame 256 --fps 1.0 --sim-threshold 0.9 \
        --aligned-model pe_core_vit_l \
        --query "In what order was the Cat first seen across the Office, the Bedroom/Toy Room, and the Industrial Room?" \
        --qa \
        --working-dir outputs/debug/vsi_cat_0000 \
        --plot-graph outputs/debug/vsi_cat_0000/graph.png

Pilot batch (event-eval IDs):
    python -m morph_graph.tests.debug_event_graph_pilot

Qwen3-VL drives TSM segmentation + values; PE-Core (or SigLIP2) supplies
text-aligned event keys for retrieval. `--qa` loads stored frame_indices,
stitches decoded pixels, and runs Qwen3-VL on the stitched clip.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

from ezcolorlog import root_logger as logger

from morph_graph.events import TSMEventSegmenter
from morph_graph.video_models import get_extractor, get_video_llm
from morph_graph.video_models.video_io import decode_frames
from morph_graph.graphrag.vl_graphrag import EventGraph, VLQueryParam
from morph_graph.graphrag.vl_graphrag.vl_segments import _video_id_from_path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_VIDEO = _REPO_ROOT / "data" / "wearable-ai" / "egolongqa" / "val" / "00cdb8ca10c069f8.mp4"
_DEFAULT_LLM = str(
    Path(os.environ.get("MORPH_GRAPH_MODEL_ROOT", _REPO_ROOT / "model_zoo"))
    / "Qwen3-VL-8B-Instruct"
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Smoke test the event graph (C+D).")
    ap.add_argument("--video", type=Path, default=_DEFAULT_VIDEO)
    ap.add_argument("--model", default="qwen3_vl_8b")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--frame-batch-size",
        type=int,
        default=128,
        help="maximum frames per Qwen vision forward; bounds GPU VRAM",
    )
    ap.add_argument("--tokens-per-frame", type=int, default=256)
    ap.add_argument(
        "--feature-stage",
        choices=["post", "pre"],
        default="post",
        help="Qwen3 vision stage for TSM: post-projection (4096) or pre-merger",
    )
    ap.add_argument("--kernel-width", type=int, default=6)
    ap.add_argument("--peak-height", type=float, default=1.0)
    ap.add_argument("--sim-threshold", type=float, default=0.5)
    ap.add_argument("--sim-top-k", type=int, default=None)
    ap.add_argument(
        "--no-store-patch-tokens",
        dest="store_patch_tokens",
        action="store_false",
        default=True,
        help="disable cached-token QA storage (retrieval-only/debug use)",
    )
    ap.add_argument(
        "--working-dir",
        type=Path,
        default=None,
        help="persist graph cache here (wiped automatically on each run)",
    )
    ap.add_argument(
        "--keep-working-dir",
        action="store_true",
        help="do not wipe --working-dir before indexing (default: clear when set)",
    )
    ap.add_argument(
        "--aligned-model",
        default=None,
        help="aligned extractor for retrieval keys (siglip2_vit_l | pe_core_vit_l)",
    )
    ap.add_argument("--query", default=None, help="text query to run after indexing (step E)")
    ap.add_argument(
        "--qa",
        action="store_true",
        help="run F+G: load values, stitch frames, answer with Qwen3-VL (needs --query)",
    )
    ap.add_argument(
        "--llm-model-path",
        default=_DEFAULT_LLM,
        help="VLM for --qa (fallback only; reuses the step-A Qwen3-VL by default)",
    )
    ap.add_argument("--retrieval-top-k", type=int, default=3)
    ap.add_argument("--max-segments-for-qa", type=int, default=5)
    ap.add_argument(
        "--max-qa-frames",
        type=int,
        default=128,
        help="global frame budget for stitched QA clip",
    )
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--plot-graph",
        type=Path,
        default=None,
        help="save an event-graph figure to this path (.png or .html)",
    )
    return ap


def run_debug_event_graph(
    args: argparse.Namespace,
    *,
    ext: Any | None = None,
    aligned: Any | None = None,
) -> dict[str, Any]:
    """Run A→G for one video. Pass preloaded ``ext`` / ``aligned`` to reuse models."""
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    if ext is None:
        logger.info(
            f"[load] extractor={args.model} tokens_per_frame={args.tokens_per_frame} "
            f"stage={args.feature_stage}"
        )
        ext = get_video_llm(
            args.model,
            tokens_per_frame=args.tokens_per_frame,
            feature_stage=args.feature_stage,
        )

    frames, frame_indices = decode_frames(
        str(args.video), fps=args.fps, max_frames=args.max_frames
    )
    feats = ext.encode_frames(frames, frame_batch_size=args.frame_batch_size)
    logger.info(
        f"[encode] pooled={tuple(feats.pooled.shape)} n_units={feats.n_units} dim={feats.dim}"
    )

    seg = TSMEventSegmenter(kernel_width=args.kernel_width, peak_height=args.peak_height)
    result = seg.segment(feats.pooled)
    logger.info(f"[segment] n_events={result.n_events} boundaries={result.boundaries}")

    aligned_pooled = None
    text_encoder = None
    key_dim = feats.dim
    if args.aligned_model:
        if aligned is None:
            logger.info(f"[aligned] extractor={args.aligned_model}")
            aligned = get_extractor(args.aligned_model)
        aligned_pooled = aligned.encode_frames(frames).pooled
        text_encoder = aligned.encode_text
        key_dim = int(aligned_pooled.shape[-1])
        logger.info(f"[aligned] pooled={tuple(aligned_pooled.shape)} dim={key_dim}")

    working_dir = (
        str(args.working_dir) if args.working_dir else tempfile.mkdtemp(prefix="event_graph_")
    )
    video_id = _video_id_from_path(str(args.video))
    eg = EventGraph(
        embedding_dim=key_dim,
        text_encoder=text_encoder,
        working_dir=working_dir,
        semantic_similarity_threshold=args.sim_threshold,
        semantic_similarity_top_k=args.sim_top_k,
        store_patch_tokens=args.store_patch_tokens,
        clear_working_dir=bool(args.working_dir) and not args.keep_working_dir,
    )
    eg.insert(
        video_id,
        str(args.video),
        feats,
        result,
        frame_indices,
        aligned_pooled=aligned_pooled,
        fps=args.fps,
    )
    logger.info(f"[graph] working_dir={working_dir}")

    summary = eg.summary()
    logger.info(
        f"[graph] nodes={summary['n_nodes']} edges={summary['n_edges']} "
        f"by_type={summary['edge_types']}"
    )
    assert summary["n_nodes"] == result.n_events, "node count != event count"
    exp_temporal = max(result.n_events - 1, 0)
    assert summary["edge_types"].get("temporal_adjacent", 0) == exp_temporal, (
        "unexpected temporal edge count"
    )

    from morph_graph.graphrag._utils import always_get_an_event_loop

    node_id = f"{video_id}::seg_0000"
    record = always_get_an_event_loop().run_until_complete(eg.segments_kv.get_by_id(node_id))
    logger.info(
        f"[kv] {node_id} -> unit_range={record['unit_range']} "
        f"frame_range={record['frame_range']} n_units={record['n_units']} values=in_memory"
    )
    value = eg.value_store.load(node_id)
    logger.info(
        f"[value] frame_indices={tuple(value['frame_indices'].shape)}"
        + (
            f" patch_tokens={tuple(value['patch_tokens'].shape)}"
            if "patch_tokens" in value
            else ""
        )
    )

    resp = None
    if args.query:
        if text_encoder is None:
            logger.warning(
                "[query] skipped: --query needs an --aligned-model for text-aligned keys"
            )
        else:
            query_param = VLQueryParam(
                mode="local",
                top_k=args.retrieval_top_k,
                max_segments_for_qa=args.max_segments_for_qa,
                max_frames_for_video_unit=args.max_qa_frames,
            )
            if args.qa:
                resp = eg.answer(
                    args.query,
                    param=query_param,
                    qa_model=ext,
                    max_new_tokens=args.max_new_tokens,
                )
                if resp.get("qa_error"):
                    logger.warning(f"[qa] failed: {resp['qa_error']}")
                else:
                    logger.info(f"[qa] answer={resp.get('answer')!r}")
            else:
                resp = eg.query(args.query, query_param)
                logger.info(f"[query] '{args.query}' -> seeds={resp['seed_node_ids']}")
                for s in resp["segments"]:
                    logger.info(
                        f"  {s['node_id']} score={s.get('retrieval_score')} "
                        f"unit_range={s.get('unit_range')} frame_range={s.get('frame_range')}"
                    )

    plot_paths: list[str] = []
    if args.plot_graph is not None:
        from morph_graph.graphrag.vl_graphrag import plot_graph

        args.plot_graph.parent.mkdir(parents=True, exist_ok=True)
        plot_graph(
            eg,
            str(args.plot_graph),
            html=args.plot_graph.suffix.lower() == ".html",
            title=f"{video_id} event graph",
            question=args.query if resp is not None else None,
            retrieved_segments=resp["segments"] if resp is not None else None,
        )
        plot_paths.append(str(args.plot_graph.resolve()))
        logger.info(f"[plot] event graph -> {args.plot_graph}")

    logger.info("[ok] event graph smoke test passed")
    return {
        "video_id": video_id,
        "video_path": str(Path(args.video).resolve()),
        "working_dir": working_dir,
        "n_events": result.n_events,
        "summary": summary,
        "query": args.query,
        "answer": None if resp is None else resp.get("answer"),
        "qa_error": None if resp is None else resp.get("qa_error"),
        "seed_node_ids": None if resp is None else resp.get("seed_node_ids"),
        "segments": None if resp is None else resp.get("segments"),
        "plot_paths": plot_paths,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_debug_event_graph(args)


if __name__ == "__main__":
    main()
