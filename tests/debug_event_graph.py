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

VSI-Morph 
    python -m morph_graph.tests.debug_event_graph \
        --video morph_graph/data/vsi_morph/vsi_constructed/Cat/0000.mp4 \
        --tokens-per-frame 256 --fps 1.0 --sim-threshold 0.9 \ 
        --aligned-model pe_core_vit_l \
        --query "In what order was the Cat first seen across the Office, the Bedroom/Toy Room, and the Industrial Room?" \
        --qa \
        --working-dir outputs/debug/vsi_cat_0000 \
        --plot-graph outputs/debug/vsi_cat_0000/graph.png


Qwen3-VL drives TSM segmentation + values; PE-Core (or SigLIP2) supplies
text-aligned event keys for retrieval. `--qa` loads stored frame_indices,
stitches decoded pixels, and runs Qwen3-VL on the stitched clip.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

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


def main() -> None:
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
    ap.add_argument("--feature-stage", choices=["post", "pre"], default="post",
                    help="Qwen3 vision stage for TSM: post-projection (4096) or pre-merger")
    ap.add_argument("--kernel-width", type=int, default=6)
    ap.add_argument("--peak-height", type=float, default=1.0)
    ap.add_argument("--sim-threshold", type=float, default=0.5)
    ap.add_argument("--sim-top-k", type=int, default=None)
    ap.add_argument("--no-store-patch-tokens", dest="store_patch_tokens", action="store_false",
        default=True, help="disable cached-token QA storage (retrieval-only/debug use)")
    ap.add_argument("--working-dir", type=Path, default=None,
                    help="persist graph cache here (wiped automatically on each run)")
    ap.add_argument("--keep-working-dir", action="store_true",
                    help="do not wipe --working-dir before indexing (default: clear when set)")
    ap.add_argument("--aligned-model", default=None,
                    help="aligned extractor for retrieval keys (siglip2_vit_l | pe_core_vit_l)")
    ap.add_argument("--query", default=None, help="text query to run after indexing (step E)")
    ap.add_argument("--tasks-json", type=Path, default=None, help="Batch JSON file of tasks to run (overrides --video/--query)")
    ap.add_argument("--output-json", type=Path, default=None, help="File to save JSON output of answers")
    ap.add_argument("--qa", action="store_true",
                    help="run F+G: load values, stitch frames, answer with Qwen3-VL (needs --query)")
    ap.add_argument("--llm-model-path", default=_DEFAULT_LLM,
                    help="VLM for --qa (fallback only; reuses the step-A Qwen3-VL by default)")
    ap.add_argument("--retrieval-top-k", type=int, default=3)
    ap.add_argument("--max-segments-for-qa", type=int, default=5)
    ap.add_argument("--max-qa-frames", type=int, default=128, help="global frame budget for stitched QA clip")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--plot-graph", type=Path, default=None, help="save an event-graph figure to this path")
    args = ap.parse_args()

    tasks = []
    if args.tasks_json:
        import json
        with open(args.tasks_json, 'r') as f:
            tasks = json.load(f)
    else:
        if not args.video.exists():
            raise FileNotFoundError(f"Video not found: {args.video}")
        tasks = [{"video": str(args.video), "query": args.query}]

    # A) Initialize models ONCE
    logger.info(f"[load] extractor={args.model} tokens_per_frame={args.tokens_per_frame} stage={args.feature_stage}")
    ext = get_video_llm(
        args.model,
        tokens_per_frame=args.tokens_per_frame,
        feature_stage=args.feature_stage,
    )
    frames, frame_indices = decode_frames(str(args.video), fps=args.fps, max_frames=args.max_frames)
    feats = ext.encode_frames(frames, frame_batch_size=args.frame_batch_size)
    logger.info(f"[encode] pooled={tuple(feats.pooled.shape)} n_units={feats.n_units} dim={feats.dim}")

    aligned = None
    text_encoder = None
    if args.aligned_model:
        logger.info(f"[aligned] extractor={args.aligned_model}")
        aligned = get_extractor(args.aligned_model)
        text_encoder = aligned.encode_text

    seg = TSMEventSegmenter(kernel_width=args.kernel_width, peak_height=args.peak_height)

    qa_model = getattr(ext, "model", None)
    qa_processor = getattr(ext, "processor", None)

    last_video = None
    eg = None
    cached_feats = None
    cached_result = None
    cached_aligned_pooled = None
    cached_frames = None
    cached_frame_indices = None
    
    all_answers = []
    processed_keys = set()
    
    if args.output_json and args.output_json.exists():
        import json
        try:
            with open(args.output_json, 'r') as f:
                all_answers = json.load(f)
            for ans in all_answers:
                if 'video' in ans and ans.get('answer'):
                    processed_keys.add((ans['video'], ans.get('query')))
            logger.info(f"[resume] Loaded {len(all_answers)} existing answers from {args.output_json}")
        except Exception as e:
            logger.warning(f"[resume] Failed to load existing answers: {e}")

    # round-trip one node + its value
    from morph_graph.graphrag._utils import always_get_an_event_loop

    node_id = f"{video_id}::seg_0000"
    record = always_get_an_event_loop().run_until_complete(eg.segments_kv.get_by_id(node_id))
    logger.info(f"[kv] {node_id} -> unit_range={record['unit_range']} frame_range={record['frame_range']} "
                f"n_units={record['n_units']} values=in_memory")
    value = eg.value_store.load(node_id)
    logger.info(
        f"[value] frame_indices={tuple(value['frame_indices'].shape)}"
        + (f" patch_tokens={tuple(value['patch_tokens'].shape)}" if "patch_tokens" in value else "")
    )

    # E) text query (+ optional F+G QA)
    resp = None
    if args.query:
        if text_encoder is None:
            logger.warning("[query] skipped: --query needs an --aligned-model for text-aligned keys")
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
                key_dim = feats.dim

    # plot the graph memory (highlight retrieval + show the question when available)
    if args.plot_graph is not None:
        from morph_graph.graphrag.vl_graphrag import plot_graph

        args.plot_graph.parent.mkdir(parents=True, exist_ok=True)
        plot_graph(
            eg,
            str(args.plot_graph),
            html=args.plot_graph.suffix == ".html",
            title=f"{video_id} event graph",
            question=args.query if resp is not None else None,
            retrieved_segments=resp["segments"] if resp is not None else None,
        )
        logger.info(f"[plot] event graph -> {args.plot_graph}")

        # E) text query (+ optional F+G QA)
        resp = None
        ans_text = None
        if query:
            if text_encoder is None:
                logger.warning("[query] skipped: --query needs an --aligned-model for text-aligned keys")
            else:
                query_param = VLQueryParam(
                    mode="local",
                    top_k=args.retrieval_top_k,
                    max_segments_for_qa=args.max_segments_for_qa,
                    max_frames_for_video_unit=args.max_qa_frames,
                )
                if args.qa:
                    if qa_model is not None and qa_processor is not None:
                        logger.info("[qa] reusing extractor model/processor for VLM QA")
                    else:
                        logger.info(f"[qa] extractor has no LLM; will load from {args.llm_model_path}")
                    resp = eg.answer(
                        query,
                        param=query_param,
                        default_video_path=vid_path,
                        model=qa_model,
                        processor=qa_processor,
                        model_path=args.llm_model_path if qa_model is None else None,
                        max_frames=args.max_qa_frames,
                        max_new_tokens=args.max_new_tokens,
                        vlm_question=vlm_query,
                    )
                    if resp.get("qa_error"):
                        logger.warning(f"[qa] failed: {resp['qa_error']}")
                    else:
                        ans_text = resp.get("answer")
                else:
                    resp = eg.query(query, query_param)
                    logger.info(f"[query] '{query}' -> seeds={resp['seed_node_ids']}")
                    for s in resp["segments"]:
                        logger.info(f"  {s['node_id']} score={s.get('retrieval_score')} "
                                    f"unit_range={s.get('unit_range')} frame_range={s.get('frame_range')}")

        # Save to list
        task_res = dict(task)
        if ans_text is not None:
            task_res['answer'] = ans_text
        if resp is not None:
            # Drop very large structures if we want, but returning answer is key
            task_res['qa_error'] = resp.get("qa_error")
            if "answer" in resp:
                task_res['answer'] = resp["answer"]
            
        all_answers.append(task_res)
        
        # Streaming save
        if args.output_json:
            import json
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, 'w') as f:
                json.dump(all_answers, f, indent=2)


        # plot graph
        if args.plot_graph is not None:
            from morph_graph.graphrag.vl_graphrag import plot_event_graph
            args.plot_graph.parent.mkdir(parents=True, exist_ok=True)
            # save per query
            import hashlib
            qid = task.get("qid", hashlib.md5(query.encode()).hexdigest()[:6] if query else "no_query")
            plot_path = args.plot_graph.parent / f"{args.plot_graph.stem}_{vid_id}_{qid}{args.plot_graph.suffix}"
            plot_event_graph(
                eg,
                str(plot_path),
                html=args.plot_graph.suffix == ".html",
                title=f"{vid_id} event graph",
                question=query if resp is not None else None,
                retrieved_segments=resp["segments"] if resp is not None else None,
            )
            logger.info(f"[plot] event graph -> {plot_path}")

    if args.output_json:
        import json
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(all_answers, f, indent=2)
        logger.info(f"[ok] saved answers to {args.output_json}")

    logger.info("[ok] event graph batch testing passed\n")


if __name__ == "__main__":
    main()
