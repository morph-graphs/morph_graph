"""Orchestrate indexing, retrieval, and optional VLM answering for one benchmark run."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

from ezcolorlog import root_logger as logger
from tqdm import tqdm

from morph_graph.graphrag.vl_graphrag.vl_base import VLQueryParam
from morph_graph.inference.framework.base import EvalConfig, VLGraphRAGBenchmark
from morph_graph.inference.framework.rag import index_rag, resolve_embedding_key
from morph_graph.inference.framework.vlm import (
    build_messages_from_caption_context,
    load_model,
    model_path_to_key,
    run_inference,
)


def run_benchmark(benchmark: VLGraphRAGBenchmark, config: EvalConfig, args: Namespace) -> Path:
    samples = benchmark.load_samples(args)
    if config.max_samples is not None:
        samples = samples[: config.max_samples]

    embedding_key = resolve_embedding_key(config.embedding_key)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.rag_cache_dir, exist_ok=True)

    # Load VLM only when we will generate answers (not --retrieval_only).
    model = processor = None
    model_key = None
    if not config.retrieval_only:
        if not config.model_path:
            raise ValueError("model_path is required unless --retrieval_only is set")
        model, processor = load_model(config.model_path)
        model_key = model_path_to_key(config.model_path)

    # In-run cache avoids re-opening the same per-video working_dir across questions.
    rag_cache = {}
    results: list[dict] = []
    n_correct = 0
    query_param = VLQueryParam(
        mode=config.retrieval_mode,
        key_source=config.key_source,
        top_k=config.top_k,
        max_segments_for_qa=config.max_segments_for_qa,
        only_need_context=False,
    )
    desc = f"{benchmark.name}[{config.retrieval_mode}/{config.key_source}]"

    for sample in tqdm(samples, desc=desc):
        video_path = benchmark.get_video_path(sample, args)
        inputs = benchmark.get_index_inputs(sample, args)
        if not inputs.required_for(config.key_source):
            logger.warning(f"No {config.key_source} index data for {sample.sample_id}, skipping")
            continue

        # Index caption and/or vision when provided; retrieval uses key_source to pick VDB.
        rag = index_rag(
            video_path,
            config.rag_cache_dir,
            embedding_key,
            config.reindex,
            rag_cache,
            inputs,
        )
        retrieval = rag.query(sample.question, query_param)
        row = benchmark.build_result_row(sample, video_path, retrieval, config)

        if config.retrieval_only:
            results.append(row)
            continue

        # Caption-context QA: retrieved segment captions → Qwen3-VL text prompt.
        caption_context = str(row.get("caption_context", ""))
        if not caption_context.strip():
            row.update(pred="", parsed_pred="", correct=False)
            results.append(row)
            continue

        messages = build_messages_from_caption_context(
            caption_context,
            sample.question,
            benchmark.format_options(sample),
        )
        pred = run_inference(model, processor, messages, max_new_tokens=config.max_new_tokens)
        parsed = benchmark.parse_prediction(pred)
        row.update(pred=pred, parsed_pred=parsed, correct=(parsed == sample.ground_truth))
        n_correct += int(row["correct"])
        results.append(row)

    acc = (n_correct / len(results)) if results and not config.retrieval_only else None
    if acc is not None:
        logger.info(f"Accuracy: {n_correct}/{len(results)} = {acc:.3f}")

    out_path = config.output_dir / benchmark.results_filename(config, embedding_key, model_key)
    output = {
        "benchmark": benchmark.name,
        "embedding_key": embedding_key,
        "retrieval_mode": config.retrieval_mode,
        "key_source": config.key_source,
        "retrieval_only": config.retrieval_only,
        "top_k": config.top_k,
        "max_segments_for_qa": config.max_segments_for_qa,
        "accuracy": acc,
        "category_breakdown": benchmark.compute_breakdown(results) if not config.retrieval_only else {},
        "retrieval_metrics": benchmark.aggregate_retrieval_metrics(results),
        "results": results,
    }
    if model_key:
        output.update(model_key=model_key, model_path=config.model_path, max_new_tokens=config.max_new_tokens)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved results to {out_path}")
    return out_path
