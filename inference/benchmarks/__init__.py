"""Benchmark registry: subclass VLGraphRAGBenchmark and add here."""

from __future__ import annotations

from argparse import ArgumentParser

from morph_graph.inference.benchmarks.vsi_morph import VSIMorphBenchmark
from morph_graph.inference.framework.base import VLGraphRAGBenchmark

BENCHMARK_REGISTRY: dict[str, type[VLGraphRAGBenchmark]] = {
    "vsi_morph": VSIMorphBenchmark
}


def get_benchmark(name: str) -> VLGraphRAGBenchmark:
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {sorted(BENCHMARK_REGISTRY)}")
    return BENCHMARK_REGISTRY[name]()


def add_benchmark_argument(parser: ArgumentParser) -> None:
    parser.add_argument("--benchmark", default="vsi_morph", choices=sorted(BENCHMARK_REGISTRY))
