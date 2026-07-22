"""Query-Aware Flow Diffusion (QAFD) Retriever for NetworkX graphs.

Implements push-relabel flow diffusion over video event graphs (NetworkX),
modulating edge traversal weights dynamically by query relevance.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

from .._utils import logger


def cosine_similarity(
    vec1: np.ndarray | list[float] | None,
    vec2: np.ndarray | list[float] | None,
    mode: str = "normalized",
) -> float:
    """Cosine similarity with configurable contrast.

    Modes:
        "normalized": (cos + 1) / 2 -> [0, 1] (standard)
        "relu":       max(0, cos)    -> [0, 1]
        "relu_sq":    max(0, cos)^2  -> [0, 1] (sharpest contrast)
    """
    if vec1 is None or vec2 is None:
        return 0.0
    v1 = np.asarray(vec1, dtype=np.float64)
    v2 = np.asarray(vec2, dtype=np.float64)
    if v1.size == 0 or v2.size == 0:
        return 0.0

    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    if m1 == 0 or m2 == 0:
        return 0.0

    raw = float(np.dot(v1, v2) / (m1 * m2))
    if mode == "relu":
        return max(0.0, raw)
    elif mode == "relu_sq":
        r = max(0.0, raw)
        return r * r
    else:  # "normalized"
        return max(0.0, (raw + 1.0) / 2.0)


class NetworkXQAFDRetriever:
    """Query-Aware Weighted Flow Diffusion retriever for NetworkX graphs.

    Uses push-relabel flow diffusion to rank nodes by relevance.
    Mass injected at seed nodes diffuses through the graph; importance scores x
    accumulate at nodes along query-aligned paths.
    """

    def __init__(
        self,
        graph: nx.Graph,
        seed_scores: dict[str, float],
        node_embeddings: dict[str, Any] | None = None,
        query_embedding: Any | None = None,
        alpha: float = 10.0,
        epsilon: float = 1e-4,
        max_iterations: int = 500,
        step_size: float = 0.2,
        weight_scheme: str = "multiply",
        sim_mode: str = "normalized",
        use_node_degree: bool = True,
        random_seed: int = 42,
        qa_sink_gamma: float = 0.0,
        qa_warm_delta: float = 0.0,
        qa_warm_steps: int = 2,
        qa_accum_gamma: float = 0.0,
        qa_post_lambda: float = 0.0,
        batch_push: bool = False,
    ) -> None:
        self.graph = graph
        self.seed_scores = seed_scores
        self.node_embeddings = node_embeddings or {}
        self.query_embedding = (
            np.asarray(query_embedding, dtype=np.float64)
            if query_embedding is not None
            else None
        )
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.weight_scheme = weight_scheme
        self.sim_mode = sim_mode
        self.use_node_degree = use_node_degree
        self.random_seed = random_seed
        self.qa_sink_gamma = qa_sink_gamma
        self.qa_warm_delta = qa_warm_delta
        self.qa_warm_steps = qa_warm_steps
        self.qa_accum_gamma = qa_accum_gamma
        self.qa_post_lambda = qa_post_lambda
        self.batch_push = batch_push

        self.mass: dict[str, float] = defaultdict(float)
        self.sink_capacity: dict[str, float] = defaultdict(float)
        self.x: dict[str, float] = defaultdict(float)
        self.edge_weights_cache: dict[tuple[str, str], float] = {}
        self.node_sim_cache: dict[str, float] = {}

        random.seed(self.random_seed)

    def _get_node_sim(self, node: str) -> float:
        if node in self.node_sim_cache:
            return self.node_sim_cache[node]
        if self.query_embedding is None or node not in self.node_embeddings:
            sim = 0.0
        else:
            sim = cosine_similarity(
                self.node_embeddings[node], self.query_embedding, mode=self.sim_mode
            )
        self.node_sim_cache[node] = sim
        return sim

    def get_edge_weight(self, u: str, v: str) -> float:
        """Get query-aware edge weight: w'(u,v) = w(u,v) * f(sim(u,q), sim(v,q))."""
        cache_key = (u, v) if u <= v else (v, u)
        if cache_key in self.edge_weights_cache:
            return self.edge_weights_cache[cache_key]

        edge_data = self.graph.get_edge_data(u, v, default={})
        w = float(edge_data.get("weight", 1.0))

        if w <= 0:
            self.edge_weights_cache[cache_key] = 0.0
            return 0.0

        if (
            self.weight_scheme == "none"
            or not self.node_embeddings
            or self.query_embedding is None
        ):
            self.edge_weights_cache[cache_key] = w
            return w

        s1 = self._get_node_sim(u)
        s2 = self._get_node_sim(v)

        if self.weight_scheme == "multiply":
            smart_weight = w * s1 * s2
        elif self.weight_scheme == "add":
            smart_weight = (w + s1 + s2) / 3.0
        else:  # "original" / "hybrid"
            smart_weight = w * (1.0 + 0.5 * (s1 + s2) / 2.0)

        self.edge_weights_cache[cache_key] = smart_weight
        return smart_weight

    def initialize(self) -> None:
        """Initialize sink capacities, source mass, and warm-start x."""
        nodes = list(self.graph.nodes())
        if not nodes:
            return

        # 1. Normalize seed weights
        sw = {node: max(0.0, float(self.seed_scores.get(node, 0.0))) for node in nodes}
        sw_total = sum(sw.values())
        if sw_total > 0:
            sw = {node: val / sw_total for node, val in sw.items()}
        else:
            sw = {node: 1.0 / len(nodes) for node in nodes}

        # 2. Sink capacities
        for node in nodes:
            if self.use_node_degree:
                self.sink_capacity[node] = max(float(self.graph.degree(node)), 1.0)
            else:
                self.sink_capacity[node] = 1.0

        total_sink = sum(self.sink_capacity.values())
        if total_sink > 0:
            for node in nodes:
                self.sink_capacity[node] = 10.0 * self.sink_capacity[node] / total_sink

        if self.qa_sink_gamma > 0:
            for node in nodes:
                self.sink_capacity[node] *= 1.0 + self.qa_sink_gamma * self._get_node_sim(node)

        total_sink = sum(self.sink_capacity.values())

        # 3. Inject mass at seed nodes
        self.mass.clear()
        for node in nodes:
            if sw[node] > 0:
                self.mass[node] = self.alpha * total_sink * sw[node]
            else:
                self.mass[node] = 0.0

        # 4. Warm-start x via lazy random walk
        if self.qa_warm_delta > 0:
            x_init = {
                node: sw[node] * (1.0 + self.qa_warm_delta * self._get_node_sim(node))
                for node in nodes
            }
            x_sum = sum(x_init.values())
            if x_sum > 0:
                x_init = {k: v / x_sum for k, v in x_init.items()}
        else:
            x_init = dict(sw)

        curr_x = dict(x_init)
        for _ in range(self.qa_warm_steps):
            x_next = defaultdict(float)
            for node, val in curr_x.items():
                if val > 0:
                    neighbors = list(self.graph.neighbors(node))
                    if neighbors:
                        spread = val / len(neighbors)
                        for neighbor in neighbors:
                            x_next[neighbor] += spread
            curr_x = {
                node: (sw.get(node, 0.0) + x_next.get(node, 0.0)) / 2.0
                for node in nodes
            }

        self.x = defaultdict(float, curr_x)

    def push(self, node: str) -> bool:
        """Push excess mass from node to neighbors."""
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return False

        w_qa = sum(self.get_edge_weight(node, nbr) for nbr in neighbors)
        if w_qa == 0:
            return False

        excess = self.mass[node] - self.sink_capacity[node]
        if excess <= 0:
            return False

        w_struct = sum(
            float(self.graph.get_edge_data(node, nbr, default={}).get("weight", 1.0))
            for nbr in neighbors
        )
        if w_struct == 0:
            w_struct = w_qa

        accum = self.step_size * excess / (w_struct + 1e-8)
        if self.qa_accum_gamma > 0:
            accum *= 1.0 + self.qa_accum_gamma * self._get_node_sim(node)

        self.x[node] += accum
        self.mass[node] = self.sink_capacity[node]

        for nbr in neighbors:
            w_ij = self.get_edge_weight(node, nbr)
            if w_ij > 0:
                self.mass[nbr] += excess * w_ij / (w_qa + 1e-8)

        return True

    def flow_diffusion(self) -> dict[str, float]:
        """Run push-relabel flow diffusion until convergence."""
        iterations = 0
        pushes = 0
        nodes = list(self.graph.nodes())

        while iterations < self.max_iterations:
            iterations += 1

            excess_nodes = [
                node
                for node in nodes
                if self.mass[node] > self.sink_capacity[node] + self.epsilon
            ]

            if not excess_nodes:
                logger.debug(f"QAFD converged in {iterations} iterations ({pushes} pushes)")
                break

            if self.batch_push:
                for node in excess_nodes:
                    if self.push(node):
                        pushes += 1
            else:
                node = random.choice(excess_nodes)
                if self.push(node):
                    pushes += 1

            if iterations % 10 == 0:
                remaining_excess = sum(
                    max(0.0, self.mass[n] - self.sink_capacity[n]) for n in nodes
                )
                if remaining_excess < self.epsilon:
                    logger.debug(
                        f"QAFD converged in {iterations} iterations ({pushes} pushes)"
                    )
                    break

        if iterations >= self.max_iterations:
            logger.warning(f"QAFD did not converge after {self.max_iterations} iterations")

        return dict(self.x)

    def retrieve(self) -> dict[str, float]:
        """Execute QAFD retrieval and return dict of node scores."""
        if self.graph.number_of_nodes() == 0:
            return {}

        self.initialize()
        scores = self.flow_diffusion()

        if self.qa_post_lambda > 0 and self.query_embedding is not None:
            scores = {
                node: val * (1.0 + self.qa_post_lambda * self._get_node_sim(node))
                for node, val in scores.items()
            }

        return scores
