"""Unit tests for Query-Aware Flow Diffusion (QAFD) graph retrieval."""

from __future__ import annotations

import asyncio
import tempfile
import unittest

import networkx as nx
import numpy as np

from morph_graph.graphrag.vl_graphrag.qafd_retriever import (
    NetworkXQAFDRetriever,
    cosine_similarity,
)
from morph_graph.graphrag.vl_graphrag.vl_base import VLQueryParam
from morph_graph.graphrag.vl_graphrag.vl_ops import (
    get_node_embeddings_from_vdb,
    vl_query,
)


class CosineSimilarityTests(unittest.TestCase):
    def test_normalized_similarity_bounds(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(v1, v2, mode="normalized"), 1.0)

        v3 = np.array([-1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(v1, v3, mode="normalized"), 0.0)

        v4 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(v1, v4, mode="normalized"), 0.5)

    def test_relu_similarity(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([-1.0, 0.0, 0.0])
        self.assertEqual(cosine_similarity(v1, v3, mode="relu"), 0.0)

    def test_empty_or_zero_vector(self) -> None:
        self.assertEqual(cosine_similarity(None, [1.0]), 0.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)


class QAFDRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        # Build a 3-node path graph: node_A -- node_B -- node_C
        self.graph = nx.Graph()
        self.graph.add_edge("node_A", "node_B", weight=1.0)
        self.graph.add_edge("node_B", "node_C", weight=1.0)

        self.query_emb = np.array([1.0, 0.0, 0.0])
        # node_A & node_B align with query, node_C is orthogonal
        self.embeddings = {
            "node_A": np.array([1.0, 0.0, 0.0]),
            "node_B": np.array([0.9, 0.1, 0.0]),
            "node_C": np.array([0.0, 1.0, 0.0]),
        }
        self.seed_scores = {"node_A": 1.0, "node_B": 0.5}

    def test_query_aware_edge_weighting(self) -> None:
        retriever = NetworkXQAFDRetriever(
            graph=self.graph,
            seed_scores=self.seed_scores,
            node_embeddings=self.embeddings,
            query_embedding=self.query_emb,
            weight_scheme="multiply",
        )
        w_AB = retriever.get_edge_weight("node_A", "node_B")
        w_BC = retriever.get_edge_weight("node_B", "node_C")

        # A-B edge should have much higher query-aware weight than B-C edge
        self.assertGreater(w_AB, w_BC)

    def test_flow_diffusion_runs_and_converges(self) -> None:
        retriever = NetworkXQAFDRetriever(
            graph=self.graph,
            seed_scores=self.seed_scores,
            node_embeddings=self.embeddings,
            query_embedding=self.query_emb,
            alpha=10.0,
            epsilon=1e-4,
            max_iterations=500,
            weight_scheme="multiply",
        )
        scores = retriever.retrieve()
        self.assertIn("node_A", scores)
        self.assertIn("node_B", scores)
        self.assertIn("node_C", scores)
        self.assertGreater(scores["node_A"], 0.0)

        # Due to query alignment and seeding, node_A & node_B score higher than node_C
        self.assertGreater(scores["node_B"], scores["node_C"])

    def test_batch_push_mode(self) -> None:
        retriever = NetworkXQAFDRetriever(
            graph=self.graph,
            seed_scores=self.seed_scores,
            node_embeddings=self.embeddings,
            query_embedding=self.query_emb,
            batch_push=True,
        )
        scores = retriever.retrieve()
        self.assertIn("node_A", scores)
        self.assertGreater(scores["node_A"], 0.0)


class MockVectorStorage:
    def __init__(self, items: list[dict], dim: int = 4) -> None:
        self.items = items
        self.dim = dim
        self._NanoVectorDB__storage = {
            "data": [{"__id__": item["id"]} for item in items],
            "matrix": np.array([item["vector"] for item in items], dtype=np.float32),
        }

    async def query(self, query: str, top_k: int = 3) -> list[dict]:
        return [{"id": item["id"], "distance": item["distance"]} for item in self.items[:top_k]]

    async def embedding_func(self, texts: list[str]) -> list[np.ndarray]:
        return [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for _ in texts]


class MockKVStorage:
    def __init__(self, data: dict[str, dict]) -> None:
        self.data = data

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        return [self.data.get(id_) for id_ in ids]


class MockGraphStorage:
    def __init__(self, graph: nx.Graph) -> None:
        self._graph = graph


class VLQueryQAFDIntegrationTests(unittest.TestCase):
    def test_vl_query_qafd_mode(self) -> None:
        g = nx.Graph()
        g.add_edge("seg_0001", "seg_0002", weight=1.0)
        g.add_edge("seg_0002", "seg_0003", weight=1.0)

        vdb_items = [
            {"id": "seg_0001", "distance": 0.95, "vector": [1.0, 0.0, 0.0, 0.0]},
            {"id": "seg_0002", "distance": 0.85, "vector": [0.9, 0.1, 0.0, 0.0]},
            {"id": "seg_0003", "distance": 0.30, "vector": [0.0, 1.0, 0.0, 0.0]},
        ]
        vdb = MockVectorStorage(vdb_items)
        kv = MockKVStorage({
            "seg_0001": {"caption": "First segment", "segment_idx": 1},
            "seg_0002": {"caption": "Second segment", "segment_idx": 2},
            "seg_0003": {"caption": "Third segment", "segment_idx": 3},
        })
        graph_storage = MockGraphStorage(g)

        param = VLQueryParam(mode="qafd", top_k=2, max_segments_for_qa=3)

        response = asyncio.run(vl_query("test query", param, vdb, kv, graph_storage))

        self.assertIsInstance(response, dict)
        self.assertEqual(response["mode"], "qafd")
        self.assertGreater(len(response["selected_node_ids"]), 0)
        self.assertIn("seg_0001", response["selected_node_ids"])


if __name__ == "__main__":
    unittest.main()
