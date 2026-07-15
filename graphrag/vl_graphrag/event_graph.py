"""EventGraph: key-value store (C) + event linking (D) on nano-graphrag.

Mirrors `VLGraphRAG` but for precomputed event keys/values from steps A/B:
  - keys   -> NanoVectorDBStorage (via a precomputed/identity embedding func)
  - values -> EventValueStore (tensors on disk) + metadata in JsonKVStorage
  - nodes/edges -> NetworkXStorage

Reuses the `vl_ops` builders unchanged:
  - `upsert_segment_values_and_graph` : KV metadata + graph nodes + in-batch
    temporal edges (consecutive events).
  - `upsert_temporal_edges`           : cross-batch temporal linking.
  - `upsert_segment_embeddings_with_semantic_edges` : cosine kNN semantic edges
    over event keys + VDB upsert.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Type

from .._storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
from .._utils import always_get_an_event_loop, logger
from ..base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from .event_segments import build_event_segments
from .event_qa import (
    answer_from_retrieval,
    fetch_event_values,
    stitch_retrieved_event_values,
)
from .event_store import EventValueStore, aligned_embedding_func
from .vl_base import VLQueryParam
from .vl_ops import (
    upsert_segment_embeddings_with_semantic_edges,
    upsert_segment_values_and_graph,
    upsert_temporal_edges,
    vl_query,
)

_EMBEDDING_SOURCE = "event_key"


@dataclass
class EventGraph:
    # Dimensionality of the event key vectors (== aligned extractor dim).
    embedding_dim: int

    # Text->key encoder for query-time seeding (step E), e.g. an
    # AlignedFeatureExtractor.encode_text. Returns (N, embedding_dim) in the same
    # (L2-normalized) space as the indexed keys. If None, text queries raise.
    text_encoder: Callable[[list[str]], Any] | None = None

    working_dir: str = field(
        default_factory=lambda: f"./event_graph_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    semantic_similarity_threshold: float = 0.5
    semantic_similarity_top_k: int | None = None
    query_better_than_threshold: float = 0.2
    embedding_batch_num: int = 32

    # Persist full patch-token grids as values (large); off by default.
    store_patch_tokens: bool = False
    always_create_working_dir: bool = True
    # Wipe `working_dir` before init so re-runs don't merge stale graph/VDB/KV state.
    clear_working_dir: bool = False

    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    def __post_init__(self):
        if self.clear_working_dir and os.path.isdir(self.working_dir):
            logger.info(f"Clearing working directory {self.working_dir}")
            shutil.rmtree(self.working_dir)
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.embedding_func = aligned_embedding_func(self.embedding_dim, self.text_encoder)
        global_config = {
            "working_dir": self.working_dir,
            "embedding_batch_num": self.embedding_batch_num,
            "query_better_than_threshold": self.query_better_than_threshold,
        }

        self.segments_relation_graph = self.graph_storage_cls(
            namespace="event_relation", global_config=global_config
        )
        self.segments_kv = self.key_string_value_json_storage_cls(
            namespace="events", global_config=global_config
        )
        self.event_key_vdb = self.vector_db_storage_cls(
            namespace="event_keys",
            global_config=global_config,
            embedding_func=self.embedding_func,
            meta_fields={"video_id", "segment_idx"},
        )
        self.value_store = EventValueStore(self.working_dir)

    # --- insert -------------------------------------------------------------

    def insert(
        self, video_id, video_path, features, segmentation, frame_indices,
        *, aligned_pooled=None, fps=1.0,
    ):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.ainsert(
                video_id, video_path, features, segmentation, frame_indices,
                aligned_pooled=aligned_pooled, fps=fps,
            )
        )

    async def ainsert(
        self, video_id, video_path, features, segmentation, frame_indices,
        *, aligned_pooled=None, fps=1.0,
    ):
        segments = build_event_segments(
            video_id,
            video_path,
            features,
            segmentation,
            frame_indices,
            aligned_pooled=aligned_pooled,
            fps=fps,
            store_patch_tokens=self.store_patch_tokens,
        )
        if not segments:
            logger.warning(f"No events to insert for {video_id}")
            return

        logger.info(f"[EventGraph] inserting {len(segments)} events for {video_id}")

        # Persist tensor values off to the value store; pull keys out for the VDB.
        key_data: dict[str, dict] = {}
        for seg in segments:
            value = seg.pop("_value")
            key_vec = seg.pop("_key")
            seg["value_path"] = self.value_store.save(seg["node_id"], value)
            key_data[seg["node_id"]] = {
                "content": key_vec,
                "video_id": seg["video_id"],
                "segment_idx": seg["segment_idx"],
            }

        await self._insert_start()
        try:
            # C: KV metadata + graph nodes + in-batch temporal edges
            await upsert_segment_values_and_graph(
                segments, self.segments_kv, self.segments_relation_graph
            )
            # D (temporal): cross-batch links to previously-inserted events
            await upsert_temporal_edges(
                segments, self.segments_kv, self.segments_relation_graph
            )
            # D (semantic): cosine kNN edges over event keys + store key vectors
            await upsert_segment_embeddings_with_semantic_edges(
                self.event_key_vdb,
                key_data,
                self.segments_relation_graph,
                self.semantic_similarity_threshold,
                self.semantic_similarity_top_k,
                embedding_source=_EMBEDDING_SOURCE,
            )
        finally:
            await self._insert_done()

    # --- query (step E) -----------------------------------------------------

    def query(self, text: str, param: VLQueryParam | None = None):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(text, param))

    async def aquery(self, text: str, param: VLQueryParam | None = None):
        """Seed via text->key vector search, optionally expand 1-hop, rank+cap.

        Returns the `vl_query` response dict (selected node ids + event records
        with retrieval scores). Requires `text_encoder` for the seed embedding.
        """
        param = param or VLQueryParam(mode="local")
        return await vl_query(
            text,
            param,
            self.event_key_vdb,
            self.segments_kv,
            self.segments_relation_graph,
        )

    # --- QA (steps F+G) ------------------------------------------------------

    def fetch_values(self, segments: list[dict]) -> list[dict]:
        """Load stored tensor values for retrieved segments (step F)."""
        return fetch_event_values(self.value_store, segments)

    def answer(
        self,
        question: str,
        *,
        param: VLQueryParam | None = None,
        default_video_path: str | None = None,
        max_frames: int = 128,
        max_new_tokens: int = 128,
        model=None,
        processor=None,
        model_path: str | None = None,
    ) -> dict:
        """Retrieve events, stitch decoded frames from values, run VLM QA."""
        return answer_from_retrieval(
            self,
            question,
            param=param,
            default_video_path=default_video_path,
            max_frames=max_frames,
            max_new_tokens=max_new_tokens,
            model=model,
            processor=processor,
            model_path=model_path,
        )

    # --- introspection ------------------------------------------------------

    def summary(self) -> dict:
        """Node/edge counts (by edge_type) for quick inspection."""
        graph = self.segments_relation_graph._graph
        edge_types: dict[str, int] = {}
        for _, _, data in graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1
        return {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "edge_types": edge_types,
        }

    async def _insert_start(self):
        await self.segments_relation_graph.index_start_callback()

    async def _insert_done(self):
        for storage in (
            self.segments_kv,
            self.event_key_vdb,
            self.segments_relation_graph,
        ):
            await storage.index_done_callback()
