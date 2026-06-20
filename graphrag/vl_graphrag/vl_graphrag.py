import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Type, cast, Optional

from .._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)

from .._utils import (
    EmbeddingFunc,
    limit_async_func_call,
    always_get_an_event_loop,
    logger,
)

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
)

from .vl_base import VLQueryParam
from ._vlm_embed import bind_embed_mode, resolve_embedding_func
from .vl_ops import insert_text_nodes, insert_vision_nodes, vl_query


@dataclass
class VLGraphRAG:

    # vision-language embedding: pass embedding_key (registry) and/or embedding_func
    embedding_key: Optional[str] = "qwen3_vl_embedding_8b"
    embedding_func: Optional[EmbeddingFunc] = None

    # key source: caption or video
    key_source: str = "caption"

    working_dir: str = field(
        default_factory=lambda: f"./vl_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2
    semantic_similarity_threshold: float = 0.5
    semantic_similarity_top_k: int | None = None

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    always_create_working_dir: bool = True

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"VLGraphRAG init with param:\n\n  {_print_config}\n")

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        registry_embedding_func = resolve_embedding_func(self.embedding_key, None)
        text_base = self.embedding_func or registry_embedding_func
        limit_embedding = limit_async_func_call(self.embedding_func_max_async)

        def _limit(func: EmbeddingFunc) -> EmbeddingFunc:
            limited = limit_embedding(func)
            limited.embedding_dim = func.embedding_dim
            limited.max_token_size = func.max_token_size
            return cast(EmbeddingFunc, limited)

        text_embedding_func = bind_embed_mode(_limit(text_base), "text")
        video_embedding_func = bind_embed_mode(
            _limit(registry_embedding_func), "video"
        )
        self.embedding_func = cast(EmbeddingFunc, text_embedding_func)

        # initialize graph for inter-segment relations (i.e., semantic similarity and temporal proximity)
        self.segments_relation_graph = self.graph_storage_cls(
            namespace="segment_relation",
            global_config=asdict(self),
        )

        # Shared segment records. Values should stay JSON-serializable: store
        # video paths, frame ranges, captions, and metadata rather than tensors.
        self.segments_kv = self.key_string_value_json_storage_cls(
            namespace="segments",
            global_config=asdict(self),
        )

        # Caption-key Vector Database: embeds text_key strings.
        self.caption_segments_vdb = self.vector_db_storage_cls(
            namespace="caption_segments",
            global_config=asdict(self),
            embedding_func=text_embedding_func,
            meta_fields={"video_id", "segment_idx", "scene_label"},
        )

        # Video-key Vector Database: embeds vision_key dicts; queries stay text.
        self.video_clip_segments_vdb = self.vector_db_storage_cls(
            namespace="video_clip_segments",
            global_config=asdict(self),
            embedding_func=video_embedding_func,
            meta_fields={"video_id", "segment_idx"},
        )

    def insert_text_nodes(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_text_nodes(string_or_strings))

    def insert_vision_nodes(self, video_clip_or_clips):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_vision_nodes(video_clip_or_clips))

    def query(self, query: str, param: VLQueryParam = VLQueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: VLQueryParam = VLQueryParam()):
        if param.mode == "global":
            raise NotImplementedError("Global VLGraphRAG retrieval is not implemented yet")
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")

        if param.key_source == "caption":
            vdb = self.caption_segments_vdb
        elif param.key_source == "video":
            vdb = self.video_clip_segments_vdb
        else:
            raise ValueError(f"Unknown VLGraphRAG key_source: {param.key_source}")

        response = await vl_query(
            query,
            param,
            vdb,
            self.segments_kv,
            self.segments_relation_graph,
        )
        await self._query_done()
        return response

    async def ainsert_text_nodes(self, string_or_strings):
        await self._insert_start()
        try:
            await insert_text_nodes(
                string_or_strings,
                self.segments_kv,
                self.segments_relation_graph,
                self.caption_segments_vdb,
                self.semantic_similarity_threshold,
                self.semantic_similarity_top_k,
            )
        finally:
            await self._insert_done()

    async def ainsert_vision_nodes(self, video_clip_or_clips):
        await self._insert_start()
        try:
            await insert_vision_nodes(
                video_clip_or_clips,
                self.segments_kv,
                self.segments_relation_graph,
                self.video_clip_segments_vdb,
                self.semantic_similarity_threshold,
                self.semantic_similarity_top_k,
            )
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.segments_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.segments_kv,
            self.caption_segments_vdb,
            self.video_clip_segments_vdb,
            self.segments_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        pass
