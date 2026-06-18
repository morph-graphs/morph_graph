import asyncio
import os
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB

from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def embed_data(self, data: dict[str, dict]) -> tuple[list[dict], np.ndarray]:
        logger.info(f"Embedding {len(data)} records for {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return [], np.array([])
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, item in enumerate(list_data):
            item["__vector__"] = embeddings[i]
        return list_data, embeddings

    async def upsert_prepared(self, list_data: list[dict]):
        if not list_data:
            return []
        logger.info(f"Inserting {len(list_data)} vectors to {self.namespace}")
        return self._client.upsert(datas=list_data)

    async def upsert(self, data: dict[str, dict]):
        list_data, _ = await self.embed_data(data)
        return await self.upsert_prepared(list_data)

    async def query_with_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        better_than_threshold: float | None = None,
    ) -> list[dict]:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        threshold = (
            better_than_threshold
            if better_than_threshold is not None
            else self.cosine_better_than_threshold
        )
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=threshold,
        )
        return [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        return await self.query_with_embedding(
            embedding[0],
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )

    async def index_done_callback(self):
        self._client.save()
