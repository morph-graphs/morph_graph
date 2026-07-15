# Relevant Papers

## Continuous Latent Compression for RAG

| Title | Identifier | Venue | Summary |
| --- | --- | --- | --- |
| CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning | [2511.18659](https://arxiv.org/abs/2511.18659) | Preprint | Compresses documents into continuous latent tokens and jointly optimizes retrieval and generation end-to-end in a shared latent space. |
| PISCO: Pretty Simple Compression for Retrieval-Augmented Generation | [2501.16075](https://arxiv.org/abs/2501.16075) | ACL Findings 2025 | Soft-compresses RAG documents into continuous tokens via sequence-level distillation, reaching ~16× compression with little accuracy loss and no large-scale pretraining. |
| Context Embeddings for Efficient Answer Generation in RAG (COCOM) | [2407.09252](https://arxiv.org/abs/2407.09252) | WSDM 2025 | Compresses multiple retrieved contexts into a small set of context embeddings that the same LLM consumes for much faster RAG generation. |
| xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token | [2405.13792](https://arxiv.org/abs/2405.13792) | NeurIPS 2024 | Maps frozen dense retrieval embeddings into the LLM input space as a single continuous token for extreme RAG context compression. |

## Query-Aware Video Token Compression

| Title | Identifier | Venue | Summary |
| --- | --- | --- | --- |
| Small Vision-Language Models are Smart Compressors for Long Video Understanding | [2604.08120](https://arxiv.org/abs/2604.08120) | ECCV 2026 | Uses a small VLM (Tempo) as a query-aware compressor that dynamically allocates visual tokens to intent-critical video segments for long-video understanding. |
| LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding | [2410.17434](https://arxiv.org/abs/2410.17434) | ICML 2025 | Adaptively reduces long-video tokens via frame-redundancy removal, text-guided selection, and spatiotemporal compression under a fixed context budget. |

## Latent Memory / KV-Cache

| Title | Identifier | Venue | Summary |
| --- | --- | --- | --- |
| IndexMem: Learned KV-Cache Eviction with Latent Memory for Long-Context LLM Inference | [2605.25475](https://arxiv.org/abs/2605.25475) | ICML 2026 | Learns KV-cache eviction with a token indexer and compensates discarded entries via a compact online-updated latent memory. |
| Latent Collaboration in Multi-Agent Systems (LatentMAS) | [2511.20639](https://arxiv.org/abs/2511.20639) | ICML 2026 Spotlight | Lets LLM agents collaborate in continuous latent space by generating latent thoughts and transferring them via shared KV-cache working memory, without text mediation. |

## Query-Aware Retrieval Papers

| Title | Identifier | Venue | Summary |
| --- | --- | --- | --- |
| Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees | [2605.18775](https://arxiv.org/abs/2605.18775) / [OpenReview](https://openreview.net/forum?id=n28wnc2QTc) | ICLR 2026 | Diffuses query-weighted flow over a knowledge graph to recover query-relevant subgraphs with statistical retrieval guarantees. |
| Breaking the Static Graph: Context-Aware Traversal for Robust Retrieval-Augmented Generation (CatRAG) | [2602.01965](https://arxiv.org/abs/2602.01965) | Preprint | Query-adaptive KG traversal over HippoRAG-style graphs (dynamic edge weights / symbolic anchoring) to fix the “static graph fallacy” and recover complete multi-hop evidence paths. |
| HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation | [2503.21322](https://arxiv.org/abs/2503.21322) | NeurIPS 2025 | Represents n-ary relational facts as hyperedges for graph-based RAG, improving completeness over binary GraphRAG relations. |
| LightRAG: Simple and Fast Retrieval-Augmented Generation | [2410.05779](https://arxiv.org/abs/2410.05779) | EMNLP 2025 | Builds a dual-level entity/relation graph index and supports local/global/mix retrieval with cheaper incremental updates than GraphRAG. |
| From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2) | [2502.14802](https://arxiv.org/abs/2502.14802) | ICML 2025 | Uses query-seeded Personalized PageRank over a knowledge graph plus passages to improve associative multi-hop retrieval without hurting simple factual RAG. |
| HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models | [2405.14831](https://arxiv.org/abs/2405.14831) | NeurIPS 2024 | Seeds Personalized PageRank from query-linked KG concepts to perform multi-hop GraphRAG-style retrieval inspired by hippocampal indexing. |
