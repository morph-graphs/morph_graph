# Bites — Quick Highlights

Terse, scannable notes — one line per item. Fuller detail lives in `proj_description.md`, `proj_relevant_papers.md`, and `proj_streaming_video_papers.md`.

## Video Memory Mechanisms (how prior work remembers a video)

| Family | What's stored | Access | Examples |
| --- | --- | --- | --- |
| A. Retrieval over stored KV | per-frame/clip KV, unbounded (offloaded to RAM/disk) | similarity retrieval of blocks | ReKV (ICLR'25) |
| B. Fixed-size KV compression | bounded KV: salient tokens + frame prototypes | in-context after compression | StreamMem, Video-XL-2, HERMES (ACL'26) |
| C. Consolidated / recurrent memory banks | merged memory tokens / recurrent state (short→long term) | always-in-context state | MovieChat (CVPR'24), Flash-VStream, VideoLLaMB (ICCV'25) |
| D. Symbolic text + object memory | captions + object DB (language space) | LLM tool-use queries | VideoAgent (ECCV'24), Ego-R1 H-RAG |
| E. Structured graph memory (**ours**) | events + relational edges | query-modulated traversal | Morph Graphs; cf. Vgent, VideoDetective, EgoSelf |

**Common gap (A–D):** units stored independently, accessed by query→unit similarity; no inter-event relational structure to traverse.

## Events — definition basis

- **Event Segmentation Theory** (Radvansky & Zacks 2017): humans parse activity into events; boundaries occur where situational features change — movement, location, characters, objects, causes, goals.
- **Intrinsic** context: goals, actions, causes. **Extrinsic** context: location, spatial surroundings, entities.
- Caveat: EST boundaries are *prediction-error* driven; segmenting by embedding-similarity is a different operationalization.

## Query-adaptivity — two levers (keep distinct)

- **Node-level**: modulate *which facets* of an event are salient (not its meaning). ~ RQ2/RQ3 (general manifold → task-specific via learnable transform).
- **Edge-level**: modulate edge weights → query-adaptive traversal. ~ Proposed idea / RQ1 (does cross-memory relation help?).

## Most related papers (shortlist)

| Paper | Similar | Different |
| --- | --- | --- |
| **SelectStream** ([2606.16353](https://arxiv.org/abs/2606.16353)) | Streaming; fixed-capacity memory **graph**; **query-conditioned** graph retrieval; budgeted evidence allocation | Latent-token memory inside a frozen VLM; graph budgets *what to keep*, not explicit inter-event relational semantics |
| **VideoDetective** ([2603.22285](https://arxiv.org/abs/2603.22285)) | Segment graph (temporal + visual affinity); query-guided **graph propagation** to localize clue segments; sparse observe → stitch → answer | Offline inference plug-in; graph is **static** — query updates a relevance field (diffusion), not edge/traversal structure |
| **GraphThinker** ([2602.17555](https://arxiv.org/abs/2602.17555)) | **Event scene graph** with intra-/inter-event relations as first-class structure | Offline RL “graph thinking”; relations used for generation grounding, not a retrieval substrate with query-adaptive traversal |
| **G2F-RAG** ([2604.04372](https://arxiv.org/abs/2604.04372)) | Offline video KG → retrieve minimal subgraph → render evidence → LLM | Problem-agnostic offline KG; subgraph rendered as one visual frame, not query-modulated relational traversal |
| **Vgent** ([2510.14032](https://arxiv.org/abs/2510.14032)) | Offline clip **graph** + retrieval before generation | Semantic clip graph + verification denoising; no inter-event relational edges or query-adaptive traversal |
| **QAFD** ([2605.18775](https://arxiv.org/abs/2605.18775)) | **Query-weighted flow diffusion** over a graph to recover query-relevant subgraphs; retrieval guarantees | Text KG RAG (ICLR'26), not video; diffuses over a fixed KG topology rather than event-graph construction from video |
| **CatRAG** ([2602.01965](https://arxiv.org/abs/2602.01965)) | **Query-adaptive graph traversal** (dynamic edge weights / anchoring) | Text KG RAG, not video; symbolic HippoRAG-style traversal vs. our visual event graph |
| **EventMemAgent** ([2602.15329](https://arxiv.org/abs/2602.15329)) | Streaming **event-centric** memory (STM/LTM) + tool retrieval | Event hierarchy, not an explicit relational graph traversed by the query |

**Baselines (same problem, different family):** HERMES / FluxMem / StreamMem — streaming compression or KV retrieval, no explicit event graph.
