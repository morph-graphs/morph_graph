# GraphRAG — Core Functionality

What exists today in `morph_graph/graphrag/` and how it connects to eval (`morph_graph/inference/`, `morph_graph/eval/`).

## Overview

The package has **two layers**:

| Layer | Entry point | Purpose |
| --- | --- | --- |
| **VLGraphRAG** (active) | `graphrag/vl_graphrag/vl_graphrag.py` | Video segment memory: index captions or decoded clip frames, retrieve segments for QA |
| **GraphRAG** (legacy fork) | `graphrag/graphrag.py` | Text nano-graphrag: chunk → entity extract → community reports → local/global query |

**Morph Graphs work uses VLGraphRAG.** The text `GraphRAG` class is a vendored [nano-graphrag](https://github.com/gusye1234/nano-graphrag) fork (entity/community pipeline) and is lazy-imported so eval does not pull in heavy optional deps (`hnswlib`, etc.).

---

## VLGraphRAG — mental model

**Unit of memory = a video segment** (one scene/event span with `frame_range`, optional `scene_label`, caption).

Each segment becomes:
- a **node** in a relation graph (`segments_relation_graph`)
- a **KV record** with full metadata (`segments_kv`)
- a **vector** in either the caption VDB or the video-clip VDB

```
Video
  └─ segment nodes  (node_id = "{video_id}::seg_{idx:04d}")
        ├─ caption_segments_vdb     ← embed text_key (caption + scene/frame prefix)
        ├─ video_clip_segments_vdb  ← embed decoded frames via Qwen3-VL-Embedding
        └─ segments_relation_graph
              ├─ temporal_adjacent  (consecutive segments; undirected)
              └─ semantic_sim       (embedding similarity; undirected)
```

---

## Storage (per `working_dir`)

| Store | Class | Namespace | Contents |
| --- | --- | --- | --- |
| Segment metadata | `JsonKVStorage` | `segments` | `video_id`, `segment_idx`, `frame_range`, `caption`, `scene_label`, `video_path`, … |
| Caption vectors | `NanoVectorDBStorage` | `caption_segments` | `content` = text key string; meta: `video_id`, `segment_idx`, `scene_label` |
| Video vectors | `NanoVectorDBStorage` | `video_clip_segments` | `content` = `{video: tensor, video_metadata}`; meta: `video_id`, `segment_idx` |
| Relation graph | `NetworkXStorage` | `segment_relation` | Nodes = segments; edges = temporal + semantic |

Indexes persist under a per-video cache dir (see `inference/framework/rag.py`).

---

## Segment inputs

Normalized in `vl_graphrag/vl_segments.py`.

### Text / caption path (`insert_text_nodes`)

Accepts:
- A **bullet-caption blob** (format from `per_video_segment_captions_*.json`):

  ```
  - [Kitchen, frames 120-180] Person opens the fridge.
  ```

- Or a **single segment dict** with `text_key` / `caption` + optional `frame_range`, `scene_label`.

Produces `text_key` like `[Kitchen, frames 120-180] Person opens the fridge.` for embedding.

### Vision path (`insert_vision_nodes`)

Accepts segment dicts with `video_path` + `frame_range` (or explicit `vision_key`).

At index time:
1. Decodes frames in each segment's `frame_range` (subsampled via `sample_frame_indices`)
2. Builds a tensor + `video_metadata` (fps, frame count)
3. Embeds via Qwen3-VL-Embedding in **video** mode

---

## Indexing pipeline

Both paths share `vl_ops.py`:

1. **`normalize_*_segments`** → list of segment dicts with `node_id`
2. **`upsert_segment_values_and_graph`** → write KV + graph nodes; add **in-batch** temporal edges between consecutive `segment_idx` within the same video
3. **`upsert_temporal_edges`** → connect first segment of this batch to the **previous** segment already in KV (cross-batch continuity)
4. **`upsert_segment_embeddings_with_semantic_edges`**:
   - Embed all new segment keys
   - Query existing VDB for neighbors above `semantic_similarity_threshold` (default 0.5, top-k default 32)
   - Also pairwise within the incoming batch
   - Upsert **semantic_sim** edges (bidirectional; keep max weight per direction)
   - Write vectors to VDB

Configurable on `VLGraphRAG`: `semantic_similarity_threshold`, `semantic_similarity_top_k`, `embedding_batch_num`, `embedding_func_max_async`.

---

## Embeddings

`vl_graphrag/_vlm_embed.py` — registry of Qwen3-VL-Embedding checkpoints:

| Key | Default path | Dim |
| --- | --- | --- |
| `qwen3_vl_embedding_2b` | `$MORPH_GRAPH_MODEL_ROOT/Qwen3-VL-Embedding-2B` | 2048 |
| `qwen3_vl_embedding_8b` | `$MORPH_GRAPH_MODEL_ROOT/Qwen3-VL-Embedding-8B` | 4096 |

`MORPH_GRAPH_MODEL_ROOT` defaults to `morph_graph/models/`.

Two bound embedding funcs per `VLGraphRAG` instance:
- **text** mode → caption VDB (queries are text strings)
- **video** mode → video VDB (queries are still text; index entries are video tensors)

---

## Query / retrieval

`VLQueryParam` (`vl_base.py`):

| Field | Default | Meaning |
| --- | --- | --- |
| `mode` | `"naive"` | `naive` = VDB top-k only; `local` = expand 1-hop graph neighbors |
| `key_source` | `"caption"` | Which VDB to query: `caption` or `video` |
| `top_k` | `3` | Initial vector search seeds |
| `max_segments_for_qa` | `5` | Cap after expansion + score ranking |
| `only_need_context` | `false` | If true, return caption context string only |

### `vl_query` flow (`vl_ops.py`)

1. **Vector search** → `seed_node_ids` + scores
2. If `mode == "local"`: **`expand_node_ids`** — 1-hop neighbors on `segments_relation_graph` (temporal + semantic edges)
3. **`rank_and_cap_node_ids`** — dedupe, sort by retrieval score, cap at `max_segments_for_qa`
4. Load segment records from KV
5. **`format_caption_context`** — bullet list sorted by `(video_id, segment_idx)` for downstream VLM prompt

Returns a dict: `seed_node_ids`, `selected_node_ids`, `segments`, `scores_by_id`, `caption_context`.

### Modes not implemented

- **`global`** — raises `NotImplementedError` (no Leiden community traversal yet for VL)
- **Query-adaptive edge weighting** — edges are static at index time; query does not modulate graph traversal weights (planned Morph Graphs idea)

---

## Post-retrieval: QA modes

Eval runner (`inference/framework/runner.py`) after retrieval:

| `qa_mode` | What the VLM sees |
| --- | --- |
| `caption` | Retrieved segment captions as text context → Qwen3-VL MCQ prompt |
| `video` | **`stitch_retrieved_segments_video`** — decode frames from selected segments, concatenate into one tensor (budget `max_frames_for_video_unit`, default 128) → Qwen3-VL video QA |

`key_source` (what you index/search) and `qa_mode` (what you feed the answer model) are **independent** — e.g. retrieve on caption index, answer on stitched video.

---

## QA and oracle integration

`video_qa/` is the supported benchmark and Slurm execution layer. EventGraph
builds events from raw video; `video_qa/oracle.py` indexes externally supplied
or pseudo-GT caption/vision segments for controlled retrieval experiments.

```bash
python -m morph_graph.video_qa.cli \
  --benchmark vsi_morph \
  --output-json outputs/vsi.json \
  --cache-dir outputs/vsi_cache
```

---

## Utilities

| Module | Role |
| --- | --- |
| `vl_ops.stitch_retrieved_segments_video` | Decode + concat frames for video QA |
| `graphrag/vl_graphrag/event_viz.py` | Visualize live or persisted relation graphs |
| `tests/debug_oracle_retrieval.py` | Oracle/pre-segmented retrieval debugging |

---

## File map (VL path)

```
graphrag/
├── vl_graphrag/
│   ├── vl_graphrag.py      # VLGraphRAG dataclass: init stores, insert_*, query
│   ├── vl_ops.py           # index/query ops, graph edges, video stitch
│   ├── vl_segments.py      # segment normalization + caption parsing
│   ├── vl_base.py          # VLQueryParam
│   └── _vlm_embed.py       # Qwen3-VL-Embedding registry + embed funcs
├── _storage/               # JsonKV, NanoVectorDB, NetworkX (+ optional HNSW/Neo4j)
├── graphrag.py             # legacy text GraphRAG (not used in VL eval)
└── _op.py, entity_extraction/, …  # legacy text pipeline
```

---

## Current limitations (honest scope)

- **Graph at query time = 1-hop expansion only** (`local` mode); no multi-hop traversal, no query-conditioned edge weights
- **Semantic edges** are symmetric similarity at index time, not relational/event semantics
- **Global mode** unimplemented for VL
- **Small eval surface**: primary benchmark adapter is `vsi_morph`; public benches (CG-Bench, SuperMemory-VQA, …) not wired yet
- **Segment definition** is upstream (scene annotations or caption bullets) — GraphRAG does not yet *construct* event graphs from raw video; it indexes pre-segmented units and links them temporally + by embedding similarity

---

## Relation to Morph Graphs (planned)

Today's VLGraphRAG is the **retrieval substrate**: segments ≈ proto event nodes, temporal + semantic edges ≈ proto graph structure. The research gap (see `morph_graph/proj_context/proj_notes/proj_description.md`) is **query-adaptive traversal** over **inter-event relations** — edge weights modulated by the query, not fixed similarity at index time. The current `local` mode is a minimal step (1-hop neighbor expansion after vector seeding).
