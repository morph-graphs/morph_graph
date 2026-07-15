# POC v1 — Event-Graph Video Memory (Functional Outline)

Pure functionality outline for the POC. Describes *what* each component must do, not
implementation. Flags what exists in `graphrag/vl_graphrag/` today vs. what is new.

---

## Pipeline (end to end)

```
frames → [vision encoder] → frame embeddings
                                  │
                                  ▼  TSM event segmentation (Qwen3 pooled)
                       event boundaries → k/V dict
                       key   = event retrieval embedding (aligned CLIP, pooled/event)
                       value = frame embeddings (+ frame indices) for that event
                                  │
                                  ▼  link events
                       graph edges: temporal adjacency + semantic similarity (on keys)
                                  │
                                  ▼  query
                       query → graph traversal → selected event nodes
                                  │
                                  ▼  retrieve values
                       fetch frame embeddings for selected nodes
                                  │
                                  ▼  stitch → LLM
                       order + assemble → QA answer
```

---

## A. Frame Embedding (vision encoder)
- Encode a stream/sequence of frames into per-frame embeddings.
- Input: video path or frame stream + sampling policy (fps / stride).
- Output: ordered `[frame_idx → embedding]` with timestamps.
- **Status:** encoder exists (Qwen3-VL-Embedding registry, `_vlm_embed.py`), but only
  used to embed whole clips/segments — no per-frame embedding export. **New: needed.**

## B. Event Segmentation (TSM / VidParse)
- Consume the per-frame embedding stream, emit event boundaries.
- Produce per-event: `frame_range`, member frame indices, aggregated key embedding.
- Boundary signal: temporal similarity drop / shift metric (TSM).
- Output: list of event records (the k/V entries).
- **Status:** does not exist. Segments today come from upstream annotations/captions.
  **New: core of the POC.**

## C. k/V Store (event → frames)
- **Two keys per event** (see "key alignment" below):
  - *structure key* — mean Qwen3-VL unit `pooled` (used for TSM only, step B).
  - *retrieval key* — mean of aligned per-frame embeddings (SigLIP2/PE-Core) over
    the event's frame span, L2-normalized. This is the indexed/searchable key.
- Value: member unit `pooled` (+ optional `patch_tokens`) + frame indices.
- Persist keys (VDB) separately from values (tensor store) and metadata (JSON KV).
- **Status:** **DONE (v1).** `EventGraph` + `EventValueStore` (tensors on disk) +
  `event_keys` NanoVectorDB; metadata in `segments_kv`.

## D. Event Linking (graph edges)
- Temporal edges: consecutive events — **single undirected `temporal_adjacent`**
  edge (NetworkXStorage is undirected; forward/back collapse — ordering recovered
  from `segment_idx`).
- Semantic edges: retrieval-key cosine similarity, top-k above threshold; bidirectional,
  weighted, keep max weight per direction.
- **Status:** **DONE (v1).** Reuses `vl_ops.py` (`upsert_temporal_edges`,
  `upsert_segment_embeddings_with_semantic_edges`) on event keys.

## E. Query Traversal (retrieval)
- Embed query with the aligned model's **text encoder** (same space as retrieval
  keys), seed via vector search, expand 1-hop, rank + cap.
- **Status:** **DONE (v1)** via `EventGraph.query` → `vl_query` (`local` = seed +
  1-hop + rank/cap). Multi-hop / query-conditioned traversal still **new**.

## Key alignment (why two keys)
- Qwen3-VL vision-pooled features are **not** contrastively text-aligned, so a text
  query can't cosine-match them (same dim ≠ aligned; VLM alignment is in deep
  attention, not the input-embedding space). Rejected using LLM input-space
  retrieval for this reason.
- Instead, a small CLIP-family encoder (**SigLIP2-ViT-L** / **PE-Core-ViT-L**,
  in `features/`) provides text-aligned per-frame embeddings; pooled per event by
  the Qwen3/TSM boundaries → retrieval key. Cheap (~400M) vs. embedding whole
  events with the 8B embedding model, and query = its text encoder.

## F. Value Retrieval
- For each selected event node, fetch its stored frame embeddings (+ indices).
- **Status:** `EventValueStore.load` / `fetch_event_values` + `EventGraph.fetch_values`.

## G. Stitch + LLM QA
- Assemble retrieved frame embeddings into a single ordered context.
- Feed to LLM with the question; return answer.
- **Status (v1):** `stitch_retrieved_event_values` loads stored `frame_indices`,
  re-decodes pixels, stitches → `EventGraph.answer` / `answer_from_retrieval` runs
  Qwen3-VL. Embedding injection path still **new**.

---

## H. Streaming Wrapper (HERMES) — optional for v1
- Wrap A–D for online ingestion: chunks in, events indexed incrementally.
- Not required for a batch POC; can be added after the offline path works.

---

## Things to be aware of (open decisions / risks)

- **Stitching & positional information.** When frame embeddings from
  non-contiguous events are concatenated, temporal position can be (a) **dropped**
  (bag of frames), (b) **preserved via original timestamps/indices**, or (c)
  **re-encoded** with fresh positions after stitching. Choice affects whether the LLM
  can reason about ordering and duration across events. Decide and make it explicit.

- **Embeddings vs. pixels into the LLM.** The plan feeds *frame embeddings* to the
  LLM. Qwen3-VL expects **pixels/visual tokens**, not arbitrary encoder embeddings.
  Either (a) keep frame **indices** as values and re-decode pixels at stitch time
  (works with current models), or (b) build an adapter that injects embeddings
  directly (more research, alignment risk). Pick one for v1.

- **Key pooling method.** How the event key is pooled from frame embeddings
  (mean / attention / first-last) changes both semantic-edge quality and retrieval.
  Simple mean-pool for v1; note as a tunable.

- **TSM boundary sensitivity.** Threshold controls event granularity: too fine →
  many tiny events (graph blowup); too coarse → events span topic changes (poor
  retrieval). Needs a tunable + sanity check on event count per video.

- **Storage size.** Storing raw frame-embedding tensors as values is far larger than
  today's JSON metadata. Consider on-disk tensor store, dtype (fp16), and per-event
  frame subsampling budget.

- **Value granularity for stitching.** Return all member frames vs. a subsampled
  budget per event. Interacts with the LLM context/frame budget
  (`max_frames_for_video_unit`).

- **Positional/temporal edges after retrieval.** Selected events may be temporally
  distant; whether the graph's temporal edges are surfaced to the LLM (e.g. ordering
  hint) or only used for retrieval is a separate choice from stitching order.

- **Dedup / overlap.** Overlapping or near-duplicate events (semantic edges) may
  retrieve redundant frames; dedup at value-retrieval or stitch stage.

---

## Build vs. reuse summary

| Stage | Reuse | New | Status |
| --- | --- | --- | --- |
| A frame embed | encoder checkpoints | per-frame embedding export | done |
| B TSM events | — | segmentation module | done |
| C k/V store | KV + graph scaffolding | embedding-valued store + aligned keys | done (v1) |
| D linking | `vl_ops` temporal + semantic ops | re-point onto event keys | done (v1) |
| E traversal | seed + 1-hop `local` (`vl_query`) | aligned text-key seeding | done (v1) |
| F value fetch | KV lookup | embedding value fetch | partial (value store exists) |
| G stitch + QA | pixel stitch path | embedding stitch / QA path | new |
| H streaming | — | HERMES wrapper (optional) | new |
