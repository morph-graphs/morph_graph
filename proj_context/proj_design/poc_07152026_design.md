# POC 2026-07-15 — Design Decisions

Companion to [poc_07152026_overview.md](poc_07152026_overview.md) (functional outline). This records
the concrete design decisions made while building the POC: step A (visual feature
extraction, `features/`), step B (offline TSM segmentation, `events/`), steps C+D
(event k/V store + graph linking on nano-graphrag, `graphrag/vl_graphrag/`), and
step E (text-aligned query retrieval).

---

## Evaluation scripts

Run on a GPU node with the `morph_graphs_qwen3_cu126` env, from the repo root.

### Feature extraction (step A)

```bash
python -m morph_graph.tests.debug_extract \
    --video morph_graph/data/wearable-ai/egolongqa/val/00cdb8ca10c069f8.mp4 \
    --tokens-per-frame 256 --fps 1.0
```

### Event segmentation (step B)

```bash
python -m morph_graph.tests.debug_events \
    --video morph_graph/data/wearable-ai/egolongqa/val/00cdb8ca10c069f8.mp4 \
    --tokens-per-frame 256 --fps 1.0 --kernel-width 6 --peak-height 1.0 \
    --plot morph_graph/tests/out/seg_00cdb8ca10c069f8.png
```

### Graph + QA (steps C–G)

VSI Cat (ordering question):

```bash
python -m morph_graph.tests.debug_event_graph \
    --video morph_graph/data/vsi_morph/vsi_constructed/Cat/0000.mp4 \
    --tokens-per-frame 256 --fps 1.0 --sim-threshold 0.9 \
    --aligned-model pe_core_vit_l \
    --query "In what order was the Cat first seen across the Office, the Bedroom/Toy Room, and the Industrial Room?" \
    --qa \
    --working-dir morph_graph/tests/out/vsi_cat_0000 \
    --plot-graph morph_graph/tests/out/vsi_cat_0000/graph.png
```

EgoLongQA walk (egocentric QA):

```bash
python -m morph_graph.tests.debug_event_graph \
    --video morph_graph/data/wearable-ai/egolongqa/val/00cdb8ca10c069f8.mp4 \
    --tokens-per-frame 256 --fps 1.0 --sim-threshold 0.9 \
    --aligned-model pe_core_vit_l \
    --query "After I left the grassy trail that ran alongside the flooded marsh, what was the first man-made structure I encountered on the paved section of my walk?" \
    --qa \
    --working-dir morph_graph/tests/out/eg_00cdb8ca \
    --plot-graph morph_graph/tests/out/eg_00cdb8ca/graph.png
```

---

# A: Model-Agnostic Visual Feature Extraction (Qwen3-VL)

## Scope of step A

- Produce, for a video, the visual features that later steps consume:
  - `**pooled**` `(n_units, dim)` — mean-pooled per-unit embedding, input to TSM.
  - `**patch_tokens**` `(n_units, tokens_per_unit, dim)` — patch-level "values",
  reused after retrieval.
- Event generation (TSM), graph, retrieval, QA are separate later steps.

## Model & space

- MLLM = **Qwen3-VL-8B-Instruct**; we use **its vision encoder** so features live
in the same space the LLM consumes (needed for later embedding re-injection).
- Env: `morph_graphs_qwen3_cu126` (transformers 5.0.0). Model at
`morph_graph/model_zoo/Qwen3-VL-8B-Instruct`.

## Unit of embedding = one temporal unit (2 frames)

- Qwen3-VL vision `temporal_patch_size = 2` fuses consecutive frame pairs at the
encoder's input `Conv3d` (kernel/stride include the temporal dim). This is part
of the **vision encoder**, not the projector. `spatial_merge_size = 2` is the
projector/merger (spatial 2x2), producing `out_hidden_size = 4096` tokens.
- Decision: **keep the native video path** and encode frame pairs (no per-frame
image-encoding trick). So `video_grid_thw = [t, h, w]` with `t = n_frames/2`
temporal units; our leading dim `n_units = t`.

## Key/event scheme (downstream, for context)

- Frame/unit embedding = mean of that unit's patch tokens.
- TSM runs on `pooled` to produce event boundaries; an event's **key** = mean of
the `pooled` rows inside it. Step A only emits `pooled` + `patch_tokens`.

## Feature stage: pre- vs post-projection (`feature_stage`)

- The vision forward returns both `pooler_output` (**post**-merger, dim
`out_hidden_size`=4096, LLM input space -- our reusable values) and
`last_hidden_state` (**pre**-merger, dim `hidden_size`, `merge**2`=4x more
tokens/unit). `feature_stage="post"` (default) vs `"pre"` swaps which one feeds
`pooled`/`patch_tokens`, so TSM can be A/B'd on either. `"pre"` is analysis-only
(not in LLM space); default stays `"post"`.

## Vision/LLM separation

- Mirror FluxMem's separation: call the vision tower directly
`self.visual(pixel_values_videos, grid_thw=video_grid_thw)` and read
`pooler_output` (merged tokens, dim 4096). We do **not** adopt FluxMem's
hierarchical memory.
- `get_video_features`/`self.visual` return `BaseModelOutputWithDeepstackFeatures`;
`pooler_output` = merged tokens, `deepstack_features` = extra features.

## Dimensions (avoid the G/T letter clash)

- `n_units` = temporal count = `n_frames / 2` (leading dim).
- `tokens_per_unit` = `(h/merge) * (w/merge)` = spatial `h*w` in merged-token space.
- `dim` = 4096. So `patch_tokens` is `(n_units, tokens_per_unit, dim)`.

## Tokens-per-frame <-> pixels

- One merged token covers a `factor x factor` px region, `factor = patch_size * spatial_merge_size = 16 * 2 = 32`. So `tokens_per_unit = per_frame_area / 32**2`,
i.e. per-frame area for TPF tokens = `TPF * 32**2`.
- The Qwen3-VL **video** processor budget is expressed via `size`:
`{"shortest_edge": min, "longest_edge": max}`, and these are **totals across all
frames** (compared against `t_bar * h * w` in `smart_resize`, where
`t_bar = ceil(n_frames/2)*2`). Proof: the rescale sets total -> max_pixels, so
per-frame area -> `max_pixels / n_frames`.
- To pin TPF: `budget = TPF * 32**2 * t_bar`, passed as
`size = {"shortest_edge": budget, "longest_edge": budget}` (equal => pinned).
Implemented as the `tokens_per_frame` knob (default 128); `_size_for(n_frames)`
computes it. Manual fallback: `min_pixels` / `total_pixels` (also totals),
mirroring the embeddings-code setup (`min = 64*32*32`, `total = 20480*32*32`).

## Frame cap (temporary)

- Total vision tokens grow linearly: `total_tokens = TPF * (n_frames / 2)`, and the
encoder runs one forward — long videos OOM.
- For now: hard cap `MAX_SAMPLED_FRAMES = 1024` in
[video_io.py](../../features/video_io.py). FIXME: replace with chunked/streaming
vision encoding (FluxMem precedent: `_visual_forward_videos_chunked`) so long
videos keep full temporal coverage.

## Decoding

- **torchcodec only, never decord.** `VideoDecoder` imported lazily inside
functions so the package stays importable where torchcodec's native lib fails to
load (e.g. login nodes).

## Interface & package layout (`morph_graph/features/`)

- `video_io.py` — torchcodec decode + fps sampling (+ frame cap).
- `base.py` — `VisualFeatures` dataclass, `VisualFeatureExtractor` ABC
(`load_video` and `encode_frames` kept separate; `encode_video` composes them),
and a string-keyed extractor registry (`register_extractor` / `get_extractor`)
so any MLLM can plug in.
- `qwen3_vl.py` — `Qwen3VLFeatureExtractor` (first implementation), auto-registered
as `"qwen3_vl_8b"`.

## Dropped / deferred

- **Dropped**: per-unit timestamps and frame_ranges (uniform 2-frame units => the
unit index encodes position).
- **Deferred**: persisting `patch_tokens` to disk (large: `n_units * tokens_per_unit * 4096` fp16) — designed later alongside retrieval so values can
be lazily loaded post-retrieval.
- **Deferred**: DeepStack features (kept in the model output but dropped by the
extractor) — only needed when re-injecting embeddings into the LLM (steps F/G).

## Things to watch

- Variable frame resolution -> variable `tokens_per_unit`; pinning `tokens_per_frame`
keeps grids uniform. Otherwise `patch_tokens` must be a list.
- Odd frame counts: `temporal_patch_size=2` rounds up to even (processor pads the
last frame), yielding one extra unit.
- Loading the full 8B to use only the vision tower is wasteful but fine for v1
(the LLM is reused later for QA).

---

# B: Event Segmentation (offline TSM)

## Scope of step B

- Consume `pooled` `(n_units, dim)` from step A and emit event boundaries + unit
spans. Values (`patch_tokens`) are untouched here; the key-value store (B->C) is
a later step.
- Output = `EventSegmentation(boundaries, events, novelty, n_units)` where
`events` are half-open `(start_unit, end_unit)` spans covering all units.

## Offline vs online (why offline now)

- The reference `OnlineTSMEdgeDetector` (matching_tsme_online.py) is streaming:
it buffers a window, scores novelty at the trailing center, and calls
`find_peaks` incrementally. **We start offline**: one pass over all units, one
`find_peaks` over the full curve. The novelty math is identical, so the online
swap later only changes buffering/peak bookkeeping — not the segmentation logic.

## Method = Foote checkerboard novelty

- L2-normalize per-unit embeddings, build a local `(2w x 2w)` cosine
self-similarity around each center, and correlate with a Gaussian-tapered
checkerboard kernel (`checkerboard_kernel`, kept identical to the reference).
Peaks in the resulting novelty curve = event boundaries.
- Novelty only defined for centers with a full window (`c in [w, n-w)`); edges
are zero-padded. Short clips (`n < 2w+1`) -> single event, no boundaries.

## Units, not frames

- Everything is in **temporal units** (1 unit = 2 frames, per step A). So
`kernel_width` and `peak_distance` are measured in units, and boundaries index
into `pooled` rows directly.

## Peak thresholding (video-agnostic)

- The reference uses a raw `height=5`, which scales with `kernel_width` (novelty
~ sum over `(2w)^2` kernel). Instead we default `normalize=True`: **z-score the
novelty** and threshold in std units (`peak_height=1.0`). This keeps the knob
meaningful across videos/kernel sizes.
- Defaults: `kernel_width=6` units, `peak_height=1.0` (z), `peak_distance=kernel_width`.
Peak picking via `scipy.signal.find_peaks` (adds a scipy dependency).

## Interface & package layout (`morph_graph/events/`)

- `base.py` — `EventSegmentation` dataclass, `EventSegmenter` ABC, string-keyed
registry (`register_segmenter` / `get_segmenter`), and `events_from_boundaries`
(boundaries -> contiguous gap-free spans). Mirrors the `features/` pattern so
other boundary methods (adjacency, Otsu, ...) plug in later.
- `tsm.py` — `TSMEventSegmenter` (offline), auto-registered as `"tsm_offline"`;
exposes `novelty_curve()` separately for reuse/debug.
- `viz.py` — `plot_segmentation()`: 2-panel figure (unit-level cosine SSM with
boundary lines, top-down `origin="upper"`; novelty curve with peaks + threshold
and alternating shaded event spans). Agg backend, matplotlib imported lazily.

## Things to watch

- `find_peaks` `distance` merges nearby boundaries; too small over-segments busy
scenes, too large misses fast cuts. Tie it to `kernel_width` by default.
- Z-scoring assumes some novelty variance; a near-constant video yields a flat
curve and (correctly) no boundaries.
- Offline z-score uses global mean/std — not available in the streaming setting,
where a running/local normalization will be needed (flagged for the online port).

---

# C+D: Event k/V Store + Linking (on nano-graphrag)

## Decision: build on nano-graphrag (reuse, don't reinvent)

- We reuse the existing VLGraphRAG storage stack + `vl_ops` edge builders rather
than a bespoke in-memory graph, for drop-in compatibility with the existing
query/traversal path. New orchestrator `EventGraph` mirrors `VLGraphRAG`.
- Stores (per `working_dir`): `NetworkXStorage` (event_relation graph),
`JsonKVStorage` (event metadata), `NanoVectorDBStorage` (event keys),
`EventValueStore` (value tensors on disk).

## Node scheme = reuse `seg`_ (event == segment)

- Events use `{video_id}::seg_{idx:04d}` (not `evt_`) so `build_temporal_edges`'
cross-batch reconstruction works verbatim. In this path "segment" *is* an event.

## Keys vs values split

- **Value tensors don't fit JSON KV.** `EventValueStore` saves per-event tensors
(fp16) to disk (`pooled`, optional `patch_tokens`, `frame_indices`); JSON KV holds
a `value_path` pointer + metadata (`unit_range`, `frame_range`, `time_range_sec`).
- `patch_tokens` persistence is opt-in (`store_patch_tokens`, default off — large).

## Precomputed / aligned embedding func

- Event keys are precomputed vectors, so the VDB uses `aligned_embedding_func`
instead of a model: **passthrough** for indexed key vectors, and **text_encoder**
for text queries (both must be L2-normalized → cosine = dot). No text_encoder =>
text queries raise (keys not in a text-aligned space).

## Temporal edges are undirected

- `NetworkXStorage` is an undirected `nx.Graph`, so the old
`temporal_next`/`temporal_prev` pair collapsed to one edge (last write won —
every cached graph showed `temporal_prev`). Fixed `_temporal_edge` to emit a
single `**temporal_adjacent*`* edge; direction is recoverable from `segment_idx`.
Applies to `EventGraph` and `VLGraphRAG` alike.

## Semantic edges

- Cosine kNN over event **retrieval keys** (top-k above `similarity_threshold`,
default 0.5), bidirectional, keep max weight per direction — reused
`build_semantic_similarity_edges` (in-batch matrix + cross-batch VDB).
- **Skip temporally-adjacent pairs.** A semantic edge on an already-adjacent pair
`(seg_i, seg_{i+1})` would overwrite the `temporal_adjacent` edge (single
undirected slot; semantic is written after temporal), punching a hole in the
temporal spine. So `build_semantic_similarity_edges` skips pairs with same video +
`|Δsegment_idx| == 1`. Semantic edges then carry only the useful signal:
long-range recurrence between non-adjacent events. Applies to `VLGraphRAG` too.

## Future: typed / parallel edges (not done)

- Current model assumes **untyped edges**: at most one edge per unordered node
pair, with `edge_type` as a *string attribute* (backing store is a plain
`nx.Graph`). So a pair can be temporal **or** semantic, not both — hence the
adjacency skip above is a workaround, not a real multi-relation graph.
- If we later want a pair to be *both* temporal and semantic (each with its own
weight), options in increasing fidelity: (1) merge onto one edge —
`edge_types=[...]` + per-type weights (GraphML-safe as a delimited string; cheap,
but viz/traversal/summary must read the list); (2) `nx.MultiGraph` keyed by
`edge_type` for true parallel edges (touches storage, serialization,
`get_nodes_edges_batch`, dedup, viz); (3) `nx.MultiDiGraph` to also restore
direction (temporal_next vs temporal_prev). Deferred — revisit if retrieval wants
to specially weight "adjacent **and** recurring" pairs.

# E: Query Retrieval (text-aligned keys)

## Why not the vision-pooled keys / LLM input space

- Qwen3-VL vision-pooled features are **post-projection** (LLM input dim 4096) but
**not contrastively text-aligned**: same dimensionality ≠ retrievable by text
cosine. VLM cross-modal alignment lives in deep attention layers, not the input
embedding space, and a mean-pooled text query there is degenerate. Rejected
"Idea 1" (retrieve in LLM input space) for this reason — it would run but retrieve
poorly, and the failure is silent.

## Two keys per event

- **structure key** (Qwen3-VL vision-pooled): TSM segmentation (B) + values (F/G).
- **retrieval key** (CLIP-family, aligned): text-query seeding (E) + semantic
edges (D). Mean of per-frame aligned embeddings over the event's frame span.

## Aligned extractors (`features/`)

- `AlignedFeatureExtractor` base: `supports_text_query=True`,
`temporal_patch_size=1` (per-frame), abstract `encode_text`. `pooled` is the
L2-normalized joint-space image embedding; `patch_tokens = pooled.unsqueeze(1)`
(no meaningful grid in joint space).
- `siglip2_vit_l` — SigLIP2-ViT-L via transformers (`get_image_features` /
`get_text_features`; processor lowercases + pads text to 64).
- `pe_core_vit_l` — PE-Core-ViT-L via Meta `perception_models` (`encode_image` /
`encode_text`), lazily imported + guarded (non-standard dep).
- Chosen over the 8B Qwen3-VL-Embedding for scalability (~400M, cheap per-frame).

## Retrieval path

- `EventGraph(text_encoder=aligned.encode_text)` seeds `vl_query` with the query's
text embedding, expands 1-hop (`mode="local"`), ranks + caps. Returns event
records (node_id, scores, unit/frame ranges) for value fetch (F) + stitching (G).

# F+G: Value fetch, stitch, LLM QA (v1 pixel path)

## F — fetch stored values

- `EventValueStore.load` / `load_many`; `fetch_event_values` attaches `values` dict
  to each retrieved segment. Values include `pooled` (Qwen3 unit embeddings),
  `frame_indices` (source indices at indexing fps), `unit_range`.

## G — stitch + answer (pixels, not embedding inject)

- **Decision (v1):** re-decode **pixels** at stitch time using stored
  `frame_indices` (not `frame_range` subsampling — avoids gaps between sampled
  frames). Global frame budget (`max_frames`, default 128) split proportionally
  across retrieved events; uniform subsample within each event's indices.
- `stitch_retrieved_event_values` → `(video_tensor, video_metadata, stitch_log)`.
- `answer_from_retrieval` / `EventGraph.answer`: E → F → stitch →
  `run_inference_on_video_tensor` (Qwen3-VL). Open-ended prompts via
  `build_messages_open_qa`.
- **Deferred:** concat `pooled`/`patch_tokens` and inject into vision forward
  (DeepStack / skip vision tower).

## Things to watch

- Aligned keys pool per-frame embeddings by the **frame span** of each event
(units×`temporal_patch_size`), not unit indices — the CLIP model and Qwen3 must
run on the **same sampled frames**.
- SigLIP2-L not yet in `model_zoo/` (downloads on first use); PE-Core needs the
`perception_models` package.
- Key pooling is mean over frames (tunable: attention / first-last).

