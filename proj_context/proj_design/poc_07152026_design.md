# POC 2026-07-15 — Design Decisions

Companion to [poc_07152026_overview.md](poc_07152026_overview.md) (functional outline). This records
the concrete design decisions made while building the POC: step A (visual feature
extraction, `video_models/`), step B (offline TSM segmentation, `events/`), steps C+D
(event k/V store + graph linking on nano-graphrag, `graphrag/vl_graphrag/`), and
step E (text-aligned query retrieval).

---

## Production QA pipeline and debug scripts

Run on a GPU node with the `morph_graphs_qwen3_cu126` env, from the
`morph_graph/` directory. Debug commands use the top-level `tests` package,
which adds the repository parent before absolute package imports; no
`PYTHONPATH` export is required.

```bash
module load cuda/12.6.3
module load ffmpeg/7.1
```

### Supported benchmark execution

`morph_graph.video_qa` is the maintained QA boundary. It uses one
`UnifiedVideoGraph` API with two indexing adapters: raw-video EventGraph events
(feature extraction → TSM → aligned retrieval keys → event values) and the
pre-segmented caption/vision records used by VLGraphRAG. Both adapters share the
same graph retrieval contract.

Benchmarks live in `morph_graph/video_qa/benchmarks/`, not under GraphRAG:

- `vsi_morph` — VSI-Morph MCQ data;
- `egolongqa` — EgoLongQA validation JSONL.

Pre-segmented caption/vision records are called **oracle segments**. They are
externally supplied annotations or pseudo-GT events and are indexed through
`morph_graph.video_qa.oracle`; they are distinct from EventGraph's raw-video
TSM event construction.

`VideoLLM` is the model boundary for raw-video QA. A VideoLLM owns a visual
extractor that defines its persisted event-value schema and implements QA from
those retrieved values. EventGraph remains model-agnostic: it retrieves ordered
event values and delegates answer generation to the selected VideoLLM.

The first production runner uses the raw EventGraph adapter and stores a
retrieval trace for every question (rank, node ID, event/frame range, score,
and graph source), separately from answer correctness:

```bash
python -m morph_graph.video_qa.cli \
    --benchmark egolongqa \
    --output-json outputs/egolongqa_eventgraph.json \
    --cache-dir outputs/egolongqa_cache \
    --frame-batch-size 128
```

Use `python -m morph_graph.video_qa.slurm` to generate a resumable, video-grouped
Slurm array. It writes per-chunk `answers_chunk_*.json` files and only submits
when `--submit` is set. Cluster setup is deliberately passed through
`--preamble`/`--sbatch-arg`, rather than embedded in source.

### Debug-only scripts

### Feature extraction (step A)

```bash
python -m tests.debug_extract \
  --video data/wearable-ai/egolongqa/val/00cdb8ca10c069f8.mp4 \
  --tokens-per-frame 256 \
  --fps 1.0
```

### Event segmentation (step B)

```bash
python -m tests.debug_events \
  --video data/wearable-ai/egolongqa/val/00cdb8ca10c069f8.mp4 \
  --tokens-per-frame 256 \
  --fps 1.0 \
  --kernel-width 6 \
  --peak-height 1.0 \
  --plot outputs/debug/seg_00cdb8ca.png
```

### Graph + QA smoke test (steps C–G)

VSI Cat (ordering question):

```bash
python -m tests.debug_event_graph \
    --video data/vsi_morph/vsi_constructed/Cat/0000.mp4 \
    --tokens-per-frame 256 --frame-batch-size 128 --fps 1.0 --sim-threshold 0.9 \
    --aligned-model pe_core_vit_l \
    --query "In what order was the Cat first seen across the Office, the Bedroom/Toy Room, and the Industrial Room?" \
    --qa \
    --working-dir outputs/debug/vsi_cat_0000 \
    --plot-graph outputs/debug/vsi_cat_0000/graph.png
```

EgoLongQA walk (egocentric QA):

```bash
python -m tests.debug_event_graph \
  --video data/wearable-ai/egolongqa/val/00cdb8ca10c069f8.mp4 \
  --tokens-per-frame 256 \
  --frame-batch-size 128 \
  --fps 1.0 \
  --kernel-width 6 \
  --peak-height 1.0 \
  --sim-threshold 0.9 \
  --aligned-model pe_core_vit_l \
  --query "After I left the grassy trail that ran alongside the flooded marsh, what was the first man-made structure I encountered on the paved section of my walk?" \
  --qa \
  --working-dir outputs/debug/eg_00cdb8ca \
  --plot-graph outputs/debug/eg_00cdb8ca/graph.png
```

---

# A: Model-Agnostic Visual Feature Extraction (Qwen3-VL)

## Scope of step A

- Produce, for a video, the visual features that later steps consume:
  - `**pooled**` `(n_units, dim)` — mean-pooled per-unit embedding, input to TSM.
  - `**patch_tokens**` `(n_units, tokens_per_unit, dim)` plus DeepStack streams —
  cached Qwen visual values replayed after retrieval without a second vision pass.
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

## Decoding

- **torchcodec only, never decord.** `VideoDecoder` imported lazily inside
functions so the package stays importable where torchcodec's native lib fails to
load (e.g. login nodes).

## Interface & package layout (`morph_graph/video_models/`)

- `video_io.py` — torchcodec decode + fps sampling (+ frame cap).
- `base.py` — `VisualFeatures` dataclass, `VisualFeatureExtractor` ABC
(`load_video` and `encode_frames` kept separate; `encode_video` composes them),
and a string-keyed extractor registry (`register_extractor` / `get_extractor`)
so any MLLM can plug in.
- `qwen3_vl.py` — `Qwen3VLFeatureExtractor` (first implementation), auto-registered
as `"qwen3_vl_8b"`.

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
(boundaries -> contiguous gap-free spans). Mirrors the `video_models/` pattern so
other boundary methods (adjacency, Otsu, ...) plug in later.
- `tsm.py` — `TSMEventSegmenter` (offline), auto-registered as `"tsm_offline"`;
exposes `novelty_curve()` separately for reuse/debug.
- `viz.py` — `plot_segmentation()`: 2-panel figure (unit-level cosine SSM with
boundary lines, top-down `origin="upper"`; novelty curve with peaks + threshold
and alternating shaded event spans). Agg backend, matplotlib imported lazily.

---

# C+D: Event k/V Store + Linking (on nano-graphrag)

## Decision: one graph pipeline on nano-graphrag (reuse, don't reinvent)

- We reuse the existing VLGraphRAG storage stack + `vl_ops` edge builders rather
than a bespoke in-memory graph. `UnifiedVideoGraph` is the public API and
dispatches to an EventGraph raw-event adapter or VLGraphRAG pre-segmented
adapter; both preserve the same retrieval response and node schema.
- Stores (per `working_dir`): `NetworkXStorage` (event_relation graph),
`JsonKVStorage` (event metadata), `NanoVectorDBStorage` (event keys),
`EventValueStore` (CPU-resident value tensors).

## Node scheme = reuse `seg`_ (event == segment)

- Events use `{video_id}::seg_{idx:04d}` (not `evt`_) so `build_temporal_edges`'
cross-batch reconstruction works verbatim. In this path "segment" *is* an event.

## Keys vs values split

- **Value tensors don't fit JSON KV.** `EventValueStore` saves per-event tensors
(fp16) in CPU memory (required `patch_tokens` + DeepStack streams and
`frame_indices`) for one EventGraph worker; JSON KV holds only event metadata
(`unit_range`, `frame_range`, `time_range_sec`).
- `patch_tokens` and DeepStack persistence are enabled by default
(`store_patch_tokens=True`) because cached-token QA is the supported path.

## Precomputed / aligned embedding func

- Event keys are precomputed vectors, so the VDB uses `aligned_embedding_func`
instead of a model: **passthrough** for indexed key vectors, and **text_encoder**
for text queries (both must be L2-normalized → cosine = dot). No text_encoder =>
text queries raise (keys not in a text-aligned space).

## Temporal edges are undirected

- `NetworkXStorage` is an undirected `nx.Graph`, so the old
`temporal_next`/`temporal_prev` pair collapsed to one edge (last write won —
every cached graph showed `temporal_prev`). Fixed `_temporal_edge` to emit a
single `**temporal_adjacent`** edge; direction is recoverable from `segment_idx`.
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

## Aligned extractors (`video_models/`)

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

# F+G: Value fetch, token stitch, LLM QA

## F — fetch stored values

- `EventValueStore.load` / `load_many`; `fetch_event_values` attaches `values` dict
to each retrieved segment. Values include `patch_tokens`, all Qwen DeepStack
streams, `frame_indices`, and `unit_range`; pooled TSM features are not stored.

## G — stitch cached tokens + answer (no vision re-encode)

- **Decision:** concatenate the retrieved events' cached post-vision
`patch_tokens` and Qwen3-VL DeepStack streams on CPU, then transfer only that
selected token sequence to GPU. Qwen's normal placeholder, M-RoPE, and
DeepStack injection path consumes it; the vision tower is not run a second
time and source frames are not re-decoded.
- `stitch_retrieved_event_tokens` returns the concatenated token streams and
retrieval-aligned stitch log. All indexed videos must use a fixed
`tokens_per_frame` so retrieved events share a token grid.
- `answer_from_retrieval` / `EventGraph.answer`: E → F → cached-token stitch →
`run_inference_on_cached_video_features`. Open-ended prompts use
`build_messages_open_qa`.

---

# Limitations & considerations (current POC)

Cross-cutting gaps, deferred work, and things to watch while the POC is in flight.
Grouped by pipeline step.

## A — Feature extraction

- **Long-video VRAM.** Qwen extraction supports `frame_batch_size` (default 128
in the production runner), which encodes fixed-size frame batches and trims
only padded temporal units. This bounds vision-forward VRAM, but needs
regression checks against unchunked extraction for partial final batches.
- **Variable resolution → variable `tokens_per_unit`.** Pinning `tokens_per_frame`
keeps grids uniform; otherwise `patch_tokens` must be a list.
- **Odd frame counts.** `temporal_patch_size=2` pads to even, yielding one extra unit.
- **Wasteful model load.** Loading the full 8B to use only the vision tower is fine
for v1 (LLM reused later for QA) but not scalable.
- **Dropped:** per-unit timestamps and `frame_ranges` (uniform 2-frame units =>
unit index encodes position).
- **Cached-value volume.** Patch tokens and DeepStack streams are persisted fp16
in CPU memory for the active video worker; move only retrieved blocks to GPU.

## B — Segmentation

- `**find_peaks` distance tuning.** Too small over-segments busy scenes; too large
misses fast cuts. Default ties `peak_distance` to `kernel_width`.
- **Flat novelty curves.** Z-scoring assumes variance; near-constant video =>
correctly no boundaries.
- **Offline z-score.** Global mean/std not available in streaming; online port needs
running/local normalization.

## C+D — Graph

- **Value-store capacity.** `store_patch_tokens=True` is required for no-reencode
QA and increases CPU RAM use for long videos.
- **Untyped / single-slot edges.** Backing store is a plain `nx.Graph`: at most one
edge per unordered pair, `edge_type` as a string attribute. A pair can be
temporal **or** semantic, not both — the adjacent-pair semantic skip is a
workaround, not a real multi-relation graph.
- **Deferred — parallel edge types.** To represent both temporal and semantic on the
same pair: (1) merge onto one edge with `edge_types=[...]`; (2) `nx.MultiGraph`
keyed by `edge_type`; (3) `nx.MultiDiGraph` for direction. Revisit if retrieval
should weight "adjacent **and** recurring" pairs.

## E — Retrieval

- **Vision-pooled keys are not text-retrievable.** Qwen3-VL pooled features are
post-projection (4096-d) but not contrastively text-aligned; mean-pooled text
queries there retrieve poorly and fail silently. Hence the separate CLIP-family
retrieval key.
- **Local 1-hop neighbors are not query-ranked.** `vl_query` seeds via text→key
VDB (`top_k`), expands all graph neighbors in `mode="local"`, then
`rank_and_cap_node_ids` keeps seeds first (sorted by query similarity) and
pushes neighbors without scores to the back. Among neighbors there is no
re-scoring against the query — remaining slots after the seeds are filled in
graph expansion order, not by relevance. With `top_k=3` and `max_segments_for_qa=5`,
the last 2 segments can be arbitrary 1-hop neighbors. **Better:** expand, then
re-score all seeds ∪ neighbors by query similarity (optionally with a
temporal-adjacency bias), then cap.

## F+G — QA

- **Frame-span alignment for keys.** Aligned keys pool per-frame embeddings over each
event's **frame span** (`units × temporal_patch_size`), not unit indices — CLIP and
Qwen3 must run on the **same sampled frames**.
- **Model zoo / deps.** SigLIP2-L not yet in `model_zoo/` (downloads on first use);
PE-Core needs the `perception_models` package.
- **Key pooling is a mean** over frames (tunable: attention / first-last).
- **Validation required:** compare cached-token replay against pixel re-encoding
on a small QA set. Replay preserves the original visual outputs but does not
recompute visual context over a newly stitched clip.

