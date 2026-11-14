# MemStream Codebase Overview

This documents the code that produced the results in the paper *"Going Down Memory
Lane: Scaling Tokens for Video Stream Understanding with Dynamic KV-Cache Memory"*
(MemStream, NeurIPS 2026 submission). It maps the paper's algorithmic description to
concrete code, so it can be read alongside `proj_context/memstream_neurips.pdf`.

**Source of truth for paper numbers.** The code that generated the submitted results
lives in the frozen snapshot directory:

```
.temp_code_code_snapshot_working_qwen2_5_vl_7b/
```

This is **not** the same as the current repo HEAD (`model/`, `video_qa/`, etc. at the
repo root) — HEAD has since diverged significantly (Qwen3-VL support added, KV-cache
manager refactored). See [Snapshot vs. HEAD divergence](#snapshot-vs-head-divergence)
before re-running anything for rebuttal experiments. All file:line references below
are against the **snapshot**, unless explicitly marked `[HEAD]`.

## 1. Repository layout (snapshot)

| Path | Purpose |
|---|---|
| `model/*_rekv.py` | One file per backbone (Qwen2.5-VL, Qwen2-VL, InternVL-via-LongVA, LLaVA-OneVision, CLIP, PECore, Video-LLaVA, Flash-VStream). Each subclasses `Abstract_ReKV` (video-LLM backbones) or `Abstract_Ext_ReKV` (external VL retrievers used for the retrieval ensemble). |
| `model/patch.py` | Monkey-patches HF `LlamaModel`/Qwen `Model.forward` so attention is routed through the custom KV-cache machinery. Threads ensemble kwargs (`ext_merge_method`, `ext_weights`, `ext_rrf_k`, cross-layer rank passing) from the CLI down into the attention forward pass. |
| `model/attention/kv_cache_manager.py` | **The core engine.** `ContextManager` class: sliding-window encoding, offloading, AKS-style compression dispatch, internal retrieval, RRF fusion. Everything else wires into this. (2323 lines in snapshot.) |
| `model/attention/compression_methods.py` | Free functions implementing every compression/selection variant, including AKS and all the paper's ablation baselines (A.1–A.3, B.1–B.3). (953 lines in snapshot.) |
| `model/attention/qwenvl/` | Qwen-VL-specific attention forward (`rekv_attention.py`) and M-RoPE (`rope.py`) — separate from the generic LLaMA-style versions in `model/attention/rekv_attention.py` / `rope.py` used by LongVA/LLaVA-OneVision. |
| `video_qa/` | Evaluation harness. `base.py` (shared driver + CLI), `rekv_offline_vqa.py` (CG-Bench/LVBench/VideoMME), `rekv_stream_vqa.py` (RVS-Ego/RVS-Movie streaming), `run_eval.py` (sweep launcher), `eval/*.py` (scoring). |
| `analysis/` | Scripts that produced the paper's Section 4 figures (recall-vs-token-budget, entropy, self-similarity, layer-wise recall) and `mllm_analysis/` for raw attention/KK-similarity dumps. |
| `model_zoo/`, `data/` | Symlinks to `/fs/nexus-scratch/vatsalag/mem_open_vocab/ReKV/{model_zoo,data}`, **shared across all snapshot directories** (not frozen per-snapshot) — checkpoints and per-benchmark data folders. Note `data/` also has `timescope`, `vsi_super_recall`, `morph` folders not used in the paper — post-submission benchmark expansion already in progress. |

## 2. KV-cache / sliding-window core (paper §3, Preliminaries)

`ContextManager.__init__` (`kv_cache_manager.py:296-411`) takes `n_local` (sliding
window size in tokens = paper's `ω·N`), `n_init` (sink tokens), `block_size`
(tokens/frame = paper's `N`), `topk` (retrieval budget), `chunk_size`.

- **Encoding** (`_append`, `kv_cache_manager.py:1691-1807`): for each new frame,
  `_apply_compression` runs on the sliding window (line 1746), positional embeddings
  are applied to the **compressed** keys only afterward
  (`self.position_embedding(...)`, lines 1756-1768), then dense sliding-window
  attention (`Attn.append(..., sliding_window=...)`, lines 1779-1786) plus attention
  to sink/init tokens (lines 1799-1805) is computed. This is exactly the paper's
  `O^i_t = Attn(Q^i_t, [K^i_{W_t}; K^i_t], [V^i_{W_t}; V^i_t])`.
- **Eviction → representative vector**: when a frame block leaves the window,
  `_get_representative_keys` (`kv_cache_manager.py:798-822`) average-pools it — for
  `retrieval_method="rekv"`, `global_k.mean(dim=-2)` over the block's `N` tokens per
  head, matching the paper's `k^i_{t-ω-1} = (1/N)·Σ K^i_{t-ω-1}[n]`. These vectors
  live in `self.block_k[u]` (a `VectorTensor`, lines 154-289), kept resident on GPU,
  while the raw K/V for that frame is offloaded to CPU via `MemoryUnit.offload()`
  (lines 142-148).
- **Question-answering retrieval**: `get_retrieved_kv` (lines 706-795) calls
  `_calc_block_topk` for per-layer top-k frame indices, then loads from either the
  CPU-offloaded `global_blocks` or the GPU `global_remainder`, streamed via a
  dedicated CUDA stream (`GLOBAL_STREAM`). Implements the paper's
  `O^i = Attn(Q^i, [K^i_R; K^i], [V^i_R; V^i])`.

**Known landmine:** `model/attention/qwenvl/rekv_attention.py:93` has a bare
`import ipdb; ipdb.set_trace()` in the branch where `retrieved_block_indices is None`.
Not hit by any paper config (all retrieve on the fly via `query`), but would hang a
non-interactive SLURM job if ever triggered. Same issue at
`video_qa/rekv_offline_vqa.py:45` (triggers on an empty MCQA prediction string).
Worth stripping before any rebuttal re-runs.

## 3. Adaptive Key Selection — AKS (paper §5.1)

AKS is selected via the prefill-method string `"swa_kk_adjacent_<dilation_rate>"`,
dispatched in `_apply_compression` (`kv_cache_manager.py:1546-1628`) to
`apply_kk_guided_compression(..., method="adjacent", dilation_rate=r)` in
`compression_methods.py:288-620`.

The branch used for **all paper numbers** is `compression_methods.py:556-613`:

1. Reshape `sparse_k` to per-frame patch grid `(b, nh, m, h*w, d)` (line 393).
2. `sparse_k_norm = F.normalize(sparse_k, dim=-1)`;
   `sparse_kk_adjacent_sim = (sparse_k_norm[:,:,1:] * sparse_k_norm[:,:,:-1]).sum(-1)`
   (lines 397-398) — cosine similarity between each frame `t` and its immediate
   predecessor `t-1`, patch-by-patch, computed on **raw `local_k`**, i.e. *before*
   `self.position_embedding(...)` is applied back in `_append` (lines 1762-1768).
   This confirms the paper's claim that "this selection process occurs before
   applying positional embeddings" is literally true in code.
3. `sampled_indices = sparse_kk_adjacent_sim.topk(k, dim=-1, largest=False).indices`
   (line 560) — keeps the **least similar** (most distinctive) patches, matching the
   paper's "select the top-k least similar... patch features" exactly.
4. Frame `t=0` (no predecessor) falls back to dilated spatial sampling (lines 400-405)
   instead of KK-selection — a fallback not spelled out in §5.1 but consistent with
   it (the paper only defines redundancy for pairs `(K_t, K_{t-1})`).
5. `k` is a fixed count per `dilation_rate`, so **every frame keeps the same number
   of patches** — matches the paper's stated limitation that AKS "preserves a fixed
   ratio of patches" (§7, Limitations).

The `method == "full"` branch (compare each frame to a global aggregate instead of
just the adjacent frame) is `raise NotImplementedError` at line 345 — dead code,
confirming only `"adjacent"` was ever used.

**Adaptive-ratio variant already exists in code but is unreported.** Lines 388-554
contain a fully-built variable-budget variant
(`kwargs.get('adaptive', False)`, parsed via `_parse_adaptive_config`,
`kv_cache_manager.py:1182-1243`) that allocates per-frame token budgets across 3
tiers (`h*w/4`, `h*w/9`, `h*w/16`) based on content dissimilarity ranking or temporal
position. This is precisely the "future work" item the paper lists ("future work
could make this ratio adaptive"). It is plumbed through the prefill-method string
parser but not evaluated in any table. **Useful for rebuttal**: if a reviewer asks
"why not the obvious adaptive extension," the honest answer is "prototyped, not yet
mature enough to report" rather than "hadn't considered it."

### Static/ablation baselines (paper Table 4) — all implemented in `compression_methods.py`

| Paper label | Function | Dispatch prefix |
|---|---|---|
| A.1 average pooling | `apply_pooled_compression` (line 231) | `swa_pooled_` / `swa_max_pooled_` |
| A.2 dilated sampling | `apply_patch_sample_compression(method='dilated')` (line 622) | `swa_dilated_` |
| A.3 uniform frame sampling | `apply_frame_sample_compression` (line 760) | `swa_uniform_sample_` |
| B.1 token-merging (ToMe-inspired) | `apply_tome_with_frame_sampling` / `apply_patch_sample_compression(method='tome')` (line 857 / 622) | `swa_tome_patch_` |
| B.2 k-means clustering | `apply_clustered_compression` (line 41) | `swa_clustered_` |
| B.3 temporal redundancy | reuses `apply_frame_sample_compression(method='event')` | `swa_event_sample_` |

Note B.3 is **not** a bespoke function — it shares plumbing with uniform/random frame
sampling. If a reviewer probes the exact formula, double-check `method='event'`
reproduces "retains frames with greatest change, computed as average cosine
similarity between adjacent frames" precisely as described.

## 4. Retrieval ensemble / RRF (paper §5.2)

- **Internal retrieval**: `_calc_block_topk` (`kv_cache_manager.py:826-1124`)
  computes `self.similarity` per layer via cosine similarity between the mean-pooled
  query (`global_h_q.mean(dim=2)`, line 874) and representative frame vectors (lines
  899-902) — matches paper's `S^i_internal`.
- **External retrieval**: `model/pecore_rekv.py:PECore_ReKV.calc_topk`
  (lines 138-178) and `model/clip_rekv.py:CLIP_ReKV.calc_topk` (lines 152-192)
  independently encode the question and chunk-averaged frame features, compute
  cosine similarity, and produce a full sorted ranking via
  `torch.argsort(..., descending=True)` — this is `S^i_external` / `R_external`,
  computed once (layer-agnostic, unlike internal retrieval).
- **RRF fusion**: `ContextManager.reciprocal_rank_fusion`
  (`kv_cache_manager.py:605-640`) implements exactly
  `RRFScore(t) = Σ_r weight_r / (k + rank_r(t))`, defaulting `k=60` (matches
  paper §5.2). Wired in at `_calc_block_topk:1098-1104`:
  ```python
  if ext_kwargs.get('ext_merge_method') == 'rrf':
      fused = self.reciprocal_rank_fusion([current_rankings, ext_rankings],
                                           k=rrf_k, weights=ext_kwargs['ext_weights'])
  ```
  Default `ext_weights=[0.5, 0.5]` (`video_qa/base.py:476`) — main-paper numbers use
  **equal-weight** fusion, matching Table 6/7 captions.
- Parameter threading: CLI arg (`base.py:479`, `run_eval.py:862`, default `60`) →
  `ext_model_kwargs` dict (`base.py:551-552`) → per-question `ext_model_outs` dict
  (`rekv_offline_vqa.py:170`) → `model.ext_kwargs` set in `patch.py:127-136` → read
  inside the patched `model_forward` → attention forward's `_rekv_ext_kwargs` →
  `ContextManager._calc_block_topk`.
- The Table 7 "L2-Concat" early-fusion baseline is
  `get_cosine_similarity_with_external` (`kv_cache_manager.py:258-289`), selected via
  `ext_merge_method == 'l2_concat'`.

**Larger method space than the paper describes.** The code also implements
`topk_watershed_rekv`, `topk_watershed_mmr_rekv` (watershed + MMR diversity-aware
selection), `dual_stream` even/odd/interleave fusion with its own
`reciprocal_rank_stream_fusion` (lines 642-704), `uniform_X_Y_METHOD` per-layer-range
fallback sampling, and cross-layer rank passing (`ext_prev_layer_full_rankings`,
threaded through `patch.py:141-196`). None of these appear in the paper. They read as
abandoned/exploratory directions predating the final internal+external RRF design —
not evidence of cherry-picking, but be aware the code surface is larger than what's
described in §5, so "which flags reproduce Table 2" is not obvious from the code
alone without the exact sweep/SLURM configs.

## 5. Qwen2.5-VL integration specifics (paper Appendix B.2)

`model/qwen2_5_vl_rekv.py:Qwen2_5_VL_ReKV` subclasses both
`Qwen2_5_VLForConditionalGeneration` and `Abstract_ReKV`.

- **Dynamic-resolution tokenization** (Appendix B.2 item 1): `_calculate_n_frame_tokens`
  (lines 124-139) and `get_tokens_per_frame` (lines 183-191) derive the per-frame
  token count from `video_grid_thw` rather than a fixed constant.
  `initialize_cache_for_video` (lines 192-208) calls
  `ContextManager.initialize_cache_for_video` (`kv_cache_manager.py:560-582`) to
  resize `global_buffer`/`n_local` once the first chunk's actual token count is
  observed.
- **M-RoPE** (item 2): `get_rope_index` (lines 209-247) computes `(3, batch, seq)`
  M-RoPE position ids from `video_grid_thw` and `second_per_grid_ts`, feeding
  `RotaryEmbeddingQwenVL` (`model/attention/qwenvl/rope.py`).
- **Temporal-patch-size 2** (item 3): handled upstream by HF's
  `Qwen2_5_VLProcessor`'s own video processor, which merges pairs of frames before
  the ViT — ReKV code doesn't re-implement this, it consumes whatever
  `video_grid_thw` reports, so `N_kv = N/2` falls out naturally.
- **Frame-wise token budget (200–256 tokens)**: this is **not a hardcoded constant
  anywhere in the code**. It's an emergent property of `min_pixels`/`max_pixels`
  processor settings (passed through `load_model`, ~line 798+) bounding the dynamic
  tokenizer's output range; `_calculate_n_frame_tokens` reports whatever that
  yields, and `ContextManager` adapts to the realized value. If a reviewer asks "show
  me the line that enforces 256 tokens/frame" — there isn't one; it's controlled at
  the launch-config level, which is a legitimate but indirect answer worth having
  ready.

## 6. Evaluation harness

- **Offline** (`video_qa/rekv_offline_vqa.py:ReKVOfflineVQA` — CG-Bench, LVBench,
  VideoMME): `analyze_a_video` (lines 53-254): clear cache → `encode_init_prompt` →
  `encode_video` (**whole video encoded once up front**, matching Figure 6a's "we
  process the actual video stream a single time") → per-question `calc_topk` on the
  external model (line 162) → build `ext_model_outs` dict with
  `ext_merge_method`/`ext_weights`/`ext_rrf_k` (lines 163-171) →
  `question_answering(..., **ext_kwargs)`. MCQA vs. open-ended dispatch by presence
  of `'choices'` key (line 214).
  - **Hardcoded benchmark patch** (lines 219-221): if a VideoMME question's choices
    contain both `"Mendel and"` and `"Watson and"` substrings, the choices/answer are
    silently overwritten with a different hardcoded option set. Reads as a fix for a
    known malformed/duplicate entry in the VideoMME release, not benchmark gaming —
    but have a one-line explanation ready if asked "did you modify any benchmark
    data."
- **Streaming/online** (`video_qa/rekv_stream_vqa.py:ReKVStreamVQA` — RVS-Ego,
  RVS-Movie): `analyze_a_video` (lines 72-189) encodes video **incrementally** as
  questions arrive (line 124-134) — true streaming, unlike the offline path. Also
  does per-question CUDA-event latency timing (`_time_cuda_ms`, lines 96-108)
  feeding Table 3's "Running Speed" column, and tracks
  `video_encoding_time_ms_total` for the "Video Enc." FPS figure.
- **Scoring** (`video_qa/eval/`): `eval_open_ended.py:GPTService` (lines 37-70) calls
  `gpt-3.5-turbo-0125` for RVS-Ego/RVS-Movie's LLM-as-judge Acc/Score, matching the
  paper. `eval_multiple_choice.py` / `extract_characters_regex`
  (`base.py:311-319`) do regex letter extraction for CG-Bench/LVBench/VideoMME
  MCQA. `eval_egoschema.py` / `eval_mlvu_by_task.py` are benchmark-specific scorers
  feeding the Table 19/20 category breakdowns.

## 7. Analysis scripts → paper figures

| Figures/Tables | Script / mechanism |
|---|---|
| Fig. 2, 4a, 9, 10 (recall/similarity vs. token budget, per-layer query-frame curves) | `analysis/run_similarity_analysis.py` (driver) + `analysis/similarity_plotter.py`. Data captured by monkey-patching `ContextManager` at runtime via `SimilarityLogger`/`patch_rekv_for_logging` (imported in both `rekv_offline_vqa.py` and `rekv_stream_vqa.py`), which intercepts and dumps `self.similarity` per layer per question. |
| Fig. 3 (attention entropy vs. token budget) | Computed **live inside `_append`** via `compute_attention_entropy` (`kv_cache_manager.py:1814-1841`), gated behind `store_attn_scores in ["prefill","all"]` (`STORE_ATTN_SCORES=True` env var, line 319-320). Not a separate script — instrumented directly into the hot path. |
| Fig. 4b, 11, 12 (self-similarity matrices) | `VectorTensor.get_blockwise_cosine_similarity` (`kv_cache_manager.py:204-211`), active when `store_full=True` (`STORAGE_ON=True` env, line 314-317), consumed by `analysis/compare_pairwise_sims.py` / `similarity_plotter.py`. |
| Fig. 5, Table 18/19 (CG-Bench recall @ clue frames) | `analysis/compute_alignment_scores.py` — computes `union_alignment_score`, `average_alignment_score`, `rrf_alignment_score` against ground-truth temporal windows (`analysis/temporal_utils.get_temporal_windows`) from `SimilarityLogger` JSON dumps. A genuinely separate post-hoc script. |
| Deep-dive entropy/redundancy diagnostics (§4 discussion) | `analysis/mllm_analysis/run_qwen_analysis.py`, `run_qwen3_analysis.py`, `compute_kk_sim_single_layer.py`, `compute_kk_sim_from_saved.py` — standalone scripts that load a model, forward a handful of frames, and dump raw attention/KK-similarity tensors for offline plotting. |

## 8. Snapshot vs. HEAD divergence

**Do not assume current repo HEAD reproduces the paper's numbers.** Diff summary:

- `model/attention/kv_cache_manager.py`: snapshot 2323 lines vs. HEAD 2538 lines
  (~50%+ churn). HEAD added a large block of Qwen3-VL "adaptive block size" support
  (`_get_adaptive_timestamp_bounds`, `_get_adaptive_full_block_start`,
  `_load_retrieved_blocks_from_cpu`, packed timestamp/boundary-token loading for
  variable per-frame block sizes) — none of this existed at submission time.
- `model/attention/compression_methods.py`: snapshot 953 → HEAD 760 lines (net
  shrink). `apply_kk_guided_compression`'s signature changed (`dense_len` kwarg
  replaces implicit sizing, new `top_p` option); the snapshot's
  `append_frames`/hybrid-uniform-sampling logic appears removed/simplified in HEAD.
- `model/qwen2_5_vl_rekv.py`: 856 → 898 lines (266 diff lines).
- `model/patch.py`: 536 → 428 lines (shrink, 411 diff lines).
- HEAD has an entirely new `model/qwen3_vl_rekv.py` that doesn't exist in the
  snapshot — consistent with the recent commit "more updates for qwen3-vl."

**What's stable:** `reciprocal_rank_fusion` is byte-for-byte unchanged between
snapshot and HEAD. AKS's core adjacent-frame cosine-similarity-and-topk logic is
conceptually intact in HEAD too, just with refactored parameter plumbing.

**Practical implication:** for any rebuttal re-runs (new ablations, new baselines,
sanity checks reviewers might request), use the snapshot directory
(`.temp_code_code_snapshot_working_qwen2_5_vl_7b/`) or verify the specific function
being invoked hasn't changed — don't assume HEAD is a drop-in equivalent.

## 9. Quick reference: how a paper-table run is configured

The full method (AKS + retrieval ensemble) is selected by combining:
- Prefill/encoding method string: `swa_kk_adjacent_<dilation_rate>` (AKS).
- Retrieval method: `rekv` (internal) or `ensemble` with `ext_merge_method=rrf`,
  `ext_rrf_k=60`, `ext_weights=[0.5, 0.5]`, and an external model
  (`clip` or `pecore`) supplying `ext_model_outs`.

These are threaded via CLI flags in `video_qa/run_eval.py` → `video_qa/base.py` →
per-video/per-question kwargs in `rekv_offline_vqa.py` / `rekv_stream_vqa.py` →
`model/patch.py` → `ContextManager` in `kv_cache_manager.py`. See `slurm/*.sh` for
the exact sweep invocations used to launch jobs on the cluster.
