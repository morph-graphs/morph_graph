# The sliding-window RoPE table is never rebuilt: a stale-cache bug in the Qwen3-VL port

**Status:** CONFIRMED, NOT FIXED as of 2026-07-28. Qwen3-VL only. Silent — no crash, no warning.

## Summary

Once the local window starts sliding, every layer reads a cos/sin table built for a *different*
window position. The cache-invalidation key tests the window's **shape** when it needs to test the
window's **position**, and shape is exactly the thing that does not change when a window slides.

## Cause

`model/attention/qwenvl/rope_qwen3.py:248-252`, in `_recompute_cos_sin_tables_for_window`:

```python
_skip = (
    self.curr_block_sizes is not None
    and list(block_sizes) == list(self.curr_block_sizes)
)
if _skip:
    return
```

`block_sizes` is the list of per-frame token counts in the current window. At a **fixed token
budget every frame has the same token count**, so the list is `[131, 131, ...]` at budget 128 and
`[490, 490, ...]` at budget 512 — byte-identical on every call, even after the window has advanced.
The skip therefore fires on every sliding read, and the table stays anchored to an old position.

Aggravated by `model/attention/rekv_attention.py:65-72`: the `deepcopy` of `position_bias` is
**device-gated**, so all layers resident on the master table's GPU share one object. A layer that
has already slid rebuilds the shared table down to its trailing window; layers that have not yet
slid then read that window-relative table while attending the full cache.

## Evidence

Instrumented runs (`REKV_ROPE_TRACE=1`), CG-Bench, budgets 128 and 512:

| observation | budget 128 | budget 512 |
|---|---|---|
| skip-path reads using a stale table | **50,000 / 50,000** | every logged skip |
| rebuilds | 11,359 | — |
| window rows with wrong relative position | ~94% (8594/9114) | ~93% (32354/34804) |
| rows sharing the query block's shift | 5.7% | 7.0% |
| `max_dcos` | ≈2.0 (saturated) | ≈2.0 |
| positional drift | ~400 units | 600–1120 units |

Only ~6–7% of rows share the query's positional shift, so the error does **not** cancel the way a
row-uniform shift would. Drift measured in *frames* is roughly common-mode across budgets (~15
blocks), but drift in *RoPE phase* scales with tokens/frame — so cross-budget comparisons are not
protected by cancellation either.

## Measured impact

A/B on one shard (174 questions, budget 128), stale table vs `REKV_ROPE_FORCE_REBUILD=1`:

- accuracy **52.30% vs 52.30%** — identical
- prediction agreement **93.68%** (11/174 flips), correctness agreement 95.40%
- retrieved-set Jaccard **0.767** (median 0.753, only 3.5% identical)

So retrieval composition genuinely changes while aggregate accuracy does not move. At n=174 a
shift of ≲2–3 points could hide.

## What is and is not affected

**Retrieval and redundancy metrics are structurally insulated.** The KV cache stores **pre-RoPE**
keys (`rekv_attention.py:113-118` builds `h_k_cache` before `position_bias(h_q_, h_k_)` at :128),
the retrieval query is pre-RoPE (`global_q = h_q`, :90), block scoring is a dot product of the two
(`kv_cache_manager.py:1168`), and block self-similarity reads the stored keys
(`kv_cache_manager.py:266`). None of it touches the cos/sin table.

The only leak is indirect: the query hidden state at layer *L* has passed through layers 0..*L*-1,
whose **attention** used the bad table. At layer 0 there is no such path, so layer-0 retrieval is
exactly clean. This reconciles the structural claim with the observed Jaccard of 0.767.

**Affected:** generated answers, and hence QA accuracy, on any Qwen3-VL run whose window slides.
**Not affected:** any other backbone. `_recompute_cos_sin_tables_for_window` exists only in
`rope_qwen3.py`; `rope.py` (`RotaryEmbeddingESM`, used by Qwen2.5-VL / InternVL3 / LLaVA-OV) has no
such function and no such skip. Main paper results are not in question.

**Never fires** when the window does not slide — videos short enough to fit inside `n_local`.

## Why validation missed it

The 97.87% match against stock HF ran `--target_frames 128 --n_local 9100`: 64 blocks against a
69-block window. It never slid once. That number certifies the port's arithmetic, not its
streaming regime.

## Fix (not yet applied)

Key invalidation on window **position** — the window's start block index, or the position ids
themselves — instead of on `block_sizes`.

Cost is near zero: the A/B pair ran 25:06 (stale) vs 25:25 (rebuild on every read), about **1.3%**
slower. The correct behaviour already exists behind `REKV_ROPE_FORCE_REBUILD=1`.

The reason to fix it is correctness, not speed. Any Qwen3-VL accuracy number produced by a sliding
run — the tokens-per-frame ladder, the AKS deltas — carries this defect. The measured effect is nil
at n=174, but that is a statement about power, not about the computation being right.

## Reproducing

```bash
REKV_ROPE_TRACE=1          # log stale-vs-fresh skips and per-event damage
REKV_ROPE_FORCE_REBUILD=1  # disable the skip (the "correct" A/B arm)
```

Both are env-gated; default behaviour is byte-identical without them. Note the `SKIP_SUMMARY`
aggregate line fires on `_rt_n_skip % 5000` **per `position_bias` object**, which never trips at
budget 512 — use the per-event `STALE_SKIP` lines there, or convert the counter to an at-exit dump.
