# External retrieval crashes on odd frame counts: the temporal-group rounding mismatch

**Status:** fixed 2026-07-28. Affected every `--ext_model` run on Qwen3-VL-8B. Cannot affect
LLaVA-OneVision or InternVL3 — see "Why only Qwen3" below.

## Symptom

Any run combining Qwen3-VL with an external retriever (`--ext_model pecore_l14_336
--ext_merge_method rrf`) dies on roughly half of all videos with a bare, message-less
`AssertionError`:

```
File "model/attention/kv_cache_manager.py", line 1352, in _calc_block_topk
    assert len(current_rankings) == len(ext_rankings)
AssertionError
```

The wrapper reports `Chunk N completed successfully!` regardless, and the child traceback is not
propagated to the array task's stderr, so the array drains looking healthy while producing
partial shards. See also `csv-tmp-is-resume-state` — the shard is left as a `.csv.tmp`.

## Cause

The QA model and the external model are handed the **same** frame tensor
(`video_qa/rekv_offline_vqa.py:107` and `:111`), then round the trailing partial temporal group in
**opposite directions**.

Qwen3-VL **truncates** it (`model/qwen3_vl_rekv.py:359-366`):

```python
remainder = num_frames % self.temporal_factor      # temporal_factor = 2
if remainder:
    video = video[: num_frames - remainder]        # the tail frame is DISCARDED
    num_frames -= remainder
```

so the KV cache holds `floor(T / 2)` blocks. (That truncation is itself deliberate: a chunk whose
frame count is not a multiple of `temporal_factor` gets an inflated pixel budget from
`process_video` and resizes to a different grid — see `qwen3_vl_grid_distortion.md`.)

PECore **appended** it as an extra block (`model/pecore_rekv.py:calc_topk`, pre-fix):

```python
merge_size = max(1, self.chunk_size * self.temporal_patch_size)    # = 2 for Qwen3-VL
remainder_size = int(num_frames % merge_size)
cropped_size = num_frames - remainder_size
chunked = kv_cache_features[:cropped_size].reshape(-1, merge_size, dim).mean(1)
if remainder_size > 0:
    remainder_feat = kv_cache_features[-remainder_size:].mean(dim=0, keepdim=True)
    chunked = torch.cat([chunked, remainder_feat], dim=0)          # ceil()
```

so it ranked `ceil(T / 2)` blocks. For odd `T` the two differ by exactly one, and the RRF fusion
in `_calc_block_topk` asserts they match.

The off-by-one is the visible failure, but the deeper problem is that the appended block **indexes
KV that does not exist**. Even where the lengths happened to agree, fusing a ranking built over
`ceil` blocks into one built over `floor` blocks would misalign the tail.

## Why only Qwen3

`merge_size = chunk_size * temporal_patch_size`, and `video_qa/base.py:599-603` reads
`temporal_patch_size` from the backbone's `vision_config` for Qwen models, while an explicit
`elif "llava" in args.model or "intern" in args.model` branch pins it to 1 ("process frames
individually — no temporal compression"):

| backbone | `temporal_patch_size` | `merge_size` | remainder possible? |
|---|---|---|---|
| Qwen3-VL-8B | **2** (from `vision_config`) | 2 | **yes** |
| LLaVA-OneVision-7B | 1 (explicit) | 1 | no — `T % 1 == 0` always |
| InternVL3-8B | 1 (explicit) | 1 | no |

This is why the LLaVA-OV `swa`/`swa_kk_adjacent_2` + pecore arms ran fine and the bug stayed
hidden until external retrieval was first tried on Qwen3-VL.

## Blast radius

At `--sample_fps 0.5`, **58 of 103 LVBench videos (56%)** sample to an odd frame count and would
trip the assert. CG-Bench has a comparable split.

## Fix

Drop the remainder instead of appending it, in **both** places in `model/pecore_rekv.py`:

- `calc_topk` — the retrieval path that feeds the RRF fusion.
- `get_block_and_adjacent_similarities` — the dump path feeding `block_sims` / `adjacent_sims`.
  Left unfixed, these dumps span one more block than the retrieval indexing, so block *i* in a
  similarity figure is not block *i* in the cache. That silently corrupts the alignment/recall
  analysis rather than crashing.

Truncating is the correct direction, not merely the convenient one: the QA model genuinely never
encoded those frames.

`kv_cache_manager.py:1352` also got a real assertion message naming the likely cause. A bare
`assert` is why this was invisible in the logs.

## Verification

- Block counts agree for T = 100..103 (odd and even) after the fix.
- Four LVBench arrays at two budgets: **zero** assertion failures; `200_256` AKS+pecore and
  `482_536` AKS+pecore both completed 52/52 shards with 103/103 videos.

**Not verified:** the `dropping N trailing frame(s)` log line the fix emits was never observed in
the captured job logs, despite ~56% of videos having odd `T`. Either the logger does not propagate
at that level to captured stdout, or `merge_size` was not 2 in those runs. The arms completed
cleanly either way, so this run set is sound — but the fix has been shown *not violated* rather
than shown *exercised*. Confirm by grepping a worker's stderr for the `merge_size` value before
relying on the fix in a new configuration.

## Related

- `qwen3_vl_grid_distortion.md` — why the tail truncation exists at all.
- Config landmine, not a bug: `topk` must not exceed `n_local // full_block_size`. The guard at
  `kv_cache_manager.py:680` raises `ValueError: topk=64 exceeds n_local_frames=63`. It fires on
  the *highest-resolution* videos at tight budgets (e.g. `n_local=10780` at `--max_tokens 160`),
  so exclusions are systematic, not random.
