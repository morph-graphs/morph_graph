# Long videos die at `vm.max_map_count`: the KV-offload mapping leak

**Status:** fixed 2026-07-26. Affected the LVBench `tpf=448_512` arm on Qwen3-VL-8B; the same
code path exists on every other backbone, so it will recur there.

## Symptom

Long videos only. Chunks containing a video needing >= 115 GiB of KV cache failed; the largest
that succeeded needed 114.3 GiB. Perfect separation, no flakiness. On LVBench that is the 7
chunks holding the 116-140 minute videos.

Two error messages, depending on which allocation happened to be next in line:

```
# site A -- model/attention/kv_cache_manager.py, MemoryUnit.__init__
RuntimeError: CUDA error: out of memory

# site B -- model/attention/kv_cache_manager.py, append(), sliding_window_pos_ids torch.cat
RuntimeError: [enforce fail at alloc_cpu.cpp:118] DefaultCPUAllocator: can't allocate memory:
  you tried to allocate 20521056 bytes. Error code 12 (Cannot allocate memory)
```

Both are victims. The process had exhausted its VMA quota, so the *next* allocation of any size
failed — which is why the reported site moved whenever the code changed, and why a 20 MB request
failed on a node with 1.8 TB free.

## Cause

`vm.max_map_count = 65530` — the kernel's per-process limit on distinct memory mappings.

`MemoryUnit.__init__` offloads each KV block to the host, and it has **two independent
`cudaHostAlloc` doors**, each costing ~0.5 mappings per block (one ~2 MB `rw-s /dev/zero`
mapping serves two ~0.94 MB blocks):

| door | call | cost |
|---|---|---|
| 1 | `.to("cpu", non_blocking=True)` — async D2H allocates a **pinned** destination | 0.500 maps/op |
| 2 | `.pin_memory()` on the result (`'pin_memory': True` in every backbone's config) | 0.500 maps/op |
| — | `.to("cpu", non_blocking=False)` — sync D2H, ordinary heap | **0.000** |

A 120-minute LVBench video at 480 tokens/frame is 1801 blocks x 2 tensors x 36 layers x 0.5
= **~64.8k mappings** against the 65530 ceiling. Measured failures landed at maps=65483..65531.

The arithmetic is why short videos are fine and long ones are not: mapping count scales linearly
with video length x tokens-per-frame.

## Fix

Close **both** doors. Either one alone accomplishes nothing, because the other keeps pinning.

1. `kv_cache_manager.py`, `MemoryUnit.__init__`: `non_blocking=False` on the CPU copy.
2. At launch: pass `--no_pin_memory` **and/or** export `REKV_DISABLE_PIN=1`.

```bash
sbatch --array=9,10,12,15,19,35,46 --export=ALL,ENABLE_LOGGING=1,REKV_DISABLE_PIN=1 \
  slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_tpf \
  --n_local 36000 --prefill_method swa --retrieval_method rekv \
  --encode_chunk_size 32 --min_tokens 448 --max_tokens 512 \
  --no_pin_memory --resume --enable_similarity
```

Verified on chunk 34 (`f8DKD78BrQA`, 120 min):

| | before | after |
|---|---|---|
| peak mappings | 65531 (at ceiling) | **2905**, flat for the whole video |
| `/dev/zero` mappings | ~57000 | collapsed |
| result | never finished | `52_34.csv`, 28 answers, 0 errors |

**Correctness:** pinning and copy synchronicity affect host-transfer mechanics only, never
values. Output is bit-identical, so chunks run with and without the fix mix freely inside one
arm's results directory. That is deliberate — see the `save_dir` comment in `run_eval.py`.

**Cost:** the offload path is synchronous, so encoding is slower. Acceptable; correctness and
completion first.

## Traps

- **Door 2 is conditional on a launch-time flag.** Forgetting `--no_pin_memory` /
  `REKV_DISABLE_PIN=1` silently reopens it and the bug returns with no warning. The code fix
  alone is not sufficient.
- **`--no_pin_memory` is gated to qwen3 models** in `video_qa/base.py` (raises otherwise). The
  same `MemoryUnit` code is shared by `internvl3`, `llava_onevision`, `qwen2_vl`, `qwen2_5_vl`
  and `video_llava`, all of which set `'pin_memory': True`. That guard will block this fix on
  those backbones the first time someone runs a long-video arm.
- **Slurm `State=COMPLETED` is meaningless here.** `video_qa/run_eval.py` runs the work in a
  `multiprocessing.Process` and never checks `p.exitcode`, so dead chunks report success and
  write to `success_*.log`. The only trustworthy signal is the final CSV count.
- **`--mem` is irrelevant to this bug.** The cgroup's `memory.max` is literally `max`; every
  `--mem` value was a no-op. Do not "fix" mapping exhaustion by asking for more RAM.

## Ruled out by measurement — do not re-try

GPU memory (118 GiB free on a 139.8 GiB H200 at failure; torch reserved 20.7 GiB) · cgroup limit
(unlimited) · node capacity (1.8 TB available) · job contention (`%2` array throttle) ·
`pin_memory=False` alone · `REKV_DISABLE_PIN=1` alone · `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
· `MALLOC_MMAP_THRESHOLD_` / `MALLOC_TRIM_THRESHOLD_` (peak maps unchanged, failed *earlier*) ·
`PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True` (converts `/dev/zero` maps to anon
maps at the same ~0.5/op rate — `cudaHostRegister` splits the malloc'd VMAs — same wall).

## If async+pinned speed is needed later

The `640` and `768` tokens-per-frame rungs in `analysis/tpf_budgets.py` produce *more* blocks per
video and hit this ceiling harder. No env var avoids it. The structural fix is to **pool the CPU
KV storage**: one large growable per-layer pinned buffer with each block a view into it, so block
count stops driving mapping count. That preserves async pinned transfers and removes the ceiling
outright.

Independently worth fixing: `sliding_window_pos_ids` in `append()` accumulates the *full* video's
position ids per layer via repeated `torch.cat` and is never trimmed (~740 MB across 36 layers,
O(n^2) allocation traffic) even though only the sliding window is ever read.

## How to diagnose a recurrence

Peak maps approaching 65530 is the signature. On a live process, check `wc -l /proc/<pid>/maps`
against `vm.max_map_count`, plus RSS and cgroup usage. Once maps > 58000, dump backing files
(`awk '{print ($6==""?"[anon]":$6)}' /proc/<pid>/maps | sort | uniq -c | sort -rn | head`)
to see whether `/dev/zero` pinned allocations are piling up.

**Methodological note.** This took five failed fixes, all from reasoning about the symptom
("CUDA error: out of memory") instead of measuring. What actually worked: the `/proc/<pid>/maps`
breakdown, then per-arm isolation probes. One in-process probe that seemed to prove
`non_blocking=False` cost zero mappings was **confounded** — it ran the async arm first, warming
the host pool, so the sync arm reused it. Allocation-path comparisons must run each arm in a
**separate, fresh process**.
