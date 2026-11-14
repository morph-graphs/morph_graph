# Qwen3-VL-8B on LVBench — runbook

Every command run in this investigation, why it was run, and what it produced.
Branch `dev/qwen3-vl-integration`, worktree `/fs/cfar-projects/mem_open_vocab/ReKV_qwen3_wt`.

---

## 0. Why

A ReKV LVBench run scored ~44.5% against a reported ~50.6% for Qwen3-VL-8B
(VideoMem, [arXiv:2512.04540](https://arxiv.org/pdf/2512.04540), Table 1). The question was
whether our Qwen3-VL integration is broken or the protocol differs. It is not broken.

---

## 1. Environment

```bash
source /etc/profile.d/modules.sh
module load cuda/12.6.3 ffmpeg/7.1
source /fs/cfar-projects/mem_open_vocab/miniconda3/etc/profile.d/conda.sh
conda activate memstream_qwen3_cu126     # torch 2.6.0+cu126, transformers 5.0.0
cd /fs/cfar-projects/mem_open_vocab/ReKV_qwen3_wt
```

`memstream_qwen3_cu126` is mandatory. The older `memstream` env cannot import
`Qwen3VLForConditionalGeneration`; because model registration in `video_qa/base.py` is wrapped in
`try/except`, that failure surfaces only as `KeyError: 'qwen3_vl_8b'`. (Now fixed — see §6.)

---

## 2. Slurm scripts written for this

| script | partition | gres | used for |
|---|---|---|---|
| `run_single_gpu_array_a5000_vulcan_qwen3.sh` | vulcan-scavenger | a5000×2 | first ReKV attempt (OOM'd) |
| `run_single_gpu_array_a5000_scav_qwen3_4gpu.sh` | **scavenger** | **a5000×4** | ReKV full-length |
| `run_qwen3_baseline_lvbench.sh` | vulcan-scavenger | a5000×2 | first stock baselines |
| `run_qwen3_baseline_lvbench_scav.sh` | **scavenger** | **a5000×1** | stock baselines (fast) |

All use `--array=1-52` + `NUM_ARRAY_JOBS=52`. **52, not 100**: LVBench has 103 videos and
`video_qa/base.py:get_chunk` raises `IndexError` on chunks whose slice is empty.

`scavenger` has ~271 nodes vs vulcan-scavenger's ~24 — moving there took the baseline from ~8
concurrent tasks to ~49.

Snapshots freeze `model/ video_qa/ analysis/` and symlink back `data model_zoo results output`.
**Re-snapshot after every code edit**, and keep `--save_dir` under `results/` or output lands
inside `.temp_code_*` and looks like the job produced nothing.

```bash
./slurm/setup_code_snapshot.sh <name>
```

---

## 3. The runs

### 3a. Stock-HF baseline (no ReKV) — `analysis/eval_qwen3_baseline.py`

Plain `Qwen3VLForConditionalGeneration.generate`. Mirrors the ReKV path exactly (frame sampling,
resolution budget, prompt, answer parsing) so the only variable is our KV machinery. Evaluates two
prompts per question in one pass: `ours` (the repo's forced `"Best option: ("` prefix) and
`standard` (lmms-eval `"The best answer is:"`).

```bash
# 128 frames @ 120 tok/frame  -> 45.00%
sbatch slurm/run_qwen3_baseline_lvbench.sh qwen3_baseline

# 128 frames @ 252 tok/frame  -> 46.35%
sbatch slurm/run_qwen3_baseline_lvbench.sh qwen3_baseline \
  --min_tokens 200 --max_tokens 256 \
  --save_dir results/qwen3_vl_8b_baseline/lvbench/target_frames=128-tpf=200_256

# 64 frames, both resolutions (scavenger, 1 GPU) -> 41.45% / 42.29%
for cfg in "100 128" "200 256"; do set -- $cfg
  sbatch slurm/run_qwen3_baseline_lvbench_scav.sh qwen3_baseline \
    --target_frames 64 --min_tokens $1 --max_tokens $2 --resume \
    --save_dir results/qwen3_vl_8b_baseline/lvbench/target_frames=64-tpf=$1_$2
done
```

### 3b. ReKV, dense-equivalent (the faithfulness check)

128 frames → 64 blocks; `retrieve_size 64` retrieves all of them and nothing offloads, so ReKV
reduces to dense and can be compared against stock directly.

```bash
sbatch slurm/run_single_gpu_array_a5000_vulcan_qwen3.sh qwen3_vl_8b lvbench qwen3_repro \
  --n_local 9100 --target_frames 128 --prefill_method swa --retrieval_method rekv \
  --min_tokens 100 --max_tokens 128 --resume --enable_similarity
```
`--n_local 9100`: must satisfy `n_local // (tok_per_frame + ~10) >= topk`, else the guard aborts.

### 3c. ReKV, full-length (the real protocol)

Full video at 0.5 FPS (~916 blocks median), retrieve 64 — retrieval actually selects.

```bash
./slurm/setup_code_snapshot.sh qwen3_rekv256
sbatch slurm/run_single_gpu_array_a5000_scav_qwen3_4gpu.sh qwen3_vl_8b lvbench qwen3_rekv256 \
  --n_local 18500 --prefill_method swa --retrieval_method rekv \
  --min_tokens 200 --max_tokens 256 --resume --enable_similarity

# 7 chunks (the longest videos) OOM'd on host RAM at 96G -> rerun, --resume skips the other 45
sbatch --mem=192G slurm/run_single_gpu_array_a5000_scav_qwen3_4gpu.sh qwen3_vl_8b lvbench qwen3_rekv256 \
  --n_local 18500 --prefill_method swa --retrieval_method rekv \
  --min_tokens 200 --max_tokens 256 --resume --enable_similarity
```
`--n_local 18500` is the minimum clearing the guard at 252 tok/frame (`64 x 262 = 16,768`).
**4 GPUs, not 2** — 2×A5000 hit *CUDA* OOM in the MLP. `--mem` is host RAM and was never the
binding constraint at this stage.

### 3d. Merge + score (no array task does this, by design)

```bash
python -m video_qa.run_eval --model qwen3_vl_8b --dataset lvbench --num_chunks 52 \
  --n_local 18500 --retrieve_size 64 --min_tokens 200 --max_tokens 256 \
  --prefill_method swa --retrieval_method rekv --only_eval
```
**`qa_acc` in the CSVs is already ×100** — mean it directly. A naive `df.qa_acc.mean()` on a
correct 27.8% chunk reads 2777%.

---

## 4. Results

| run | frames | tok/f | acc |
|---|---|---|---|
| stock HF | 64 | 120 | 41.45 |
| stock HF | 64 | 252 | 42.29 |
| stock HF | 128 | 120 | 45.00 |
| stock HF | 128 | 252 | 46.35 |
| ReKV swa (dense-equiv) | 128 | 120 | 44.54 |
| **ReKV swa, full video, top-64 of ~916** | full | 252 | **49.63** (89/103 videos) |
| *VideoMem paper baseline* | *64* | *?* | *50.6* |

**The implementation is faithful.** ReKV vs stock at matched config: 44.54 vs 45.00, agreeing on
**97.87% of all 1549 questions**. The difference is bf16 non-determinism, not a bug.

**Prompt is not the explanation** — `standard` tracks `ours` within ~0.7 points everywhere.

**Fewer frames is worse, not better.** 64 frames loses ~4 points against 128 at both resolutions.
An earlier hypothesis in the opposite direction, argued from the paper's Table 4 / Figure 5, was
wrong: those vary *turns over fixed content*, not frame count.

**Coverage is the lever.** Full-video ReKV with top-64 retrieval beats dense-64 by ~8 points using
the same number of frames in context — they are just chosen from the whole video.

⚠️ **49.63 is provisional**: measured on 89/103 videos, and the 14 missing ones are the *longest*
(median 99 min vs 58). Expect the final number to fall. Job 7135906 is completing them.

Unresolved: at their stated 64 frames we get 41–42%, not 50.6. Their eval resolution, prompt and
harness are not stated (§4.1 gives `128*32*32` for **training** only; the appendix is missing —
§3.2/§3.3 cite "Appendix ??").

---

## 5. Layout

```
results/qwen3_vl_8b_baseline/lvbench/target_frames={64,128}-tpf={100_128,200_256}/52_*.csv
    columns: qa_acc_ours, qa_acc_standard, pred_choice_*, plus qa_acc aliased to `ours`
results/qwen3_vl_8b/lvbench/64-0.5-target_frames=128-tpf=100_128/swa-rekv/52_*.csv
results/qwen3_vl_8b/lvbench/64-0.5-tpf=200_256/swa-rekv/52_*.csv
```

---

## 6. Code fixes this required

`video_qa/run_eval.py`
- `qwen3_vl_8b` added to `--model` choices (argparse rejected it outright)
- `--min_tokens/--max_tokens` added and forwarded to workers
- token budget encoded into `save_dir` (`-tpf=200_256`) so sweeps cannot overwrite each other
- **array tasks no longer fail spuriously**: both the all-chunks-exist assertion and the merge are
  now gated on `args.chunk_idx is None`. Previously every array task wrote its chunk, then died on
  the assertion — which masked a real CUDA OOM for hours.

`video_qa/base.py`
- same two flags on the worker, gated to the qwen2_5_vl / qwen3_vl loaders
- records why each model import failed and raises a readable error naming the likely cause,
  instead of `KeyError: '<model>'`

`model/attention/kv_cache_manager.py`
- `_remainder_visual_mask()`: two sites indexed `global_remainder` with `visual_mask_buffer`, which
  mirrors `local_k`, not the remainder. Only fires when nothing is offloaded — i.e. exactly the
  dense-equivalent LVBench config.

`model/qwen3_vl_rekv.py`
- `--min_tokens/--max_tokens` plumbed into `load_model`
- trailing odd frame dropped: a chunk whose frame count is not a multiple of `temporal_factor` gets
  a different resolution from the processor, breaking the cache's fixed `block_size`
- `max_frames >= topk + 1`

`slurm/run_single_gpu_array_a5000_vulcan_qwen3.sh` and the scav variants capture `RC=$?`
immediately; the original re-read `$?` after an `echo`, so failures always logged "exit code 0".

---

## 7. Gotchas, in the order they cost time

1. Wrong conda env → `KeyError: 'qwen3_vl_8b'` with the cause swallowed.
2. `--num_chunks 100` on 103 videos → `IndexError` for chunks 52-99.
3. `--save_dir` outside `results/` → output written inside the snapshot; job looks empty.
4. Array tasks marked FAILED by the merge/existence checks even when the chunk was fine.
5. `--mem` is host RAM; the ReKV OOM was **CUDA** memory. Raising `--mem` to 128G fixed nothing and
   made the job unschedulable on a 24-node partition.
6. `qa_acc` is pre-multiplied by 100.
7. Stale snapshot after a code edit → argparse rejects the very flags you just added.

---

## 8. AKS-2 (`swa_akv2_*`) — Phase 1 validation + Phase 2 ladder (2026-08-31)

Method: `apply_budgeted_kk_compression` (count-targeted, budget-reallocated,
query-guided key selection). Name grammar `swa_akv2_<avg_keep>[_a<alpha*100>][_f<floor*100>]`.
Snapshot: `qwen3_akv2` (arms), `qwen3_akv2_fid` (fidelity diag — includes
`_record_proxy_fidelity`, gated on `MEMSTREAM_PROXYFID_DIR`).

Required flags for every arm, copied from the validated arms' exec lines — omitting
either of the first two fails in ways that still print "completed":
`--n_local 39000 --min_tokens 482 --max_tokens 536 --resume --enable_similarity --no_pin_memory`

### Phase 1 (all on vulcan46 unless noted)

```bash
# proxy-fidelity diagnostic (swa arm, video iA_69g87Ilw) — job 7409070, COMPLETED 27 min
sbatch --partition=vulcan-ampere --account=vulcan-abhinav --qos=vulcan-high slurm/run_proxyfid_diag.sh
# -> output/proxyfid_tpf510_swa/csv/  median frame-Spearman 0.36 overall, 0.60-0.64 at layers 20-24 (peak l21 0.644)

# equivalence gate vs swa_kk_keep_2-ropefixed — job 7409222, COMPLETED 13.6 min, chunk 39
sbatch --array=39 --partition=vulcan-ampere --account=vulcan-abhinav --qos=vulcan-high \
  --gres=gpu:h200-sxm:2 --mem=128G slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_255_a100_f100 --n_local 39000 --min_tokens 482 --max_tokens 536 \
  --resume --enable_similarity --no_pin_memory
# -> 79.3% choice agreement (23/29), acc 72.4 vs 62.1, 0 degenerate; divergence = bf16/fp32
#    tie-breaking on near-static content (CPU cross-check: 95.7% patch overlap on distinct
#    frames, 57-65% under heavy ties). Integration PASS.

# smoke, default config, chunk 0 (long videos) — job 7409285, COMPLETED 15 min, MaxRSS 91.6G
sbatch --array=1 --partition=vulcan-ampere --account=vulcan-abhinav --qos=vulcan-high \
  --gres=gpu:h200-sxm:2 --mem=128G slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_120_a50 --n_local 39000 --min_tokens 482 --max_tokens 536 \
  --resume --enable_similarity --no_pin_memory
# -> 35 Q, 0 degenerate, acc 37.14 == swa == kk_2 on this chunk (89%/74% agreement)

# Failed precursors, for the record: jobs 7409014/15/16 + 7409068 (missing --n_local 39000
# -> topk>n_local_frames guard, wrapper still said COMPLETED); 7409068 also OOM-killed at
# exactly the vulcan-high 128G cap before the flag fix (--no_pin_memory was also missing
# until 7409162/7409222).
```

### Phase 2 — LVBench ladder, avg_keep=120 (kk_2-matched budget), alpha sweep

References: swa-ropefixed 49.19, kk_2-ropefixed 51.00 (n=1,549).

```bash
# a100 (pure novelty under the budget) — job 7409554, cml-scavenger, full 52-array
sbatch slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_120_a100 --n_local 39000 --min_tokens 482 --max_tokens 536 \
  --resume --enable_similarity --no_pin_memory

# a50 (default mix) — job 7409555, vulcan-scavenger (256G QoS cap -> --mem=250G)
sbatch --partition=vulcan-scavenger --account=vulcan-abhinav --qos=vulcan-scavenger \
  --gres=gpu:h200-sxm:2 --mem=250G slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_120_a50 --n_local 39000 --min_tokens 482 --max_tokens 536 \
  --resume --enable_similarity --no_pin_memory

# a0 (pure query guidance), array SPLIT across the two non-preemptible vulcan H200 QoS
# (plain vulcan-medium caps mem at 64G < the measured 91.6G need; the -h200 variants
# allow 1T / 512G. Both MaxJobsPU=2, so these trickle 2 tasks at a time.)
# chunks 1-26  — job 7409556, vulcan-high-h200
sbatch --array=1-26 --partition=vulcan-ampere --account=vulcan-abhinav --qos=vulcan-high-h200 \
  --gres=gpu:h200-sxm:2 --mem=400G slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_120_a0 --n_local 39000 --min_tokens 482 --max_tokens 536 \
  --resume --enable_similarity --no_pin_memory
# chunks 27-52 — job 7409557, vulcan-medium-h200
sbatch --array=27-52 --partition=vulcan-ampere --account=vulcan-abhinav --qos=vulcan-medium-h200 \
  --gres=gpu:h200-sxm:2 --mem=400G slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_120_a0 --n_local 39000 --min_tokens 482 --max_tokens 536 \
  --resume --enable_similarity --no_pin_memory
```

Verify each arm by counting 52 CSVs / n=1,549 dedup'd questions — never the wrapper log.

Rebalance (2026-09-01): a50 was starved on vulcan-scavenger (0 running behind vulcan46
paid traffic) and a0 trickled at the 2-job MaxJobsPU of the -h200 QoS pair. Cancelled
7409555 (fully pending), 7409557, and the pending part of 7409556 (kept its running
task), then resubmitted the missing chunks on cml-scavenger with identical args:

```bash
sbatch --array=2-52  slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_120_a50 --n_local 39000 --min_tokens 482 --max_tokens 536 \
  --resume --enable_similarity --no_pin_memory   # job 7411092
sbatch --array=10-52 slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_120_a0  --n_local 39000 --min_tokens 482 --max_tokens 536 \
  --resume --enable_similarity --no_pin_memory   # job 7411093
```

### Disambiguator arms (2026-09-01)

Motivation: at 51/52 chunks, `swa_akv2_120_a100` reads −0.07 (±1.78) vs swa but
−1.51 (±1.90) vs kk_2 — it gives back most of kk_2's margin. Two candidate causes,
one arm each:
- budget gap: avg_keep=120 vs kk_2's realized stride yield of 135 (~11% fewer tokens)
  → `swa_akv2_135_a100` (exact count match)
- floor too aggressive: floor 0.25×120=30 tokens ≈ 17× compression on static frames,
  and the compression ladder says ~14× costs 2.2–2.5
  → `swa_akv2_120_a100_f50` (floor 60 ≈ 8.5×, inside the safe zone)

Split across the four routes (no chunk double-submitted per arm). FLAGS =
`--n_local 39000 --min_tokens 482 --max_tokens 536 --resume --enable_similarity --no_pin_memory`

```bash
# swa_akv2_135_a100
sbatch --array=1-17  --partition=vulcan-ampere --account=vulcan-abhinav --qos=vulcan-high-h200   \
  --gres=gpu:h200-sxm:2 --mem=400G slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_135_a100 $FLAGS     # job 7414674
sbatch --array=18-34 --partition=vulcan-ampere --account=vulcan-abhinav --qos=vulcan-medium-h200 \
  --gres=gpu:h200-sxm:2 --mem=400G slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_135_a100 $FLAGS     # job 7414675
sbatch --array=35-52 slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_135_a100 $FLAGS     # job 7414676 (cml-scavenger)

# swa_akv2_120_a100_f50
sbatch --array=1-26  --partition=vulcan-scavenger --account=vulcan-abhinav --qos=vulcan-scavenger \
  --gres=gpu:h200-sxm:2 --mem=250G slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_120_a100_f50 $FLAGS # job 7414677
sbatch --array=27-52 slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 \
  --prefill_method swa_akv2_120_a100_f50 $FLAGS # job 7414678 (cml-scavenger)
```

Rebalance 2 (2026-09-01): live GPU sampling inside a running akv2 task (srun --overlap,
raw id) showed peak ~27 GiB GPU0 / ~19 GiB GPU1 of 141 — H200s are overkill for
compressed arms. Moved every pending chunk to the 48GB scavenger pools (A6000/L40S,
--mem=200G, same FLAGS; running tasks left in place). f50 submitted first:
  7416106 f50 1-52 (a6000)   7416107 a0 10-52 (l40s)   7416108 m135 12-17,21-52 (a6000)
  7416109 a50 33,43-52 (l40s) + 7416111 a50 11 (l40s)  7416110 a100 35 (a6000)

Rebalance 3 (2026-09-01, user-directed): A6000 arrays via vulcan-scavenger (not plain
scavenger); L40S as gpu:l40s:1 + --gpus_per_chunk 1 (2 co-located L40S are near-impossible
to schedule). Kept running tasks (7416107_10 which did land 2xL40S, a50 cml x2, m135
vulcan46 x1). Final layout:
  7416132 f50 1-52 (vulcan-scav a6000x2)   7416133 m135 12-17,21-52 (vulcan-scav a6000x2)
  7416134 a100 35 (vulcan-scav a6000x2)    7416135 a0 11-52 (scav l40s x1)
  7416136 a50 11,33,43-52 (scav l40s x1)

Rebalance 4 (2026-09-01, user-directed — Hopper pools freer now): f50 split 1-26
cml-scavenger (7416303) + 27-52 vulcan-scavenger h200-sxm/vulcan46 (7416304); m135
remainder to cml-scavenger (7416305; task 11 cancelled — chunk 10 still running on
vulcan-ampere from 7414674); a100 straggler 35 to vulcan46 (7416306). a0/a50 remain
on 1x L40S (7416135/7416136) — first L40S chunks completing fine.

Rebalance 5 (2026-09-01, user-directed): added the non-preemptible vulcan46 lanes.
f50 chunks 43-52 moved from vulcan-scavenger to vulcan-high-h200 (7416943); m135
chunks 21-32 moved from cml-scavenger to vulcan-medium-h200 (7416944); both
--mem=400G, 2 concurrent tasks each (MaxJobsPU=2). Plain vulcan-high/medium not used:
mem caps 128G/64G vs the 60-150G per-chunk host need.
Rebalance 6 (2026-09-01): f50 remaining chunks 30-42,51-52 consolidated onto cml-scavenger (launcher defaults) as job 7418882; cancelled 7416304/7416943.
Rebalance 7 (2026-09-01): a0/a50 1xL40S tasks all CUDA-OOM'd instantly at ecs=64 (44GB card; worker died, wrapper+sacct said COMPLETED — detected via 2-min elapsed + missing CSVs). Resubmitted with --encode_chunk_size 16: a0 11-52, a50 11,33,43-52.
Rebalance 8 (2026-09-01): plain-scavenger useless (start times next-day) — a0 pending 12-52 -> vulcan-scavenger rtxa6000:2 (ecs default, 2-way shard fits 48GB); a50 11,33,43-52 -> cml-scavenger defaults. a0 task 11 left running on L40S/ecs16.
Rebalance 9 (2026-09-01): added non-preemptible base-QoS lanes on tron/nexus A6000s — a0 12-31 -> qos=high (a6000:2, 125G; job 7419022), a0 32-42 -> qos=medium (a6000:2, 62G — host-RAM marginal for long videos, requeue+resume mops OOMs; job 7419023); a0 43-52 stays vulcan-scavenger (7418954). qos=default skipped (1 GPU/32G cannot run this).
kk_2 LVBench replication for sim curves (2026-09-01): swa_kk_adjacent_2 + suffix ropefixed_repl2, vulcan-scavenger a6000:2 — regenerates the purged similarity dumps so akv2-vs-kk_2 curve comparisons are possible on LVBench.
OOM cleanup (2026-09-01): a0 chunk 11 OOM'd on tron high at exactly 125G (49min) -> resubmitted on cml-scav (job above). Medium slice 32-42 pulled back to vulcan-scav a6000 preemptively (62G usable < typical 60-130G chunk footprints). High lane kept (128G covers most chunks; stragglers mopped at end).
tron-high lane abandoned (2026-09-01): chunks 11+12 both OOM'd at exactly 125G (~1 GPU-h each; long-video stretch). a0 13-31 -> vulcan-scav a6000 200G. Base-QoS verdict: high (128G) marginal for long-video chunks, medium (62G) unusable — this workload needs 200G+ lanes.
repl2 split (2026-09-01): chunks 1-26 -> cml-scavenger (job above), 27-52 stay vulcan-scav (7419331).
Rebalance 10 (2026-09-02): cml-scav finishing fastest — a0 all missing (11,13-52) and a100 straggler (35) consolidated onto cml-scav; repl2 27-52 stays vulcan-scav.
Rebalance 11 (2026-09-02): a0 33-52 -> 1xL40S ecs=16 (the validated single-card recipe) to parallelize with cml; a0 11,13-32 stay cml (7420527).
Rebalance 12 (2026-09-02): a0 23-32 -> vulcan-high-h200 (vulcan46, 400G, 2 concurrent) per user; a0 on cml now 11,13-22.
Rebalance 13 (2026-09-02): qos=high rejected on scavenger partition (tron-only, no L40S there). a0 34-52 -> L40S via cml-scavenger (cml34) instead of plain scavenger.
swa repl2 launched (2026-09-02): 1-26 cml-scav, 27-52 vulcan-scav h200 (swa is the heavy config -> H200 lanes only).
akv2 VERDICT (2026-09-02, user): FAILED — does not match v1 (kk_2) on LVBench; v1's benefit is corroborated everywhere. Remaining a0 arms cancelled (query channel dead). m135 finishing for the record; repl2/swa-repl2 continue (analysis value). Iteration protocol going forward: v1 is the base; any v2 encoding idea gets ONE single-change LVBench arm with kill criterion >= kk_2.
CORRECTION (2026-09-02): user wants all arms COMPLETED for the record — a0 cancellation reversed; resubmitted 11,13-22 (cml h200), 23-32 (vulcan-high-h200), 33-52 (cml l40s ecs16). Verdict on akv2 stands; runs complete regardless.
GT-alloc oracle prototype (2026-09-02): swa_gtalloc_f60 on BV1Dx4y1p77a (idx 303), union 125/781 blocks dense + floor 60 -> avg ~132 == kk_2 budget. 4xA6000 vulcan-high. Code: apply_gt_alloc_compression + swa_gtalloc_ dispatch (unit-tested; akv2 dispatch regression green).

## CRITICAL CORRECTION (2026-09-02, user-caught): code-version confound voids the akv2 verdict

repl2 (current code) does NOT reproduce the original kk_2-ropefixed arm (July-31
snapshot): 78.6% answer agreement, acc 50.68 vs 52.33 on 22 matched chunks. ~900
lines changed between snapshots incl. the committed RoPE-table fix (7c84f1d) and the
kk-path refactor (keep_k/dense_len/top_p). Deterministic generation ⇒ the path
changed, not sampling.

⇒ All akv2-vs-kk_2 comparisons used OLD-code references while akv2 ran NEW code.
Against SAME-code kk_2 (repl2, n=665): f50 +1.20(±1.70), a100 +0.30, a50 −0.15 —
the "reallocation gives back kk_2's margin" conclusion was the CODE DRIFT (−1.65),
not allocation. Within-code comparisons (old kk_2 vs old swa +1.68; akv2-family
internal) remain valid; every cross-code number is void, including the Phase-1
equivalence gate's 79% (CPU-level selection-overlap check unaffected).

Rebuild plan: complete kk_2-repl2 + swa-repl2, re-score all arms against same-code
references. Standing rule going forward: NEVER compare arms across snapshots —
references must be regenerated whenever model-path code changes.

RESOLUTION of the code-version confound (2026-09-02): direction determined. The July-31
snapshot CARRIES the stale-window-RoPE-table bug (confirmed 7-28, ~93-94% of window
rows at wrong relative positions, no cancellation — see docs/qwen3_vl_rope_stale_window_table.md);
the LVBench references were run Aug 1-2, PRE-fix (7c84f1d, Aug 4). All akv2-era arms are
post-fix. ⇒ The old references are the corrupted side; kk_2-repl2 / swa-repl2 (post-fix)
become canonical. CG-Bench regenerated its post-fix references weeks ago (the existing
_repl2 dir); LVBench never did until now. V1's own LVBench margin (+1.68) must be
re-established on fixed code. Vintage rule: no comparison crosses the Aug-4 fix line.
ECS-isolation test (2026-09-02): repl2 ran ecs=64 but ALL original tpf=482_536 references ran ecs=8 (verified from logs) — the 21% repl2 divergence may be preprocessing config, not code. Test: kk_2 chunk 0 on current code at ecs=8 (suffix replE8) vs original(ecs8,July) vs repl2(ecs64,current).
repl3 (2026-09-02, user): kk_2 chunk 0 from the JULY SNAPSHOT itself (qwen3_ropefixed) at its original ecs=8, suffix ropefixed_repl3 — direct reproducibility test of the reference. Together with replE8 (current code @ ecs8) this gives the full 2x2: {July code, current code} x {ecs8}.

GT-alloc oracle readout (2026-09-02, 1 video/12 q, user-reviewed sheets): dense GT-union
encoding does NOT lift GT-window scores or recall (oracle .565 vs kk_2 .588 vs swa .615);
query-frame curves change non-directionally. Combined with akv2-vs-kk2 curve near-identity
and the ecs/numerics sensitivity: query-alignability is not steerable by encoding budget
placement — allocation axis closed with an oracle-grade null. Optional: ~30-video subset
to make the negative citable.
CORRECTION (same night, user): the oracle null is CONFOUNDED — budget-matching forced
8.5x on all non-GT frames (incl. dynamic ones kk_2 treats at 4x), so it tests "GT gain
vs collateral damage", not placement per se. Clean ceiling = generous oracle (GT@510 +
rest@135, avg ~195, budget-unmatched by design) + reverse control (GT@60 + rest@135).
Morning runs; needs rest_keep param in swa_gtalloc.
Generous oracle (2026-09-02, user): swa_gtalloc_f135 = GT union dense(510) + rest at kk_2's 135 (avg ~195, deliberately budget-unmatched ceiling), same video BV1Dx4y1p77a, 4xA6000 vulcan-high. No code change: f<floor> already parameterizes the non-GT keep.
Generous oracle result (2026-09-02, job 7422051): INVERSION, not null. GT@510 + rest@135
(no collateral damage, surplus budget): recall .546 < kk_2 .588 < swa .615; GT mean-z
+1.26 vs 1.55 for both uniform arms (f60 oracle: same pattern). Densifying evidence
frames REDUCES their retrieval contrast — V1's redundancy-dilutes-representations
thesis operating at frame granularity. Next single-job test: reverse control (GT@60 +
rest@135) — if recall RISES, compression-sharpens-evidence confirmed directionally.
Caveats: 1 video/12q/l21, z is curve-relative; recall ordering is measure-independent.
repl3 config corrected per user: July snapshot @ ecs=64 (not 8) — completes the 2x2: orig(J,8), replE8(C,8), repl2(C,64), repl3(J,64).

2x2 VERDICT (2026-09-02, chunk 0): CODE EXONERATED. repl3(July,64) == repl2(current,64)
at 100.0% — bit-identical; nothing since July changes the kk_2 path. ecs 8->64 flips
~17% (82.9%) on either vintage (acc 37.1->40.0 on this chunk). Residual 11.4%
(orig(J,8) vs replE8(C,8)) = ENVIRONMENT drift since Aug 1 (conda/driver). The whole
repl2-vs-original divergence = ecs numerics + env vintage, both zero-mean chaos.
Canonical references = the repl2 pair; rule: never compare across vintages of code,
config, OR environment.
repl3 extended to full 52 chunks (2026-09-02, user): July snapshot @ ecs=64 across the whole benchmark — at-scale test that repl3(J,64) == repl2(C,64) everywhere, isolating ecs+env from code conclusively.
Reverse control launched (2026-09-02): swa_gtalloc_f135_d60 = GT frames COMPRESSED to 60, rest at 135 (avg ~123). Prediction if compression-sharpens-evidence holds: recall RISES above kk_2. dense_keep param added + unit-tested; snapshot qwen3_gtalloc2.
REVERSE CONTROL RESULT (2026-09-02, job 7424966): CONFIRMED INVERSION, monotone dose-
response. GT@60/rest@135: recall .599, GT-z +1.70 — beats kk_2 (.588/+1.55) at LOWER
budget; generous (GT@510) worst (.546/+1.26). Per-frame compression rate is a
DIRECTIONAL retrieval-sharpness dial (visibility vs storage decoupled: QA still reads
full KV). First controllable sub-knob effect found on the encoding axis. Next: 4-arm
dose curve on ~20-30 CG videos (recall channel) before building a method on it.
replEXACT (2026-09-02): TRUE reproduction attempt of the original kk_2 chunk 0 — July snapshot, ecs=8, 1 GPU, NO enable_similarity, NO no_pin_memory, cml35-class H200. Original exec line recovered from log 7182511_1; earlier replE8 mismatched gpus(2), +enable_similarity, +no_pin_memory.

## Replication matrix for swa_kk_adjacent_2 (documented 2026-09-02, user-requested)

The original kk_2 reference and its replications, with EVERY launch difference. The
original exec line was recovered from output/logs/memstream_qwen3_h200_array_7182511_1.log
(job 7182511, submitted 2026-08-01T12:39, cml35).

| arm (suffix) | code snapshot | ecs | GPUs | enable_similarity | no_pin_memory | status |
|---|---|---|---|---|---|---|
| -ropefixed (ORIGINAL) | qwen3_ropefixed (Jul 31) | 8 | 1 | no | no | 52/52, acc 51.00 |
| -ropefixed_repl2 | qwen3_akv2 (Aug 31) | 64 | 2 | yes | yes | in progress |
| -ropefixed_repl3 | qwen3_ropefixed | 64 | 2 | yes | yes | CANCELLED 09-02 (12 chunks done — code exonerated, no longer needed) |
| -ropefixed_repl4 | qwen3_ropefixed | 8 | 1 | no | no | VERDICT IN — EXACT original command; partial run then cancelled (see below) |
| -ropefixed_replE8 (chunk 0 only) | qwen3_akv2 | 8 | 2 | yes | yes | DELETED 09-02 (user request; its 88.6%-vs-original number lives only in this doc) |
| -ropefixed_repl5 | qwen3_ropefixed | 8 | 1 | **yes** | **yes** | job 7425819, chunks 0-5,7,23,29,45 — isolates the two flags vs ORIGINAL |
| -ropefixed_repl6 | qwen3_ropefixed | **64** | 1 | yes | yes | job 7425874, same chunks — vs repl5 isolates ecs alone |

What each pairwise comparison isolates:
- repl3 vs repl2: CODE (July vs current) at matched launch config — 100.0% identical so far (156+/156 exact answers)
- repl4 vs ORIGINAL: full reproduction — same code, same config, same node class; any
  divergence here is substrate-vintage (driver/system) or nondeterminism, nothing else
- repl4 vs repl3: launch config (ecs 8→64, 1→2 GPUs, +similarity logging, +no_pin) on identical code
- replE8 vs ORIGINAL (chunk 0): 88.6% — confounded by GPUs(2)+enable_similarity, superseded by repl4
Findings so far: chunk-0 2x2 gave repl3==repl2 at 100.0%; ecs 8↔64 flips ~17%;
conda env verifiably unchanged since Mar 27 (torch Feb 2026); original ran on cml35,
current driver 595.71.05 (no historical driver logged — logging gap to fix).
Priority change (2026-09-02): repl3 pending chunks held (scontrol hold 7424903) so repl4 (exact-command reproduction, 1-GPU tasks) takes all cml slots; auto-release repl3 when repl4 drains.

### Command ledger — exact submit line per repl arm (from sacct SubmitLine)

All from the worktree root; script = slurm/run_single_gpu_array_h200_qwen3.sh
(script default ecs when the flag is absent = 64; script default GPUs = 2).

- **ORIGINAL** (job 7182511, 2026-08-01, cml35; first of the resume submissions):
  `sbatch --array=1-26 --time=12:00:00 --partition=cml-scavenger --qos=cml-scavenger --account=cml-scavenger --gres=gpu:h200-sxm:1 --export=ALL,GPUS_PER_CHUNK=1 --mem=250G slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_ropefixed --prefill_method swa_kk_adjacent_2 --n_local 39000 --min_tokens 482 --max_tokens 536 --encode_chunk_size 8 --resume --save_dir_suffix ropefixed`
- **repl2** (job 7419331; ecs=64 + 2 GPUs via script defaults):
  `sbatch --array=1-52 --partition=vulcan-scavenger --account=vulcan-abhinav --qos=vulcan-scavenger --gres=gpu:rtxa6000:2 --mem=200G slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_akv2 --prefill_method swa_kk_adjacent_2 --save_dir_suffix ropefixed_repl2 --n_local 39000 --min_tokens 482 --max_tokens 536 --resume --enable_similarity --no_pin_memory`
- **repl3** (job 7424903, CANCELLED at 12 chunks):
  `sbatch --array=2-52 slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_ropefixed --prefill_method swa_kk_adjacent_2 --save_dir_suffix ropefixed_repl3 --n_local 39000 --min_tokens 482 --max_tokens 536 --resume --enable_similarity --no_pin_memory --encode_chunk_size 64`
- **repl4** (job 7425193 array 1-52, then continuation 7425773 array 1,8,24,30,46):
  `sbatch --array=1-52 --gres=gpu:h200-sxm:1 slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_ropefixed --prefill_method swa_kk_adjacent_2 --save_dir_suffix ropefixed_repl4 --n_local 39000 --min_tokens 482 --max_tokens 536 --resume --encode_chunk_size 8 --gpus_per_chunk 1`
- **repl5** (job 7425819; = repl4 + the two flags, chunks 0-5,7,23,29,45):
  `sbatch --array=1-6,8,24,30,46 --gres=gpu:h200-sxm:1 slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_ropefixed --prefill_method swa_kk_adjacent_2 --save_dir_suffix ropefixed_repl5 --n_local 39000 --min_tokens 482 --max_tokens 536 --resume --encode_chunk_size 8 --gpus_per_chunk 1 --enable_similarity --no_pin_memory`
- **repl5/repl6 FULL-BENCHMARK extension** (2026-09-02, user-directed): jobs 7428702
  (repl5) / 7428703 (repl6), --array=7,9-23,25-29,31-45,47-52 (the chunks not in the
  original 10), same commands. Purpose: the definitive PAIRED ecs test at n=1549
  (paired SD ~±0.95) with no dependence on the single original run — decides
  "ecs is a zero-mean coin" vs "ecs costs ~1-2 pts" (pooled flip test was p=0.17,
  all-negative signs confounded by the shared orig anchor).
- **repl6** (job 7425874; = repl5 + ecs=64 — repl6-vs-repl5 is the PURE ecs comparison):
  `sbatch --array=1-6,8,24,30,46 --gres=gpu:h200-sxm:1 slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_ropefixed --prefill_method swa_kk_adjacent_2 --save_dir_suffix ropefixed_repl6 --n_local 39000 --min_tokens 482 --max_tokens 536 --resume --encode_chunk_size 64 --gpus_per_chunk 1 --enable_similarity --no_pin_memory`
- **replE8** (chunk 0 forensic, current code, ecs=8, 2 GPU, both flags — results DELETED
  2026-09-02 at user request; recorded verdict: 88.6% agreement vs original on chunk 0)

Single-delta chain: ORIGINAL → repl4 (nothing changed; 100%) → repl5 (+two flags)
→ repl6 (+ecs 8→64). Each adjacent pair isolates one knob; GPU count 1→2 is the
only knob left un-isolated (attributable by elimination).

### FINAL DECOMPOSITION (2026-09-02, interim n=195–285 per pair; all zero-mean)
- repl4 vs ORIGINAL: 280/285 = 98.2% — exact command reproduces bit-perfectly; ALL
  5 flips sit in videos adjacent to a crash-resume boundary (2 = original's chunk-0
  resume history — repl5's clean single pass agrees with repl4 on them; 3 = original's
  chunk-29 4-submission history). Uninterrupted chunks: 100%.
- repl5 vs repl4: 195/195 = 100.0% — --enable_similarity + --no_pin_memory are
  numerically NEUTRAL.
- repl6 vs repl5: 180/225 = 80.0% — PURE ecs 8→64 effect: ~20% answer flips.
- repl6 vs repl2: 213/225 = 94.7% — GPU count (1→2) + hardware (H200→A6000) + code
  (exonerated) together: ~5% flips.
- Scoreboard: ecs ≈20%, hardware/GPU ≈5%, flags = 0, code = 0, substrate-month = 0,
  resume boundaries ≈1–2 flips per crashed chunk. The founding 78.6% repl2-vs-orig
  mystery is fully explained; accuracy never moved (chaos is zero-mean).

### REPL4 VERDICT (2026-09-02) — exact command reproduces the original BIT-PERFECTLY

repl4 (job 7425193, cml35, exact original exec line incl. ecs=8 / 1 GPU / no
enable_similarity / no no_pin_memory, July qwen3_ropefixed snapshot) vs ORIGINAL:
**95/95 = 100.0% choice agreement across chunks 0–3; matched accuracy identical
(54.74 vs 54.74).** Update at 212 matched rows: 210/212 (99.1%) — the only 2 flips
are both in chunk 0's crash-resumed video (q01CUy_gwdU, rows 19+ written post-resume);
never-crashed chunks are 100%. I.e., determinism is exact within an uninterrupted
process; a mid-chunk resume re-encode is itself a small bf16 perturbation. Not
pursued further (user: 99% is fine). Zero degenerate outputs in any repl arm (scanned for the
GPU-corruption quote signature, empty preds, invalid choice letters).

Conclusion of the whole forensic thread:
- The pipeline is EXACTLY deterministic when code + config + node class all match
  — even a month of substrate aging (Aug 1 → Sep 2, driver 595.71.05) changed nothing.
- Therefore ALL orig-vs-repl divergence is launch config: ecs 8→64 alone flips
  ~17% of answers (kernel/tile selection differs per batch shape; bf16 non-associative
  reductions compound through the streaming KV cache). The remaining ~11% seen in
  replE8 belongs to some mix of {2 GPUs, enable_similarity, no_pin_memory} — never
  separated one-at-a-time; pin memory is the least plausible actor (byte-exact DMA).
- All flips are zero-mean chaos (accuracy unmoved); the vintage rule stands:
  references are only comparable at EXACTLY matched code+config.
- Wind-down (user-directed): repl3 scancel'd entirely; repl4 pending tasks [7-52]
  scancel'd, running tasks 4–6 (chunks 3–5) left to finish for a final tally.
  repl4 mid-chunk .csv.tmp files retained (resume state — never delete).
- Revised (same day, user-directed): repl4 EXTENDED to 10 chunks total — chunks 0–5
  (sequential, already run/running) + 4 drawn randomly from 6–51 (python
  random.seed(20260902) → chunks 7, 23, 29, 45). Continuation job 7425773
  (--array=1,8,24,30,46; task 1 resumes the crashed chunk 0), identical command:
  `sbatch --array=1,8,24,30,46 --gres=gpu:h200-sxm:1 slurm/run_single_gpu_array_h200_qwen3.sh
  qwen3_vl_8b lvbench qwen3_ropefixed --prefill_method swa_kk_adjacent_2
  --save_dir_suffix ropefixed_repl4 --n_local 39000 --min_tokens 482 --max_tokens 536
  --resume --encode_chunk_size 8 --gpus_per_chunk 1`
- Note: repl4 task 1 died mid-chunk-0 at 18/35 with wrapper reporting COMPLETED —
  faithful reproduction of the original's mapping-wall crash-and-resume behavior
  (the exact command predates --no_pin_memory).

## Allocation-search harness (evolutionary masking, 2026-09-02)

User idea: search the masking/allocation space instead of hand-designing. Prototype
on BV1Dx4y1p77a (12 q); fitness = retrieval channel (clue recall@64 layers 17-24 +
GT mean-z via analysis/allocvec_search.py score), NEVER 12-question accuracy.
Fitness is deterministic per candidate (fixed config => bit-reproducible encode).

Machinery (all tested green, incl. bit-exact uniform==floor equivalence):
- swa_allocvec_<tag>: arbitrary per-frame keep counts via MEMSTREAM_ALLOCVEC_JSON
  ({video: {"keeps": [...]}}); v1 selection inside frames. kv_cache_manager both
  branches + apply_gt_alloc_compression(keep_vector=). tests/test_allocvec.py.
- analysis/allocvec_search.py: gen0 (8 budget-matched candidates @135 mean:
  uniform / novelty-prop x2 / anti-novelty x2 [dose-response direction] /
  budget-matched random placement x2 / GOP keyframe probe; pixel-novelty features,
  1562 frames) + score. Population at output/allocvec/gen0/.
- Snapshot .temp_code_qwen3_allocvec1 (worktree code + symlinked assets);
  slurm/run_gtalloc_oracle.sh passes ALLOCVEC_JSON through.
- Launch: the 8 sbatch lines printed by `python analysis/allocvec_search.py gen0`
  (vulcan-high, 4xA6000 each, ~2 concurrent under the QoS cap).
Caveats: 1 video = harness validation + oracle-style insight, not a result;
winners must be re-validated on 10-25 videos; deployable policies may only use
GT-free features (pixel novelty, run structure).

## Qwen3-VL-4B lane (2026-09-02)

Rationale: user idea — smaller family member unlocks the A5000/A4000 fleet for
search-scale work (evolution generations, TPF sweep at a second scale). Within-family
transfer assumed (all observed sign-flips were cross-family). NOTE: 4B has the 8B's
exact KV geometry (36L/8kv/128hd) so KV-cache and host-RAM costs are UNCHANGED;
savings are weights (9.4G), q-compute, and a ~30% smaller vision tower.

Setup: weights at model_zoo/Qwen3-VL-4B-Instruct; registered as qwen3_vl_4b
(base.py + run_eval.py); eval_qwen3_baseline.py MODEL_PATH now env-overridable
(QWEN3_BASELINE_MODEL); snapshot qwen3_4b_smoke.

Faithfulness smoke (the §3a/3b protocol) at the 144-tpf ladder rung
(min 144 / max 160 / n_local 10780 — low-distortion 16:9 control):
- 7429172 stock-HF 4B baseline, 128 frames (scavenger A5000):
  `QWEN3_BASELINE_MODEL=model_zoo/Qwen3-VL-4B-Instruct sbatch --export=ALL slurm/run_qwen3_baseline_lvbench_scav.sh qwen3_baseline --target_frames 128 --min_tokens 144 --max_tokens 160 --resume --save_dir results/qwen3_vl_4b_baseline/lvbench/target_frames=128-tpf=144_160`
- 7429173 ReKV 4B dense-equivalent (vulcan-scav A5000, doubles as the fit test):
  `sbatch slurm/run_single_gpu_array_a5000_vulcan_qwen3.sh qwen3_vl_4b lvbench qwen3_4b_smoke --n_local 10780 --target_frames 128 --prefill_method swa --retrieval_method rekv --min_tokens 144 --max_tokens 160 --resume --enable_similarity`
Pass = ReKV ≈ stock acc, ~95%+ per-question agreement (bf16-chaos-calibrated).

### ECS DEGRADES RETRIEVAL QUALITY (2026-09-03, 3-video single-delta isolation)

kk_2 GT-scored on CG-Bench at matched everything except --encode_chunk_size
(single-video runner, A5000): @8 beats @16 on 3/3 videos, both channels
(BV1s94y1G7RD .506/1.49 vs .467/1.32; N03ddXnwAls .274/.48 vs .202/.35;
tKoKRd66hP8 .754/1.55 vs .734/1.40). @8-single == repl2@8-multi on 3/3
(hardware + 13-video-per-process both exonerated). Plus the 10/10 sweep
(repl2@8 vs val@16: recall +4.5 pts, z +0.12 aggregate).

NOT semantic by any known channel: LLM cache path is per-frame (verified), and
Qwen3-VL ViT attention is per-frame-pair (cu_seqlens boundaries every h*w -
verified in the env's modeling_qwen3_vl.py). Remaining mechanism candidates:
(1) noise-flattens-contrast - batch-shape rounding noise is zero-mean on answers
but systematically compresses cosine contrast (GT-z); (2) M-RoPE chunk-relative
position subtleties. The staged fp32 probe (slurm/run_ecs_probe.sh) separates
them: noise collapses in fp32, positions don't.

Consequences: run everything at SMALL ecs (8; never 64); every ecs=64 arm is
retrieval-handicapped; predicts repl6(ecs=64) < repl5(ecs=8) on LVBench accuracy;
the original CG swa(16)-vs-kk_2(8) comparison was cross-ecs but the effect
replicated at matched ecs=16 (10-video validation, 8/10, +0.13 z).

## ARCHIVE: pre_sep3_ecs_fix (2026-09-03, user-directed)

All Qwen3 ReKV results produced before the position-stitch fix carry the
ecs-dependent position inflation and are archived to
results/pre_sep3_ecs_fix/{qwen3_vl_8b,qwen3_vl_4b}. Kept live: the posfix_*/clean_*
fixed-code arms and qwen3_vl_4b_baseline (stock HF, no ReKV machinery, unaffected).
Cancelled as moot on bugged code: repl5/repl6 extensions (7428702/03), repl2
back-halves (7419331/7420901), a0 (7421052) — the ecs-accuracy question they were
answering is resolved by the fix (encode is now ecs-invariant: dz 0.0001-0.02 on
the acceptance pairs vs 0.13-0.15 broken). Archived numbers remain internally
consistent within a vintage (same code+config) but are superseded for any absolute
claim; regenerate references on the qwen3_clean1 vintage.

## LVBench regeneration on the fixed vintage (2026-09-03)

Snapshot naming convention change (user): date-based, no more opaque suffixes.
The bit-certified fixed+cleaned code is snapshot **qwen3_0903_postfix**
(renamed from qwen3_clean1; identical bytes).

Regeneration launched (references for the V2 era; all future arms compare here):
- 7435515: kk_2 — `sbatch --array=1-52 --gres=gpu:h200-sxm:1 slurm/run_single_gpu_array_h200_qwen3.sh qwen3_vl_8b lvbench qwen3_0903_postfix --prefill_method swa_kk_adjacent_2 --save_dir_suffix 0903_postfix --n_local 39000 --min_tokens 482 --max_tokens 536 --resume --enable_similarity --no_pin_memory --encode_chunk_size 8 --gpus_per_chunk 1`
- 7435516: swa — same with --prefill_method swa
Save dirs: results/qwen3_vl_8b/lvbench/64-0.5-tpf=482_536/{swa,swa_kk_adjacent_2}-rekv-0903_postfix
Key question they answer: V1's kk_2−swa margin on position-correct code.

## REGENERATION VERDICT (2026-09-03, fixed vintage qwen3_0903_postfix)

LVBench n=1513 matched (51/52 swa chunks; chunk 14 mop-up in flight):
**kk_2 50.50 vs swa 51.42 — margin −0.93 ± 0.96.** The AKSv1 accuracy margin
does NOT survive the position fix (vintage arc: paper +1.68 → same-vintage
broken +0.90 → fixed −0.93). Interpretation: the compression advantage was
substantially position-artifact repair — sharpening retrieval over corrupted
keys; with correct rotary positions dense encoding retrieves on its own.
kk_2 absolute level unchanged (50.55 vs broken 51.00, within floor); the fix
helped swa, not hurt kk_2. Mandatory before any paper claim: re-check the
CG-Bench compression-recall replication on fixed code (the 10-video validation
predates the fix).

### CG-Bench recall recheck on fixed code (2026-09-03)

10-video swa-vs-kk_2 at ecs=8 on qwen3_0903_postfix: **kk_2 0.573/1.296 vs swa
0.554/1.244 — kk_2 wins 7/10 on both channels (+1.9 recall pts, +0.052 z).**
Pre-fix the gap was +4.0 recall/+0.12 z (8/10). So the AKS retrieval benefit is
REAL but roughly HALVED by the position fix; and both arms improved absolutely
(swa recall 0.472→0.554 — the fix helps dense most). Synthesis with the LVBench
accuracy verdict (−0.93): a ~+2-recall-point retrieval edge no longer converts
to accuracy (consistent with the ORACLE-A ceiling: perfect retrieval bought only
+2.7). Entropy comparison in flight (kk2 side rerun on qwen3_0903_diag1 with the
entropy-only fallback for non-tileable compressed windows).

### Encode-time attention entropy, fixed code (2026-09-03, 3 videos, all layers/heads)

swa vs kk_2, perplexity=2^H medians (raw bits, per memory protocol): L21 149.0 vs
56.2 (2.7x concentration from 3.5x fewer keys: 37,960 vs 10,960/window); early
layers purely mechanical (ppl/key identical 0.0035), mid/late layers show kk_2's
surviving keys carrying +25-36% more per-key mass (genuinely less redundant).
AKS DOES reduce encode entropy; the concentration -> sharper representatives ->
the halved-but-real recall gain. Data: output/entropy_postfix/{swa,kk2}/;
kk2 side via the entropy-only fallback (qwen3_0903_diag1).

### Compression ratio/pattern grid, fixed code (2026-09-03, 10 videos + L21 entropy)

Recall ladder: dense .554/1.244 -> 255 .573/1.289 -> 135 .573/1.296 (PLATEAU)
-> 85 .562 -> 57 .553 -> 32 .540/1.186. Optimum stays at 2-4x; monotone decline
beyond (no deeper-compression free lunch; matches gen-0 c4 overshoot).
Pattern control at matched 135 keys: dilated_2 .557/1.264 ~= DENSE, kk_2 .573 —
on fixed code the ENTIRE gain is selection content, key-count alone buys nothing
(pre-fix it looked valuable because fewer keys = less position inflation).
Entropy at matched count: kk_2 50.4 vs dilated 48.3 (equal) — selection does NOT
concentrate attention beyond the mechanical key-count effect; entropy reduction
is trivial. Mechanism sentence: novelty selection picks better anchor keys; it
does not sharpen attention. Data: grid_*/gridE_* dirs + output/entropy_postfix/.

### Matched-budget token-scaling closure (2026-09-03, CG recall, 10 videos, fixed code)

swa@144 (native coarse) 0.549/1.234 vs kk_2@510->135 (fine-then-selected)
0.573/1.296 vs swa@510 0.554/1.244. (1) 144->510 dense: recall flat (+0.005)
while attended keys grow 2.7x (56->149 effective; entropy runs) — dilution
confirmed on the recall channel. (2) At identical key budget AND identical
attention profiles (56.2 effective each), fine-then-selected beats native-coarse
+2.4 recall pts (8/10) — the resolution's value is real but only accessible via
content selection. V2 mechanism chain complete on certified code.
LVBench swa@144 accuracy rung (7440764) in flight for the accuracy-channel twin.

### Top-k attention + post-RoPE novelty verdicts (2026-09-03, 10 videos, fixed code)

Top-k encode attention (MEMSTREAM_TOPK_ATTN, exact top-k pre-softmax in the torch
path; dense storage, per-query dynamic): k=256 -> 0.5724/1.3433, k=1024 ->
0.5726/1.3415 vs kk_2 0.5729/1.2962, swa 0.5543/1.2444. MATCHES kk_2 recall with
zero deletion and the best GT-z measured (7/10 vs kk_2); k-insensitive (matches
the ~150-effective-keys entropy picture). Mechanism refinement: curating encode
attention INPUTS is the lever — static novelty masks and dynamic top-k both
implement it; content-blind lattice doesn't. Cost: torch path 2-3x slower.
Post-RoPE novelty criterion (swa_kk_adjacent_2_postrope): NULL — 0.567/1.279,
criterion is basis-insensitive. Snapshots qwen3_0903_{postrope1,topk1}.

Note (2026-09-03): LVBench@144_160 chunks 2/8 hit the topk guard (odd-aspect
videos realize ~160 tpf -> 63-frame window < topk=64 at n_local=10780); chunks
2, 8, 16 rerun at n_local=11200 (~65-frame window; 2-frame delta vs the arm's
other chunks, inside the ladder's documented 67-75 spread). tpf_budgets' 8%
slack is insufficient at the 160 budget edge — raise the table's n_local to
11200 for future 160-budget arms.

### Token-scaling dilution, precise form (2026-09-03, frame-unit analysis)

frames_at_90 is INVARIANT across tpf (L21: 33 vs 33; f@50: 5 vs 5) while
effective keys grow 2.65x (56->149) — so resolution scaling fragments the same
temporal attention over finer sub-patches of the SAME frames (4.5 vs 1.7
effective keys/attended-frame); total attended pixel-area actually shrinks ~25%
(tokens are 3.5x smaller). With flat performance + the matched-budget selection
result, the claim: tpf scaling fragments attention without broadening it, and
the finer patches' value is only accessible via selection. Tokens are NOT
comparable units across tpf — use frames (invariant) and pixel-area for
cross-resolution attention claims; never raw key counts alone.
