# Qwen3-VL similarity / retrieval analysis — runbook

How to regenerate the CG-Bench similarity plots: tokens-per-frame sweep (64 / 256 / 512) and
AKS vs `swa` at 256. Branch `dev/qwen3-vl-integration`,
worktree `/fs/cfar-projects/mem_open_vocab/ReKV_qwen3_wt`.

For the LVBench accuracy runs see `docs/qwen3_vl_lvbench_runbook.md`.

---

## 1. Environment

```bash
source /etc/profile.d/modules.sh
module load cuda/12.6.3 ffmpeg/7.1
source /fs/cfar-projects/mem_open_vocab/miniconda3/etc/profile.d/conda.sh
conda activate memstream_qwen3_cu126
cd /fs/cfar-projects/mem_open_vocab/ReKV_qwen3_wt
```

These are single-node runs, not slurm. The 512 budget needs **4 GPUs**; 64/128/256 fit on 2.

---

## 2. Generate the data

```bash
./analysis/run_sim_sweep.sh results_sim/sweep BV1cF411o72x
```

That runs the four configs below. Each is two separate passes:

```bash
COMMON="--model qwen3_vl_8b --mode single \
  --anno_path data/cgbench/full_mc.json \
  --video_path data/cgbench/videos/BV1cF411o72x.mp4 \
  --retrieve_size 64 --sample_fps 0.5"

# pass 1 -> encoding/  (block self-similarity; STORAGE_ON retains full K/V)
STORAGE_ON=True ENABLE_LOGGING=True python -m analysis.run_similarity_analysis $COMMON \
  --n_local 18500 --min_tokens 200 --max_tokens 256 --prefill_method swa \
  --encoding_only --save_dir results_sim/sweep/tpf256

# pass 2 -> q_000../  (per-question retrieval indices + query-block similarity)
ENABLE_LOGGING=True python -m analysis.run_similarity_analysis $COMMON \
  --n_local 18500 --min_tokens 200 --max_tokens 256 --prefill_method swa \
  --qa_only --save_dir results_sim/sweep/tpf256
```

| config | `--min_tokens/--max_tokens` | `--n_local` | `--prefill_method` | actual tok/frame |
|---|---|---|---|---|
| tpf 64 | `60 / 64` | 5000 | `swa` | 60 |
| tpf 256 | `200 / 256` | 18500 | `swa` | 252 |
| tpf 512 | `448 / 512` | 36000 | `swa` | 464 |
| AKS @ 256 | `200 / 256` | 18500 | `swa_kk_adjacent_4` | 252 |

### Two things that will silently corrupt the comparison

**`n_local` must scale with the budget.** It is sized so every config holds ~70 frames in its
sliding window, which (a) keeps `topk=64` legal and (b) puts all budgets in the *same offload
regime*. Leave it fixed at 17000 and the 64-tpf arm keeps the whole video resident and never
offloads or retrieves anything — that point then measures something different from the others.
Values live in `analysis/tpf_budgets.py`; edit that table, not the shell script.

```bash
python -m analysis.tpf_budgets        # prints the table and validates topk fits
```

**Run the two passes separately.** Combined, the 256 budget was OOM-killed (exit 137) on its last
question, leaving 6 of 7 and skewing that budget's mean. `--encoding_only` needs `STORAGE_ON`;
`--qa_only` must not have it, since that is the memory hog.

---

## 3. Plot

```bash
export SIM_ROOT=$PWD/results_sim/sweep

python results_sim/tpf_scaling_k64/tpf_curves.py      # recall@64 per layer + recall vs budget
python results_sim/tpf_scaling_k64/mech_fig.py        # recall / key uniformity / temporal drift
python results_sim/tpf_scaling_k64/selfsim_fig.py     # 240x240 self-sim matrices, budget x layer
python results_sim/tpf_scaling_k64/perq_fig.py        # per-question recall, every question
python results_sim/tpf_scaling_k64/simprofile_fig.py  # similarity across video + drift slope
python results_sim/aks_at_tpf512/aks_both.py          # AKS vs swa, per question, both budgets
```

`SIM_ROOT` defaults to `<worktree>/results_sim` if unset. (The scripts previously hardcoded an
ephemeral `/tmp/claude-*/scratchpad` path and would not have run elsewhere.)

⚠️ **The `RUNS` / `SRC` dicts at the top of each script still name the original directory layout**
— `results_tpf64k/tpf{64,128,256,512}` and `aks_at_tpf{256,512}/{swa,swa_kk_adjacent_4}`. The sweep
script writes `tpf64 / tpf256 / tpf512 / tpf256_aks`, so either edit those dicts or choose matching
`--save_dir` names. They are not yet parameterized.

---

## 4. Reading the output — two traps

**Use the dispersion of the off-diagonal block similarity, not its mean.** The paper's claim is
that self-similarity matrices become *more uniform* at high tokens-per-frame, which is a statement
about spread. Across our four budgets the off-diagonal **mean is flat (0.574–0.592)** and tests
nothing, while the **std falls monotonically 0.0392 → 0.0335**. Measuring the mean is an easy way
to wrongly conclude the mechanism is absent. Note the two statistics *disagree about AKS* (AKS
lowers both), so treat block similarity as supporting evidence and let recall be the outcome.

**Temporal drift is the reproducible signal; recall is noisy.** The drift slope (rise of
query-block similarity across the video) went +0.09 → +0.14 → +0.22 → +0.39 with the budget and was
positive for **7/7 questions individually**. Per-question recall deltas ranged −0.61 to +0.13 with
two questions moving the *opposite* way, so the recall mean is a population tendency, not something
most questions exhibit.

**Retrieval is causal — never compare recall across runs with different `topk`.** What layer L
retrieves changes the hidden states entering layer L+1, hence layer L+1's query and its retrieved
set. Only **layer 0** is invariant: its keys and queries come straight from the input embeddings
and never pass through attention. Measured between `retrieval_curves` (topk=32) and
`aks_at_tpf512` (topk=64), both at 474 tpf, the fraction of the top-32 set still contained in the
top-64 set falls 1.000 (L0, L1) → 0.987 (L2) → **0.625 (L8)** → ~0.85 (L18–35); under a
non-causal reading it would be 1.000 at every layer. Layer-0 bit-identity is still a valid check
that two runs used the same tokens/frame — it pins the visual tokenization.

Corollary for AKS: **its delta at layer 0 is exactly 0.000** (identical similarities and retrieved
sets, all 7 questions). `swa_kk_adjacent_4` compresses `local_k/local_v`, the encoding-time sliding
window, while retrieval scores against `self.block_k` built from the *uncompressed*
`global_remainder` (`model/attention/kv_cache_manager.py:864`). AKS cannot re-rank blocks directly;
every recall gain it shows is an indirect effect of altered hidden states at layers ≥1.

Also: `results_sim/tpf_scaling_k64/raw/tpf512` and `results_sim/aks_at_tpf512/swa` are **byte
identical** (same md5, different inode, mtimes 2.3 h apart) — one is a copy. They are not two
independent runs and cannot serve as an FP-noise baseline.

---

## 5. What these runs showed (one video, 7 questions — directional only)

| tok/frame | mean recall@64 | off-diag std | mean retrieved block (midpoint 119.5) |
|---|---|---|---|
| 60 | 0.539 | 0.0392 | 130.9 |
| 130 | 0.525 | 0.0372 | 135.3 |
| 262 | 0.486 | 0.0335 | 146.1 |
| 474 | 0.400 | 0.0337 | 160.4 |

AKS vs `swa`, matched at `topk=64`: **+0.021** at 252 tok/frame and **+0.023** at 474 — i.e. AKS
does *not* help more at the higher budget, contrary to the initial hypothesis. Which questions it
helps is stable across budgets, which is the more interesting result: both counting questions lose
at both budgets ("how many cucumbers", 11 clue blocks, −0.116 / −0.124).

95% CI on the AKS mean is [−0.062, +0.106] — it includes zero. Everything here is one video.
Powering the AKS effect needs ~20 videos; the TPF contrast needs ~4 (its per-question sd is larger,
0.258 vs 0.091, but so is the effect).

---

## 6. Files

```
analysis/run_sim_sweep.sh          driver for all four configs
analysis/tpf_budgets.py            budget -> n_local table + topk validator
analysis/run_similarity_analysis.py  the analysis entrypoint
analysis/run_tpf_pilot.py          multi-video sweep, per-cell failure isolation (unused so far)

results_sim/tpf_scaling_k64/       topk=64 sweep + its 5 plot scripts   <- the current figures
results_sim/tpf_scaling/           earlier topk=32 sweep (superseded: fixed n_local)
results_sim/aks_at_tpf256/         AKS vs swa @ 252 tok/frame
results_sim/aks_at_tpf512/         AKS vs swa @ 464 tok/frame + aks_both.py
results_sim/retrieval_curves/      swa vs AKS recall curves + plot_recall.py
```

Published views of these figures:
- TPF scaling — https://claude.ai/code/artifact/1d2828b0-30de-4fd3-9757-60d3ca0d21db
- AKS vs swa — https://claude.ai/code/artifact/8e969c25-0d7a-4843-82e3-14beb0b72d57
- retrieval curves — https://claude.ai/code/artifact/de6a9951-5202-4fb4-a794-7fe68a8a3026
