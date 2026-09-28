# Zaratan: GPU partitions, nodes, QoS and running jobs

Reference for the `abhinav2-prj-cmsc` and `abhinav2-prj-aac` accounts. The data comes from
`sinfo`, `scontrol`, `sacctmgr`, `sbalance`, and `sbatch --test-only`, collected on 2026-09-24.
Limits change over time, so re-run the [refresh commands](#8-refresh-commands) to update it.

---

## Rules of thumb (read first)

1. **Default to `abhinav2-prj-cmsc`.** Always submit with `-A abhinav2-prj-cmsc`. Use
   `abhinav2-prj-aac` **only with explicit permission from the user**, and never as a silent
   fallback.
2. **Test new environments on MIG slices first.** When checking that a new environment, CUDA
   build, or GPU-dependent import works (`torch.cuda.is_available()`, flash-attn, custom kernels,
   etc.), use `-p gpu-a100_1g.5gb --gres=gpu:a100_1g.5gb:1`. It costs 7 SU/hr, versus 48 for an
   A100 and 144 for an H100, and still exercises the real CUDA stack. Move to full GPUs once the
   environment is confirmed working.
3. **Memory: A100 = 40 GB, H100 = 80 GB.** If a job needs more than 40 GB of GPU memory, first
   try spreading it across **multiple A100s** (data/tensor/pipeline parallel, FSDP, etc.). Suggest
   H100s only if A100 availability is low. Use H100s sparingly: they cost 3× an A100 per GPU-hour.
4. **Check availability before recommending a GPU type.** The cluster is shared across the whole
   university, and A100/H100 availability swings a lot. Check free GPUs and the pending queue
   (command below) first:
   - Many A100s free: suggest **2+ A100s** (e.g. 2× A100 = 80 GB total for 96 SU/hr, vs 1× H100 = 80 GB for 144 SU/hr).
   - A100s scarce but H100s free: suggest H100s, **with a note that SU usage goes up**.
   - Both scarce: say so, and suggest either queueing on A100s or cutting the job size.
5. **Keep debug and interactive jobs to 4 hours or less.** Idle interactive sessions still burn
   SUs (an idle H100 node for 8 hours costs about 4.6 kSU). Use `-t` of 4 hours or less for `srun --pty`
   and debug runs, and exit the session when you're done.

Check current GPU availability:

```bash
sinfo -h -N -p gpu-a100,gpu-h100,gpu-a100_1g.5gb -O "Partition:20,StateCompact:10,Gres:40,GresUsed:50" \
  | awk '{split($3,t,":"); split($4,u,":"); tot=t[3]+0; used=u[3]+0; up=($2!~/drain|drng|down|maint|resv/);
          T[$1]+=tot; F[$1]+=(up?tot-used:0)} END{for(p in T) printf "%-18s free %3d / %3d GPUs\n",p,F[p],T[p]}'
squeue -h -p gpu-a100,gpu-h100,gpu-a100_1g.5gb -t PD -o "%P" | sort | uniq -c   # pending jobs per partition
```

Example output (2026-09-24): `gpu-a100 free 23 / 76`, `gpu-h100 free 0 / 32`,
`gpu-a100_1g.5gb free 21 / 28`. In that state the recommendation would be multiple A100s.

Availability is not the same as your start time: fairshare on cmsc can be low, so also
compare estimated starts before choosing (nothing is submitted):

```bash
sbatch --test-only -A abhinav2-prj-cmsc -p gpu-a100 --gres=gpu:a100:2 -c 8 --mem=64G -t 4:00:00 --wrap=hostname
sbatch --test-only -A abhinav2-prj-cmsc -p gpu-h100 --gres=gpu:h100:1 -c 8 --mem=64G -t 4:00:00 --wrap=hostname
```

On 2026-09-27 this put 1× H100 four to six days out and 2× A100 about one day out.

---

## 1. Our accounts

| Account | Parent | SU limit | Used (2026-09-24) | Allowed QoS | Default QoS |
|---|---|---|---|---|---|
| `abhinav2-prj-aac`  | `aac`  | 550 kSU (`GrpTRESMins billing=33,000,000`) | 1.18 kSU (0.2%) | `normal`, `scavenger` | `normal` |
| `abhinav2-prj-cmsc` | `cmsc` | 300 kSU (`GrpTRESMins billing=18,000,000`) | 27.98 kSU (9.3%) | `normal`, `scavenger` | `normal` |

- 1 SU = 1 billing-unit-hour (the limit is stored in billing-minutes, so 33,000,000 / 60 = 550k).
- Check live balances with `sbalance`.
- Always pass `-A abhinav2-prj-cmsc`. `abhinav2-prj-aac` requires explicit user permission (see rule 1).

---

## 2. GPU hardware

| Partition | Nodes | # nodes | GPUs / node | Total GPUs | CPU | Cores / node | RAM / node | Local NVMe | InfiniBand |
|---|---|---|---|---|---|---|---|---|---|
| `gpu-h100` | `gpu-a6-[2-9]` | 8 | 4× H100 (80 GB) | 32 | Intel Xeon 8468 (2 × 48c) | 96 | 500 GB | yes (~12 TB tmp) | yes |
| `gpu-a100` | `gpu-b9-[1-7]`, `gpu-b10-[1-7]`, `gpu-b11-[1-5]` | 19 | 4× A100 (40 GB) | 76 | AMD EPYC 7763 (8 NUMA × 16c) | 128 | 500 GB | yes (~12 TB tmp) | yes |
| `gpu-a100_1g.5gb` | `gpu-b11-6` | 1 | 28× A100 MIG slice `1g.5gb` (1/7 of an A100, 5 GB) | 28 | AMD EPYC 7763 | 128 | 500 GB | yes | yes |
| `gpu-v100` | `gpu-a5-1` | 1 | 4× V100 | 4 | Intel Xeon 6248 (2 × 20c) | 40 | 1.5 TB | no (120 GB tmp) | **no** |
| `gpu` (umbrella) | all of the above | 29 | — | 140 | — | — | — | — | — |

Notes:
- The `1g.5gb` MIG profile only exists on 40 GB A100s, which is how we know the A100s are the
  40 GB version. The H100s have 80 GB each. Slurm does not report V100 memory; run `nvidia-smi`
  inside a job to see it.
- `gpu-a5-1` (V100) has **no InfiniBand**, so it is not suitable for multi-node jobs.
- Useful `--constraint` features: `nvme`, `beeond`, `ib`, `epyc_7763`, `xeon_8468`.

---

## 3. GPU partitions

Every GPU partition has `MaxTime=7-00:00:00`, `DefaultTime=00:15:00` (**always set `-t`**),
`AllowAccounts=ALL`, and `AllowQos=ALL`.

| Partition | Partition QoS (applied automatically) | Billing weights (per unit) |
|---|---|---|
| `gpu`             | `gpu`             | CPU=1.0, Mem=0.25/GB, a100=48, a100_1g.5gb=7, v100=12, h100=144 |
| `gpu-h100`        | `gpu-h100`        | CPU=6, Mem=1.125/GB, h100=144 |
| `gpu-a100`        | `gpu-a100`        | CPU=1.5, Mem=0.375/GB, a100=48 |
| `gpu-a100_1g.5gb` | `gpu-a100_1g.5gb` | CPU=1.5, Mem=0.375/GB, a100_1g.5gb=7 |
| `gpu-v100`        | `gpu-v100`        | CPU=1.2, Mem=0.03125/GB, v100=12 |

**`-p gpu` is rerouted.** A submit filter moves a job sent to `-p gpu` into the matching
per-type partition. For example, `-p gpu --gres=gpu:a100:1` went to `gpu-a100` in a
`--test-only` run. You can target the specific partition directly instead.

---

## 4. QoS table

Our associations list only `normal` and `scavenger`. On GPU partitions, the **partition QoS**
(section 3) is added on top of the job QoS and is where the GPU caps come from.

| QoS | Priority | MaxWall | Per-user limit (`MaxTRESPU`) | MaxSubmit / user | Usage factor | Can preempt | Relevant to us? |
|---|---|---|---|---|---|---|---|
| `normal`          | 10 | 14-00:00:00 | — | 5000 | 1.0 | `scavenger` | **Yes** (default) |
| `scavenger`       | 1  | 14-00:00:00 | — | 5000 | **0.0 (free)** | — | CPU only (see below) |
| `gpu`             | 10 | 14-00:00:00 | gpu=40, a100=38, h100=29, a100_1g.5gb=10 | 5000 | 1.0 | `scavenger` | Yes (partition QoS of `gpu`) |
| `gpu-h100`        | 10 | 7-00:00:00  | h100=16 | 5000 | 1.0 | `scavenger` | Yes (partition QoS) |
| `gpu-a100`        | 10 | 7-00:00:00  | a100=38 | 5000 | 1.0 | `scavenger` | Yes (partition QoS) |
| `gpu-a100_1g.5gb` | 10 | 7-00:00:00  | a100_1g.5gb=10 | 5000 | 1.0 | `scavenger` | Yes (partition QoS) |
| `gpu-v100`        | 10 | 7-00:00:00  | — (all 4) | 5000 | 1.0 | `scavenger` | Yes (partition QoS) |
| `high`            | 20 | 3-00:00:00  | — | 5000 | 1.0 | `scavenger` | No (not in our association) |
| `gpu-h100-hi`     | 10 | 7-00:00:00  | h100=16 | 5000 | 1.0 | `scavenger` | No |
| `gpu-a100-hi`     | 10 | 7-00:00:00  | a100=20 | 5000 | 1.0 | `scavenger` | No |
| `gpu-v100-hi`     | 20 | 7-00:00:00  | v100=1  | 5000 | 1.0 | `scavenger` | No |
| `gpu-a100_1g_5gb` | 10 | 7-00:00:00  | a100_1g.5gb=28 | 5000 | 1.0 | `scavenger` | No (appears unused; note the underscore spelling) |

### Effective per-user GPU caps

| GPU type | Cap per user (all running jobs together) |
|---|---|
| H100 | **16** |
| A100 | **38** |
| A100 MIG 1g.5gb | **10** |
| V100 | 4 (the whole node) |
| Any GPU type combined | 40 (from the `gpu` QoS) |

An unthrottled array still runs at most this many GPUs at once: a 61-task 2× A100 array
has at most 19 tasks running (38 A100s).

### No free GPU scavenging
`scavenger` QoS is free (UsageFactor 0) and preemptible, but GPU jobs are rejected from it:

```
sbatch: error: GPU jobs may not be submitted to the scavenger partition.
```

This happens with both `-p scavenger` and `--qos=scavenger` on `-p gpu`. Every GPU job is
charged to the allocation.

---

## 5. How GPU jobs are charged

The cluster uses `PriorityFlags=MAX_TRES`, so a job's billing rate is the **largest** of its
weighted CPU, memory, and GPU values, not their sum.

| Partition | 1 GPU = | "Free" CPU per GPU | "Free" memory per GPU | Full node / hr |
|---|---|---|---|---|
| `gpu-h100`        | 144 SU/hr | 24 cores | 128 GB | 576 SU |
| `gpu-a100`        | 48 SU/hr  | 32 cores | 128 GB | 192 SU |
| `gpu-a100_1g.5gb` | 7 SU/hr   | ~4 cores | ~18 GB | 196 SU |
| `gpu-v100`        | 12 SU/hr  | 10 cores | 384 GB | 48 SU (CPU-bound: 40 × 1.2) |

As long as you stay within the "free" CPU and memory per GPU, you pay only the GPU rate. If you
go over, the CPU or memory term becomes the largest and sets the rate.

Rough budget comparison:
- **H100:** 144 SU/hr. The cmsc allocation (300 kSU) is about 2,080 H100-hours.
- **A100:** 48 SU/hr. The cmsc allocation is about 6,250 A100-hours.
- **MIG 1g.5gb:** 7 SU/hr. The cmsc allocation is about 42,800 slice-hours.
- An H100 costs **3×** an A100 per hour, so use it only when it runs more than 3× faster or you
  need the extra memory.

---

## 6. Environment: network, containers, paths

- **Network.** Only login nodes reach the internet. Compute nodes cannot `git fetch`/`push`,
  `pip install` or download models, so do all of that on a login node.
- **Containers.** GPU work runs in the Apptainer image built from
  `proj_context/setup/morph_graphs.def` (build notes: `proj_context/setup/container_setup.md`).
  - `apptainer` is a module, not on `PATH` by default. Job scripts must
    `module load apptainer`: `sbatch` copies the submitting shell's environment, so a script
    that relies on it works from one shell and fails from another.
  - Run with `apptainer exec --nv --cleanenv --bind /scratch …`. `--cleanenv` keeps host
    module/compiler variables from breaking Triton inside the image; pass anything you need
    with `--env` (e.g. `--env PYTHONPATH=…` for pure-Python extras installed outside the image).
- **ffmpeg** is inside the image. On the host it is `module load ffmpeg/6.0/gcc/11.3.0/`.
- **Paths** (`~/scratch` = `/scratch/zt1/project/abhinav2-prj/user/vatsalag`):

  | what | where |
  |---|---|
  | image | `~/scratch/morph_graph/morph_graphs.sif` |
  | extra pure-Python packages | `~/scratch/pyextra_morph_graphs` |
  | model weights | `~/scratch/model_hub/` (Qwen3.8-27B, Qwen3-VL-8B-Instruct, perception_models) |
  | HD-EPIC videos | `~/scratch/datasets/HD-EPIC/videos/P0x/` |
  | epic_eval runs + shared encoded clips | `~/scratch/epic_eval_runs/` (`media/` holds the clips) |

---

## 7. Example submissions

```bash
# Step 1: test the environment on a cheap MIG slice
srun -A abhinav2-prj-cmsc -p gpu-a100_1g.5gb --gres=gpu:a100_1g.5gb:1 -c 4 --mem=16G -t 1:00:00 --pty bash
#   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1× A100 batch job
sbatch -A abhinav2-prj-cmsc -p gpu-a100 --gres=gpu:a100:1 -c 32 --mem=120G -t 1-00:00:00 job.sh

# More than 40 GB of GPU memory needed: prefer 2× A100 (80 GB total, 96 SU/hr)
sbatch -A abhinav2-prj-cmsc -p gpu-a100 --gres=gpu:a100:2 -c 64 --mem=240G -t 1-00:00:00 job.sh

# Only when A100s are scarce: 1× H100 (80 GB, 144 SU/hr)
sbatch -A abhinav2-prj-cmsc -p gpu-h100 --gres=gpu:h100:1 -c 24 --mem=120G -t 1-00:00:00 job.sh

# Interactive/debug A100 session (4 hours or less)
srun -A abhinav2-prj-cmsc -p gpu-a100 --gres=gpu:a100:1 -c 8 --mem=64G -t 2:00:00 --pty bash

# Estimate start time without submitting
sbatch --test-only -p gpu-h100 --gres=gpu:h100:1 -t 1:00:00 --wrap=hostname
```

For epic_eval, the orchestrator writes and submits the job scripts from the config's `slurm`
block; see `epic_eval/README_local_models.md`.

---

## 8. Refresh commands

```bash
sinfo -o "%P|%a|%l|%D|%G|%c|%m|%f"                   # partitions + GRES
sinfo -s                                              # alloc/idle/other/total
scontrol show partition gpu-h100                      # partition details + billing weights
scontrol show node gpu-a6-2                           # node details
sacctmgr -P show qos format=Name,Priority,MaxWall,MaxTRESPU,MaxSubmitPU,Preempt,UsageFactor
sacctmgr -P show assoc where user=$USER format=Account,QOS,DefaultQOS
sbalance                                              # SU balances
squeue -u $USER                                       # your jobs
```
