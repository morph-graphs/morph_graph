# Building the `morph_graphs` Apptainer container on Zaratan

This documents how the `morph_graphs_qwen38_cu126` conda environment was turned into a SIF container, the cluster-specific problems that came up, and how to run it. The short version: build a sandbox on the login node (the only place with internet), convert it to a `.sif` on a compute node, and always run with `--cleanenv`.

## Environment being containerized

The container reproduces this working conda env, with versions taken from `pip freeze`:

| Package | Version |
|---|---|
| Python | 3.11 |
| torch / torchvision / torchaudio | 2.8.0 / 0.23.0 / 2.8.0 (cu126 wheels) |
| torchcodec | 0.7.0 (cu126) |
| triton | 3.4.0 |
| transformers | 5.8.0 |
| accelerate | 1.15.0 |
| huggingface_hub | 1.32.0 |
| datasets | 5.0.1 |
| qwen-vl-utils | 0.0.14 |
| flash-attn | 2.8.3 |
| causal-conv1d | 1.7.0 |
| fla-core / flash-linear-attention | 0.5.2 |
| ffmpeg (conda-forge) | 6.1.2 |

## Cluster constraints that shaped the approach

Three facts about the cluster drove most of the decisions:

1. **Only the login node has internet**, and it has no GPU. Compute nodes have GPUs but no internet. So all downloading happens on the login node, and GPU testing happens on compute nodes.
2. **No root, and the user isn't in `/etc/subuid`.** `apptainer build --fakeroot` still works: Apptainer falls back to a root-mapped namespace and runs `%post` under the `fakeroot` command.
3. **The login node caps each user at 256 processes/threads** (`ulimit -u` = 256) on a 128-core machine. Any tool that sizes its thread pool to the core count (mksquashfs, micromamba, parallel compiles) blows through this limit. This caused most of the failures.

No GPU or host CUDA is needed to build. The CUDA 12.6 toolkit comes from the `nvidia/cuda:12.6.3-devel` base image, the torch wheels bundle their own CUDA runtime, and compiling CUDA extensions with `nvcc` is pure CPU work. `TORCH_CUDA_ARCH_LIST` tells the compiler which GPUs to target, since there's no GPU to auto-detect.

## The definition file

`morph_graphs.def`, next to this file (pinned versions in `pinned.txt`):

```singularity
Bootstrap: docker
From: nvidia/cuda:12.6.3-devel-ubuntu22.04

%environment
    export PATH=/opt/env/bin:$PATH
    export CUDA_HOME=/usr/local/cuda
    export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
    # Not in the current image -- add on next rebuild (see "Host environment leakage")
    export CC=gcc
    export CXX=g++

%files
    /scratch/zt1/project/abhinav2-prj/user/vatsalag/micromamba /usr/local/bin/micromamba

%post
    # Stay under the login node's 256-process limit -- must come FIRST
    export MAMBA_EXTRACT_THREADS=1
    export MAMBA_DOWNLOAD_THREADS=2
    export MAX_JOBS=2
    export OMP_NUM_THREADS=1

    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates bzip2 git build-essential && \
        rm -rf /var/lib/apt/lists/*

    # micromamba (copied in via %files)
    chmod +x /usr/local/bin/micromamba
    micromamba --version

    # Conda layer first: python + ffmpeg (shared libs for torchcodec)
    micromamba create -y -p /opt/env -c conda-forge python=3.11 "ffmpeg=6.1.2"
    micromamba clean --all --yes

    export PATH=/opt/env/bin:$PATH
    export CUDA_HOME=/usr/local/cuda
    # A100=8.0, A6000/A5000=8.6, L40S/4090=8.9, H100=9.0 -- trim to what your cluster has
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

    # Constraints stop later installs from silently swapping torch/triton
    cat > /opt/constraints.txt <<'EOF'
torch==2.8.0+cu126
torchvision==0.23.0+cu126
torchaudio==2.8.0+cu126
torchcodec==0.7.0+cu126
triton==3.4.0
EOF

    # PyTorch stack (cu126 wheels)
    pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchcodec==0.7.0 triton==3.4.0 \
        --index-url https://download.pytorch.org/whl/cu126

    # HF stack
    pip install --no-cache-dir -c /opt/constraints.txt \
        transformers==5.8.0 \
        accelerate==1.15.0 \
        huggingface_hub==1.32.0 \
        datasets==5.0.1 \
        qwen-vl-utils==0.0.14 \
        ezcolorlog   # not in the current image -- see "Adding a package you forgot"

    # Build deps for the CUDA extensions
    pip install --no-cache-dir ninja packaging psutil wheel

    # CUDA extensions (try prebuilt wheels first, fall back to source build)
    pip install --no-cache-dir -c /opt/constraints.txt flash-attn==2.8.3 --no-build-isolation
    pip install --no-cache-dir -c /opt/constraints.txt causal-conv1d==1.7.0 --no-build-isolation

    # Gated DeltaNet kernels
    pip install --no-cache-dir -c /opt/constraints.txt \
        fla-core==0.5.2 flash-linear-attention==0.5.2

%test
    python -c "import torch, triton, transformers, flash_attn, fla, causal_conv1d, torchcodec; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('triton', triton.__version__, 'transformers', transformers.__version__); \
print('flash_attn', flash_attn.__version__)"
```

Notes on the design:

- **Conda packages before pip.** Installing ffmpeg via conda after the pip packages risks conda swapping shared libraries underneath them, so the conda layer comes first.
- **No `module load`.** Environment modules don't exist inside the container. The base image supplies nvcc 12.6, which matches the cu126 torch build exactly (the original env compiled extensions with nvcc 12.3).
- **The constraints file** is passed with `-c` to every pip install after the PyTorch step, so no later package can silently replace torch or triton. flash-linear-attention is sensitive to the triton version.
- **micromamba is copied in via `%files`** rather than downloaded during the build (see troubleshooting).

## Step-by-step

### 1. Download micromamba on the login node

```bash
cd /scratch/zt1/project/abhinav2-prj/user/vatsalag
curl -fsSL -o micromamba \
  https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
chmod +x micromamba
./micromamba --version
```

### 2. Build a sandbox on the login node

```bash
export APPTAINER_CACHEDIR=/scratch/.../apptainer_cache
export APPTAINER_TMPDIR=/scratch/.../apptainer_tmp

taskset -c 0-3 apptainer build --fakeroot --notest --sandbox morph_graphs_sandbox/ <morph_graph checkout>/proj_context/setup/morph_graphs.def
```

`--sandbox` produces a plain directory and skips mksquashfs, which can't run under the login node's thread limit. `taskset -c 0-3` pins the build to 4 cores so tools that size themselves to the visible core count stay small. `--notest` skips `%test`, which could fail on a node without a GPU even when the container is fine.

### 3. Convert to SIF on a compute node

Converting needs no internet. In an interactive job:

```bash
cd /scratch/zt1/project/abhinav2-prj/user/vatsalag
apptainer build --fakeroot morph_graphs.sif morph_graphs_sandbox/
```

Don't use `taskset -c 0-3` here: inside a Slurm job you can only pin to the cores the job was allocated, and the job's allocation already limits mksquashfs. Check `nproc` and `grep Cpus_allowed_list /proc/self/status` if you want to pin.

The conversion first copies the whole sandbox into `APPTAINER_TMPDIR` ("Extracting local image..."), then compresses it. The copy is I/O-bound and slow on scratch, while the compression is CPU-bound. Expect 15 to 45 minutes. Make sure the temp location has room for a full copy of the sandbox (check with `df -h`); node-local disk is faster than scratch if available. Monitor progress from a second terminal on the same node:

```bash
du -sh ${APPTAINER_TMPDIR:-/tmp}/build-temp-*
top -u $USER
```

Once the `.sif` works, delete the sandbox (`rm -rf morph_graphs_sandbox/`). It's hundreds of thousands of small files.

### 4. Verify on a GPU node

```bash
apptainer exec --nv --cleanenv morph_graphs.sif python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); \
from flash_attn import flash_attn_func; import fla, causal_conv1d, torchcodec; print('ok')"

apptainer exec --nv --cleanenv morph_graphs.sif python -c \
  "import triton; print(triton.runtime.driver.active.get_current_target())"
```

Expected: `True NVIDIA A100-SXM4-40GB`, `ok` with no fla warnings, and `GPUTarget(backend='cuda', arch=80, warp_size=32)`. Follow up with a short Qwen forward pass on one video, which actually launches the flash-attn and DeltaNet kernels.

## Running jobs

GPU nodes are offline, so download models from the login node first into a cache on scratch. Set `HF_HOME` to a scratch or project path rather than the default `~/.cache/huggingface`: home quotas are small, Qwen checkpoints are tens of GB, and the cache must be readable from compute nodes. If a cache already exists from the conda setup, point `HF_HOME` at it instead of re-downloading.

```bash
export HF_HOME=/scratch/zt1/project/abhinav2-prj/user/vatsalag/hf_cache
apptainer exec morph_graphs.sif hf download <model-id>
```

In sbatch scripts, always use `--cleanenv` and pass the needed variables with the `APPTAINERENV_` prefix (Apptainer strips the prefix inside the container):

```bash
export APPTAINERENV_HF_HOME=/scratch/zt1/project/abhinav2-prj/user/vatsalag/hf_cache
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_HF_DATASETS_OFFLINE=1
apptainer exec --nv --cleanenv -B /path/to/data morph_graphs.sif python run.py
```

`--cleanenv` also strips any `HF_HOME` set in `.bashrc`, which is why it has to be passed explicitly.

## Adding a package you forgot to put in the image

The `.sif` is read-only, so a package missing from the def can't be installed into it after the build. This happened with `ezcolorlog`: the repo imports it everywhere (including `epic_eval/backends/smoke_qwen38.py`), but it wasn't in the def when the image was built, so the first run died with `ModuleNotFoundError: No module named 'ezcolorlog'`. To check whether an image has a package: `apptainer exec morph_graphs.sif python -c "import ezcolorlog"`.

A full rebuild takes hours because of the flash-attn and causal-conv1d compiles. For **pure-Python packages**, install them into a directory on scratch instead and put it on `PYTHONPATH` at run time. Run this on the login node, the only node with internet:

```bash
SIF=/scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graph/morph_graphs.sif
PYEXTRA=/scratch/zt1/project/abhinav2-prj/user/vatsalag/pyextra_morph_graphs

apptainer exec --cleanenv --bind /scratch "$SIF" pip install --target "$PYEXTRA" ezcolorlog
```

`--bind /scratch` is required. Without it, `/scratch` inside the container is part of the read-only image, and pip fails with `OSError: [Errno 30] Read-only file system: '/scratch'`. Using the container's own pip means packages are resolved for the container's Python 3.11.

Then pass the directory in with `--env` on every run:

```bash
apptainer exec --nv --cleanenv --bind /scratch --env PYTHONPATH="$PYEXTRA" "$SIF" python run.py
```

Use `--env` or `APPTAINERENV_PYTHONPATH`, not a plain `export PYTHONPATH`, which `--cleanenv` strips. `epic_eval/backends/smoke_qwen38.slurm` does this, with the directory overridable via `PYEXTRA=...`.

Caveats:

- **Watch the dependencies pip pulls in.** `pip install --target` installs the package's dependencies into the directory too, and anything on `PYTHONPATH` takes precedence over the image's own site-packages. Check `ls "$PYEXTRA"` afterwards. If it pulled in something the image already has (numpy, torch, transformers), delete the directory and reinstall with `--no-deps`, installing only the missing pieces.
- **Compiled or CUDA packages need a rebuild.** Anything that builds against torch or CUDA (like flash-attn) must be compiled in the image with the constraints file. If the sandbox from step 2 still exists, a faster path is to install into it and re-run only the SIF conversion (step 3):
  ```bash
  apptainer exec --fakeroot --writable morph_graphs_sandbox/ \
    pip install --no-cache-dir -c /opt/constraints.txt <package>
  ```
- **This is a stopgap.** Add the package to the def right away so the next rebuild includes it, then drop the `PYTHONPATH` directory.

## Troubleshooting: errors hit along the way

| Symptom | Cause | Fix |
|---|---|---|
| `bzip2: Compressed file ends unexpectedly` when fetching micromamba | `curl -Ls` via the micro.mamba.pm redirect returned empty/HTML; `-s` hid the error | Download the plain binary from GitHub releases on the host and copy it in with `%files` |
| `cp: cannot stat 'micromamba'` during `%files` | Paths reversed | `%files` syntax is `<host path> <container path>`, and the host path should be absolute |
| `mksquashfs ... FATAL ERROR: Failed to create thread` at build start | mksquashfs starts ~128 threads, exceeding `ulimit -u` of 256 | Build with `--sandbox` under `taskset -c 0-3`; convert to SIF on a compute node |
| `unknown flag: --mksquashfs-args` | Not supported by this Apptainer version | Use `taskset` or a sandbox build instead |
| libmamba `Resource temporarily unavailable`, half-extracted package | micromamba's parallel threads hit the same process limit | `MAMBA_EXTRACT_THREADS=1`, `MAMBA_DOWNLOAD_THREADS=2`, set at the **top** of `%post` (setting them after `micromamba create` has no effect) |
| `taskset: failed to set pid's affinity: Invalid argument` on a compute node | Slurm restricts the job to specific cores, and 0-3 weren't among them | Drop `taskset` inside Slurm jobs, or pin to cores from `Cpus_allowed_list` |
| fla warns `Triton is not supported on current platform, roll back to CPU` | Triton failed to compile its runtime helper (see next two rows) | Run with `--cleanenv` |
| Triton `FileNotFoundError` for `/cvmfs/.../gcc-11.3.0/bin/gcc` | Host `CC` (from a loaded module) leaked into the container, pointing at a compiler that only exists on the host | `--cleanenv`; long-term, `export CC=gcc` in `%environment` |
| Triton `fatal error: stddef.h: No such file or directory` | Other host compiler variables from modules and conda leaked in and redirected gcc's include search | `--cleanenv` |
| `ModuleNotFoundError` for a package the repo needs (e.g. `ezcolorlog`) | Package was never added to the def before the build | Install into a scratch directory on `PYTHONPATH` (see "Adding a package you forgot"), then add it to the def |
| pip `OSError: [Errno 30] Read-only file system: '/scratch'` | `/scratch` wasn't bound, so pip tried to write into the read-only image | Add `--bind /scratch` |
| `Could not find any nv files on this host!` and `0 active drivers` | Ran a GPU test on the login node | GPU tests only work on GPU nodes |

## Host environment leakage

Apptainer forwards the host environment into the container by default. On this cluster, loaded modules (the spack gcc 11.3.0 pulled in by the cuda/ffmpeg modules) and the active conda base env set compiler and path variables that point at host-only locations. The torch and flash-attn imports worked regardless, which made this easy to miss, but Triton compiles a small C helper at runtime and broke. Running with `--cleanenv` fixed it completely. The alternative is `module purge && conda deactivate` before `apptainer exec`, but `--cleanenv` doesn't depend on shell state and is more reproducible.

## Next rebuild checklist

1. Add `export CC=gcc` and `export CXX=g++` to `%environment` (already in the def file above).
2. `ezcolorlog` is now in the def. After rebuilding, drop `--env PYTHONPATH=...` from job scripts and delete `pyextra_morph_graphs/`.
3. Trim `TORCH_CUDA_ARCH_LIST` to the GPUs actually in use (`nvidia-smi --query-gpu=name,compute_cap --format=csv`) to speed up any source compiles.
4. For the SIF conversion, point `APPTAINER_TMPDIR` at node-local disk and request more cores, since the copy stage is I/O-bound and the compression stage scales with cores.
5. `proj_context/setup/morph_graphs.def` and `pinned.txt` are the source of truth for rebuilding the image or changing versions; update them in the repo along with any rebuild.