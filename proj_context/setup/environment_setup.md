# Packaging a conda, uv or venv environment as an Apptainer image (Zaratan)

A generic recipe for turning a Python environment that works on your machine or
the cluster into a `.sif` image that every job runs the same way. It generalizes
the `morph_graphs` build ([container_setup.md](container_setup.md),
[morph_graphs.def](morph_graphs.def)); read that for the full story and the
error table.

**Tested:** the conda path (that is how `morph_graphs.sif` was built). The uv and
plain-venv variants use the same build steps and the same fixes, but have not
been built yet. Treat their `%post` blocks as a starting point.

## Why a container, and what shapes the recipe

- Everyone runs the same stack regardless of what they use locally (conda, uv,
  a venv), and jobs don't depend on modules or on whatever is in your `.bashrc`.
- Three Zaratan facts drive every choice below:
  1. **Only login nodes have internet**, and they have no GPU. Download and build there;
     test on GPU nodes.
  2. **No root.** `apptainer build --fakeroot` works anyway.
  3. **The login node caps you at 256 processes/threads.** Anything that starts one
     thread per core (mksquashfs, micromamba, uv, parallel compiles) must be throttled.

## 1. Freeze the environment you have

Write down exactly what works, before building anything.

| tool | export | notes |
|---|---|---|
| conda | `conda env export --no-builds > environment.yml` | Delete the `prefix:` line. Keep the `pip:` section; micromamba installs it. |
| uv | commit `pyproject.toml` + `uv.lock` (or `uv pip compile requirements.in -o requirements.lock`) | The lock file is the source of truth. |
| venv | `pip freeze > requirements.txt` | Remove editable/local entries (`-e ...`, `file://...`). |

Then pull these out of the list, because they need special handling in the image:

- **torch and friends** (`torch`, `torchvision`, `torchaudio`, `torchcodec`, `triton`):
  installed from the PyTorch wheel index for your CUDA (`.../whl/cu126`), first,
  and then pinned with a constraints file so nothing replaces them.
- **CUDA extensions that compile** (`flash-attn`, `causal-conv1d`, custom kernels):
  installed last, with `--no-build-isolation`, after torch is in place.
- **System libraries** (e.g. ffmpeg for torchcodec): from conda-forge or apt, before pip.

Keep a `pinned.txt` of the exact versions next to the def file
(see [pinned.txt](pinned.txt)).

## 2. Pick the base image

Use `nvidia/cuda:<cuda>-devel-ubuntu22.04` with the **same CUDA as your torch
wheels** (cu126 → `12.6.3`). `devel`, not `runtime`: compiling extensions needs
`nvcc`. No GPU or host CUDA is needed to build; set `TORCH_CUDA_ARCH_LIST` to the
GPUs you will run on (Zaratan: `8.0` A100, `9.0` H100; add `8.6;8.9` for
A6000/L40S elsewhere).

## 3. Write the definition file

All three variants share this skeleton; only the "install Python + packages"
part differs. Copy your lock/requirements files in with `%files`, and put the
package-manager binary there too (downloading it inside the build is fragile).

```singularity
Bootstrap: docker
From: nvidia/cuda:12.6.3-devel-ubuntu22.04

%environment
    export PATH=/opt/env/bin:$PATH
    export CUDA_HOME=/usr/local/cuda
    export CC=gcc CXX=g++                  # host compilers must not leak in (see section 6)
    export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

%files
    /scratch/.../<tool binary> /usr/local/bin/<tool>     # host path (absolute) -> image path
    /scratch/.../<your lock or env file> /opt/<file>

%post
    # Stay under the login node's 256-process limit. Must come FIRST.
    export MAX_JOBS=2 OMP_NUM_THREADS=1
    export MAMBA_EXTRACT_THREADS=1 MAMBA_DOWNLOAD_THREADS=2                          # micromamba
    export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_INSTALLS=2 UV_CONCURRENT_BUILDS=1 # uv

    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates bzip2 git build-essential && rm -rf /var/lib/apt/lists/*

    # ---- install Python + your packages: pick ONE variant below ----

    export PATH=/opt/env/bin:$PATH CUDA_HOME=/usr/local/cuda
    export TORCH_CUDA_ARCH_LIST="8.0;9.0"

    # Torch first, from the CUDA index; then pin it for every later install.
    cat > /opt/constraints.txt <<'EOF'
torch==2.8.0+cu126
torchvision==0.23.0+cu126
torchaudio==2.8.0+cu126
triton==3.4.0
EOF
    # (variant-specific torch install here; see below)

    # CUDA extensions last, against the installed torch.
    pip install --no-cache-dir -c /opt/constraints.txt ninja packaging wheel
    pip install --no-cache-dir -c /opt/constraints.txt flash-attn==2.8.3 --no-build-isolation

%test
    python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Variant A: conda (micromamba)

Download the static micromamba binary on the login node
(`https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64`)
and `%files` it in with your `environment.yml`.

```bash
    chmod +x /usr/local/bin/micromamba
    # conda layer first: python + system libs (e.g. ffmpeg), so conda can't swap
    # shared libraries under pip packages later
    micromamba create -y -p /opt/env -c conda-forge -f /opt/environment.yml
    micromamba clean --all --yes
    pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 triton==3.4.0 \
        --index-url https://download.pytorch.org/whl/cu126
```

If your `environment.yml` already lists torch under `pip:`, remove it there and
let the line above install it, so it comes from the CUDA index.

### Variant B: uv

Download the uv binary on the login node
(`https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz`,
take the `uv` file) and `%files` it in with your lock file.

```bash
    chmod +x /usr/local/bin/uv
    uv venv /opt/env --python 3.11           # uv downloads a standalone Python (needs internet: build on the login node)
    export VIRTUAL_ENV=/opt/env
    uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 triton==3.4.0 \
        --index-url https://download.pytorch.org/whl/cu126
    uv pip install -c /opt/constraints.txt -r /opt/requirements.lock
    uv pip install pip                       # so the shared "pip install ... flash-attn" lines work
```

With a `pyproject.toml` + `uv.lock` project, copy both in and use
`uv sync --frozen` instead of the requirements line. Configure the PyTorch index
in `pyproject.toml` (`[[tool.uv.index]]`), so the lock already resolves torch
from cu126.

### Variant C: plain venv + pip

Ubuntu 22.04's system Python is 3.10. If you need another version, use variant A
or B to provide the interpreter, or install it from the deadsnakes PPA.

```bash
    apt-get update && apt-get install -y --no-install-recommends python3-venv python3-dev
    python3 -m venv /opt/env
    /opt/env/bin/pip install --upgrade pip
    pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 triton==3.4.0 \
        --index-url https://download.pytorch.org/whl/cu126
    pip install --no-cache-dir -c /opt/constraints.txt -r /opt/requirements.txt
```

## 4. Build (same for every variant)

```bash
# login node: build a sandbox directory, pinned to 4 cores, tests skipped (no GPU here)
export APPTAINER_CACHEDIR=/scratch/zt1/project/abhinav2-prj/user/$USER/apptainer_cache
export APPTAINER_TMPDIR=/scratch/zt1/project/abhinav2-prj/user/$USER/apptainer_tmp
taskset -c 0-3 apptainer build --fakeroot --notest --sandbox my_env_sandbox/ my_env.def

# compute node (CPU is fine; no taskset inside SLURM): convert to a single .sif
apptainer build --fakeroot my_env.sif my_env_sandbox/      # 15-45 min; needs room for a full copy in APPTAINER_TMPDIR
rm -rf my_env_sandbox/                                      # once the .sif works
```

`--sandbox` avoids mksquashfs on the login node, where it dies on the thread limit.
CUDA extensions compile during step 1 and can take an hour or more at `MAX_JOBS=2`.

## 5. Verify on a GPU (use a MIG slice: 7 SU/hr)

```bash
srun -A abhinav2-prj-cmsc -p gpu-a100_1g.5gb --gres=gpu:a100_1g.5gb:1 -c 4 --mem=16G -t 1:00:00 --pty bash
module load apptainer
apptainer exec --nv --cleanenv my_env.sif python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); import flash_attn; print('ok')"
apptainer exec --nv --cleanenv my_env.sif python -c \
  "import triton; print(triton.runtime.driver.active.get_current_target())"
```

Then run one real step of your workload (a forward pass, one batch), since some
kernels only fail when they actually launch.

## 6. Run it

```bash
module load apptainer        # not on PATH by default, including inside sbatch scripts
apptainer exec --nv --cleanenv --bind /scratch \
    --env PYTHONPATH=<optional dir of extra pure-Python packages> \
    my_env.sif python your_script.py
```

- `--cleanenv` is required. Host modules and conda set compiler and path variables
  (`CC`, include paths) that point at host-only locations, and Triton breaks on
  them when it compiles at runtime. Pass what you need with `--env NAME=value` or
  `APPTAINERENV_NAME=value`; a plain `export` does not get through.
- `--bind /scratch` makes models, data and outputs visible; otherwise `/scratch`
  inside the container is the read-only image.
- For epic_eval, put the image in the config's `container` block
  (`epic_eval/README_local_models.md`) and the job scripts do all of the above.

## 7. Changing the image later

- **Pure-Python package missing:** don't rebuild. On a login node, install it into a
  directory on scratch with the image's own pip and add that directory via
  `--env PYTHONPATH=...`:
  `apptainer exec --cleanenv --bind /scratch my_env.sif pip install --target <dir> <pkg>`.
  Check what it pulled in; use `--no-deps` if it brought numpy/torch along.
  Then add the package to the def file for the next rebuild.
- **Anything compiled or torch-related:** rebuild. If the sandbox still exists,
  `apptainer exec --fakeroot --writable my_env_sandbox/ pip install -c /opt/constraints.txt <pkg>`
  and redo only the SIF conversion.
- Commit the def file, `pinned.txt` and your lock/env file to the repo
  (`proj_context/setup/`), so the image can be rebuilt by anyone.

## Quick reference: errors seen building `morph_graphs`

| symptom | fix |
|---|---|
| `mksquashfs ... Failed to create thread` | sandbox build under `taskset -c 0-3`; convert on a compute node |
| `Resource temporarily unavailable` while installing | thread-limit variables at the very top of `%post` |
| `bzip2: Compressed file ends unexpectedly` fetching a tool | download the binary on the host and `%files` it in |
| `taskset: failed to set pid's affinity` | don't use `taskset` inside a SLURM job |
| Triton: gcc path under `/cvmfs/...` not found, or `stddef.h` missing | run with `--cleanenv`; set `CC=gcc CXX=g++` in `%environment` |
| `ModuleNotFoundError` for a package you forgot | `--target` install + `PYTHONPATH` (section 7), then fix the def file |
| `apptainer: command not found` in a job | `module load apptainer` in the job script |
| `Read-only file system: '/scratch'` | add `--bind /scratch` |

Full explanations: [container_setup.md](container_setup.md#troubleshooting-errors-hit-along-the-way).
