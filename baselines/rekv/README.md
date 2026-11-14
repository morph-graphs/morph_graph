# baselines/rekv: ReKV on EPIC-Morph, with Qwen3-VL-8B

> **The vendored snapshot may be re-synced.** `vendor/MemStream` is a subset (15 files) of Vatsal Agarwal's
> working tree (his commit `0efaa6e` plus uncommitted changes, `vendor/SOURCE.md`). Which state is canonical is being
> confirmed; once it is, the snapshot is copied again. Every answer row records a hash of the snapshot
> (`memory_hash`, over `vendor/SOURCE.sha256` and the settings), so answers made with another snapshot count as
> not done and are redone.

[ReKV](https://github.com/Becomebright/ReKV) (Streaming Video Question-Answering with In-context Video KV-Cache
Retrieval, ICLR 2025, arXiv 2503.00540) is training-free: a Video-LLM reads the video as a stream with
sliding-window attention, the key-value cache of every frame is offloaded to RAM, and a question retrieves the
caches of the most relevant frames, which the model then attends to as if it had just read them. There is no
second model and no API: the memory model is the answering model. Our sheet lists ReKV with Qwen3-VL-8B.

Upstream supports LLaVA-OneVision, LongVA and Video-LLaVA only. **The Qwen3-VL port used here is Vatsal
Agarwal's**, from his ReKV fork MemStream; the files a run needs are copied into `vendor/MemStream`, unmodified
except for import-only edits in 4 of them (`vendor/MemStream.patch`: imports of code for other models and
research features are removed, and the branches that would use it raise "not vendored"). It is run in its plain
ReKV configuration: `prefill_method swa`, `retrieval_method rekv`, none of the fork's research knobs
(`MEMSTREAM_*`, compression, ToMe, top-p attention, external retrievers); `epic/run_memstream.py` refuses to
run when one is set. Nothing imports upstream ReKV, so it is not a submodule.

| setting | paper / upstream | here |
|---|---|---|
| sampling | 0.5 fps, 196 tokens per frame (LLaVA-OneVision) | 0.5 fps, the paper's rate, at 448 px. Qwen3-VL pools two frames into one feature, so one stored feature is 196 visual tokens spanning 4 s (plus the model's `<t seconds>` stamp and vision start/end tokens, ~8) |
| sliding window `n_local` | 15000 tokens | 15000 |
| retrieval | per layer, the mean question query against the mean key of each frame, top 64 frames, question text only | the same over stored features: top 64 features = 128 frames (256 s of video). Scored by cosine similarity, which is required here because Qwen3 normalises queries and keys (QK-norm) |
| answer | `Question/Options/Only give the best option.` + prefill `Best option: (`, greedy, 16 tokens | the same |
| frames per forward pass (`encode_chunk_size`) | 64 | 16, to fit 48 GB of GPU memory. Open question for Vatsal: his default is 64, and he measured that this setting changes answers through numerics alone |

## Layout

| path | what |
|---|---|
| `vendor/MemStream/` | the vendored port: `model/qwen3_vl_rekv.py` and what a run of it needs (15 files). `vendor/SOURCE.md`: source path, commit, date, the kept files and their state; `vendor/MemStream.patch`: the only differences from the source (4 files, imports); `vendor/SOURCE.sha256`: the source's hashes of the kept files, and ours of the 4 edited ones |
| `epic/run_memstream.py` | one video: encode once with the port, answer a bank's questions with the streaming cutoff; `--blind` |
| `run_rekv.sh` | one video, one arm, one stage (prep, build, qa, blind, upload, all) |
| `env/` | the environment, pinned to the versions the port was developed with (torch 2.6.0, transformers 5.0.0): `requirements.txt`, `setup_env.sh` |
| `docker/` | the image (`Dockerfile`), `build.sh`, `run.sh` (one job in a container; the job board's entry point), `image_tag.sh` |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset |

Arm `m8` = Qwen/Qwen3-VL-8B-Instruct (bfloat16): 48 GB of GPU memory, as one 48 GB GPU or two 24 GB ones (the
port loads with `device_map="auto"`), and host RAM that grows with the video: 18 GB for a 27 min video, 36 GB for
the longest one, 75 min ("On the job board" below). There is no `m27` arm:
the port is written against Qwen3-VL's decoder.

## Running

**Off our clusters, run only in the Docker image** (rule in `AGENTS.md`): `ghcr.io/morph-graphs/rekv:m8-<env
hash>`, built from `docker/Dockerfile` by `.github/workflows/baseline-images.yml`. The code is the mounted
checkout of `main`; `docker/run.sh` picks the image whose tag matches the checkout's environment files:

```bash
docker pull ghcr.io/morph-graphs/rekv:$(bash baselines/rekv/docker/image_tag.sh m8)
REKV_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/rekv/docker/run.sh qa P05-20240425-201444 m8 batch10_agree2
REKV_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/rekv/docker/run.sh blind P05-20240425-201444 m8 batch10_agree2
REKV_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/rekv/docker/run.sh upload P05-20240425-201444 m8
```

`REKV_RUNTIME=apptainer` runs the same image with Apptainer; the job board (`coord/worker.py`) calls `run.sh`
with `MERIT_GPU` / `MERIT_RUNTIME` / `MERIT_SIF_DIR` / `MERIT_HF_CACHE` / `MERIT_WORK`, which it takes when the
`REKV_` ones are unset. On our own clusters, a virtualenv:

```bash
bash baselines/rekv/env/setup_env.sh <venv dir>     # Python 3.11 (PYTHON=...), CUDA 12.6 wheels
export PY=<venv>/bin/python EPIC_MORPH_DIR=<epic-morph checkout>      # ffmpeg and ffprobe on PATH
bash baselines/rekv/run_rekv.sh qa    P05-20240425-201444 m8 batch10_agree2
bash baselines/rekv/run_rekv.sh blind P05-20240425-201444 m8 batch10_agree2
bash baselines/rekv/run_rekv.sh upload P05-20240425-201444 m8
```

The model is `Qwen/Qwen3-VL-8B-Instruct` from the Hugging Face cache, or a local copy in `REKV_MODELS=<dir>`
(`<dir>/Qwen3-VL-8B-Instruct`; `REKV_MODEL_DIR` for `docker/run.sh`).

**One pass, nothing on disk.** `qa` encodes the video once (question-independent) and answers every question
from the cache in host RAM (147 kB per token, about 6.7 MB per second of video at these settings). `build`
therefore has nothing to do and is kept for the common interface; a rerun of `qa` skips answered questions
but encodes the video again. `memory.json` records what was encoded (settings, stored features, sizes, times,
and the snapshot hash). `upload` uses `hf_results.py --memory none`: the folder is `memory.json`, the answers
and `runs.jsonl`.

**Streaming.** The port encodes a whole video per call (its Qwen3-VL path does not support incremental
encoding), so the cutoff is applied at retrieval: a question asked at `t_ask` retrieves only among the stored
features whose last frame is at or before `t_ask` (`run_memstream.py` wraps the port's `_calc_block_topk` and
asserts the result). Attention is causal, so those features' cache is what a stream stopped at `t_ask` would
hold.

`blind` answers the same prompt with the unpatched model and no video.

**Known properties of the port** (from Vatsal's notes in his repository, not re-measured here). Its sliding-window
RoPE table was once reused stale after the window moved; the snapshot rebuilds it whenever the window's position
ids change. The number of frames per forward pass changes answers through numerics alone (he measured about 17%
of answers flipping between 8 and 64 on one benchmark chunk, with no net accuracy change), so `encode_chunk_size`
is part of the recorded settings and of the snapshot hash. His equality check against the unmodified model was
run with every feature inside the window and retrieved.

## What comes back (in the Hugging Face dataset)

Per video, at `rekv/<arm>/<video>/`:
- `memory.json`: version, settings, model, frames, stored features (`blocks`), tokens per feature, KV-cache size,
  encode time, GPU peak, snapshot hash;
- `answers_<bank>.json.gz`: one row per question in bank order, keyed by the bank's `key` (plus `qhash`), with the
  chosen option (`response`), the raw text, `evaluate`, `t_ask`, how many features were visible, and `retrieved`
  (start second of each retrieved feature -> number of layers that retrieved it), `version` and `memory_hash`;
- `blind_<bank>.json.gz`: the same rows with no video;
- `runs.jsonl`: one line per finished stage (commits, model, torch and transformers versions, GPUs, seconds).

## On the job board

`coord/master.py` lists one `qa` task per bank of its `BANKS` and video (`rekv-m8-qa-<bank>-<video>`), with no
build to wait for: a task is done when `rekv/m8/<video>/answers_<bank>.json.gz` is in the dataset. `blind` is not a
board stage. A worker takes the tasks when started with `rekv` in `--baselines`; it needs this baseline's image
(with Apptainer, `rekv_m8.sif` in `MERIT_SIF_DIR`) and the model in `MERIT_HF_CACHE`. In a worker's queues file the
plain `m8` names this arm as it does the other baselines': one 48 GB GPU and 40G of RAM are enough.

Measured through `docker/run.sh` with Apptainer, as the worker runs a task (2026-11-14, upload to the outbox):

| video | GPU | stored features, KV-cache | GPU peak | process RAM peak (anonymous) | job | answers |
|---|---|---|---|---|---|---|
| P05-20240425-201444, 27 min, `batch10_agree2` | RTX A6000, job with 40G | 407, 11.0 GB | 26.6 GB | 18.0 GB (14.2 GB) | 8 min | 59 of 59, all with a valid option |
| P03-20240216-185832, 75 min (the longest), `t2_agree2` | A100 80 GB, job with 96G, then 40G | 1121, 30.2 GB | 26.7 GB | 36.0 GB (35.3 GB) | 13 min, 11 min | 139 of 139, all valid; the two runs gave the same answers and retrieved features |
