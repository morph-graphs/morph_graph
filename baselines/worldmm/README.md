# baselines/worldmm: WorldMM on EPIC-Morph

[WorldMM](https://github.com/wgcyeo/WorldMM) (Yeo et al., CVPR 2026; Apache-2.0) is a memory agent for long
videos. It builds three memories of a video: episodic (captions at 10 s, 30 s, 3 min and 10 min, indexed by a
HippoRAG graph per granularity), semantic (triples consolidated as the video goes on) and visual (VLM2Vec clip
embeddings). It answers by deciding, over up to five rounds, which memory to search and with what query, then
answering from what it retrieved. Here it runs on EPIC-Morph's HD-EPIC videos and multiple-choice questions,
with one local model for every role (captions, memory construction, retrieval, answers), as WorldMM's
`qwen3vl-8b` setting. It is a port of Vatsal's batch10 adapter.

## Layout

| path | what |
|---|---|
| `WorldMM/` | submodule: [morph-graphs/WorldMM](https://github.com/morph-graphs/WorldMM), branch `epic-morph` = upstream `3a55b65` + our patches (one commit) |
| `epic/prepare_epic.py` | one HD-EPIC video -> a 512 px copy and WorldMM's 10 s segment skeleton; a bank -> WorldMM's question rows |
| `epic/caption_epic.py` | WorldMM's 10 s captions (its own captioning code; HD-EPIC has no transcript) |
| `epic/build_epic.py` | the rest of the memory with WorldMM's own code (`script/videomme/3_build_memory.sh` step by step), then the memory files into the results folder |
| `epic/eval_epic.py` | answers a video's questions, each from the memory up to its `t_ask`; `--blind`: the answer prompt with no context |
| `epic/llm_server.py` | the backbone in its own process (batches WorldMM's concurrent calls; its own transformers) |
| `epic/wmm_paths.py` | puts the submodule's code on `sys.path`; the memory's file names |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset, or download them |
| `run_worldmm.sh` | one video, one arm, one stage (prep, build, qa, blind, upload, all) |
| `env/` | the two environments, pinned: `wheels-*`, `requirements-*`, `install_requirements.sh`, `setup_env.sh` |
| `docker/` | one image per arm with those environments (`Dockerfile`, `image_tag.sh`, `build.sh`), and `run.sh` to run `run_worldmm.sh` in it with Docker or Apptainer |

Our patches (in the submodule's `epic-morph` branch):
- **Backbone server.** With `WORLDMM_SERVER` set, `Qwen3VLModel` loads no model and sends each call (after its
  own prompt preprocessing) to `epic/llm_server.py`, which batches them. Qwen3.8-27B is added as `qwen38-27b`
  (server only, instruct mode).
- **Tolerant JSON parsing** of the model's structured replies (a bare list for a one-list schema, prose or fences
  around the JSON, non-string scalars in string lists): without schema-constrained decoding, Qwen's NER and
  triple replies otherwise fail validation and the memory loses them.
- **Strict streaming.** HippoRAG's OpenIE is computed once per video at build time; at answer time each
  granularity's HippoRAG starts empty and takes OpenIE rows only for the captions it indexes
  (`EpisodicMemory.openie_sources`). Visual retrieval by time range looks only at indexed clips and never past
  the indexed time (upstream scanned every clip, future ones included), with offsets into the one video file.
- **Speed and memory:** the four granularities' episodic searches run in threads; `WORLDMM_EMB_BATCH` caps
  Qwen3-Embedding's batch (OOM on 16 GB GPUs for long videos).
- `WORLDMM_MODEL_DIR`: load models from a folder of model folders instead of Hugging Face ids.
- Each visual search round records which clips it returned (`retrieved_clips`), for evidence scoring.

Settings are WorldMM's Video-MME pipeline defaults (10 s base captions at 1 fps, 30 s / 3 min / 10 min
multiscale captions with the egocentric prompts, 10 frames per clip for VLM2Vec) and its eval defaults
(top-k episodic 3, semantic 10, visual 3; at most 5 rounds; the `qa_egolife` answer template). Decoding is greedy.

**Streaming.** The build stage needs no questions: the memory covers the whole video and is causal by
construction (a caption or semantic snapshot is keyed by the time it ends). Each question then sees only what
ends by its `t_ask`: questions are answered in `t_ask` order, `WorldMemory.index` only grows, and every answer
row records how much of each memory was indexed (`indexed`). So memories can be built for every video before
its questions exist.

**Where things live.** The code is this repo (`git clone --recurse-submodules`, or `git submodule update --init
baselines/worldmm/WorldMM`). The questions are an epic-morph checkout (`EPIC_MORPH_DIR`): banks from its
`outputs/v3/banks/` (generated there with `v3/export_bank.py <bank>` when missing), videos from its
`data/HD-EPIC/videos/` (or `HDEPIC_VIDEOS`). Results are written to `baselines/worldmm/results/<arm>/<video>/`
(ignored) and the `upload` stage puts them in the private Hugging Face dataset
[morph-graphs/epic-morph-results](https://huggingface.co/datasets/morph-graphs/epic-morph-results) at
`worldmm/<arm>/<video>/`.

## Two arms, two environments

| arm | model (every role) | GPU per job | environments |
|---|---|---|---|
| `m8` | Qwen/Qwen3-VL-8B-Instruct | one GPU with >= 40 GB (8B server + the 4B text and 2B visual embedding models) | `client` (also runs the server) |
| `m27` | Qwen/Qwen3.8-27B | one 80 GB GPU, or 2x 48 GB (the server spread over both, WorldMM on the second) | `client` + `server27` |

WorldMM's VLM2Vec code needs transformers 4.x and Qwen3.8 needs 5.8 (5.2 loads it but generates garbage), so
the 27B backbone runs in its own environment. Both are virtualenvs on Python 3.11 with torch 2.6.0 (CUDA 12.4)
and MERIT's flash-attn wheel:

```bash
bash baselines/worldmm/env/setup_env.sh client   <dir>/worldmm-client     # transformers 4.57.6
bash baselines/worldmm/env/setup_env.sh server27 <dir>/worldmm-server27   # transformers 5.8.0; CUDA_HOME for causal-conv1d
```
(`PYTHON` picks the Python 3.11 interpreter.) WorldMM is not installed into them: the scripts put the
submodule's `src/` on the path. The models (`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3.8-27B`,
`Qwen/Qwen3-Embedding-4B`, `VLM2Vec/VLM2Vec-V2.0` and its base `Qwen/Qwen2-VL-2B-Instruct`) come from the
Hugging Face cache, or from `WORLDMM_MODEL_DIR` (a folder of model folders; VLM2Vec's `adapter_config.json`
must then name the local Qwen2-VL-2B-Instruct as its base).

## Running

```bash
export EPIC_MORPH_DIR=<epic-morph checkout> PY=<dir>/worldmm-client/bin/python
bash baselines/worldmm/run_worldmm.sh build P05-20240425-201444 m8                   # the memory, once per video and arm
bash baselines/worldmm/run_worldmm.sh qa    P05-20240425-201444 m8 batch10_agree2    # answers of a bank
bash baselines/worldmm/run_worldmm.sh all   P05-20240425-201444 m8 batch10_agree2    # prep + build + qa + blind
SERVER_PY=<dir>/worldmm-server27/bin/python bash baselines/worldmm/run_worldmm.sh all <video> m27 batch10_agree2
bash baselines/worldmm/run_worldmm.sh upload P05-20240425-201444 m8                  # after the stages finished
```
ffmpeg must be on `PATH` for `prep`. Each GPU stage starts the backbone server once and stops it on exit.
**Build a memory once, reuse it everywhere:** `build` first takes the memory from the dataset when it is there
(`WORLDMM_FETCH=0` to skip that, e.g. offline); every bank and variant (`WORLDMM_TAG=<name>` writes
`answers_<bank>_<name>.json.gz`) answers from it. Every stage skips finished work, so rerun the same command
after a failure; a question that fails is retried by the next pass (three per run). The 512 px videos, build
intermediates, per-question rows and HippoRAG caches stay in `baselines/worldmm/work/` (`WORLDMM_WORK`).

## Docker (off our clusters)

One image per arm, built from the repo root with the same `env/` install as the virtualenvs: `m8` has the client
environment in the image's Python; `m27` adds the server27 environment as a virtualenv at `/opt/server` (the
image sets `PY` and `SERVER_PY`). The code is not in the image: the checkout is mounted at `/repo`.

```bash
bash baselines/worldmm/docker/build.sh m8          # = docker build -f baselines/worldmm/docker/Dockerfile --build-arg ARM=m8 .
WORLDMM_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/worldmm/docker/run.sh all P05-20240425-201444 m8 batch10_agree2
WORLDMM_RUNTIME=apptainer WORLDMM_IMAGE=worldmm_m27.sif EPIC_MORPH_DIR=... bash baselines/worldmm/docker/run.sh all <video> m27 batch10_agree2
```
Tags are `ghcr.io/morph-graphs/worldmm:<arm>-<hash of the environment files>` (`docker/image_tag.sh`), so a
code-only commit needs no new image.

## What comes back (in the Hugging Face dataset)

Per video, at `worldmm/<arm>/<video>/`:
- the memory: `<video>_{10sec,30sec,3min,10min}.json` (captions), `<video>_semantic.json` (semantic snapshots),
  `<video>_visual.pkl` (VLM2Vec clip embeddings), `<video>_openie_{10sec,30sec,3min,10min}.json` (HippoRAG's
  OpenIE per granularity);
- `answers_<bank>.json.gz`: one row per question, in bank order, keyed by the bank's question `key` (and `ID`,
  its 1-based position): the chosen option (`pred`, and the raw `response`), `evaluate` (correct or not), the
  rounds (`round_history`) and the retrieved items per round with their time spans in seconds (`retrieved`;
  episodic captions and visual clips have spans, semantic triples have none), `indexed`;
- `blind_<bank>.json`, when the blind stage ran (WorldMM's final answer prompt with no context);
- `runs.jsonl`: one line per finished stage (this repo's commit, the WorldMM and epic-morph commits, model,
  torch and transformers versions of WorldMM and of the server, GPUs, seconds).

## Checking a new machine first

Run one video into a scratch folder and compare accuracy with ours (fetched with `hf_results.py download
--baseline worldmm`). Decoding is greedy, but captions differ across GPU models (see MERIT's README), and the
answers follow the captions, so compare accuracy, not answers.
