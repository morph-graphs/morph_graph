# baselines/magic: MAGIC-Video on EPIC-Morph

[MAGIC-Video](https://github.com/lijiazheng0917/MAGIC-video) (arXiv 2605.08271; Apache-2.0; a fork of WorldMM with
EgoRAG and HippoRAG) builds a multimodal memory graph of a video (30 s captions, 3 min / 10 min / 1 h summaries,
OpenIE entities, consolidated semantic triples, VLM2Vec clip embeddings) plus a Narrative Memory Chain (topic and
event chains), and answers by retrieving from it over several rounds. Here every LLM role (summaries, extraction,
retrieval, answering) is the arm's backbone, served locally; no paid API.

## Layout

| path | what |
|---|---|
| `MAGIC-video/` | submodule: [morph-graphs/MAGIC-video](https://github.com/morph-graphs/MAGIC-video), branch `epic-morph` = upstream `632d7fc` + our patches |
| `epic/hf_server.py` | a small OpenAI-compatible server over the arm's backbone (HF transformers, batched); MAGIC's OpenAI client talks to it |
| `epic/build_memory.py` | the question-independent memory of one video (captions, summaries, OpenIE, semantic memory, clip embeddings, graph) |
| `epic/chains.py` | the Narrative Memory Chain at one question cutoff, from the memory before it |
| `epic/run_qa.py` | answers one video's questions of a bank, grouped by cutoff; writes `answers_<bank>.json.gz` |
| `run_magic.sh` | one video, one arm, one stage (prep, build, qa, upload, all) |
| `env/` | the environments, pinned: `wheels-client.txt`, `requirements-client.txt`, `install_requirements.sh` (shared by the image and `setup_env.sh`), `setup_env.sh` |
| `docker/` | the image (`Dockerfile`), `build.sh`, `run.sh` (one job in a container; the job board's entry point), `image_tag.sh` |
| `../merit/` | MAGIC reuses MERIT's clips (`prep`), MERIT's 30 s captions and MERIT's question conversion |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset, or download them |

Our patches (in the submodule's `epic-morph` branch): with `MAGIC_LLM_BASE_URL` set, every OpenAI-client call
(WorldMM's `OpenRouterModel`, the time parser, and the upstream preprocess/eval scripts) goes to that server,
in non-thinking mode, seeded, with bounded output length; lenient JSON parsing for local backbones (a truncated
answer is closed after its last complete element, a bare list is taken for a one-field schema, malformed triples
are dropped instead of failing a passage). MAGIC's EgoLife chain prompts are rewritten for one camera wearer in
their own kitchen (`epic/chains.py`). Settings are `eval/eval_egolife.py`'s paper defaults (unified graph,
`chain_mode=facts`, 5 rounds, top-k episodic 3 / semantic 10 / visual 3, 16 frames per clip).

**Where things live.** The code is this repo (`git submodule update --init baselines/magic/MAGIC-video
baselines/merit/MERIT`). The questions are an epic-morph checkout (`EPIC_MORPH_DIR`): banks from its
`outputs/v3/banks/` (generated with `v3/export_bank.py <bank>` when missing), videos from its
`data/HD-EPIC/videos/` (or `HDEPIC_VIDEOS`). Results go to `baselines/magic/results/<arm>/<video>/` (ignored;
`MAGIC_OUT`) and the `upload` stage puts them in the private dataset
[morph-graphs/epic-morph-results](https://huggingface.co/datasets/morph-graphs/epic-morph-results) at
`magic/<arm>/<video>/`.

## Two arms, two environments

| arm | backbone (every LLM role) | GPUs per job | environments |
|---|---|---|---|
| `m8` | Qwen/Qwen3-VL-8B-Instruct | one GPU >= 40 GB (server and MAGIC share it), or 2 GPUs (server on the first, MAGIC on the second) | `client` only |
| `m27` | Qwen/Qwen3.8-27B (non-thinking) | one 80 GB GPU, or 3+ GPUs (server on all but the last) | `client` + `m27` (server) |

MAGIC's own process needs transformers 4.x (VLM2Vec-V2.0, the visual embedder, does not load with 5.x), so the
backbone runs in a separate server process. The `client` environment (transformers 4.57.6) runs MAGIC for both
arms and also serves Qwen3-VL-8B. The `m27` environment serves Qwen3.8-27B: it is exactly MERIT's m27 environment
(transformers 5.8; with 5.2 Qwen3.8 loads but generates garbage), installed by MERIT's `env/install_requirements.sh`.

**Off our clusters, run only in the Docker images** (rule in `AGENTS.md`): `ghcr.io/morph-graphs/magic:<arm>-<env
hash>`, built from `docker/Dockerfile` by `.github/workflows/baseline-images.yml`. Both images hold the client
environment; `m27`'s also holds the m27 server environment in `/opt/m27` (a venv over the client's torch, with
MERIT's m27 pins), so one container runs the server and MAGIC. The code is the mounted checkout of `main`;
`docker/run.sh` picks the image whose tag matches the checkout's environment files (`docker/image_tag.sh`):

```bash
docker pull ghcr.io/morph-graphs/magic:$(bash baselines/magic/docker/image_tag.sh m8)
MAGIC_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/magic/docker/run.sh build P05-20240425-201444 m8
MAGIC_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/magic/docker/run.sh qa P05-20240425-201444 m8 batch10_agree2
MAGIC_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/magic/docker/run.sh upload P05-20240425-201444 m8
```

`docker/run.sh` mounts this checkout at `/repo`, the epic-morph checkout at `/epic-morph`, the videos at `/videos`
and the Hugging Face cache (`MAGIC_HF_CACHE`, default `~/.cache/huggingface`) at `/hf`; the token comes from
`HF_TOKEN` or that cache. `MAGIC_RUNTIME=apptainer` runs the same image with Apptainer (clusters without Docker;
the GPUs are the job's). The job board (`coord/worker.py`) calls it with `MERIT_GPU` / `MERIT_RUNTIME`, which
`run.sh` takes when the `MAGIC_` ones are unset. Download the models into that cache once (list below).

On our own clusters, virtualenvs:

```bash
bash baselines/magic/env/setup_env.sh client <venv dir>   # Python 3.11 (PYTHON=...), CUDA 12.4 wheels
bash baselines/magic/env/setup_env.sh m27 <venv dir>      # compiles causal-conv1d: CUDA_HOME with nvcc
```

Models: `Qwen/Qwen3-VL-8B-Instruct` or `Qwen/Qwen3.8-27B`, `Qwen/Qwen3-Embedding-4B`, `VLM2Vec/VLM2Vec-V2.0` (a LoRA
on `Qwen/Qwen2-VL-2B-Instruct`), from the Hugging Face cache, or from a folder of local copies named after the
repos (`MAGIC_MODELS=<dir>`: `<dir>/Qwen3-VL-8B-Instruct`, ...; the local VLM2Vec copy's `adapter_config.json`
must point at the local Qwen2-VL-2B-Instruct).

## Running

```bash
export PY=<client venv>/bin/python EPIC_MORPH_DIR=<epic-morph checkout>
# m27 also: MAGIC_SERVER_PY=<m27 venv>/bin/python
bash baselines/magic/run_magic.sh build P05-20240425-201444 m8                  # the memory, no questions needed
bash baselines/magic/run_magic.sh qa    P05-20240425-201444 m8 batch10_agree2   # chains per cutoff + answers
bash baselines/magic/run_magic.sh upload P05-20240425-201444 m8
```

**The memory is question-independent; the chains are per cutoff.** `build` makes everything that does not
depend on the questions, so memories can be built for every video before its questions exist. It first takes
the memory from the dataset when it is there (`MAGIC_FETCH=0` to skip that). Its 30 s captions are MERIT's for
the same arm (same backbone): taken from the dataset's `merit/<arm>/<video>/`, or, when MERIT has none, built
here by `../merit/run_merit.sh build` (with `MERIT_PY`, default the server's python). MAGIC's Narrative Memory
Chain is a whole-video scan upstream, which would show a streaming question the future; so `qa` builds the chains
at each question cutoff (ceil(t_ask)) from the captions, 3 min summaries and semantic groups that end before it,
caches them in `chains/c<seconds>/`, and every bank asking at the same second reuses them. The graph is filtered
to nodes ending before the cutoff and the semantic memory to the last snapshot before it (MAGIC's own query-time
filter). So a question sees only memory up to its `t_ask`.

`qa` retries questions MAGIC failed on (up to three passes; rerun the same command for more). Every stage skips
finished work. `MAGIC_TAG=<name>` writes `answers_<bank>_<name>.json.gz`. There is no blind stage: the
no-memory floor depends only on the backbone and the questions, and MERIT's `blind` stage for the same arm (same
model, same question rows) already measures it (`merit/<arm>/<video>/blind_<bank>.json` in the dataset). The clips, MERIT's files, index caches and server logs stay in `baselines/magic/work/`
(`MAGIC_WORK`).

## What comes back (in the Hugging Face dataset)

Per video, at `magic/<arm>/<video>/`:
- the memory: `<video>_{30sec,3min,10min,1h}.json`, `openie_results_local.json`,
  `semantic_{extraction,consolidation}_results_local.json`, `visual_embeddings.pkl`, `graph.pkl` (clip paths in
  it are relative to the clips folder);
- `chains/c<seconds>/{topic_chains,step3_enriched_chains}.json` and `chains/topic_cache.json`;
- `answers_<bank>.json.gz`: one row per question in bank order, keyed by the bank's `key` (plus `qhash`, a hash of
  question and options, and the positional `ID`), with the chosen option (`response`), `evaluate`, the cutoff and
  MAGIC's retrieval rounds (`round_history`: each round's query and the retrieved items with their time stamps),
  `version` and the `memory_hash` the answer was made from (a row from another version or memory is redone);
- `runs.jsonl`: one line per finished stage (commits of this repo, the submodule and epic-morph; backbone; torch
  and transformers versions of MAGIC and of the server; GPUs; seconds).
