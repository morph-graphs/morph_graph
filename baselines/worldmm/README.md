# baselines/worldmm: WorldMM on EPIC-Morph

[WorldMM](https://github.com/wgcyeo/WorldMM) (Yeo et al., CVPR 2026; Apache-2.0) is a memory agent for long
videos. It builds three memories of a video: episodic (captions at 10 s, 30 s, 3 min and 10 min, indexed by a
HippoRAG graph per granularity), semantic (triples consolidated as the video goes on) and visual (VLM2Vec clip
embeddings). It answers by deciding, over up to five rounds, which memory to search and with what query, then
answering from what it retrieved. Here it runs on EPIC-Morph's HD-EPIC videos and multiple-choice questions,
with one local model for every role (captions, memory construction, retrieval, answers), as WorldMM's
`qwen3vl-8b` setting (arms `m8`, `m27`), or with WorldMM's published OpenAI models (the official arms `g5m`,
`g5m-q8`, `g5m-g5`, below). It is a port of Vatsal's batch10 adapter.

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
| `epic/openai_usage.py` | official arms: counts every OpenAI call's tokens and the model id that answered, for `runs.jsonl` |
| `epic/cost_official.py` | official arms: the cost estimate below, from video lengths and question counts |
| `epic/mock_openai.py`, `test_official.sh` | official arms end to end on a slice of one video against a stand-in API (no paid call) |
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

Local arms (the official arms above run in the `client` environment):

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

## Official arms: WorldMM's published models (OpenAI API, paid)

WorldMM's own setting builds the memory with `gpt-5-mini` (`script/videomme/3_build_memory.sh --model gpt-5-mini`)
and answers with `--retriever-model gpt-5-mini --respond-model gpt-5` (`script/videomme/4_eval.sh`). Our results
sheet has two WorldMM rows, and three arms cover them:

| arm | what | memory model | retriever | respond model | sheet row |
|---|---|---|---|---|---|
| `g5m` | the memory, built once per video, at `worldmm/g5m/<video>/` | gpt-5-mini | | | both rows read it |
| `g5m-q8` | answers at `worldmm/g5m-q8/<video>/` | (reads `g5m`) | gpt-5-mini | Qwen3-VL-8B, local | "Memory Model GPT-5-mini, Reasoning Model Qwen3-VL-8B" |
| `g5m-g5` | answers at `worldmm/g5m-g5/<video>/` | (reads `g5m`) | gpt-5-mini | gpt-5 | "Memory Model GPT-5-mini, Reasoning Model GPT-5" |

**How the sheet's two columns map to WorldMM's three roles (decided by Anubhav, 2026-11-13; it matches the paper's
own rows).** "Memory model" is everything that builds the memory (10 s captions from 10 frames each, the 30 s /
3 min / 10 min captions, episodic triples, semantic extraction and consolidation, HippoRAG's OpenIE) plus the
*retriever* model, which only filters what a search returns (HippoRAG's fact filter per granularity and the
multiscale caption filter). "Reasoning model" is WorldMM's *respond* model: in `WorldMemory.answer` it decides
each round whether to search, which memory and with what query, and then writes the answer, seeing the retrieved
captions, triples and frames. The embedding models (Qwen3-Embedding-4B, VLM2Vec-V2) stay local in every arm, so
each stage still needs a GPU (they take about 13 GB; `g5m-q8` needs 40 GB, for the 8B server too).

Everything else is the same code path as `m8`: WorldMM's prompts and defaults, the strict-streaming patches
(a question's HippoRAG graphs, semantic snapshot and visual clips hold only what ends by its `t_ask`), the
tolerant row format. WorldMM's `OpenAIModel` is used unchanged: `MODEL_DICT` pins `gpt-5-mini-2025-08-07` and
`gpt-5-2025-08-07`, images go as 512 px JPEGs at the API's default detail, and no reasoning effort is set (the
API's default). **OpenAI lists both models for shutdown on 2026-12-11** (deprecations page, read 2026-11-13;
replacements named there: gpt-6.1-terra, gpt-6.1-sol): after that day these arms cannot run as published.

```bash
export OPENAI_API_KEY=...          # in the environment only: never an argument, never written to a file or log
bash baselines/worldmm/run_worldmm.sh build P05-20240425-201444 g5m                       # the memory, once
bash baselines/worldmm/run_worldmm.sh qa    P05-20240425-201444 g5m-g5 batch10_agree2     # answers, respond gpt-5
bash baselines/worldmm/run_worldmm.sh qa    P05-20240425-201444 g5m-q8 batch10_agree2     # answers, respond Qwen3-VL-8B
bash baselines/worldmm/run_worldmm.sh upload P05-20240425-201444 g5m-g5                   # the g5m memory and this arm's answers
```
With Docker the arms run in the `m8` image (`docker/image_tag.sh g5m` prints its tag), the key passed by name:
`OPENAI_API_KEY=... WORLDMM_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/worldmm/docker/run.sh qa <video> g5m-g5 batch10_agree2`.
On the job board these are the `worldmm g5m*` tasks: they carry `needs_env: [OPENAI_API_KEY]`, are listed for pool
`max` only, and a worker takes them only when that variable is set in its environment (`coord/DESIGN.md`).

- **Shared memory.** `build` with any of the three names builds (or fetches) `g5m`; `qa` with an answer arm reads it
  and writes `<video>_memory_ref.json` (the memory's file hashes) next to its answers. If the memory is ever
  rebuilt, the next `qa` or `upload` of an answer arm removes the answers made from the old one.
- **Resuming costs nothing twice.** Captions are saved every 20 segments, every later step skips finished files,
  answered questions are kept per row, and WorldMM's own reply cache (`work/openai_cache/<video>/`, keyed by the
  full prompt) serves any text call that was already paid for. HippoRAG's 10 s OpenIE repeats the episodic-triple
  prompts and comes from that cache.
- **What is recorded.** Each stage's `runs.jsonl` line gets `respond_model` and `openai`: per pipeline step
  (`captions`, `multiscale`, `triples`, `semantic`, `openie`) or role (`retriever`, `respond`), per model id *as the
  API returned it*: calls, input, cached, output and reasoning tokens, service tier; summed over resumed attempts.
- `WORLDMM_OPENAI_SERVICE_TIER=flex` sends every call at the flex tier (half price, slower, same models).
- `blind` with `g5m-g5` is gpt-5 on the answer prompt alone (one call per question); with `g5m-q8` it equals `m8`'s.

### Cost estimate

Estimated 2026-11-13 with `epic/cost_official.py` (standard tier; nothing was run against the API):

| scope | memory `g5m` (once) | answers `g5m-q8` | answers `g5m-g5` | both sheet rows | range |
|---|---|---|---|---|---|
| batch10 (10 videos, 5.7 h, 598 questions) | $14.50 | $18.65 | $57.85 | **$91** | $59 - $171 |
| tranche 2 (20 videos, 12.8 h, 1,499 questions) | $32.68 | $47.18 | $145.43 | **$225** | $147 - $421 |
| the two banks together (what the job board lists) | $47.18 | $65.83 | $203.28 | **$316** | $206 - $592 |
| memories for all 156 videos (41.3 h) + the 2,097 questions | $106.24 | $65.84 | $203.28 | **$375** | $242 - $708 |

Per video the memory costs about $2.55 per video-hour ($1.45 for a 34 min video); per question `g5m-q8` about
$0.031 (the gpt-5-mini retriever) and `g5m-g5` about $0.097 (retriever + gpt-5). `WORLDMM_OPENAI_SERVICE_TIER=flex`
halves all of it. By step, for batch10 (calls; input / output tokens in millions; dollars):

| step | model | calls | input | output | $ |
|---|---|---|---|---|---|
| build: 10 s captions (10 frames each) | gpt-5-mini | 2,042 | 6.85 | 1.20 | 4.12 |
| build: 30 s / 3 min / 10 min captions | gpt-5-mini | 843 | 0.82 | 0.75 | 1.70 |
| build: NER + triples (episodic triples, HippoRAG OpenIE) | gpt-5-mini | 5,770 | 2.74 | 2.32 | 5.32 |
| build: semantic extraction + consolidation | gpt-5-mini | 4,010 | 2.12 | 1.41 | 3.35 |
| qa: retriever (fact filter x4 + multiscale filter per episodic search), per answer arm | gpt-5-mini | 8,252 | 38.03 | 4.57 | 18.65 |
| qa: respond (4.4 rounds + the answer per question), `g5m-g5` only | gpt-5 | 3,235 | 8.83 | 2.82 | 39.19 |

How it was computed, and what is measured or assumed:
- Prices: gpt-5-mini $0.25 in / $2.00 out, gpt-5 $1.25 in / $10.00 out per 1M tokens (flex and batch: half);
  <https://developers.openai.com/api/docs/pricing>, read 2026-11-13. Cached-input discounts are not counted.
- Calls per video follow the pipeline from the video's length (`ffprobe`, all 156 videos): one caption call per
  10 s; one summary per 30 s, 3 min and 10 min; NER + triples per caption of every granularity (HippoRAG's 10 s
  pass is served by the reply cache); one semantic extraction per 10 captions; one consolidation per extracted
  semantic triple.
- Image tokens: a caption call carries 10 frames and a visual search adds 33 frames to the answer call, each a
  512 x 512 JPEG (`OpenAIModel` `max_size=(512, 512)`, no `detail`): 308 tokens for gpt-5-mini (16 x 16 patches
  x 1.2), 210 for gpt-5 (70 + 140 per tile);
  <https://developers.openai.com/api/docs/guides/images-vision>, read 2026-11-13.
- Fixed prompt text per call: WorldMM's templates, counted with tiktoken (e.g. the fact filter's examples are
  2,559 tokens, sent four times per episodic search: the retriever's input is the largest token count).
- Measured on the Qwen3-VL-8B run of P05-20240425-201444 (163 captions, 59 questions, 2026-11-13): caption,
  entity and triple sizes per granularity, 19.2 semantic triples per group, 4.41 rounds and 2.76 episodic /
  0.53 semantic / 0.68 visual searches per question, 7,757 characters of retrieved text per question.
  **Assumed** to carry over: gpt-5-mini may write longer captions and gpt-5 may search differently (5 rounds at
  most), and one video stands for all.
- **Assumed, and the main uncertainty:** reasoning tokens, which are billed as output and which no run without
  the API can measure. The estimate takes 250 - 1,000 per call depending on the call (`REASONING` in the script);
  the range column scales that by 0.4 and 2.5. The first video run for real settles it: its `runs.jsonl` has
  `reasoning_tokens` per step (P05-20240425-201444, the test video: about $1.2 for the memory and $7.6 for its
  59 questions on both answer arms).

### Test without the API

`test_official.sh <scratch dir>` runs prep, `build g5m`, `qa` and `blind` for `g5m-g5`, `qa g5m-q8` (the real
local Qwen3-VL-8B as respond model) and a rerun, on the first 7 minutes and the 3 earliest questions of
P05-20240425-201444, with `OPENAI_BASE_URL` pointing at `epic/mock_openai.py`. The stand-in returns captions that
name the seconds they cover, so the test can check that nothing ending after a question's `t_ask` reached a
retriever or respond call, that a rerun makes no call, that `runs.jsonl` counts every call, and that the key is in
no file. It needs one GPU and no key.

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
  torch and transformers versions of WorldMM and of the server, GPUs, seconds; official arms: the respond model
  and the OpenAI usage).
- official answer arms: `<video>_memory_ref.json` instead of the memory files (which are at `worldmm/g5m/<video>/`).

## Checking a new machine first

Run one video into a scratch folder and compare accuracy with ours (fetched with `hf_results.py download
--baseline worldmm`). Decoding is greedy, but captions differ across GPU models (see MERIT's README), and the
answers follow the captions, so compare accuracy, not answers.
