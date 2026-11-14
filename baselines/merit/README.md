# baselines/merit: MERIT on EPIC-Morph

[MERIT](https://github.com/choi-yeeun/MERIT) (Choi et al., arXiv 2608.07663; Apache-2.0) builds a memory of a
video once (a caption per 30 s clip, plus retrieval keys) and answers questions by retrieving from it over
several rounds. It is the first memory baseline of the EPIC-Morph benchmark paper. The batch10 numbers
(598 questions, epic-morph bank `batch10_agree2`) were made with this code: MERIT 8B 36.3%, MERIT 27B 49.0%.

## Layout

| path | what |
|---|---|
| `MERIT/` | submodule: [morph-graphs/MERIT](https://github.com/morph-graphs/MERIT), branch `epic-morph` = upstream `6edea6e` + our patches (one commit) |
| `epic/prepare_epic.py` | one HD-EPIC video -> 30 s clips (448 px), 1 fps frames, the clip list; a bank -> MERIT's question format |
| `epic/caption_clips_qwen.py` | MERIT's memory, step 1: one caption per clip with the arm's memory model (a local Qwen, or gpt-5-mini) |
| `epic/openai_usage.py` | the g5m arms: counts every OpenAI call (role, model id answered, tokens billed) for `runs.jsonl` |
| `epic/estimate_api_cost.py`, `epic/test_api_arms.py` | the g5m arms: the cost estimate from the Qwen runs; the pipeline run against a local stand-in for the API |
| `epic/blind_qa.py` | MERIT's answer prompt with no context: what the model gets from question and options alone |
| `epic/collect_answers.py`, `epic/compare_answers.py` | merge a video's answer files (reusing those of unchanged questions) and list unanswered IDs; compare two runs |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset, or download them |
| `run_merit.sh` | one video, one arm, one stage (prep, build, qa, blind, upload, all) |
| `env/` | the two environments, pinned: `wheels-*`, `requirements-*`, `install_requirements.sh` (shared by the image and `setup_env.sh`) |
| `docker/` | the image (`Dockerfile`), `build.sh`, `run.sh` (one job in a container), `image_tag.sh` |
| `build_order.txt` | the videos still without a MERIT memory, in the order to build them |

Our patches (in the submodule's `epic-morph` branch): Qwen3.8-27B as `qwen38-27b` (transformers 5.8, instruct
mode); loading from the local Hugging Face snapshot when cached; `MERIT_DEVICE_MAP` and, on several GPUs,
memory caps per GPU with a sequential layout (Qwen3.8-27B on 4x 24 GB); `MERIT_NO_FRAMES=1` (answer from
retrieved captions only, for memory probes). Settings are MERIT's EgoLife defaults except top-k 5, max 5
rounds, 512 image tokens, keys event + object + summary, 448 px.

**Where things live.** The code is this repo (`git clone --recurse-submodules`, or `git submodule update
--init baselines/merit/MERIT`). The questions are an epic-morph checkout, named by `EPIC_MORPH_DIR` as for
`epic_eval`: banks from its `outputs/v3/banks/` (generated there with `v3/export_bank.py <bank>` when missing:
banks are not committed, and the same commit gives the same bank byte for byte), videos from its
`data/HD-EPIC/videos/` (or `HDEPIC_VIDEOS`).
The results are not in git: they are written to `baselines/merit/results/<arm>/<video>/` (ignored) and the
`upload` stage puts them in the private Hugging Face dataset
[morph-graphs/epic-morph-results](https://huggingface.co/datasets/morph-graphs/epic-morph-results) at
`merit/<arm>/<video>/` (a member's token with write access: `HF_TOKEN`, or `hf auth login`).

## Two arms, two images

| arm | model (captions and answers) | GPU per job | environment |
|---|---|---|---|
| `m8` | Qwen/Qwen3-VL-8B-Instruct | one GPU with >= 40 GB (the 4B embedding model loads next to it) | transformers 4.57.3 |
| `m27` | Qwen/Qwen3.8-27B | one 80 GB GPU (H100); also runs on 2x 48 GB or 4x 24 GB | transformers 5.8.0 (5.2 loads it but generates garbage) |

**Off our clusters, run only in the Docker images** (rule in `AGENTS.md`): `ghcr.io/morph-graphs/merit:<arm>-<env
hash>`, built from `docker/Dockerfile` by `.github/workflows/baseline-images.yml`. The image holds the environment
(`env/`: prebuilt CUDA wheels, pinned requirements, causal-conv1d compiled for m27); the code is the mounted
checkout of `main`. `docker/run.sh` picks the image whose tag matches the checkout's environment files
(`docker/image_tag.sh`):

```bash
export EPIC_MORPH_DIR=<epic-morph checkout>
docker pull ghcr.io/morph-graphs/merit:$(bash baselines/merit/docker/image_tag.sh m27)
MERIT_GPU=0 bash baselines/merit/docker/run.sh build P03-20240218-190556 m27            # one H100
MERIT_GPU=1 bash baselines/merit/docker/run.sh build P03-20240218-190556 m8             # one GPU >= 40 GB
MERIT_GPU=2 bash baselines/merit/docker/run.sh qa P07-20240531-201703 m8 batch10_agree2
```
`docker/run.sh` mounts this checkout at `/repo`, the epic-morph checkout at `/epic-morph`, the Hugging Face
cache (`MERIT_HF_CACHE`, default `~/.cache/huggingface`, which also holds a `hf auth login` token) and the
videos read-only; it passes `HF_TOKEN` when set, runs as your user and shows the container only the GPU(s)
in `MERIT_GPU`. After a video's stages finish: `MERIT_GPU=0 bash baselines/merit/docker/run.sh upload <video> <arm>`. Download the models into that cache once
(`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3.8-27B`, `Qwen/Qwen3-Embedding-4B`). `docker/build.sh <arm>` builds
the same image locally. On our own clusters, a virtualenv from `env/setup_env.sh <arm> <dir>` (Python 3.11;
CUDA_HOME for m27's causal-conv1d) and `PY=<dir>/bin/python bash baselines/merit/run_merit.sh ...` run the same
stack without a container.

## Reported-configuration arms (GPT-5-mini memory)

The results sheet has three more MERIT rows, with the models of the paper. What the paper and the released code
do (arXiv 2608.07663v2, Sec. 4.2 and App. A; `MERIT/src/merit/merit_memory.py`, `MERIT/.env.example`):

| role | published model | where in the code |
|---|---|---|
| caption of each 30 s clip (1 fps frames) | GPT-5-mini | `benchmarks/*/preprocess/caption_clips.py`; here `epic/caption_clips_qwen.py` |
| the retrieval keys of each caption | GPT-5-mini ("the same model") | `preprocess/extract_keys.py` |
| key and query embeddings | Qwen3-Embedding-4B, local | `MultiKeyMemory` |
| search rounds (search or answer, the query), neighbour filtering, the final answer | one model for all three ("all inference-time roles ... a single identical MLLM backbone"): Qwen3-VL-8B (top-k 5), GPT-5 (top-k 10) or Gemini 2.5 Pro | `MeritMemory.respond_llm_model`, `--respond-model` |
| frames | only in the final answer request: 6 per clip of round 1, 6 per top-5 clip of round 2, 1 per clip after | `MeritMemory._get_frame_count` |

**The mapping used here (correct it if the sheet means something else):** "memory model" = the model that writes
the memory, i.e. captions and keys, at build time only; "reasoning model" = MERIT's respond model, i.e. search
rounds, neighbour filtering and the answer. Nothing at answer time uses the memory model, so a row with a Qwen
reasoning model makes no API call when answering.

| sheet row | arm | memory (captions, keys) | reasoning (search, filter, answer) | top-k | calls the OpenAI API in | runs in image |
|---|---|---|---|---|---|---|
| (the memory of the three rows) | `g5m` | gpt-5-mini | - | - | `build` | m8 |
| MERIT (reported): GPT-5-mini / Qwen3-VL-8B | `g5m-q8` | from `g5m` | Qwen3-VL-8B, local | 5 | nothing | m8 |
| MERIT (reported): GPT-5-mini / GPT-5 | `g5m-g5` | from `g5m` | gpt-5 | 10 | `qa` | m8 |
| MERIT: GPT-5-mini / Qwen3.8-27B | `g5m-q27` | from `g5m` | Qwen3.8-27B, local | 5 | nothing | m27 |

- One memory per video, at `merit/g5m/<video>/`, shared by the three answer arms. An answer arm keeps its answers
  in its own folder (`merit/g5m-q8/<video>/answers_<bank>.json.gz`, `runs.jsonl`) with `<video>_memory_ref.json`,
  the hashes of the memory they were made from: when the memory changes, `qa` drops the old answers and `upload`
  removes them from the dataset. An answer arm never builds: without a `g5m` memory (here or in the dataset) it stops.
- Everything else is as for `m8` / `m27`: our first-person caption prompt, 448 px clips, 1 fps frames (30 a
  caption), keys event + object + summary, 5 rounds at most, questions asked at `t_ask` (MERIT only retrieves,
  expands to and shows clips that end by then). Differences from the paper's own runs: top-k follows the paper per
  reasoning model (10 for GPT-5; `MERIT_TOP_K` overrides it and names the variant `top<k>`), and the API's model
  ids are the snapshots MERIT pins (`gpt-5-mini-2025-08-07`, `gpt-5-2025-08-07`; temperature 1, the only one
  GPT-5 accepts, so answers are not reproducible call by call).
- **Shutdown: OpenAI retires `gpt-5`, `gpt-5-mini` and both snapshots on 2026-12-11** (deprecations page, read
  2026-11-13: notice 2026-06-11, replacements `gpt-6.1-sol` / `gpt-6.1-terra`). These arms cannot be run or
  extended after that date.
- The key: `OPENAI_API_KEY` in the environment of whoever runs `g5m` build or `g5m-g5` qa, never on a command line
  or in a file. `docker/run.sh` hands it to the container as an environment variable for those two arms only, and
  `run_merit.sh` removes it from every other stage's environment. On the job board those tasks go to pool `max`
  only and carry `needs_env: [OPENAI_API_KEY]`; a worker without the key leaves them.
- What was spent: each `build` / `qa` line of `runs.jsonl` has `openai`: per role (`captions`, `keys`; `search`,
  `filter`, `answer`) and per model id the API answered with, the calls, input, cached, output and reasoning
  tokens and the service tier. MERIT's reply cache (in the work folder) keeps a rerun from paying twice for the
  same text request; a stage that fails partway keeps its counts for the line written when it finishes.

### Through OpenRouter (our own builds)

With `OPENROUTER_API_KEY` in the environment and `OPENAI_API_KEY` not, the same calls go through OpenRouter, so
a memory can be built on our cluster without an OpenAI account. Nothing else is set; `epic/openai_usage.py`
does the switch.

```bash
export OPENROUTER_API_KEY=...      # typed into the shell (or read -s): never an argument, a file, a log or a commit
unset OPENAI_API_KEY GOOGLE_API_KEY GEMINI_API_KEY HF_TOKEN       # no other provider's key in the job
export PY=<m8 environment>/bin/python EPIC_MORPH_DIR=<epic-morph checkout> MERIT_FETCH=0
bash baselines/merit/run_merit.sh prep  P05-20240425-201444 g5m   # no GPU: ffmpeg clips and 1 fps frames
bash baselines/merit/run_merit.sh build P05-20240425-201444 g5m   # no GPU: one caption and one key call per 30 s clip
# upload (no GPU) needs HF_TOKEN back; it makes no API call:  bash baselines/merit/run_merit.sh upload <video> g5m
```
For an sbatch job, export the key in the submitting shell and let `sbatch` pass the environment (its default,
`--export=ALL`); do not write it into the script. `g5m` needs no GPU at any stage. Answering needs one: `g5m-q8`
and `g5m-q27` as `m8` / `m27` (no API call), `g5m-g5` one GPU of 16 GB for the embedding model (its gpt-5 calls
take the same route, as `openai/gpt-5`).

- **What changes in the request.** The endpoint (`https://openrouter.ai/api/v1`; `OPENROUTER_BASE_URL` overrides
  it, for the stand-in test), the key, the model id (`openai/gpt-5-mini`, `openai/gpt-5`) and one added field,
  `provider: {"only": ["openai"], "allow_fallbacks": false}`: OpenRouter also lists Azure for these models, and
  with the pin OpenAI answers or the call fails. Prompts, frames, image size, temperature, seed and
  reasoning effort (none sent: the model's default, `medium` on both routes) are what the port sends to OpenAI.
- **What may still differ from the direct API** (OpenRouter's model list and API reference, read 2026-11-14; no
  call was made, there is no key yet):
  - the snapshot is not ours to pin: `openai/gpt-5-mini` is OpenRouter's alias, whose `canonical_slug` today is
    `openai/gpt-5-mini-2025-08-07` (`openai/gpt-5`: `openai/gpt-5-2025-08-07`), the snapshots MERIT pins. The
    reply's `model` is recorded per call; whether it shows the snapshot or the alias is not known yet;
  - OpenRouter lists two OpenAI endpoints, standard and `openai/flex` (half price, slower). We send no service
    tier unless `MERIT_SERVICE_TIER` is set; the tier each reply reports is recorded;
  - `temperature` is not among the parameters OpenRouter lists for these models and may be dropped before OpenAI
    sees it. The port sends 1 or nothing, which is the model's only setting, so nothing changes;
  - image tokens are counted by OpenAI either way (the images are passed on as data URLs), but OpenRouter could
    re-encode or reject an oversized request before that: compare `input_tokens` of the first clips with the
    estimate before building many videos;
  - the price per token is OpenAI's ($0.25 in / $2.00 out per million for gpt-5-mini, $1.25 / $10 for gpt-5);
    OpenRouter adds its fee when credits are bought, not per call, so `cost` in `runs.jsonl` is before that fee;
  - both models carry `expiration_date` 2026-12-11 on OpenRouter too: the same shutdown.
- **What is recorded.** The stage's `runs.jsonl` line says which gateway made the memory: each `openai` row is
  keyed by the model id the endpoint returned and has `endpoint`, and through OpenRouter also `provider` (who
  served the call) and `cost` (dollars charged, summed). The memory folder stays `merit/g5m/<video>/`: the same model
  through another gateway. The reply cache keys on the model id, so a memory started on one gateway and resumed
  on the other reuses what is already written to its files but not the other gateway's cached replies; its
  `runs.jsonl` lines then show both.
- **One gateway per run.** With both keys set OpenAI's is used and `OPENROUTER_API_KEY` is removed from the
  stage's environment; stages that call no API get neither key.
- **Check without a key**: `epic/test_api_arms.py --work <scratch> --gateway openrouter` runs the same test with
  only `OPENROUTER_API_KEY` set (a dummy) against the stand-in, and asserts the path, the `openai/...` ids, the
  provider pin and the recorded endpoint, provider and cost; without `--gateway` it asserts that the OpenAI route
  is what it was (path, snapshots, no `provider` field) even when an OpenRouter key is also set.

```bash
export EPIC_MORPH_DIR=<epic-morph checkout> OPENAI_API_KEY=...        # the key only in the environment
docker pull ghcr.io/morph-graphs/merit:$(bash baselines/merit/docker/image_tag.sh m8)
MERIT_GPU=0 bash baselines/merit/docker/run.sh build P05-20240425-201444 g5m                    # API; the GPU is not used
MERIT_GPU=0 bash baselines/merit/docker/run.sh upload P05-20240425-201444 g5m
MERIT_GPU=0 bash baselines/merit/docker/run.sh qa P05-20240425-201444 g5m-g5 batch10_agree2     # API; a GPU >= 16 GB (embeddings)
MERIT_GPU=0 bash baselines/merit/docker/run.sh upload P05-20240425-201444 g5m-g5
MERIT_GPU=1 bash baselines/merit/docker/run.sh qa P05-20240425-201444 g5m-q8 batch10_agree2     # local, no key needed
```
`MERIT_SERVICE_TIER=flex` halves the price (slower, and requests can be refused when capacity is short);
`MERIT_API_CONCURRENCY` (default 8) and `MERIT_API_THREADS` (default 4) set how many requests run at once.
With the board, `coord/worker.py --pool max ...` started with `OPENAI_API_KEY` set does all of this.

**Cost** (standard tier, prices of 2026-11-13 from https://developers.openai.com/api/docs/pricing: gpt-5-mini $0.25
in / $2.00 out, gpt-5 $1.25 in / $10.00 out per 1M tokens; flex and batch half; a 448 px frame is 236 tokens for
gpt-5-mini and 210 for gpt-5). Output of `epic/estimate_api_cost.py --proxy m27 --bank batch10_agree2 --top-k 10
--count <banks>/batch10_agree2 <banks>/t2_agree2`, run 2026-11-13 on the Qwen3.8-27B runs in the results dataset
(156 memories, 5,030 clips; 598 answered questions). No API call was made: the request sizes are rebuilt from those
runs and counted with tiktoken.

| | requests | input tokens | output tokens (visible) | $ | $ with 0x / 3x the assumed reasoning | $ flex |
|---|---|---|---|---|---|---|
| `g5m` build, P05-20240425-201444 (55 clips; the smallest of batch10) | 55 captions + 55 keys | 0.44M | 0.012M | 0.22 | 0.13 / 0.40 | 0.11 |
| `g5m` build, batch10 (10 videos, 684 clips) | 1,368 | 5.46M | 0.138M | 2.74 | 1.64 / 4.92 | 1.37 |
| `g5m` build, tranche 2 (20 videos, 1,539 clips) | 3,078 | 12.33M | 0.330M | 6.20 | 3.74 / 11.13 | 3.10 |
| `g5m` build, all 156 videos (5,030 clips) | 10,060 | 39.93M | 1.062M | 20.16 | 12.11 / 36.25 | 10.08 |
| `g5m-g5` qa, per question | 4.45 search + 4.09 filter + 1 answer | 87.4k | 1.5k | 0.187 | 0.124 / 0.313 | 0.094 |
| `g5m-g5` qa, batch10_agree2 (598 questions) | 5,704 | 52.25M | 0.897M | 111.93 | 74.29 / 187.23 | 55.97 |
| `g5m-g5` qa, t2_agree2 (1,499 questions, at batch10's mean per question) | 14,298 | 130.98M | 2.249M | 280.58 | 186.22 / 469.32 | 140.29 |
| `g5m-q8`, `g5m-q27` qa | 0 | 0 | 0 | 0 (GPU time as `m8` / `m27` qa) | | |

What is measured and what is assumed:
- Measured from the proxy runs: clips per video, caption and key lengths, and per question the search rounds, the
  clips each retrieved, the filter replies and the answer, from which each request is rebuilt with MERIT's prompts.
- Assumed: gpt-5-mini captions are as long as Qwen3.8-27B's; gpt-5 searches as many rounds as Qwen3.8-27B did
  (4.45 a question here; the paper reports fewer for GPT-5, which would make qa cheaper); a round's text and frames
  scale from top-k 5 to top-k 10, up to the clips visible at the ask time; and the hidden reasoning tokens per call,
  billed as output (`REASONING` in the script: 500 captions, 300 keys, 500 search, 800 filter, 800 answer), which
  nothing short of a real call can measure. The `$` column includes them; the next column shows 0x and 3x.
- The first real build settles the reasoning term: its `runs.jsonl` line has `reasoning_tokens` per role.

**Check without the API**: `epic/test_api_arms.py --work <scratch>` runs `build g5m` and `qa g5m-g5` on one real
video against a local stand-in for the endpoint (a dummy key) and checks the calls per role, that no request
holds a clip ending after the question's ask time, the frames per answer request, the answer rows and that the
key is in no file (`--qwen` adds `qa g5m-q8` with the real model).

## Running

**Build a memory once, reuse it everywhere.** A memory depends only on the video and the model, so `build`
first takes it from the dataset when it is there (CPU only, seconds; `MERIT_FETCH=0` to skip that) and every
bank, probe set and variant answers from it. Variants of the same bank get their own answer files:
`MERIT_TAG=<name>` writes `answers_<bank>_<name>.json.gz`, and `MERIT_NO_FRAMES=1` (answer from captions only,
for memory probes) defaults the tag to `noframes`. `upload` adds files to the dataset folder and merges
`runs.jsonl`; only a rebuilt memory replaces the folder (answers from the old memory are then removed).

`build_order.txt` lists the 146 videos with no memory yet: the 17 that already have object stories,
then the rest longest first. `build` makes the clips and frames first if needed. `qa` answers every question
of the bank for the video; a question MERIT fails on (it writes "Error", e.g. after a CUDA out-of-memory) is
retried by ID, up to three passes. Every stage skips finished work, so rerun the same command after a
failure.

**Question identity.** A question's ID is its bank `key` (files without keys, such as M2 probes, keep the
1-based position), and each answer row also records `key`, `qhash` (a hash of the wording and options) and
`pos` (its position in the bank file). `qa` reuses the answers file already in `MERIT_OUT` (from an earlier
tranche, or `hf_results.py download`): an answer is kept while its question's wording, options and gold letter
are unchanged, so after a tranche adds or edits questions of a video only those are answered. Answer files from
before keys (positional IDs) are matched by the question they carry, and read the same.

One job per GPU (or GPU set); stages of different videos can run side by side. The clips and frames
(~300 MB a video) and MERIT's run folders stay in `baselines/merit/work/` (not committed).

## What comes back (in the Hugging Face dataset)

Per video, at `merit/<arm>/<video>/` in morph-graphs/epic-morph-results (~0.3-0.8 MB a video):
- `<video>_30sec.json`, `<video>_30sec_4_key.json`: the memory (captions, keys);
- `answers_<bank>.json.gz`: one row per question, with MERIT's retrieval rounds (the M1 evaluation reads them);
- `blind_<bank>.json`, when the blind stage ran (`blind_<bank>_<tag>.json` when `MERIT_CONTEXT` gave it a context file, named by `MERIT_TAG`);
- `runs.jsonl`: one line per finished stage (this repo's commit, the MERIT and epic-morph commits, model,
  torch and transformers versions, GPUs, seconds).

The batch10 runs (10 videos x 2 arms, from our Nexus runs) are there in this layout.

## Checking a new machine first

Rebuild one batch10 video into a scratch folder and compare with our run, fetched from the dataset:
```bash
MERIT_GPU=0 MERIT_OUT=/repo/baselines/merit/work/check \
    bash baselines/merit/docker/run.sh all P07-20240531-201703 m27 batch10_agree2
python3 baselines/common/hf_results.py download --arm m27 --video P07-20240531-201703 --dest baselines/merit/work/ref
python3 baselines/merit/epic/compare_answers.py \
    baselines/merit/work/ref/merit/m27/P07-20240531-201703/answers_batch10_agree2.json.gz \
    baselines/merit/work/check/m27/P07-20240531-201703/answers_batch10_agree2.json.gz
```
(`hf_results.py` needs `huggingface_hub`, in the images: on the host, `pip install huggingface_hub` or run it
in the container.)
The committed P07 runs score 24/64 (m8) and 25/64 (m27), made on RTX A5000s (m8) and A5000s / a GH200 (m27).
Decoding is greedy and deterministic on one GPU model, but captions differ across GPU models (measured on
P07, 2026-11-12: this environment and the original container gave 8/8 identical captions on the same A6000,
and both only 2/8 identical to the A5000 run), and the answers follow the captions. So on another GPU model
compare accuracy, not answers: our check on an L40S gave 22/64 vs 24/64 for m8 (41/64 same option, McNemar
p = 0.80). A large gap means a setup problem (most often the wrong transformers version for m27).
