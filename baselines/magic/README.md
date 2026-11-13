# baselines/magic: MAGIC-Video on EPIC-Morph

[MAGIC-Video](https://github.com/lijiazheng0917/MAGIC-video) (arXiv 2605.08271; Apache-2.0; a fork of WorldMM with
EgoRAG and HippoRAG) builds a multimodal memory graph of a video (30 s captions, 3 min / 10 min / 1 h summaries,
OpenIE entities, consolidated semantic triples, VLM2Vec clip embeddings) plus a Narrative Memory Chain (topic and
event chains), and answers by retrieving from it over several rounds. Here every LLM role (summaries, extraction,
retrieval, answering) is the arm's backbone, served locally; no paid API. Three more arms use the paper's own
models through their API ("Reported configuration" below).

## Layout

| path | what |
|---|---|
| `MAGIC-video/` | submodule: [morph-graphs/MAGIC-video](https://github.com/morph-graphs/MAGIC-video), branch `epic-morph` = upstream `632d7fc` + our patches |
| `epic/hf_server.py` | a small OpenAI-compatible server over the arm's backbone (HF transformers, batched); MAGIC's OpenAI client talks to it |
| `epic/build_memory.py` | the question-independent memory of one video (captions, summaries, OpenIE, semantic memory, clip embeddings, graph) |
| `epic/chains.py` | the Narrative Memory Chain at one question cutoff, from the memory before it |
| `epic/run_qa.py` | answers one video's questions of a bank, grouped by cutoff; writes `answers_<bank>.json.gz` |
| `epic/roles.py` | the reported arms: which endpoint answers which LLM role (caption, memory, reason), keys from the environment, calls and tokens counted |
| `epic/mock_llm.py` | a fake OpenAI-compatible endpoint with canned replies and a request log, and the checks of a run against it (no paid call) |
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

## Reported configuration: GPT-OSS-120B memory, Qwen3.5-Flash or Qwen3.8-27B reasoning

Three more arms run MAGIC-Video with the models of its paper, for the results-sheet rows "MAGIC-Video (reported)":

| arm | what | LLM calls | GPU (MAGIC's embedders; no backbone unless noted) | image |
|---|---|---|---|---|
| `oss120` | the memory, built once per video and shared by both rows (stages `prep`, `build`, `upload`) | caption + memory roles, paid API | one GPU (24 GB should do: the two embedders only; not measured) | `m8`'s |
| `oss120-flash` | row A: answers from `oss120`'s memory, reasoning Qwen3.5-Flash | reason role, paid API; chains: memory role, paid API | one GPU (24 GB should do: the two embedders only; not measured) | `m8`'s |
| `oss120-q27` | row B: answers from `oss120`'s memory, reasoning Qwen3.8-27B served here (non-thinking, as `m27`) | none paid once the bank's chains are in the dataset | as `m27` | `m27`'s |

**What the paper and upstream publish** (arXiv 2605.08271 Sec. 4.1 and App. E; upstream README "Default models"; read
2026-11-13). Every LLM call goes through OpenRouter (`OPENROUTER_API_KEY`):

| upstream role | model (OpenRouter id) | where upstream sets it |
|---|---|---|
| 30 s captions | EgoLife: the dataset's own dense captions and transcripts, rewritten by `openai/gpt-oss-120b`. Raw video (MM-Lifelong): `qwen/qwen3.5-flash-02-23` on each clip, merged with Whisper ASR by gpt-oss-120b | `generate_fine_caption.py`; `preprocess_video.py --vlm-model` |
| 3 min / 10 min / 1 h summaries, OpenIE, semantic extraction and consolidation | `openai/gpt-oss-120b` | `TRANSLATION_MODEL` default, `--llm-model` |
| Narrative Memory Chain (topic and event chains) | `openai/gpt-oss-120b` | `extract_{topic,event}_chains.py --model` |
| retriever (`--retriever-model`) | `openai/gpt-oss-120b`; with the unified graph (the paper's setting) no code path calls it: retrieval is embeddings, BM25 and PageRank | `eval_egolife.py` |
| responder (`--respond-model`): the search/answer loop and the final answer; sees the retrieved frames | `qwen/qwen3.5-flash-02-23`, sampling temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5 | `eval_egolife.py` |
| time parser (a time named in a search query) | `qwen/qwen3.5-flash-02-23`, hard-coded | `worldmm/memory/memory.py` |
| text embeddings; clip and query embeddings | `Qwen/Qwen3-Embedding-4B`; `VLM2Vec/VLM2Vec-V2.0` (16 frames per clip), both local | as the other arms |

The paper's text calls gpt-oss-120b "the online retrieval controller" and reports temperature 0 for controller and
backbone; the released code runs the loop with the responder and the sampling above. The arms follow the code.

**Our mapping of the sheet's two names onto these roles** (`epic/roles.py`; correct it there and here):

| role | upstream roles | `oss120` / `oss120-flash` | `oss120-q27` |
|---|---|---|---|
| `memory` = "memory model" | summaries, OpenIE, semantic memory, both chains, the (uncalled) retriever | `openai/gpt-oss-120b` | the same memory |
| `reason` = "reasoning model" | responder (loop and answer) and time parser | `qwen/qwen3.5-flash-02-23` | `Qwen/Qwen3.8-27B`, local |
| `caption` | the 30 s clip captions (GPT-OSS-120B reads no video) | `qwen/qwen3.5-flash-02-23` | the same memory |

- **Captions**: HD-EPIC comes with no captions, so the memory uses the paper's raw-video recipe: each 30 s clip, as a
  360p copy, goes to Qwen3.5-Flash with upstream's prompt plus one sentence asking for first person ("I"), which the
  rest of this port reads. No ASR merge (as the other arms). `MAGIC_CAPTIONS=merit-m27` (or `merit-m8`) builds the
  memory from MERIT's captions of that arm instead (they must be in the dataset); row B then has no Qwen3.5-Flash
  text in its memory.
- **Chains**: built in `qa`, per question cutoff, as for the other arms, by the memory role (GPT-OSS-120B). They are
  kept with the memory (`magic/oss120/<video>/chains/`), so the row that runs first pays for a cutoff and the other
  reuses it. `oss120-q27` takes them from the dataset and stops, rather than build them, where the key is missing.
- **Cutoff**: unchanged. The graph filter, the semantic snapshot and the per-cutoff chains are the same code.
- The memory's file names keep the tag `local` (`openie_results_local.json`, ...); the models are in `runs.jsonl`.
- Upstream's strict JSON parsing applies to these arms (the truncated-JSON repair is for the local backbones), and a
  step or a cutoff with a failed call is not kept: WorldMM would store it as an empty result.

**Local or API**, per model (prices: https://openrouter.ai/api/v1/models, read 2026-11-13, dollars per million
tokens in / out):

| model | weights | API | local | used here |
|---|---|---|---|---|
| GPT-OSS-120B | open, Apache-2.0: `openai/gpt-oss-120b`, 117B parameters (5.1B active), 65 GB (MXFP4), text only; one 80 GB GPU (H100, H200, GH200), not one 40 or 48 GB card | `openai/gpt-oss-120b` 0.15 / 0.60 | vLLM on an 80 GB GPU: about 10-25 H100-hours for all 156 memories and the chains (estimated from the token counts below, not measured), plus a new environment | API (the paper's route). Local: `MAGIC_MEMORY_BASE_URL=http://<host>:<port>/v1 MAGIC_MEMORY_KEY_ENV=` |
| Qwen3.5-Flash | none: API only. Its model card names the open `Qwen/Qwen3.5-35B-A3B` (36B, Apache-2.0) as the corresponding weights, not the same service | `qwen/qwen3.5-flash-02-23` 0.065 / 0.26 (video and image input) | only as Qwen3.5-35B-A3B behind an OpenAI-compatible server (one 80 GB GPU); not set up | API. `MAGIC_REASON_*` / `MAGIC_CAPTION_*` point elsewhere |
| Qwen3.8-27B | open: `Qwen/Qwen3.8-27B` | `qwen/qwen3.8-27b` 0.15 / 1.875 | `epic/hf_server.py`, as arm `m27` | local. API: `MAGIC_REASON_BASE_URL=https://openrouter.ai/api/v1 MAGIC_REASON_MODEL=qwen/qwen3.8-27b` (then no backbone is served) |

**Settings.** `OPENROUTER_API_KEY` in the environment (never on a command line or in a file; `docker/run.sh`
passes it like `HF_TOKEN`). Per role `R` in `CAPTION`, `MEMORY`, `REASON`: `MAGIC_<R>_MODEL`, `MAGIC_<R>_BASE_URL`
(any OpenAI-compatible endpoint), `MAGIC_<R>_KEY_ENV` (the name of the variable holding that endpoint's key; empty
for a keyless server of ours). A stage stops before any work when a role it needs has no key.

```bash
export OPENROUTER_API_KEY=...   # in the shell's environment only
MAGIC_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/magic/docker/run.sh build  P05-20240425-201444 oss120
MAGIC_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/magic/docker/run.sh qa     P05-20240425-201444 oss120-flash batch10_agree2
MAGIC_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/magic/docker/run.sh upload P05-20240425-201444 oss120-flash
MAGIC_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/magic/docker/run.sh qa     P05-20240425-201444 oss120-q27 batch10_agree2   # 80 GB GPU
MAGIC_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/magic/docker/run.sh upload P05-20240425-201444 oss120-q27
```

`upload` of an answer arm puts the memory folder (with the chains `qa` added) at `magic/oss120/<video>/` and the
answers at `magic/<arm>/<video>/` with `memory.json` (the memory's hash: answers made from a rebuilt memory replace
the old ones). Answer rows also carry `reason_model`. `runs.jsonl` lines of these arms add `roles` (model and
endpoint per role), `captions` and `usage`: per role the calls, prompt / completion / reasoning tokens, the cost
when the endpoint reports it (OpenRouter does) and failed calls of that stage, reruns included.

On the job board (`coord/master.py`): the `oss120` builds and the `oss120-flash` answers go to pool `max` only (paid
API, Max's key); an `oss120-q27` task waits for the same video's `oss120-flash` task of that bank, then runs on the
27B pools without a key.

**Cost estimate** (2026-11-13). No run of these arms exists; no full MAGIC run exists either. Counts and text
sizes are measured on the one finished run, `m8` on P05-20240425-201444 (27.5 min, 55 clips, 59 questions at 7
cutoffs): 14 summaries, 110 OpenIE calls, 6 semantic extractions and 696 semantic triples (at most one
consolidation call each), 33 topic-chain prompts (553k characters of captions) and 78 event-step calls over the 7
cutoffs, 4.2 search rounds per question. Assumed, not measured: 4 characters per token, ~300-500 reasoning tokens
per GPT-OSS-120B call, ~7k tokens per 30 s clip and 256 per frame for Qwen3.5-Flash, 41.3 h of video in all.
Read it as right within a factor of three.

| per unit | calls | tokens in / out | dollars |
|---|---|---|---|
| build, per video hour: captions (Qwen3.5-Flash) | 120 | 0.84M / 0.01M | 0.06 |
| build, per video hour: memory (GPT-OSS-120B) | ~1,550 | 1.1M / 0.56M | 0.50 |
| qa, chains per cutoff (GPT-OSS-120B) | ~19 | 39k / 13k | 0.014 (27 min video; grows with length) |
| qa, answers per question (Qwen3.5-Flash) | ~8.4 | 69k / 5k | 0.006 |
| qa, answers per question (Qwen3.8-27B) | ~8.4 | 69k / 0.3k | local: ~50 s on an H100 (guess: 17 s measured for 8B on an A6000); API 0.011 |

| | videos, hours | questions, cutoffs | build | qa: chains | qa: Qwen3.5-Flash (row A) | qa: Qwen3.8-27B (row B) | dollars, both rows |
|---|---|---|---|---|---|---|---|
| batch10 | 10, ~6 h | 598, 131 | $3.4 | $2.6 | $3.6 | ~8.5 H100-h (API $6.6) | ~$10 |
| tranche 2 | 20, ~13 h | 1,499, 403 | $7.3 | $10 | $9 | ~21 H100-h (API $16) | ~$26 |
| all 156 memories | 156, 41.3 h | (other videos have no questions yet) | $23 | | | | $23 for the builds |

## Test without API calls

`epic/mock_llm.py serve` is a fake endpoint: replies in the format each prompt expects, derived from the prompt, and
a log of every request. Point the three roles at it and run the real stages on one video; `check` then asserts that
each kind of call came from the right role, that no call of the answer loop shows a time stamp after its question's
current time, that no stored chain mentions a time after its cutoff, the answer rows' fields, and that the usage in
`runs.jsonl` equals the calls logged.

```bash
python baselines/magic/epic/mock_llm.py serve --port 18000 --log requests.jsonl &
for r in CAPTION MEMORY REASON; do export MAGIC_${r}_BASE_URL=http://127.0.0.1:18000/v1 MAGIC_${r}_KEY_ENV=MOCK_KEY; done
export MOCK_KEY=x MAGIC_CAPTION_MODEL=mock-caption MAGIC_MEMORY_MODEL=mock-memory MAGIC_REASON_MODEL=mock-reason MAGIC_FETCH=0
export MAGIC_OUT=<scratch results folder>
bash baselines/magic/run_magic.sh build P05-20240425-201444 oss120
bash baselines/magic/run_magic.sh qa P05-20240425-201444 oss120-flash <bank>
python baselines/magic/epic/mock_llm.py check --log requests.jsonl --memory $MAGIC_OUT/oss120/P05-20240425-201444 \
    --answers $MAGIC_OUT/oss120-flash/P05-20240425-201444/answers_<bank>.json.gz \
    --runs $MAGIC_OUT/oss120/P05-20240425-201444 $MAGIC_OUT/oss120-flash/P05-20240425-201444
```

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
