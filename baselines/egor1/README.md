# baselines/egor1: Ego-R1 on EPIC-Morph

[Ego-R1](https://github.com/egolife-ai/Ego-R1) (arXiv 2506.13654, TPAMI 2026; Apache-2.0) answers questions about
very long egocentric video with a small trained agent, Ego-R1-Agent-3B (Qwen2.5-3B-Instruct, fine-tuned on
Chain-of-Tool-Thought traces, then GRPO). Each turn the agent thinks and calls one of three tools: `rag` (search
of a hierarchical text memory of the video), `video_llm` (a question about up to 10 minutes of video) and `vlm`
(a question about one frame). A summary model writes the final answer from the trace.

**This port keeps the models of the paper, with one replacement: Gemini 3.6 Flash (`gemini-3.6-flash`, arm
`gemini36flash`) stands in for the paper's Gemini 1.5 Pro, which is retired.** It affects the 30 s captions of the
memory and the `video_llm` tool; nothing is swapped for a local backbone. **The arm that is run,
`gemini36flash-gpt41`, has a second replacement: GPT-4.1 stands in for GPT-4o in the `rag` and `vlm` tools**,
because the machines that run this row have no GPT-4o (Anubhav's decision, 2026-11-14). How it came to 3.6 Flash:

- Gemini 1.5 Pro is not offered to our key (2026-11-13). Anubhav chose Gemini 2.5 Pro that day (arm `gemini25pro`).
- **Google refuses Gemini 2.5 Pro on our key**, and it is not available to the collaborators either: every
  `generate_content` call answers `404 NOT_FOUND: This model models/gemini-2.5-pro is no longer available to new
  users` (2026-11-14; Google's deprecations page: API access is limited to those who used it before). The arm's
  code stays, for a key that still has the model.
- Anubhav then chose Gemini 3.5 Flash (2026-11-14). **Google routes the `gemini-3.5-flash` endpoint to Gemini 3.6
  Flash** (its deprecations page says so, and a call on our key came back with `model_version` `gemini-3.6-flash`;
  the pricing page has no 3.5 Flash). So the arm asks for `gemini-3.6-flash` by name and is called `gemini36flash`.

| arm | captions, `video_llm` | `rag`, `vlm` | summaries, final answer | memory |
|---|---|---|---|---|
| `gemini36flash-gpt41` | Gemini 3.6 Flash | **GPT-4.1** | GPT-4.1 | `gemini36flash`'s: the same models make it, so `build` takes it from that arm's folder in the dataset (its own folder first) and builds one only when neither has it |
| `gemini36flash` | Gemini 3.6 Flash | GPT-4o | GPT-4.1 | its own |
| `gemini25pro` | Gemini 2.5 Pro (refused on our key) | GPT-4o | GPT-4.1 | its own |
| `llava` (not set up) | captions as `gemini25pro`; `video_llm` LLaVA-Video-7B, local | GPT-4o | GPT-4.1 | its own |

Each arm has its own results folder (`<arm>/<video>/`), with a copy of the memory it answered from.

Only the agent has open weights. Every other step calls a paid API, so those stages stop (exit code 5) until the
keys are set:

| step | model, as published | where upstream calls it | runs here |
|---|---|---|---|
| agent | Ego-R1/Ego-R1-Agent-3B | `Ego-R1-Agent/eval/infer_bench_summ.py` (vLLM) | locally, transformers, fp16, ~8 GB GPU |
| memory: 30 s captions | Gemini 1.5 Pro; **here Gemini 3.6 Flash** (arm `gemini25pro`: 2.5 Pro) | not in the repo (paper, H-RAG construction) | Gemini key |
| memory: 10 min / hour / day summaries | GPT-4.1 | not in the repo (paper) | OpenAI key |
| `rag` retrieval | GPT-4o; **arm `gemini36flash-gpt41`: GPT-4.1** | `api/rag/r1rag/utils.py` `call_gpt4` | OpenAI key |
| `video_llm` | Gemini 1.5 Pro; **here Gemini 3.6 Flash** (arm `gemini36flash`; arm `gemini25pro`: 2.5 Pro); LLaVA-Video-7B (arm `llava`, paper Table 4) | `api/visual_tools/egor1_videollm_{gemini,llava}` | Gemini key; or locally |
| `vlm` | GPT-4o; **arm `gemini36flash-gpt41`: GPT-4.1** (the same request: one PNG frame as an `image_url` data URL, 800 tokens, temperature 0.7) | `api/visual_tools/egor1_vlm` | OpenAI key |
| final answer | GPT-4.1 | `infer_bench_summ.py` `summ_client` | OpenAI key |

**A reply from another model than the one asked is an error on every arm**: `epic/apis.py` checks the model id in
every Gemini reply, upstream's `video_llm` calls included, and stops the run with exit code 6 (asked for
`gemini-3.5-flash`, it stops on the first reply, which names `gemini-3.6-flash`). Gemini 3.6 Flash has no shutdown
date announced (deprecations page, 2026-11-14). Gemini 2.5 Pro is listed for shutdown on 2026-12-01, after which
its requests go to another model: from that day arm `gemini25pro`'s Gemini stages refuse to run. A memory is
shared only between arms whose captions and summaries come from the same models (the table above).

## Layout

| path | what |
|---|---|
| `Ego-R1/` | submodule: upstream [egolife-ai/Ego-R1](https://github.com/egolife-ai/Ego-R1), pinned at `69177ee` (`git submodule update --init baselines/egor1/Ego-R1`) |
| `epic-morph.patch` | our changes to upstream, applied to the submodule's files by `run_egor1.sh` on first use (so `git status` shows the submodule as modified; a morph-graphs fork with this as a branch would be cleaner): `video_llm` takes the Gemini model id from `EGOR1_GEMINI_MODEL` (upstream hard-codes `gemini-1.5-pro`) and no longer prints the tail of the key; `call_gpt4` (rag) and `GPT` (vlm) accept a plain `OPENAI_API_KEY` when no Azure endpoint is set, or else `OPENROUTER_API_KEY` (a client at OpenRouter's endpoint); `call_gpt4` and `vlm` take their model id from `EGOR1_RAG_MODEL` / `EGOR1_VLM_MODEL` (upstream hard-codes `gpt-4o`; set by `epic/tools.py` from the arm). A checkout patched with an earlier version of this file has to be restored first (`git -C baselines/egor1/Ego-R1 checkout .`); `run_egor1.sh` says so |
| `epic/prepare_clips.py` | 30 s clips with audio in the EgoLife layout upstream's tools read |
| `epic/build_memory.py` | the question-independent memory: captions, then 10 min / 1 hour / 1 day summaries, in the format of upstream's released database (`Ego-R1/h-rag_database`) |
| `epic/tools.py` | the three tools, calling upstream's functions in process, plus the streaming cutoff |
| `epic/run_qa.py` | the agent loop of `infer_bench_summ.py` on one video's questions of a bank; writes `answers_<bank>.json.gz` |
| `epic/apis.py` | the model of each step, the key check, and the two API clients upstream has no reusable code for |
| `epic/plan_cost.py` | calls and cost of a run per arm, from video lengths and one or more banks; calls nothing |
| `run_egor1.sh` | one video, one arm, one stage (prep, build, qa, upload, all; and the check `firstturn`) |
| `env/` | the environment, pinned: `requirements.txt` (shared by the image and `setup_env.sh`), `setup_env.sh` |
| `docker/` | the image (`Dockerfile`), `build.sh`, `run.sh` (one job in a container; the job board's entry point), `image_tag.sh` |
| `../merit/epic/prepare_epic.py` | (reused) MERIT's question conversion: ID = the bank's `key`, asked at floor(t_ask) |
| `../common/hf_results.py` | (shared) upload a video's results to the Hugging Face dataset, or fetch its memory |

## How it maps to EPIC-Morph

- **One video is DAY1 starting at 00:00:00**, as in the MERIT port; the agent's `DAYX_HHMMSSFF` stamps are
  seconds into the video. The setting is upstream's EgoLife one (the prompt in `Ego-R1-Agent/utils/constants.py`,
  the day / hour / 10 min logs, `RagAgent.query`): it is the one the agent was trained on, and the only one the
  repo has complete. The single-video Video-MME path lacks its prompt (`constants.prompt_mme` is not defined).
- **The memory is question-independent** (`build`): `<video>_30sec.json`, `_10min.json`, `_1hour.json`,
  `_1day.json`. Upstream publishes the finished EgoLife logs but not the code or prompts that made them; the
  caption and summary prompts in `build_memory.py` are ours, written after the paper's description.
- **Streaming.** A question is asked at floor(t_ask) (MERIT's conversion). Upstream's day and hour searches read
  summaries of the whole day, which would show a question the future. So `rag` searches a view of the memory cut
  at the question's time: the windows that ended by then, and the unfinished 10 min / hour / day windows
  summarized again from the captions before it (GPT-4.1; cached in `cutoffs/c<seconds>/`, shared by every
  question asked at that second). `video_llm` ranges are clipped to the question's time and a `vlm` stamp after
  it gets an error string. Questions asked at the end of the video use the memory as built.
- **Answering** follows `infer_bench_summ.py` and `infer_bench_summ.sh`: 7 turns, temperature 0.6, top-p 0.95,
  1024 new tokens per turn; GPT-4.1 turns the trace into the letter (after the agent's `<answer>`, or at the
  last turn). Where upstream would answer "A" after three failed GPT-4.1 tries, the question is left unanswered
  and retried on the next pass, as is any question with a failed API call.
- **No blind stage**: the answer is written by GPT-4.1, so a no-memory floor would be GPT-4.1 on the bare
  question, a paid run of its own.

Differences from upstream that remain: the agent runs in transformers, not vLLM (same weights and sampling
settings, different random stream); the tools are called in process instead of through upstream's FastAPI
servers (same functions); `rag` failures of the API are detected so the question is retried.

## Running

```bash
git submodule update --init baselines/egor1/Ego-R1
bash baselines/egor1/env/setup_env.sh <venv dir>        # Python 3.11 (PYTHON=...), CUDA 12.4 wheels; ffmpeg and patch on PATH
export PY=<venv>/bin/python EPIC_MORPH_DIR=<epic-morph checkout>
bash baselines/egor1/run_egor1.sh prep      P05-20240425-201444 gemini36flash                # CPU
bash baselines/egor1/run_egor1.sh firstturn P05-20240425-201444 gemini36flash batch10_agree2   # GPU, no API: the local half
# with the keys in the environment (next section):
bash baselines/egor1/run_egor1.sh build     P05-20240425-201444 gemini36flash                # paid APIs, no GPU
bash baselines/egor1/run_egor1.sh qa        P05-20240425-201444 gemini36flash batch10_agree2   # GPU + paid APIs
bash baselines/egor1/run_egor1.sh upload    P05-20240425-201444 gemini36flash
python baselines/egor1/epic/plan_cost.py --videos <HD-EPIC videos> --bank_dir <epic-morph>/outputs/v3/banks/batch10_agree2   # Cost estimate, below
```

`firstturn` runs the agent's first turn on every question (or the first `EGOR1_LIMIT`), parses its tool call with
upstream's parser and applies the streaming check; it writes `answers_<bank>_firstturn.json` and needs neither a
memory nor a key. Every stage skips finished work. Results go to `baselines/egor1/results/<arm>/<video>/`
(ignored; `EGOR1_OUT`); `upload` puts them in the private dataset morph-graphs/epic-morph-results at
`egor1/<arm>/<video>/`. Clips, tool caches and logs stay in `baselines/egor1/work/` (`EGOR1_WORK`).

Arm `llava` needs upstream's LLaVA-Video-7B server (`api/visual_tools/egor1_videollm_llava/llava_video.py`,
LLaVA-NeXT, port 8000) in an environment of its own, given as `EGOR1_LLAVA_PY`; that environment is not set up
here yet, and the arm has not been run.

**Off our clusters, run only in the Docker image** (rule in `AGENTS.md`): `ghcr.io/morph-graphs/egor1:gemini25pro-<env
hash>`, built from `docker/Dockerfile` by `.github/workflows/baseline-images.yml`. One image serves both API arms (it
holds the environment, and keeps the first arm's name); under Apptainer both use `egor1_gemini25pro.sif`. The code is the mounted checkout
of `main`; `docker/run.sh` picks the image whose tag matches the checkout's environment files:

```bash
git submodule update --init baselines/egor1/Ego-R1
docker pull ghcr.io/morph-graphs/egor1:$(bash baselines/egor1/docker/image_tag.sh gemini36flash)
export GEMINI_API_KEY OPENAI_API_KEY HF_TOKEN        # already set in this shell; never typed into a command
export EPIC_MORPH_DIR=<epic-morph checkout>
bash baselines/egor1/docker/run.sh prep   P05-20240425-201444 gemini36flash
bash baselines/egor1/docker/run.sh build  P05-20240425-201444 gemini36flash          # the memory, shared by the two arms
bash baselines/egor1/docker/run.sh upload P05-20240425-201444 gemini36flash
EGOR1_GPU=0 bash baselines/egor1/docker/run.sh qa P05-20240425-201444 gemini36flash-gpt41 batch10_agree2   # takes that memory from the dataset
bash baselines/egor1/docker/run.sh upload P05-20240425-201444 gemini36flash-gpt41
```

`docker/run.sh` mounts this checkout at `/repo`, the epic-morph checkout at `/epic-morph`, the videos at `/videos`
and the Hugging Face cache (`EGOR1_HF_CACHE`, default `~/.cache/huggingface`) at `/hf`; the agent's weights
(`Ego-R1/Ego-R1-Agent-3B`, 13 GB, public) are downloaded there on first use. `EGOR1_RUNTIME=apptainer` runs the same
image with Apptainer. The job board (`coord/worker.py`) calls it with `MERIT_GPU` / `MERIT_RUNTIME` / `MERIT_SIF_DIR`
/ `MERIT_HF_CACHE` / `MERIT_WORK`, which `run.sh` takes when the `EGOR1_` ones are unset.

## Keys, and running the paid stages as job-board tasks

`build` and `qa` are the paid stages. On any machine, a collaborator's included, they need:

| variable | for |
|---|---|
| `GEMINI_API_KEY` | the arm's Gemini: captions (`build`), `video_llm` (`qa`) |
| `OPENAI_API_KEY` | GPT-4o and GPT-4.1 with a plain OpenAI key: summaries (`build`), `rag`, `vlm`, final answer, cutoff views (`qa`) |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | instead of `OPENAI_API_KEY`: an Azure OpenAI resource with deployments named `gpt-4o` and `gpt-4.1` (upstream's way; used when both are set) |
| `OPENROUTER_API_KEY` | instead of both, when neither is set: the same GPT-4o and GPT-4.1 calls through OpenRouter (next section). Gemini never goes through it |
| `HF_TOKEN` | `upload`, and `build`'s look in the results dataset |

- **No stage may put a key on a command line**, in a file of this repo, in a job script or in a log. Keys are read
  from the environment of the process only; the worker that starts a task exports them from its own secret
  store (for a container: `--env GEMINI_API_KEY` by name, so the value comes from the caller's environment).
  `run_egor1.sh` takes no key argument, upstream's prints are dropped, and `api_usage.jsonl` holds token counts only.
- Upstream's OpenAI clients are Azure-only; a plain OpenAI key, or OpenRouter's, works through `epic-morph.patch`
  (two small changes, above). Our own client (`apis.gpt41`) takes any of the three.
- A stage that calls no model (`prep`, `firstturn`, `upload`) drops the model keys from its environment.
- `build` needs no GPU (ffmpeg, network). `qa` needs one small GPU (the agent, about 8 GB) and network.
- A stage without its keys exits with code 5 before any call, so a task handed to a pool without keys fails at
  once instead of half-running.
- The job-board entries (`coord/master.py` `EXPERIMENTS`) do not exist yet: a paid row needs the board to know
  which pool holds keys.
- `docker/run.sh` hands the keys to the container by name (docker `-e NAME`, apptainer `APPTAINERENV_NAME`).

### Through OpenRouter (our own runs)

With `OPENROUTER_API_KEY` in the environment and neither `OPENAI_API_KEY` nor the Azure pair, the GPT-4o and
GPT-4.1 calls go through OpenRouter, so the row can be run on our cluster without an OpenAI account. Gemini
stays on Google's API with `GEMINI_API_KEY` on every route.

```bash
export GEMINI_API_KEY=... OPENROUTER_API_KEY=...   # from the job's secret store: never an argument, a file, a log or a commit
unset OPENAI_API_KEY AZURE_OPENAI_API_KEY          # one gateway per run: with either set, OpenRouter's key is dropped
bash baselines/egor1/run_egor1.sh build P05-20240425-201444 gemini36flash
bash baselines/egor1/run_egor1.sh qa    P05-20240425-201444 gemini36flash batch10_agree2
```

- **What changes in the request.** The endpoint (`https://openrouter.ai/api/v1`; `OPENROUTER_BASE_URL` overrides
  it), the key, the model id (`openai/gpt-4o`, `openai/gpt-4.1`) and one added field,
  `provider: {"only": ["openai"], "allow_fallbacks": false}`: OpenAI itself answers, or the call fails. Prompts,
  the `vlm` frame, temperature, top-p and token limits are what upstream sends.
- **Where.** The client: `epic-morph.patch` for upstream's two (`call_gpt4` of `rag`, `GPT` of `vlm`), `apis.gpt41`
  for the summaries, the cutoff views and the final answer. The model id and the provider pin: `apis.track()`,
  the one wrapper every chat completion of the port already passes through, for a client that points at OpenRouter.
- **What may differ from the direct API.** `openai/gpt-4o` and `openai/gpt-4.1` are OpenRouter's aliases; the
  snapshot behind them is OpenRouter's choice, and the id each call answered with is recorded.
- **What is recorded.** Each `api_usage.jsonl` line of a chat completion has `endpoint`, `cached` (input tokens
  billed as cached) and, through OpenRouter, `provider` (who served the call) and `cost` (dollars charged);
  `runs.jsonl` sums them per step and answering model. A `qa` line also has `q`, the question's key, so the
  calls and cost of one question can be read off. Gemini lines have tokens only (Google returns no price).
- **Checked with real calls** (2026-11-14, in the image, through OpenRouter only): `apis.gpt41`, upstream's
  `call_gpt4` and upstream's `vlm` on a frame of P05-20240425-201444 (793 input tokens, $0.0027) each answered,
  with `provider` `OpenAI` and a `cost` in the line. 
- **Pilot, arm `gemini36flash`, stopped part-way** (2026-11-14, P05-20240425-201444, 27 min, on our cluster; Anubhav
  then decided not to run the paid rows on our end, so the run was cancelled and nothing was uploaded):
  - `build`: 55 captions (2,710 input tokens per 30 s clip, about 90 per second of video; 125 output and 650
    thinking tokens each) and 5 summaries, 14 min, $0.32 ($0.27 Gemini at list price, $0.05 GPT-4.1).
  - `qa`: 43 of the 59 questions answered before the stop, $1.39, about $0.032 a question: per question 2.6
    `rag`, 1.0 `video_llm` (ranges of 60 s at the median, 73 s on average) and 1.7 `vlm` calls, 5.8 turns, 76 s;
    7.4 GB of GPU memory. Every GPT line has `provider` `OpenAI`, every Gemini line `gemini-3.6-flash`.
  - Two things the run showed about the port as it is: 8 of the 23 GPT-4.1 summaries (memory and cutoff views)
    ended at the port's 1,000-token limit, so those are cut short; and one `video_llm` range of 1,329 s was
    refused by upstream's tool (10 minutes at most), the agent getting its error string.

### Cost estimate

`epic/plan_cost.py` (calls nothing; needs `ffprobe`) prices a run from the videos' lengths and the questions of
any number of banks, with the sizes measured in that pilot (its docstring says which numbers are measured) and
list prices: Gemini 3.6 Flash $0.75 in / $3.75 out per million tokens through 2026-12-31 (thinking is billed as
output; one input rate, video and audio included; no prompt-size tier; Google's pricing page, 2026-11-14), GPT-4.1
$2 / $8. Cached-input discounts are not counted. Arm `gemini36flash-gpt41`, run on 2026-11-14:

| scope | memory | cutoff views | answers, mid (low to high) | total, mid | Gemini 3.6 Flash | GPT-4.1 |
|---|---|---|---|---|---|---|
| every bank of the board: 5,354 questions, 107 videos, 39.1 h (batch10, t2, t3, t4, t5, rescue, extra) | $29 | $29 | $152 ($94 to $214) | **$210** ($152 to $271) | $61 | $149 |

- mid is the mean tool use of the 43 pilot questions (2.5 `rag`, 1.0 `video_llm` of 44 s, 1.7 `vlm`, one final
  answer: $0.029 a question); low and high are the 10th and 90th percentile of each count.
- Checked against the pilot, with arm `gemini36flash`'s models: the planner's formulas give $0.33 for that video's memory
  (measured $0.32) and $0.032 a question before cutoff views (measured $0.028, where 61% of the `rag` input was
  billed as cached).
- One video and 43 questions carry the whole estimate. From 2027-01-01 Gemini 3.6 Flash costs twice as much.

```bash
B=<epic-morph>/outputs/v3/banks
python baselines/egor1/epic/plan_cost.py --videos <HD-EPIC videos> --arm gemini36flash-gpt41 \
    --bank_dir $B/batch10_agree2 $B/t2_agree2 $B/t3_agree2 $B/t4_agree2 $B/t5_agree2 $B/rescue_agree2 $B/extra_agree2
```

### Through another gateway to Gemini (no Files API)

What the port asks of Google's API, in both places that call Gemini (the captions, `epic/apis.py` `gemini_video`;
the `video_llm` tool, upstream's `video_llm_sync`), through the `google-genai` SDK (1.20.0 in the image):

| feature | used | where |
|---|---|---|
| Files API: `files.upload` of an mp4, then `files.get` until the file is `ACTIVE` | **yes, by default** | both. Captions upload each 30 s clip (704 px, 30 fps, with audio: 5.0 MB on average, 6.6 MB at most for P05-20240425-201444); `video_llm` uploads the range stitched by upstream at 1 fps, 448 px, with audio (measured: 4.0 MB for 60 s, 14.8 MB for 5 min, 27.4 MB for the 10 min upstream allows) |
| `files.delete` | captions only | upstream's `video_llm` leaves its uploads to Google's expiry |
| `models.generate_content` with `[file, text]`, one request and one reply | yes | both; no system instruction and no generation settings (the model's defaults, thinking included) |
| `model_version` and `usage_metadata` of the reply | yes | `apis.track()`: a reply whose `model_version` does not start with the id asked stops the run (exit 6), so **a gateway has to return it**; token counts are logged when present |
| inline video (`inline_data`) | only with `EGOR1_GEMINI_INLINE=1` | below |
| context caching, streaming, the Batch API, function calling, grounding | no | |

`EGOR1_GEMINI_INLINE=1` (opt-in; default unchanged) sends the video inside the request instead: `apis.track()`
answers the SDK's `files.upload` with a stand-in and puts the file's bytes into the `generate_content` request
as a `video/mp4` part, for the captions and for upstream's `video_llm` alike (no change to upstream's code).
`GOOGLE_GEMINI_BASE_URL`, which the SDK reads, points it at another endpoint that speaks Google's API;
`docker/run.sh` passes both variables on (and `OPENAI_BASE_URL`, which the OpenAI SDK reads with
`OPENAI_API_KEY`, for the GPT-4.1 calls). **Not tried against any endpoint**: checked only without a call, in the
image, that with the switch the request holds the clip's bytes and no upload is made, and that without it the
upload is made as before. The request grows with the range (sizes above, a third more in base64), so the
endpoint's request limit decides whether the longest ranges fit; the longest range the agent asked for in the
pilot was 249 s. A gateway that only speaks OpenAI's chat API cannot be used this way.

## What comes back

Per video, at `egor1/<arm>/<video>/`: the four memory files; `cutoffs/c<seconds>/{10min,1hour,1day}.json`;
`answers_<bank>.json.gz` (one row per question in bank order, keyed by the bank's `key` plus `qhash`, with the
chosen option, `evaluate`, the cutoff, and `turns`: each turn's agent output, parsed tool call, the call as sent on after the streaming check (`sent`) and the
returned information; `version` and `memory_hash`); `api_usage.jsonl` (one line per API call, upstream's included: the step, the model id asked and the one the API
answered with, input, output and thinking tokens); `runs.jsonl` (per finished stage: commits, the model of each
step, GPU, seconds, and `api_usage`: calls and tokens of that stage per step and answering model).

## Open decisions

- **Arm `gemini25pro` cannot run on our key** (top of this file: 404 on every call, text-only included;
  `gemini-2.5-flash` the same; the first `build` on our cluster stopped at its first caption, before any billed
  call). For a key that has the model, it ends on 2026-12-01: the whole run would have to fit before that date.
- **Gemini 3.6 Flash's price doubles on 2027-01-01** (pricing page, 2026-11-14: $0.75 in / $3.75 out per million
  tokens through 2026-12-31, then $1.50 / $7.50; output includes thinking; one input rate, no prompt-size tier).
- **Caption and summary prompts** are not published; ours could be replaced by the authors' if they share them.
- **Fork**: `Ego-R1/` is a submodule of upstream itself, patched in place at run time; a morph-graphs fork with
  the patch as its `epic-morph` branch would be cleaner.
