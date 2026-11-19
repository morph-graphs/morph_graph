# baselines/egagent: EGAgent on EPIC-Morph

[EGAgent](https://github.com/facebookresearch/egagent) (Rege et al., "Agentic Very Long Video Understanding",
ACL 2026, arXiv 2601.18157; code CC-BY-NC 4.0) answers questions about very long egocentric video with a
planning agent over three tools: frame search (SigLIP 2 embeddings of 1 fps frames), an entity scene graph
queried with SQL (people, objects and locations, and timed relations between them, extracted from captions),
and transcript search. Here it runs on EPIC-Morph's HD-EPIC videos and multiple-choice questions **with the
models it was published with**: GPT-4.1 builds the memory, and GPT-4.1 or Gemini 2.5 Pro is the agent. Those
are paid APIs; the embedder (SigLIP 2) is local. One arm is not the paper's: `gemini36flash` (Gemini 3.6 Flash
as the agent) stands in for the Gemini 2.5 Pro arm, which Google no longer serves to us (see Read first).

## Read first

- **What has run for real** (2026-11-14): the memory of all ten `batch10_agree2` videos (GPT-4.1 through
  OpenRouter, in the dataset under `egagent/gpt41/`), and the stand-in arm `gemini36flash` on the 59 questions
  of P05-20240425-201444; see Cost for what both took. The paper's two agents (GPT-4.1, Gemini 2.5 Pro) have
  only run against fake APIs (`epic/stub_check.py`): no agent call of theirs has been paid for.
- **Google refuses `gemini-2.5-pro` on our Gemini API key** (2026-11-14, before the shutdown date): the model
  is listed, and every `generateContent` call, a text-only one included, returns `404 NOT_FOUND: This model
  models/gemini-2.5-pro is no longer available to new users`. The `gemini25pro` arm's `qa` failed on every
  question that way (nothing billed) and was stopped; the collaborators' gateway does not have the model
  either. The arm's code stays as it is: it documents the published setting, and runs for anyone Google
  still serves the model to.
- **`gemini36flash` stands in for it** (Gemini 3.6 Flash as the agent, on the same `gpt41` memory). Anubhav's
  choice of stand-in (2026-11-14) was Gemini 3.5 Flash; Google routes `gemini-3.5-flash` requests to
  `gemini-3.6-flash` (its deprecations page says so, and a call on our key answered with `modelVersion`
  `gemini-3.6-flash`), so the arm asks for, and is named for, the model that answers. Everything else about
  the agent is upstream's Gemini 2.5 Pro path: prompts, tools, 1408 px frames, at most 5 plan steps,
  temperature 0. Two things had to follow the model's name, both in `epic_morph.patch`:
  - upstream tests `agent_backbone == 'gemini-2.5-pro'` in three places (the client, and the Gemini variant of
    the frame-search query schema); they now match any Gemini backbone;
  - the pinned `langchain-google-genai` (2.1.12) forces the function call of a structured output only for
    models named `gemini-1.5-*` or `gemini-2*`. Left free, 3.6 Flash answered some structured requests in
    text and the parser returned nothing: 4 of the first 5 questions failed (`'NoneType' object has no
    attribute 'sql_queries'` / `'mcq_prediction'`). Every Gemini backbone now gets the forced call that
    2.5 Pro gets; with it the 59 pilot questions parsed.
- **Both models are being retired** (the providers' deprecation pages, read 2026-11-13):
  - `gemini-2.5-pro`: shutdown **2026-12-01**; from then Google routes requests for it to `gemini-3.8-flash`
    ([Google](https://ai.google.dev/gemini-api/docs/deprecations)). The `gemini25pro` arm refuses to start
    from that date.
  - `gpt-4.1` (`gpt-4.1-2025-04-14`): deprecated 2026-10-14, shutdown **2027-01-14**
    ([OpenAI](https://developers.openai.com/docs/deprecations)). Every GPT-4.1 stage refuses from that date.
  - Every reply's model id is checked: a reply from any other model stops the run (exit 4) and nothing from
    it is kept. Gemini's client does not always report the id; such calls are counted as `unreported`.
- **Cost** (estimate, list prices, `batch10_agree2`: 10 videos, 5.66 h, 598 questions; see Cost): about $36
  for the memory, then $160 to $260 for the `gpt41` arm's answers and $180 to $300 for `gemini25pro`'s.
  Measured: the ten memories cost $40.45; `gemini36flash` answers cost $0.11 a question on the first video,
  about $67 for the 598.
- **Keys** come from the environment only: `OPENAI_API_KEY` (the memory for both arms, and the `gpt41`
  agent) and `GEMINI_API_KEY` (the `gemini25pro` and `gemini36flash` agents). A paid stage runs when its key
  is set and stops with exit 3, nothing spent, when it is not. Never put a key in a file, an argument or a log. With
  `OPENROUTER_API_KEY` and no `OPENAI_API_KEY`, the GPT-4.1 calls go through OpenRouter (see Through
  OpenRouter); Gemini is only ever called through Google's API. Each paid script is started without the
  other provider's key.
- **How Gemini is called** (2026-11-15): upstream's client (`langchain-google-genai` 2.1.12) over HTTPS + JSON
  (`transport="rest"`), as Google's `google-genai` SDK calls it in our other Gemini baselines (`api`, Ego-R1);
  upstream's default was gRPC. On Max's machines the same `GEMINI_API_KEY` that works for those baselines got
  `400 API key not valid` over gRPC, on every question (whether REST alone cures that there was not yet known
  when this was written). `GOOGLE_GEMINI_BASE_URL` (the variable that SDK reads;
  passed on by `docker/run.sh`) points the calls at another endpoint that speaks Google's API. Every Gemini
  usage row records the `endpoint` the client called.
- **A failed stage says why on its last line**: after the `API calls:` summary, `<step> failed: <exception>:
  HTTP <status> [<code>] <message>` (or `<n> questions unanswered; last error ...`, or the other model's id for
  exit 4), cut to 220 characters, so the tail of a log (a job board note keeps about 400) shows the cause.
- **What is not the authors'** (each forced by the benchmark or by what upstream released):
  - *The caption prompt is ours* (`epic/caption_epic.py`): upstream released neither its captioning script
    nor the prompt.
  - *Streaming.* A question asked at `t_ask` may only use the video up to `t_ask`. Upstream extracts one
    graph per hour (EgoLife) or per video (Video-MME); here the graph is extracted per 300 s chunk
    (`EGAGENT_CHUNK_S`), each from its own captions only, and a question sees the chunks that end by its
    `t_ask` and the frames of seconds up to `floor(t_ask)`. For each cutoff `epic/run_qa.py` writes upstream's
    two SQLite tables with only those rows, under a video name unique to the cutoff, and upstream's tools read
    those: no tool can return anything later. So the graph lags `t_ask` by up to 300 s; the frame search does not.
  - *Stitched videos.* A kitchen's stitched video (`<kitchen>-stitched`, bank `stitched`) is one video on one clock,
    and its questions are asked mid-video like any streaming one. As for MERIT and WorldMM, the kitchen's sessions
    started by the cutoff (`../common/stitched_sessions.py`: each session's date, time and where it starts in the
    video) are put before the question the agent gets; the answer row keeps the bank's wording in `question` and
    the table in `context`. Checked against the fake APIs (`epic/stub_check.py --video <kitchen>-stitched`); no
    stitched video has been answered for real yet.
  - *No transcript.* HD-EPIC has none and the method has no speech recogniser: captions are not fused with a
    transcript and the transcript tool gets upstream's "NO TRANSCRIPTS AVAILABLE", as upstream does for a
    Video-MME video without subtitles.
  - *Times.* Frames are named `HHMMSS00.jpg` (EgoLife's naming) and the question time is passed as
    `HHMMSS00`: upstream's Video-MME path names frames by frame number and passes `HHMMSS`, which its own time
    formatter prints without the seconds.
  - *The agent path* is upstream's Video-MME workflow (one video, elapsed time) on the EgoLife memory recipe
    (GPT-4.1 captions of 30 s clips); upstream's EgoLife agent is tied to seven named days and one wearer.
  - Upstream's Video-MME table builders do not run as released (`build_db_videomme` creates no table; the
    frame table is read from embedding files that were not released), so `run_qa.py` writes both tables with
    upstream's schemas.
- **Kept as released**, quirks included: the frame search compares its `HHMMSS` window against `HHMMSS00`
  time stamps as text, so the last second of a window the agent chooses is left out (it can only hide a
  frame, never show a later one).

## Layout

| path | what |
|---|---|
| `egagent/` | submodule: upstream [facebookresearch/egagent](https://github.com/facebookresearch/egagent) at `01a48a4`, unmodified in git |
| `epic_morph.patch` | our patch to upstream, applied to the submodule's files by `run_egagent.sh` before anything runs (it refuses to run if the patch does not apply): settings from the environment instead of edited files, API keys from the environment only, the workflow graph as a function (`build_app`) |
| `epic_morph_trace.patch` | applied after it, logging only: the agent's tool steps per question (plan, SQL written and rows kept, frame searches and frames found), which `run_qa.py` writes as each row's `trace` |
| `epic/prepare_epic.py` | one HD-EPIC video -> 1 fps frames, as upstream cuts them |
| `epic/embed_frames.py` | SigLIP 2 embeddings of the frames with upstream's code (GPU, local) |
| `epic/caption_epic.py` | GPT-4.1 captions of every 30 s of frames (paid) |
| `epic/build_graph.py` | the entity graph from the captions with upstream's functions and prompts (paid) |
| `epic/run_qa.py` | answers a video's questions, each from the memory up to its `t_ask` (paid); `--blind`; `--retrieval_check` (free) |
| `epic/eg_paths.py` | puts upstream on `sys.path`; the memory's file names; the paper's models; the watch on every API call (model id returned, tokens) |
| `epic/estimate_cost.py` | what the paid calls would cost, from the calls the code makes |
| `epic/stub_check.py` | runs the paid scripts against fake APIs and checks that nothing after a question's cutoff is sent |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset, or download them |
| `run_egagent.sh` | one video, one arm, one stage (prep, build, qa, blind, upload, all; check) |
| `env/` | the environment, pinned: `requirements-client.txt`, `install_requirements.sh`, `setup_env.sh` |
| `docker/` | the image with that environment (`Dockerfile`, `image_tag.sh`, `build.sh`), and `run.sh` to run `run_egagent.sh` in it with Docker or Apptainer |

After `run_egagent.sh` has run once, git shows the submodule as modified: that is the patch. `git -C
baselines/egagent/egagent checkout .` takes it off again.

## What runs where, and with which model

| step | model (as published) | where | calls |
|---|---|---|---|
| frames | - | ffmpeg, 1 fps, the video's own resolution (1408 px) | - |
| frame embeddings | `google/siglip2-giant-opt-patch16-384` | local GPU (about 5 min for a 27 min video on an A5000) | - |
| captions | GPT-4.1 | OpenAI API (or through OpenRouter) | 1 per 30 s of video, 30 frames each |
| entity graph | GPT-4.1 (LangChain `LLMGraphTransformer`, then the time-stamper) | OpenAI API (or through OpenRouter) | 2 per 300 s |
| agent, arm `gpt41` | GPT-4.1 | OpenAI API (or through OpenRouter) | about 12 per question (planner; per step its tools and a grader; answer); a frame search sends up to 50 frames |
| agent, arm `gemini25pro` | Gemini 2.5 Pro (the paper's best, 57.5% on EgoLifeQA) | Google API | the same |
| agent, arm `gemini36flash` | Gemini 3.6 Flash (not the paper's: the stand-in for `gemini25pro`) | Google API | measured: 8.9 per question (see Cost) |
| text queries for frame search | SigLIP 2 text tower | local GPU or CPU | - |

**One memory per video, shared by every arm**: `build` for another arm copies the first's (from its results
folder, or from the dataset). Settings are upstream's: 30 s captions at 1 fps, SigLIP 2 giant, at most 5 plan
steps, top 50 frames per frame search and at most 50 frames per analysis call, temperature 0, LLM transcript
search.

## Running

Off our clusters, in the image (`ghcr.io/morph-graphs/egagent:gpu-<env hash>`, one image for both arms), from a
checkout of `main` with the submodule (`git submodule update --init baselines/egagent/egagent`):

```bash
docker pull ghcr.io/morph-graphs/egagent:$(bash baselines/egagent/docker/image_tag.sh)
export EPIC_MORPH_DIR=<epic-morph checkout> EGAGENT_GPU=0
export OPENAI_API_KEY=...        # the memory (both arms) and the gpt41 agent
export GEMINI_API_KEY=...        # the gemini25pro agent
bash baselines/egagent/docker/run.sh all    P05-20240425-201444 gpt41 batch10_agree2        # prep, build, qa, blind
bash baselines/egagent/docker/run.sh all    P05-20240425-201444 gemini25pro batch10_agree2  # reuses the memory
bash baselines/egagent/docker/run.sh qa     P05-20240425-201444 gemini36flash batch10_agree2  # the stand-in: memory from the dataset, GEMINI_API_KEY, no GPU
bash baselines/egagent/docker/run.sh upload P05-20240425-201444 gpt41                       # HF_TOKEN or `hf auth login`
bash baselines/egagent/docker/run.sh upload P05-20240425-201444 gemini25pro
```

`docker/run.sh` mounts the checkout at `/repo`, the epic-morph checkout at `/epic-morph`, the videos at
`/videos` and the Hugging Face cache at `/hf` (SigLIP 2 is downloaded into it on first use, about 7 GB), and
hands the keys to the container by name. `EGAGENT_RUNTIME=apptainer` runs the same image with Apptainer; the
job board's `MERIT_GPU`, `MERIT_RUNTIME`, `MERIT_SIF_DIR` (`egagent_gpu.sif`), `MERIT_HF_CACHE` and `MERIT_WORK`
are read when the `EGAGENT_` ones are unset.

### Through OpenRouter (our own builds)

With `OPENROUTER_API_KEY` in the environment and `OPENAI_API_KEY` not, every GPT-4.1 call (the captions, the
graph extraction and its time-stamper, and the `gpt41` arm's agent) goes through OpenRouter, so the memory can
be built on our cluster without an OpenAI account. Nothing else is set; the switch is in `epic_morph.patch`
(upstream's `utils.py`, where its two OpenAI clients are made).

```bash
export OPENROUTER_API_KEY=...      # typed into the shell (or read -s): never an argument, a file, a log or a commit
unset OPENAI_API_KEY
bash baselines/egagent/docker/run.sh build P05-20240425-201444 gpt41                       # the memory: GPT-4.1 through OpenRouter
export GEMINI_API_KEY=...          # Google's own API for the Gemini agent (without it: through OpenRouter, below)
bash baselines/egagent/docker/run.sh qa    P05-20240425-201444 gemini25pro batch10_agree2  # takes the gpt41 folder's memory
```
For an sbatch job, read the key into the job's environment inside the job and do not write it into the script.

- **What changes in the request.** The endpoint (`https://openrouter.ai/api/v1`; `OPENROUTER_BASE_URL` overrides
  it), the key, the model id (`openai/gpt-4.1`) and one added field,
  `provider: {"only": ["openai"], "allow_fallbacks": false}`: with the pin OpenAI itself answers or the call
  fails. Prompts, frames and their resolution, temperature 0 and the structured-output schemas are what the
  port sends to OpenAI: the same LangChain `ChatOpenAI` client makes the same Chat Completions request.
- **What may still differ from the direct API**: the snapshot is OpenRouter's alias to resolve
  (`openai/gpt-4.1`; a reply names what answered, and any id that is not GPT-4.1 or one of its dated
  snapshots stops the run, exit 4), and OpenRouter sits between the request and OpenAI.
- **Which arm calls what.** `build` calls GPT-4.1 only. `qa` and `blind` of `gemini25pro` call Gemini 2.5 Pro
  only: every agent worker (planner, the three tools' query writers, frame analysis, transcript search,
  grader, answer) is built on the arm's backbone, and the watch on the calls stops the run if any other model
  answers. `gpt41`'s `qa` and `blind` call GPT-4.1 only.
- **What is recorded.** `api_usage.jsonl` and the stage's `runs.jsonl` line (`api`) key each row by the model
  id the endpoint returned, with calls, input / cached / output / reasoning tokens, and the `endpoint` (GPT and
  Gemini rows); through OpenRouter also `provider` (who served the calls) and `cost` (dollars OpenRouter
  charged, summed). The memory folder stays `egagent/gpt41/<video>/`: the same model through another gateway.
- **One gateway per run.** With both keys set OpenAI's is used and `OPENROUTER_API_KEY` is dropped.
- **The Gemini agent through OpenRouter, a fallback only.** With `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) set nothing
  changes: Gemini is called through Google's API as before, whatever else is set. Only with `OPENROUTER_API_KEY` and no
  Gemini key (our own test runs),
  a Gemini arm's calls go to OpenRouter as `google/<model>` (e.g. `google/gemini-3.6-flash`), pinned to the
  provider `google-ai-studio` (Google's Gemini API) with no fallback; reply ids `google/gemini-...` count as the
  model. Unlike the GPT route this is not the same client: the requests go through LangChain's `ChatOpenAI`
  instead of `langchain-google-genai`, so the structured outputs are asked the OpenAI way. Answers from this route
  are comparable with each other, not with answers made with `GEMINI_API_KEY`: give them their own `EGAGENT_TAG`.
- **An OpenAI-compatible gateway.** With `OPENAI_API_KEY`, `OPENAI_BASE_URL` (passed on by `docker/run.sh`)
  points the same requests at another endpoint, model id `gpt-4.1` unchanged and no provider pin. A gateway
  serving GPT-4.1 from Azure OpenAI answers as `azure/openai/gpt-4.1`, which counts as GPT-4.1; the `endpoint`
  of every usage row says which gateway it was. A gateway that answers with its deployment's own label:
  `OPENAI_MODEL_ALIASES="gpt-4.1=<label>"` counts that label as GPT-4.1 (the label is what gets recorded, so
  pass it through; never rewrite it to `gpt-4.1`). `OPENAI_PROVIDER_NOTE` (e.g. "azure-openai, via local
  relay") is recorded in every GPT usage row as `provider_note`, with the replies' `system_fingerprints`. A
  gateway that names the model differently (NVIDIA's: `azure/openai/gpt-4.1`; `gpt-4.1` gets a 403):
  `EGAGENT_GPT_PREFIX=azure/openai/` is put before every GPT model id the port asks for.
- **Check without a key**: `epic/stub_check.py ... --gateway openrouter` runs the paid scripts against the
  fake APIs with only an OpenRouter key set and checks, on every GPT request, the host, the `openai/gpt-4.1`
  id, the provider pin and the key used, and that the usage rows hold the endpoint, provider and cost; without
  `--gateway` it checks that the OpenAI route is what it was, also when an OpenRouter key is set as well.

On our clusters, a virtualenv (Python 3.11, torch 2.6.0 CUDA 12.4, upstream's pip pins):

```bash
bash baselines/egagent/env/setup_env.sh <dir>/egagent-client
export EPIC_MORPH_DIR=<epic-morph checkout> PY=<dir>/egagent-client/bin/python
bash baselines/egagent/run_egagent.sh build P05-20240425-201444 gpt41                  # embeddings; stops before GPT-4.1 without a key
bash baselines/egagent/run_egagent.sh check P05-20240425-201444 gpt41 batch10_agree2   # tables + frame search per cutoff, no API
```

### Parallel answering

`EGAGENT_QA_WORKERS=6` (passed on by `docker/run.sh`) has `qa` answer six questions at a time in threads; the
default, 1, answers one at a time as before. Each question's agent is already independent: a fresh graph state,
LLM clients that keep nothing between calls, SQLite connections opened per call. What `epic/run_qa.py` adds for
threads (`per_question_state`): the trace lists of `epic_morph_trace.patch` become one list per question (a
ContextVar), SigLIP 2's text encoder runs behind a lock, each cutoff's two tables are written once by the first of
its questions and deleted after its last, every question runs in a copy of the main thread's context (so the
usage watch counts its calls, under a lock, and still stops a wrong model), and only the main thread writes rows.
The answers file is in bank order either way. Measured on P01-20240203-123350, bank probes_v2 (36
questions), `gemini36flash` through OpenRouter (2026-11-16): one worker 1,505 s and 1,565 s (two runs), six workers
297 s, with the same calls (305 / 303 / 308), tokens (3.8-3.9 M in) and cost ($3.63 / $3.68 / $3.66). The two
one-worker runs agree on 17 of 36 letters, one-vs-six on 25 and 22: the agent's own run-to-run spread (API
sampling over ~8 calls a question) is too large for live runs to show a small difference. `epic/parallel_check.py`
checks the threads exactly, against the fake APIs (no cost): every question's prompts, trace, working memory and
answer are the same with one worker and with six, and every call is counted. Six workers send about one call a
second and ~0.8 M input tokens a minute per task; a rate-limited (429) call is retried by the client, then the
question by the next pass.

Every stage skips finished work; captions, graph chunks and answers resume from where they stopped, and a
question that failed is retried (three passes per run). The frames (about 0.16 MB each, one per second of
video: 253 MB for a 27 min video), per-question rows and logs stay in `baselines/egagent/work/`
(`EGAGENT_WORK`); results are written to `baselines/egagent/results/<arm>/<video>/` (`EGAGENT_OUT`) and
`upload` puts them in the private dataset
[morph-graphs/epic-morph-results](https://huggingface.co/datasets/morph-graphs/epic-morph-results) at
`egagent/<arm>/<video>/`.

## Cost

`python baselines/egagent/epic/estimate_cost.py --hours <video hours> --questions <n> [--frame_px N]` counts
the calls from the code and prices them at list prices (GPT-4.1 $2 / $8, Gemini 2.5 Pro $1.25 / $10 per
million input / output tokens). Tokens per call are estimates (image tokens from the providers' formulas;
text and output lengths assumed) until a first video has run: then `runs.jsonl` holds the measured tokens per
stage. For `batch10_agree2` (5.66 h, 598 questions):

| frames sent | memory (GPT-4.1) | answers, `gpt41` | answers, `gemini25pro` |
|---|---|---|---|
| 1408 px, as published (default) | $36 | $160 to $260 | $180 to $300 |
| 512 px (`EGAGENT_FRAME_PX=512`) | $15 | $70 to $110 | $180 to $300 (Gemini bills the same tiles) |
| 384 px (`EGAGENT_FRAME_PX=384`) | $15 | $70 to $110 | $100 to $155 |

Frames are nearly all of it: a caption call carries 30 frames and a frame search up to 50. `EGAGENT_FRAME_PX`
is not the published setting and makes a different memory (give it its own `EGAGENT_WORK` and `EGAGENT_OUT`);
the default stays the video's own resolution, which is what upstream sends.

**Measured, the memory of one video** (2026-11-14, P05-20240425-201444, 1,630 s, 1408 px frames, GPT-4.1
through OpenRouter, from its `runs.jsonl`; every reply named `openai/gpt-4.1` and the provider OpenAI):

| step | calls | input tokens | output tokens | charged |
|---|---|---|---|---|
| captions | 55 | 1,251,295 (22,751 a call: 30 frames of 765) | 34,793 (633 a call) | $2.78 |
| graph extraction + time stamps | 12 | 98,421 | 36,098 (3,008 a call) | $0.49 |
| the memory | 67 | 1,349,716 | 70,891 | **$3.27** |

That is $7.21 per video-hour, 13% above the estimate's $6.36 (the replies are about twice as long as
assumed; the frames, most of the cost, are as computed). The other nine memories, built the same day on A4000s
(9.3 GB of GPU memory at most, as on the A5000), cost $37.18 (763 calls, 15.5M input and 0.77M output tokens,
$7.04 to $7.31 per video-hour), so `batch10_agree2`'s ten memories cost **$40.45**, not the estimated $36.
With the frames to cut first, a build took 16 to 20 minutes for a 30-minute video on an A4000 (26 for the 55
minute one). OpenRouter charged the list price (tokens x $2 / $8 per million gives the same $3.27; no cached
input). The stage took 611 s on an A5000 (about 4 min of embedding with the model's load, 3.5 min of
captions, 2.7 min of graph), 9.3 GB of GPU memory at most and 6.0 GB of host RAM; the frames before it, 184 s.

**Measured, the stand-in arm's answers on that video** (`gemini36flash`, 2026-11-14, the 59 questions of
`batch10_agree2`, Gemini 3.6 Flash through Google's API, from its `runs.jsonl` and answers file; every reply
named `gemini-3.6-flash`, and the arm made no other API call):

| per question | calls | input tokens | output tokens (thinking in it) | at list price |
|---|---|---|---|---|
| mean of the 59 | 8.9 | 111,689 | 7,412 (6,212) | $0.112 |
| all 59 | 524 | 6,589,637 | 437,322 (366,485) | **$6.58** |

- Per question: 1 plan, 1.9 frame searches (a query-writing call, then an analysis call with 50 frames:
  55.9k tokens, about 1,120 a frame), 1 graph SQL call, 0.2 transcript calls, 1.9 grader calls, 1 answer. The
  frame analyses are about 95% of the tokens. Every question parsed at the first attempt.
- Price: $0.75 / $3.75 per million input / output tokens, Google's list price through 2026-12-31 ($1.50 /
  $7.50 from 2027-01-01); Google reports no charge per call, so the dollars are tokens x list price (no
  cached input was reported). At this rate the 598 questions of `batch10_agree2` come to about $67.
- 48 s a question, one at a time (2,898 s for the 59 with the model loaded twice). No GPU: SigLIP 2's text
  tower ran on a CPU node (4 CPUs, 4.4 GB of RAM used).
- The first start of this arm, before the function call was forced (Read first), spent $0.63 on 54 calls and
  answered 3 of 5 questions in two passes; those rows were set aside and are not in the answers file.

## What comes back (in the Hugging Face dataset)

Per video, at `egagent/<arm>/<video>/`:
- the memory: `<video>_frames.npy` (one 1536-d SigLIP 2 embedding per second), `<video>_captions.json`
  (`start`, `end`, `caption` per 30 s), `<video>_graph.json` (per chunk: its relationships with their time
  intervals);
- `answers_<bank>.json.gz`: one row per question in bank order, keyed by the bank's question `key` (and `ID`,
  its 1-based position): `pred`, `evaluate`, `justification`, the `plan`, the `working_memory` the tools
  filled, `total_tokens` per call, `cutoff` and what was `indexed` at it;
- `blind_<bank>.json`, when the blind stage ran;
- `api_usage.jsonl`: per script run, the model ids the APIs returned with calls and tokens (input, cached,
  output with Gemini's thinking, reasoning), a GPT row's endpoint, and through OpenRouter its provider and cost;
- `runs.jsonl`: one line per finished stage (commits of this repo, upstream and epic-morph; models; the
  stage's API calls, tokens and cost per returned model id, `api`; versions; GPUs; seconds).
