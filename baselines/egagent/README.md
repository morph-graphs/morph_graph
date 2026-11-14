# baselines/egagent: EGAgent on EPIC-Morph

[EGAgent](https://github.com/facebookresearch/egagent) (Rege et al., "Agentic Very Long Video Understanding",
ACL 2026, arXiv 2601.18157; code CC-BY-NC 4.0) answers questions about very long egocentric video with a
planning agent over three tools: frame search (SigLIP 2 embeddings of 1 fps frames), an entity scene graph
queried with SQL (people, objects and locations, and timed relations between them, extracted from captions),
and transcript search. Here it runs on EPIC-Morph's HD-EPIC videos and multiple-choice questions **with the
models it was published with**: GPT-4.1 builds the memory, and GPT-4.1 or Gemini 2.5 Pro is the agent. Those
are paid APIs; the embedder (SigLIP 2) is local.

## Read first

- **Nothing paid has run yet.** Frames, embeddings and the per-question tables have run on a real video; the
  captions, graph, answer and blind stages have only run against fake APIs (`epic/stub_check.py`). The first
  real video will show what the method costs and whether the models' replies parse.
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
- **Keys** come from the environment only: `OPENAI_API_KEY` (the memory for both arms, and the `gpt41`
  agent) and `GEMINI_API_KEY` (the `gemini25pro` agent). A paid stage runs when its key is set and stops
  with exit 3, nothing spent, when it is not. Never put a key in a file, an argument or a log.
- **What is not the authors'** (each forced by the benchmark or by what upstream released):
  - *The caption prompt is ours* (`epic/caption_epic.py`): upstream released neither its captioning script
    nor the prompt.
  - *Streaming.* A question asked at `t_ask` may only use the video up to `t_ask`. Upstream extracts one
    graph per hour (EgoLife) or per video (Video-MME); here the graph is extracted per 300 s chunk
    (`EGAGENT_CHUNK_S`), each from its own captions only, and a question sees the chunks that end by its
    `t_ask` and the frames of seconds up to `floor(t_ask)`. For each cutoff `epic/run_qa.py` writes upstream's
    two SQLite tables with only those rows, under a video name unique to the cutoff, and upstream's tools read
    those: no tool can return anything later. So the graph lags `t_ask` by up to 300 s; the frame search does not.
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
| captions | GPT-4.1 | OpenAI API | 1 per 30 s of video, 30 frames each |
| entity graph | GPT-4.1 (LangChain `LLMGraphTransformer`, then the time-stamper) | OpenAI API | 2 per 300 s |
| agent, arm `gpt41` | GPT-4.1 | OpenAI API | about 12 per question (planner; per step its tools and a grader; answer); a frame search sends up to 50 frames |
| agent, arm `gemini25pro` | Gemini 2.5 Pro (the paper's best, 57.5% on EgoLifeQA) | Google API | the same |
| text queries for frame search | SigLIP 2 text tower | local GPU or CPU | - |

**One memory per video, shared by both arms**: `build` for the second arm copies the first's (from its results
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
bash baselines/egagent/docker/run.sh upload P05-20240425-201444 gpt41                       # HF_TOKEN or `hf auth login`
bash baselines/egagent/docker/run.sh upload P05-20240425-201444 gemini25pro
```

`docker/run.sh` mounts the checkout at `/repo`, the epic-morph checkout at `/epic-morph`, the videos at
`/videos` and the Hugging Face cache at `/hf` (SigLIP 2 is downloaded into it on first use, about 7 GB), and
hands the keys to the container by name. `EGAGENT_RUNTIME=apptainer` runs the same image with Apptainer; the
job board's `MERIT_GPU`, `MERIT_RUNTIME`, `MERIT_SIF_DIR` (`egagent_gpu.sif`), `MERIT_HF_CACHE` and `MERIT_WORK`
are read when the `EGAGENT_` ones are unset.

On our clusters, a virtualenv (Python 3.11, torch 2.6.0 CUDA 12.4, upstream's pip pins):

```bash
bash baselines/egagent/env/setup_env.sh <dir>/egagent-client
export EPIC_MORPH_DIR=<epic-morph checkout> PY=<dir>/egagent-client/bin/python
bash baselines/egagent/run_egagent.sh build P05-20240425-201444 gpt41                  # embeddings; stops before GPT-4.1 without a key
bash baselines/egagent/run_egagent.sh check P05-20240425-201444 gpt41 batch10_agree2   # tables + frame search per cutoff, no API
```

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

## What comes back (in the Hugging Face dataset)

Per video, at `egagent/<arm>/<video>/`:
- the memory: `<video>_frames.npy` (one 1536-d SigLIP 2 embedding per second), `<video>_captions.json`
  (`start`, `end`, `caption` per 30 s), `<video>_graph.json` (per chunk: its relationships with their time
  intervals);
- `answers_<bank>.json.gz`: one row per question in bank order, keyed by the bank's question `key` (and `ID`,
  its 1-based position): `pred`, `evaluate`, `justification`, the `plan`, the `working_memory` the tools
  filled, `total_tokens` per call, `cutoff` and what was `indexed` at it;
- `blind_<bank>.json`, when the blind stage ran;
- `api_usage.jsonl`: per script run, the model ids the APIs returned with calls and tokens;
- `runs.jsonl`: one line per finished stage (commits of this repo, upstream and epic-morph; models; the
  stage's API calls and tokens per returned model id, `api`; versions; GPUs; seconds).
