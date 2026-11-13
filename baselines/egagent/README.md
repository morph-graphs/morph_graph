# baselines/egagent: EGAgent on EPIC-Morph

[EGAgent](https://github.com/facebookresearch/egagent) (Rege et al., "Agentic Very Long Video Understanding",
ACL 2026, arXiv 2601.18157; code CC-BY-NC 4.0) answers questions about very long egocentric video with a
planning agent over three tools: frame search (SigLIP 2 embeddings of 1 fps frames), an entity scene graph
queried with SQL (people, objects and locations, and timed relations between them, extracted from captions),
and transcript search. Here it runs on EPIC-Morph's HD-EPIC videos and multiple-choice questions **with the
models it was published with**, not ours: GPT-4.1 builds the memory, GPT-4.1 or Gemini 2.5 Pro is the agent.
Those are paid APIs, so every stage that calls one stops unless `EGAGENT_PAID_OK=1` is set.

**State (2026-11-13): the free part (frames, embeddings, per-cutoff tables and frame search) has run on one video; no paid call has been made**, so the captions,
graph, answer and blind stages are written but have never run.

## Layout

| path | what |
|---|---|
| `egagent/` | upstream [facebookresearch/egagent](https://github.com/facebookresearch/egagent) at `01a48a4` + `epic_morph.patch` applied (not committed here; to become a fork submodule, branch `epic-morph`) |
| `epic_morph.patch` | our patch to upstream: settings from the environment instead of edited files, API keys from the environment, the workflow graph as a function (`build_app`) |
| `epic/prepare_epic.py` | one HD-EPIC video -> 1 fps frames, as upstream cuts them |
| `epic/embed_frames.py` | SigLIP 2 embeddings of the frames with upstream's code (GPU, local) |
| `epic/caption_epic.py` | GPT-4.1 captions of every 30 s of frames (paid) |
| `epic/build_graph.py` | the entity graph from the captions with upstream's functions and prompts (paid) |
| `epic/run_qa.py` | answers a video's questions, each from the memory up to its `t_ask` (paid); `--blind`; `--retrieval_check` (free) |
| `epic/estimate_cost.py` | what the paid calls would cost, from the calls the code makes |
| `epic/eg_paths.py` | puts upstream on `sys.path`; the memory's file names; the paper's models |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset, or download them |
| `run_egagent.sh` | one video, one arm, one stage (prep, build, qa, blind, check, upload, all) |
| `env/` | the environment, pinned: `requirements-client.txt`, `install_requirements.sh`, `setup_env.sh` |

Until the fork exists: `git clone https://github.com/facebookresearch/egagent baselines/egagent/egagent &&
git -C baselines/egagent/egagent checkout 01a48a4 && git -C baselines/egagent/egagent apply ../epic_morph.patch`.

## What runs where, and with which model

| step | model (as published) | where | calls |
|---|---|---|---|
| frames | - | ffmpeg, 1 fps, the video's own resolution (1408 px) | - |
| frame embeddings | `google/siglip2-giant-opt-patch16-384` | local GPU | - |
| captions | GPT-4.1 | OpenAI API | 1 per 30 s of video, 30 frames each |
| entity graph | GPT-4.1 (LangChain `LLMGraphTransformer`, then the time-stamper) | OpenAI API | 2 per `EGAGENT_CHUNK_S` (300 s) |
| agent, arm `gpt41` | GPT-4.1 | OpenAI API | about 12 per question (planner, per step: tool calls and grader, answer); a frame search sends up to 50 frames |
| agent, arm `gemini25pro` | Gemini 2.5 Pro (the paper's best, 57.5% on EgoLifeQA) | Google API | the same |
| text queries for frame search | SigLIP 2 text tower | local GPU or CPU | - |

The memory is the same for both arms, so `build` for the second arm copies the first's.

Settings are upstream's: 30 s captions at 1 fps, SigLIP 2 giant, at most 5 plan steps, top 50 frames per frame
search and at most 50 frames per analysis call, temperature 0, LLM transcript search. The agent is upstream's
Video-MME workflow (one video, elapsed time), the memory recipe is EgoLife's (GPT-4.1 captions of 30 s clips).

## What differs from upstream, and why

- **The caption prompt is ours** (`epic/caption_epic.py`). Upstream released neither its captioning script nor
  the prompt (the repo starts from finished caption files; the paper's appendix does not print it).
- **No transcript.** HD-EPIC has none and the method has no speech recogniser: captions are not fused with a
  transcript, the time-stamper gets an empty transcript list and the transcript tool gets upstream's
  "NO TRANSCRIPTS AVAILABLE", as upstream does for a Video-MME video without subtitles.
- **Streaming: the graph is extracted per 300 s chunk, not per hour or per video.** A chunk's graph is made
  from its own captions only, and a question sees the chunks that end by its `t_ask`, the frames named for
  seconds up to `floor(t_ask)`, and nothing else: for each cutoff `run_qa.py` writes upstream's two SQLite
  tables with only those rows, under a video name unique to the cutoff, and the tools read those. With
  upstream's one-graph-per-hour, a video shorter than an hour would have no graph at any streaming question;
  with smaller chunks the graph lags `t_ask` by less but each extraction sees less context
  (`EGAGENT_CHUNK_S`; the frame search has no lag).
- **Times.** Frames are named `HHMMSS00.jpg` (EgoLife's naming) and the question time is passed as `HHMMSS00`:
  upstream's Video-MME path names frames by frame number and passes `HHMMSS`, which its own time formatter
  prints without the seconds (and frame numbers as if they were clock times).
- Upstream's Video-MME graph table builder (`build_db_videomme`) does not run as released (no connection or
  table is created) and its frame table for Video-MME is read from embedding files that are not released;
  `run_qa.py` writes both tables with upstream's schemas.

## Environment

One virtualenv (Python 3.11, torch 2.6.0 CUDA 12.4, upstream's pip pins):

```bash
bash baselines/egagent/env/setup_env.sh <dir>/egagent-client
```

## Running

```bash
export EPIC_MORPH_DIR=<epic-morph checkout> PY=<dir>/egagent-client/bin/python
bash baselines/egagent/run_egagent.sh prep  P05-20240425-201444 gpt41                  # frames (ffmpeg on PATH)
bash baselines/egagent/run_egagent.sh build P05-20240425-201444 gpt41                  # embeddings, then stops before GPT-4.1
bash baselines/egagent/run_egagent.sh check P05-20240425-201444 gpt41 batch10_agree2   # tables + frame search per cutoff, no API
# paid, with the keys in the environment only:
EGAGENT_PAID_OK=1 OPENAI_API_KEY=... bash baselines/egagent/run_egagent.sh all P05-20240425-201444 gpt41 batch10_agree2
EGAGENT_PAID_OK=1 OPENAI_API_KEY=... GOOGLE_API_KEY=... bash baselines/egagent/run_egagent.sh all P05-20240425-201444 gemini25pro batch10_agree2
bash baselines/egagent/run_egagent.sh upload P05-20240425-201444 gpt41
```

Every stage skips finished work; captions, graph chunks and answers resume from where they stopped. The
frames (about 0.16 MB each, one per second of video: 253 MB for a 27 min video), per-question rows and logs stay in
`baselines/egagent/work/` (`EGAGENT_WORK`); results are written to `baselines/egagent/results/<arm>/<video>/`
(`EGAGENT_OUT`) and `upload` puts them in the private dataset
[morph-graphs/epic-morph-results](https://huggingface.co/datasets/morph-graphs/epic-morph-results) at
`egagent/<arm>/<video>/`.

## Cost

`python baselines/egagent/epic/estimate_cost.py --hours <video hours> --questions <n>` counts the calls from
the code and prices them at list prices; the tokens per call are estimates until a first video has run
(`run_qa.py` records every call's `total_tokens` per question, `caption_epic.py` per caption). On 2026-11-13:
about $6 per video hour for the memory (the captions' frames are nearly all of it) and $0.27 to $0.50 per
question for the agent.

## What comes back (in the Hugging Face dataset)

Per video, at `egagent/<arm>/<video>/`:
- the memory: `<video>_frames.npy` (one 1536-d SigLIP 2 embedding per second), `<video>_captions.json`
  (`start`, `end`, `caption` per 30 s), `<video>_graph.json` (per chunk: its relationships with their time
  intervals);
- `answers_<bank>.json.gz`: one row per question in bank order, keyed by the bank's question `key` (and `ID`,
  its 1-based position): `pred`, `evaluate`, `justification`, the `plan`, the `working_memory` the tools
  filled, `total_tokens` per call, `cutoff` and what was `indexed` at it;
- `blind_<bank>.json`, when the blind stage ran;
- `runs.jsonl`: one line per finished stage (commits of this repo, upstream and epic-morph; models; versions;
  GPUs; seconds).
