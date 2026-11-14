# baselines/egor1: Ego-R1 on EPIC-Morph

[Ego-R1](https://github.com/egolife-ai/Ego-R1) (arXiv 2506.13654, TPAMI 2026; Apache-2.0) answers questions about
very long egocentric video with a small trained agent, Ego-R1-Agent-3B (Qwen2.5-3B-Instruct, fine-tuned on
Chain-of-Tool-Thought traces, then GRPO). Each turn the agent thinks and calls one of three tools: `rag` (search
of a hierarchical text memory of the video), `video_llm` (a question about up to 10 minutes of video) and `vlm`
(a question about one frame). A summary model writes the final answer from the trace.

**This port keeps the models of the paper, with one replacement: Gemini 2.5 Pro stands in for the paper's Gemini
1.5 Pro, which is retired** (not offered to our key, 2026-11-13; Anubhav's decision of that day). It affects the
30 s captions of the memory and the `video_llm` tool; nothing is swapped for a local backbone. Only the agent has
open weights. Every other step calls a paid API, so those stages stop (exit code 5) until the keys are set:

| step | model, as published | where upstream calls it | runs here |
|---|---|---|---|
| agent | Ego-R1/Ego-R1-Agent-3B | `Ego-R1-Agent/eval/infer_bench_summ.py` (vLLM) | locally, transformers, fp16, ~8 GB GPU |
| memory: 30 s captions | Gemini 1.5 Pro; **here Gemini 2.5 Pro** | not in the repo (paper, H-RAG construction) | Gemini key |
| memory: 10 min / hour / day summaries | GPT-4.1 | not in the repo (paper) | OpenAI key |
| `rag` retrieval | GPT-4o | `api/rag/r1rag/utils.py` `call_gpt4` | OpenAI key |
| `video_llm` | Gemini 1.5 Pro; **here Gemini 2.5 Pro** (arm `gemini25pro`); LLaVA-Video-7B (arm `llava`, paper Table 4) | `api/visual_tools/egor1_videollm_{gemini,llava}` | Gemini key; or locally |
| `vlm` | GPT-4o | `api/visual_tools/egor1_vlm` | OpenAI key |
| final answer | GPT-4.1 | `infer_bench_summ.py` `summ_client` | OpenAI key |

**Gemini 2.5 Pro (`gemini-2.5-pro`) is itself listed for shutdown on 2026-12-01** (Google's deprecation page, read
2026-11-13); from that day requests are routed to another model. Every Gemini call of one comparison must be made
before then. From that date the Gemini stages refuse to run, and a caption answered by another model is an error
(`epic/apis.py`, which checks the model id in every Gemini reply, upstream's `video_llm` calls included).

## Layout

| path | what |
|---|---|
| `Ego-R1/` | submodule: upstream [egolife-ai/Ego-R1](https://github.com/egolife-ai/Ego-R1), pinned at `69177ee` (`git submodule update --init baselines/egor1/Ego-R1`) |
| `epic-morph.patch` | our three changes to upstream, applied to the submodule's files by `run_egor1.sh` on first use (so `git status` shows the submodule as modified; a morph-graphs fork with this as a branch would be cleaner): `video_llm` takes the Gemini model id from `EGOR1_GEMINI_MODEL` (upstream hard-codes `gemini-1.5-pro`) and no longer prints the tail of the key; `call_gpt4` (rag) and `GPT` (vlm) accept a plain `OPENAI_API_KEY` when no Azure endpoint is set |
| `epic/prepare_clips.py` | 30 s clips with audio in the EgoLife layout upstream's tools read |
| `epic/build_memory.py` | the question-independent memory: captions, then 10 min / 1 hour / 1 day summaries, in the format of upstream's released database (`Ego-R1/h-rag_database`) |
| `epic/tools.py` | the three tools, calling upstream's functions in process, plus the streaming cutoff |
| `epic/run_qa.py` | the agent loop of `infer_bench_summ.py` on one video's questions of a bank; writes `answers_<bank>.json.gz` |
| `epic/apis.py` | the model of each step, the key check, and the two API clients upstream has no reusable code for |
| `epic/plan_cost.py` | calls and cost of a run, from video lengths and a bank; calls nothing |
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
bash baselines/egor1/run_egor1.sh prep      P05-20240425-201444 gemini25pro                  # CPU
bash baselines/egor1/run_egor1.sh firstturn P05-20240425-201444 gemini25pro batch10_agree2   # GPU, no API: the local half
# with the keys in the environment (next section):
bash baselines/egor1/run_egor1.sh build     P05-20240425-201444 gemini25pro                  # paid APIs, no GPU
bash baselines/egor1/run_egor1.sh qa        P05-20240425-201444 gemini25pro batch10_agree2   # GPU + paid APIs
bash baselines/egor1/run_egor1.sh upload    P05-20240425-201444 gemini25pro
python baselines/egor1/epic/plan_cost.py --videos <HD-EPIC videos> --bank_dir <epic-morph>/outputs/v3/banks/batch10_agree2
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
hash>`, built from `docker/Dockerfile` by `.github/workflows/baseline-images.yml`. The code is the mounted checkout
of `main`; `docker/run.sh` picks the image whose tag matches the checkout's environment files:

```bash
git submodule update --init baselines/egor1/Ego-R1
docker pull ghcr.io/morph-graphs/egor1:$(bash baselines/egor1/docker/image_tag.sh gemini25pro)
export GEMINI_API_KEY OPENAI_API_KEY HF_TOKEN        # already set in this shell; never typed into a command
export EPIC_MORPH_DIR=<epic-morph checkout>
bash baselines/egor1/docker/run.sh prep  P05-20240425-201444 gemini25pro
bash baselines/egor1/docker/run.sh build P05-20240425-201444 gemini25pro
EGOR1_GPU=0 bash baselines/egor1/docker/run.sh qa P05-20240425-201444 gemini25pro batch10_agree2
bash baselines/egor1/docker/run.sh upload P05-20240425-201444 gemini25pro
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
| `GEMINI_API_KEY` | Gemini 2.5 Pro: captions (`build`), `video_llm` (`qa`, arm `gemini25pro`) |
| `OPENAI_API_KEY` | GPT-4o and GPT-4.1 with a plain OpenAI key: summaries (`build`), `rag`, `vlm`, final answer, cutoff views (`qa`) |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | instead of `OPENAI_API_KEY`: an Azure OpenAI resource with deployments named `gpt-4o` and `gpt-4.1` (upstream's way; used when both are set) |
| `HF_TOKEN` | `upload`, and `build`'s look in the results dataset |

- **No stage may put a key on a command line**, in a file of this repo, in a job script or in a log. Keys are read
  from the environment of the process only; the worker that starts a task exports them from its own secret
  store (for a container: `--env GEMINI_API_KEY` by name, so the value comes from the caller's environment).
  `run_egor1.sh` takes no key argument, upstream's prints are dropped, and `api_usage.jsonl` holds token counts only.
- Upstream's OpenAI clients are Azure-only; a plain OpenAI key works through `epic-morph.patch` (two small
  changes, above). Our own client (`apis.gpt41`) takes either.
- `build` needs no GPU (ffmpeg, network). `qa` needs one small GPU (the agent, about 8 GB) and network.
- A stage without its keys exits with code 5 before any call, so a task handed to a pool without keys fails at
  once instead of half-running.
- The job-board entries (`coord/master.py` `EXPERIMENTS`) do not exist yet: a paid row needs the board to know
  which pool holds keys.
- `docker/run.sh` hands the keys to the container by name (docker `-e NAME`, apptainer `APPTAINERENV_NAME`).

## What comes back

Per video, at `egor1/<arm>/<video>/`: the four memory files; `cutoffs/c<seconds>/{10min,1hour,1day}.json`;
`answers_<bank>.json.gz` (one row per question in bank order, keyed by the bank's `key` plus `qhash`, with the
chosen option, `evaluate`, the cutoff, and `turns`: each turn's agent output, parsed tool call, the call as sent on after the streaming check (`sent`) and the
returned information; `version` and `memory_hash`); `api_usage.jsonl` (one line per API call, upstream's included: the step, the model id asked and the one the API
answered with, input, output and thinking tokens); `runs.jsonl` (per finished stage: commits, the model of each
step, GPU, seconds, and `api_usage`: calls and tokens of that stage per step and answering model).

## Open decisions

- **Gemini 2.5 Pro ends on 2026-12-01.** The whole run (captions of every video, and every `video_llm` call)
  has to fit before that date, or it mixes models.
- **Caption and summary prompts** are not published; ours could be replaced by the authors' if they share them.
- **Fork**: `Ego-R1/` is a submodule of upstream itself, patched in place at run time; a morph-graphs fork with
  the patch as its `epic-morph` branch would be cleaner.
