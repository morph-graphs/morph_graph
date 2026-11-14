# baselines/hippomm: HippoMM on EPIC-Morph

[HippoMM](https://github.com/linyueqian/HippoMM) (Lin et al., arXiv 2504.10739; MIT) is a hippocampus-inspired
memory for long audiovisual streams. It cuts a video into 10-30 s segments at visual changes and audio silences,
encodes every extracted frame and each segment's audio with ImageBind and transcribes the audio with Whisper,
consolidates the video into one event (the key frames are those whose ImageBind feature is less than 0.9 similar
to every earlier key frame), and "replays" it: one Qwen2.5-VL caption per key frame and a one-sentence summary.
It answers in two passes: a fast one from the summary and the captions, and, when that is not confident, a
detailed recall (ImageBind retrieval of frames, new captions of the video around them, Whisper on the audio
around them), then a reflection step that reconciles the two answers.

## Read first

- **This is a paid row.** Answering (`qa`, `blind`) calls `gpt-4o` through the OpenAI API, as upstream does; both
  stages refuse to start without `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`: the same calls through OpenRouter,
  see "Through OpenRouter"). The memory build is free (local models). `count` gives the
  calls, tokens and cost of a bank before anything is spent, and every paid stage records the model id the API
  returned and its token usage in `runs.jsonl`.
- **The method runs as released.** Our rule for baselines: run the released code as it is, and change only what
  it needs to run at all and to respect the question-time cutoff. So the settings are the repository's, not the
  paper's where the two differ, audio is used as published, and three behaviours that look like upstream bugs are
  kept (all listed under "What we changed in upstream" below). They lower what the method can do here; that is
  the released method's result, not ours to repair.
- **Upstream is a submodule at the released commit; our changes are `HippoMM.patch`.** `run_hippomm.sh` applies
  the patch to the submodule checkout before every stage (once; it stops if the patch does not apply cleanly), so
  `git status` shows the submodule as modified after a run. A fork under morph-graphs with the patch as a commit,
  like the other baselines, would be cleaner.
- **Two arms, one memory.** `q25vl7b-gpt4o` is the paper's configuration. `q25vl7b-gpt41` answers from the same
  memory with `gpt-4.1` in place of `gpt-4o` (see "Arms"). There is no m8 / m27 variant.

**The models are those of the paper and official code, not our local Qwen arms:**

| role | model | where it runs |
|---|---|---|
| frame captions, summaries, caption digests | `Qwen/Qwen2.5-VL-7B-Instruct` | locally, served by vLLM as upstream does |
| frame, audio and query features | ImageBind-Huge (weights are CC BY-NC-SA 4.0) | locally |
| speech | Whisper `distil-large-v3` (faster-whisper) | locally |
| reasoning: question type, direct answer, search query, frame selection, final answer, reflection | `gpt-4o` (arm `q25vl7b-gpt41`: `gpt-4.1`) | **OpenAI API (paid)** |

So `prep`, `build` and `count` need no API. `qa` and `blind` call the reasoning model and refuse to start without
`OPENAI_API_KEY` or `OPENROUTER_API_KEY`; `count` says beforehand how many calls and tokens a bank needs.

## Arms

| arm | reasoning model | memory | what it is |
|---|---|---|---|
| `q25vl7b-gpt4o` | `gpt-4o` | its own folder | the published setting |
| `q25vl7b-gpt41` | `gpt-4.1` | `q25vl7b-gpt4o`'s | **GPT-4.1 stands in for gpt-4o**, because the machines that run this row have no GPT-4o (Anubhav's decision, 2026-11-14). Not the paper's model: report it under this name |

- The memory does not depend on the reasoning model (Qwen2.5-VL, ImageBind and Whisper make it), so there is one,
  in `results/q25vl7b-gpt4o/<video>/` (the dataset's `hippomm/q25vl7b-gpt4o/<video>/`), with `summaries.json`
  (Qwen2.5-VL's summary per question cutoff) beside it. `q25vl7b-gpt41` has no `prep`, and its `build` only takes
  that memory from the dataset: build it with `q25vl7b-gpt4o`. Its own folder holds `count_<bank>.json` (at
  GPT-4.1 prices), the answers, `runs.jsonl` and `<video>_memory_ref.json`, the hashes of the memory files its
  answers were made from.
- Everything but the model id of the request is the same: prompts, `temperature=0`, the paths through upstream's
  code. Upstream's token counter (its 120k-token cap on a prompt's context) is still given the name `gpt-4o`:
  tiktoken 0.9.0, upstream's pin, cannot map `gpt-4.1`, and upstream's `TokenCount` would then have no encoding;
  both models use the encoding `o200k_base`.
- Both arms run in the one image (`hippomm:q25vl7b-gpt4o-<env hash>`, `hippomm_q25vl7b-gpt4o.sif`).

## Layout

| path | what |
|---|---|
| `HippoMM/` | submodule: upstream [linyueqian/HippoMM](https://github.com/linyueqian/HippoMM) at `d7000b4`, unmodified in git |
| `HippoMM.patch` | our changes to upstream, as a diff against `d7000b4`; `run_hippomm.sh` applies it to the checkout |
| `epic/hm_common.py` | upstream's config with our paths, the memory files, the memory as it stood at a question's cutoff |
| `epic/build_memory.py` | `prep` (upstream's frame and audio extraction) and `build` (upstream's `process_sequence`, then the memory files) |
| `epic/run_qa.py` | per-cutoff summaries, upstream's `QARecallSystem` per question, `--count`, `--blind`; writes `answers_<bank>.json.gz` |
| `run_hippomm.sh` | one video, one arm, one stage (prep, build, count, qa, blind, upload, all) |
| `docker/` | the image (`Dockerfile`), `build.sh`, `run.sh` (one job in a container; the job board's entry point), `image_tag.sh` |
| `env/` | the two environments, pinned: `requirements-client.txt` (upstream's pins), `requirements-vllm.txt`, `install_requirements.sh`, `setup_env.sh` |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset, or download them |

## What we changed in upstream (`HippoMM.patch`)

The released code does not run as published; these are the changes needed to run it, and the two for streaming.
Nothing else is touched: prompts, thresholds and settings are upstream's (`config/default_config.yaml`).

- `QwenVL.__init__` takes the `config` that `HippocampalMemory` passes (a `TypeError` upstream) and uses its
  `api.qwen` endpoint instead of a fixed `localhost:8000`.
- ImageBind loads `<models.imagebind_path>/imagebind_huge.pth` when it is there (the path upstream's README
  describes but its code ignored, downloading into `./.checkpoints` instead).
- The caption workers get the config's endpoints. In `replay` they fell back to `localhost:8000` and `:9000`. In
  detailed recall the pool was given a lambda, which cannot be pickled: the captions of the retrieved frames
  always failed (silently for video questions, with a `NameError` for speech-first ones). It is now a `partial`.
- A segment transcription of `None` (Whisper heard nothing) no longer breaks the summary prompt.
- Temporary frames and audio go to `tempfile.gettempdir()` instead of fixed names in `/tmp`, and
  `run_hippomm.sh` gives each run its own `TMPDIR`: two jobs on one node wrote the same `/tmp/frame_<ms>.jpg`.
- Streaming: detailed recall never reads the video or its audio past `memory.max_time` (the question's `t_ask`).

**Settings are the released code's, where the paper differs.** The paper (v2) states segments of 5-10 s,
consolidation similarity 0.85, fast-retrieval confidence 0.75 and Whisper v3 medium; the repository's
`config/default_config.yaml` and code have 10-30 s, 0.9, 0.7 and `distil-large-v3`, and those are what runs here.

Upstream behaviour we kept although it looks unintended (see "Open points"):
- `process_sequence` never puts the audio into a segment's content, so consolidation drops the segments' ImageBind
  audio features and their Whisper text: the event holds vision features, key frames, captions and the
  whole-audio transcription only. Sound-first questions therefore fall back to the whole-context prompt.
- Retrieval ranks the features of *all* frames but reads the time from the list of *key* frames at the same index.
- Consolidation makes one event per video (the paper describes several).

## Streaming

`build` needs no questions: the memory covers the whole video. A question asked at `t_ask` gets the memory
restricted to what was seen by then (`hm_common.at_cutoff`): frame features and key frames with time <= `t_ask`
(key-frame selection is greedy in time order, so these are the key frames a build of the first `t_ask` seconds
selects), their captions, and the transcription lines that ended by then. Upstream's summary is one sentence over
all captions, so it is made again per cutoff by Qwen2.5-VL with upstream's replay prompt, from the captions up to
`t_ask` (`summaries.json`, shared by every bank asking at the same time). Questions without `t_ask` see everything.
One approximation: Whisper transcribes the whole audio in 30 s windows, so a line that ends before `t_ask` may
have been decoded with up to 30 s of later audio in its window.

HD-EPIC audio that outlasts the video is harmless here: segments follow the frame times, and transcription lines
after the last frame are never before a cutoff.

## Environments

**Off our clusters, run only in the Docker image** (rule in `AGENTS.md`): `ghcr.io/morph-graphs/hippomm:<arm>-<env
hash>`, built from `docker/Dockerfile` by `.github/workflows/baseline-images.yml`. It holds both environments
below (vLLM in the base Python, HippoMM's in a virtualenv at `/opt/client`), no weights and no key. The code is
the mounted checkout of `main` (`git submodule update --init baselines/hippomm/HippoMM`); `docker/run.sh` picks the
image whose tag matches the checkout's environment files:

```bash
ARM=q25vl7b-gpt4o
docker pull ghcr.io/morph-graphs/hippomm:$(bash baselines/hippomm/docker/image_tag.sh $ARM)
export EPIC_MORPH_DIR=<epic-morph checkout> HIPPOMM_GPU=0
bash baselines/hippomm/docker/run.sh build P05-20240425-201444 $ARM                  # free
bash baselines/hippomm/docker/run.sh count P05-20240425-201444 $ARM batch10_agree2   # free: calls, tokens, cost
export OPENAI_API_KEY    # set in the shell beforehand; it reaches the container by name only
bash baselines/hippomm/docker/run.sh qa    P05-20240425-201444 $ARM batch10_agree2   # paid (gpt-4o)
bash baselines/hippomm/docker/run.sh upload P05-20240425-201444 $ARM
# the gpt-4.1 arm, from the same memory (fetched from the dataset when it is not on this machine):
bash baselines/hippomm/docker/run.sh count  P05-20240425-201444 q25vl7b-gpt41 batch10_agree2
bash baselines/hippomm/docker/run.sh qa     P05-20240425-201444 q25vl7b-gpt41 batch10_agree2   # paid (gpt-4.1)
bash baselines/hippomm/docker/run.sh upload P05-20240425-201444 q25vl7b-gpt41
```
`OPENAI_BASE_URL`, when set, is passed to the container too: the OpenAI client then talks to that
OpenAI-compatible endpoint (a gateway) with `OPENAI_API_KEY`, and `runs.jsonl` records it as `endpoint`.

`docker/run.sh` mounts this checkout at `/repo`, the epic-morph checkout at `/epic-morph`, the videos at `/videos`
and the Hugging Face cache (`HIPPOMM_HF_CACHE`, default `~/.cache/huggingface`) at `/hf`. The first `build`
downloads into that cache: Qwen2.5-VL-7B-Instruct (16 GB), the Whisper model (1.5 GB) and
`imagebind/imagebind_huge.pth` (4.5 GB). One GPU with 48 GB, 64 GB of RAM. The image has `gcc`: vLLM 0.8.5 compiles
at start-up (torch.compile; triton builds a small C module) and stops with "Failed to find C compiler" without
one, which is how the first published image (`q25vl7b-gpt4o-9d3b512389c7`) failed. `HIPPOMM_RUNTIME=apptainer` runs the
same image with Apptainer; the job board (`coord/worker.py`) calls it with `MERIT_GPU` / `MERIT_RUNTIME` /
`MERIT_SIF_DIR` / `MERIT_HF_CACHE` / `MERIT_WORK`, which `run.sh` takes when the `HIPPOMM_` ones are unset.

On our own clusters, two virtualenvs on Python 3.11 (`PYTHON` picks the interpreter):

```bash
bash baselines/hippomm/env/setup_env.sh client <dir>/hippomm-client   # torch 2.1.2 (CUDA 12.1), ImageBind, faster-whisper 1.0.3
bash baselines/hippomm/env/setup_env.sh vllm   <dir>/hippomm-vllm     # vLLM 0.8.5.post1 (its own torch 2.6)
```

`client` is upstream's `requirements.txt` for the packages its code imports, plus the ones it imports without
listing. Models: `Qwen/Qwen2.5-VL-7B-Instruct` and `Systran/faster-distil-whisper-large-v3` from the Hugging Face
cache; `imagebind_huge.pth` (4.5 GB, https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) in the folder
named by `HIPPOMM_IMAGEBIND` (default `<HF cache>/imagebind`, downloaded there on first use). ffmpeg, ffprobe and
`patch` must be on `PATH`. One GPU with 48 GB holds vLLM (0.6 of it),
ImageBind and Whisper.

## Running

```bash
export PY=<hippomm-client>/bin/python HIPPOMM_VLLM_PY=<hippomm-vllm>/bin/python EPIC_MORPH_DIR=<epic-morph checkout>
ARM=q25vl7b-gpt4o
bash baselines/hippomm/run_hippomm.sh build P05-20240425-201444 $ARM                  # the memory, no questions needed
bash baselines/hippomm/run_hippomm.sh count P05-20240425-201444 $ARM batch10_agree2   # gpt-4o calls, tokens, cost: free
OPENAI_API_KEY=... bash baselines/hippomm/run_hippomm.sh qa P05-20240425-201444 $ARM batch10_agree2   # paid
bash baselines/hippomm/run_hippomm.sh upload P05-20240425-201444 $ARM
```

Every stage skips finished work. `qa` saves after each question and answers only those without a row from this
`VERSION`, this memory (`memory_hash`) and the same question text (`qhash`); `HIPPOMM_LIMIT=N` answers N
questions. `HIPPOMM_TAG=<name>` writes `answers_<bank>_<name>.json.gz`. Work files (full-resolution frames, about
1 fps; audio; vLLM logs) stay in `baselines/hippomm/work/` (`HIPPOMM_WORK`); answering re-reads the video, not
the frames, so a memory fetched from the dataset answers without them.

### What `count` reports

`count` prices the arm's model at list price ($2.50 in / $10 out per million tokens for gpt-4o, $2 / $8 for
gpt-4.1; the cached-input discount is not counted) and writes the model and prices into `count_<bank>.json`.
Per question, upstream makes between 2 and 7 calls to it. Every question gets the classification and the
direct answer, whose prompt holds the summary and all key-frame captions up to `t_ask`. `low` is that pair
(the direct answer was confident). `high` adds the search query, the frame selection over all captions (when
ImageBind similarity is under 0.4), the final answer over all captions, and both reflection calls: the caption
list goes out three times. Prompt templates are counted as fixed sizes; the caption tokens are counted with
tiktoken from the real memory.

### Through OpenRouter (our own runs)

With `OPENROUTER_API_KEY` in the environment and `OPENAI_API_KEY` not, `qa` and `blind` send the same gpt-4o
calls through OpenRouter, so the row can be answered where there is no OpenAI account. Nothing else is set;
`epic/run_qa.py` (`meter`) does the switch, and `OPENAI_API_KEY`, when set, wins and goes to OpenAI as before.

```bash
export OPENROUTER_API_KEY=...      # typed into the shell (or read -s): never an argument, a file, a log or a commit
unset OPENAI_API_KEY GOOGLE_API_KEY GEMINI_API_KEY       # no other provider's key in the job
bash baselines/hippomm/docker/run.sh qa P05-20240425-201444 q25vl7b-gpt4o batch10_agree2
```

- **What changes in the request.** The endpoint (`https://openrouter.ai/api/v1`; `OPENROUTER_BASE_URL` overrides
  it), the key, the model id (`openai/gpt-4o`) and one added field,
  `provider: {"only": ["openai"], "allow_fallbacks": false}`: OpenRouter also lists Azure for this model, and with
  the pin OpenAI answers or the call fails. Prompts, `temperature=0` and `response_format` are what upstream
  sends to OpenAI. Upstream's own token counting (its 120k-token context cap) still uses `gpt-4o`'s tokenizer.
- **What may still differ from the direct API.** `openai/gpt-4o` is OpenRouter's alias: the snapshot behind it is
  not ours to pin (upstream pins none either: it asks for `gpt-4o`); the id each reply names is recorded. The
  price per token is OpenAI's ($2.50 in, $1.25 cached in, $10 out per million; OpenRouter's listing, read
  2026-11-14); OpenRouter adds its fee when credits are bought, not per call.
- **What is recorded.** The stage's `api` entry in `runs.jsonl` has `requested` (`openai/gpt-4o`), `endpoint`,
  the model ids returned, calls, input / cached / output tokens, and through OpenRouter `providers` (who served
  the calls) and `cost` (dollars charged, summed); each answer row has the same for its question. The results
  folder stays `hippomm/q25vl7b-gpt4o/<video>/`: the same model answers either way.

## What comes back

Per video, in `results/<arm>/<video>/` (the dataset's `hippomm/<arm>/<video>/` after `upload`; the memory and
`summaries.json` only in `q25vl7b-gpt4o`'s, `q25vl7b-gpt41`'s has `<video>_memory_ref.json` instead):
- the memory: `<video>_memory.json` (key frames, times, captions, segments with their Whisper text, the
  whole-audio transcription, the summary, the settings) and `<video>_features.npz` (ImageBind `vision` features of
  every extracted frame with `vision_times`; `audio`, `audio_times`);
- `summaries.json`: the summary per question cutoff;
- `count_<bank>.json`: the gpt-4o call and token range per question and in total;
- `answers_<bank>.json.gz`: one row per question in bank order, keyed by the bank's `key` (plus `qhash`), with
  upstream's raw answer (`response`), the option letter read from it (`pred`), `evaluate`, `t_ask` and `cutoff`,
  how much memory was visible (`indexed`), upstream's question type, confidence, reasoning, whether the direct
  answer or the reflection was used, the retrieved time spans, `version` and `memory_hash`;
- `blind_<bank>.json.gz`: the same rows for the no-memory prompt;
- `runs.jsonl`: one line per finished stage (commits, the patch's hash, models, the memory arm, vLLM and torch versions, GPU,
  seconds); for `qa` and `blind` also `api`: the model asked for, the endpoint, the model ids the API returned, the
  number of calls and questions, and input, cached, output and reasoning tokens as the API reported them (through
  OpenRouter also `providers` and `cost`). Each answer row has the same counts for its question (`api`).

## Open points

- **No call is made without a key**, and gpt-4o is replaced only in the arm named for it (`q25vl7b-gpt41`).
  Budget from `count` before running `qa`.
- The three kept upstream behaviours above change what the method can do on audio and which frames detailed
  recall looks at; fixing them would depart from the released code.
- Measured on P05-20240425-201444 (27 min, one RTX A6000, 4 CPUs): prep 8 min, build 19 min of which 7.5 min is
  vLLM loading the model; peak job memory 41.5 GB (ask for 64 GB); 1.1 GB of frames in the work folder; the
  memory files are 6 MB. Whisper turns kitchen noise into lines like "I don't know."; ImageBind text-to-frame
  similarity stayed under 0.4 for the two queries tried, the case where upstream asks gpt-4o to pick frames.
- Upstream's segment checkpoint writes each segment's audio samples as JSON text (about 1 GB per hour of video);
  `build` deletes it once the memory is written.
