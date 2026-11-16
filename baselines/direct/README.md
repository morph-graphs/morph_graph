# baselines/direct: a plain video model on EPIC-Morph

The no-memory-system baseline of the EPIC-Morph benchmark: for each question the model is shown frames sampled
from the video before the question's ask time `t_ask`, then the question, and answers. Nothing is built and
nothing is retrieved. It is MERIT's blind control (`baselines/merit/epic/blind_qa.py`) with the frames put
where MERIT puts its retrieved context, so MERIT, blind and direct differ only in that context.

## Layout

| path | what |
|---|---|
| `epic/direct_qa.py` | one video, one bank: sample the frames per question, ask, score, write the answers |
| `run_direct.sh` | one video, one arm, one stage (prep, build, qa, upload, all); `build` does nothing |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset |
| `env/setup_env.sh` | builds the model's MERIT environment (`baselines/merit/env/`): nothing extra is needed |
| `docker/run.sh` | the same stages inside MERIT's image of the model's environment (Docker or Apptainer): what the job board runs |
| `docker/image_tag.sh` | that image's tag for an arm: `merit:m8-<hash>` or `merit:m27-<hash>` |

It uses, from `baselines/merit/` (the MERIT submodule must be checked out): the prep (`epic/prepare_epic.py`:
1 fps frames, question conversion, `match_answers`), MERIT's Qwen wrapper as the loader of the two models it
knows, its final answer prompt (`final_qa_egolife`), answer parser and scorer. The other models are loaded by
`load_hf` in `epic/direct_qa.py`, set up the same way (local snapshot, bf16, flash-attention, seed 42).

## Design

**Frames.** MERIT's prep cuts one 448 x 448 frame per second (HD-EPIC is square, so nothing is cropped). A question asked at `t_ask` may use frames `0 .. floor(t_ask - 0.5)`: ffmpeg's `fps`
filter gives frame k the last picture before k + 0.5 s (measured: `t00100.jpg` is closest to the exact picture at
100.5 s), so no frame is later than `t_ask`, and frame k is stamped k + 0.5 s. The frame budget N
takes N of them, evenly spaced, first and last included, so the moment just before the question is always
shown; when the video so far is shorter than N seconds it takes all of them. One number therefore covers both
"uniform N frames over [0, t_ask]" and "1 fps up to a cap of N". In `batch10_agree2` every question is asked
at 136 s or later (median about 1,900 s), so 1 fps without a cap would be 1,600 to 3,300 frames for most.

**What the model sees.** The frames go in as one video through the model's own video input, with their true
time stamps (`<17.5 seconds>` before each pair of frames: the mean of the pair). Qwen3-VL joins 2 frames into one group of
(448 / 32)^2 = 196 tokens, so N frames cost 98 N tokens: 3.1k for N = 32, 12.5k for 128, 75k for 768. The
processor is given the exact pixel count, so it never rescales a frame (asserted per question).

**Prompt and decoding.** MERIT's final-answer step, word for word: its system prompt, `Query: ... Query Time:
... Choices: (A) ...`, `Context:`, then the video where MERIT has its retrieved captions and frames, then
"Please provide only the final answer from the choices given". Greedy decoding, up to 2,048 new tokens,
Qwen3.8 in instruct mode, MERIT's `ANSWER: X` parser and scorer. Three differences from MERIT's step remain:
1. the frames are one video (time-stamped, 2 frames per token group), not separate images;
2. the system prompt still speaks of "retrieved memory context", which here is the video;
3. `Query Time` is written as MERIT writes it (`DAY1 00:27:04`), while the frames' stamps are in seconds.

**Arms.** `<model>-f<N>`, e.g. `qwen3vl-8b-f128`; the settings are N = 32, 128 and 768 (since 2026-11-14 the board lists Qwen3-VL-8B and Qwen3.8-27B at N = 768 only). The model and frame setting are the arm, so each
setting has its own results folder and its own row in the results table ("N. Frames" = N).

| model | Hugging Face id | environment | GPU |
|---|---|---|---|
| `qwen3vl-8b` | Qwen/Qwen3-VL-8B-Instruct | MERIT m8 (transformers 4.57.3) | measured peak 17.5 GB (N = 32), 20.2 GB (128), 39.2 GB (768): a 24 GB GPU up to N = 128, a 48 GB GPU for 768 |
| `qwen38-27b` | Qwen/Qwen3.8-27B | MERIT m27 (transformers 5.8.0) | measured on 2x 48 GB at N = 128: peak 28.9 + 30.5 GB; one 80 GB GPU should also do. N = 32 and 768 not run yet (768 not sized) |
| `qwen35-9b` | Qwen/Qwen3.5-9B | MERIT m27 | measured peak 21.3 GB (N = 32 and 128), 33.9 GB (768): a 24 GB GPU up to N = 128, a 48 GB GPU for 768 |
| `qwen35-35b-a3b` | Qwen/Qwen3.5-35B-A3B | MERIT m27 | measured on 2x 48 GB: peak 32.4 + 35.7 GB (N = 32), 34.1 + 35.9 GB (128); N = 768 not run yet (70 GB of the 96 are already used at N = 128, so size it first) |
| `molmo2-8b` | allenai/Molmo2-8B | MERIT m8 (its card names transformers 4.57.1; 4.57.3 runs it) | measured peak 18.5 GB (N = 32), 24.8 GB (128), 42.0 GB (384): a 48 GB GPU. N is at most 384 (below) |

Time per question on RTX A6000s (P05-20240425-201444, a few questions each, the first one of a run takes longer):

| model | N = 32 | N = 128 | N = 768 |
|---|---|---|---|
| Qwen3-VL-8B | 11 s | 14 s | 55 s |
| Qwen3.8-27B (2 GPUs) | not run | 32 s | not run |
| Qwen3.5-9B | 3 s | 11 s | 35 s |
| Qwen3.5-35B-A3B (2 GPUs) | 8 s | 9 s | not run |
| Molmo2-8B | 4 s | 11 s | does not fit (N = 384: 37 s) |

**The Qwen3.5 models** take exactly the Qwen3-VL-8B input: same processor class, 196 tokens per 2 frames, same
time stamps, the same messages, instruct mode (`enable_thinking=False`) as Qwen3.8.

**Molmo2 differs**, because its processor and chat template do (we follow its model card):
1. no system role: MERIT's system text opens the user turn, followed by the same query text;
2. the chat template puts the video before all text, so nothing sits under "Context:";
3. every frame is resized to 378 px and costs 81 tokens, with its own time stamp before it (`17.5 `), not one per pair;
4. its context is 36,864 tokens: N = 768 does not fit (69k tokens; the run stops with that message), N = 384 is its
   largest setting and leaves room for a short answer only (generation is capped to the room left);
5. it is loaded in bf16, not the card's float32 (`dtype="auto"`), which needs 46.7 GB at N = 128: on 8 questions
   the two gave the same option every time;
6. it answers `A` then `REASONING: ...` without "ANSWER:", so a letter alone on the first line is also read as the answer
   (for every model; MERIT's scorer already scored it the same way).

## The evidence-clips ablation (arms `<model>-evclips`)

An upper bound of each reasoning model the memory methods use, given the right evidence and no memory (Anubhav,
2026-11-15): one row per model, all with this baseline's one prompt and scorer, over the `_end` banks.

**Frames.** Only the question's evidence clips: the bank's `evidence.clips` time ranges (a dict, or its str repr), merged
where they overlap and cut at `t_ask` (`epic_eval.common.evidence_spans`, shared with epic_eval's evidence mode), at
1 fps: `ceil(length)` consecutive 1 fps frames per merged range (at least 1), centred on the range, nothing in between
and no padding; a frame two ranges share is shown once. Over the 3,805 end questions: 2.8 clips a question, median 15
frames, mean 18.3, p90 33, max 177 (69,450 in all). Each answer row adds `evidence_spans`.

**Models.** The local ones as their `-f<N>` arms (`qwen3vl-8b`, `qwen38-27b`, `qwen35-9b`, `molmo2-8b`, `qwen35-35b-a3b`),
plus `vst7b`: VST's released weights (Catalan258/VST-7B) as a plain Qwen2.5-VL-7B through `load_hf`, not VST's streaming
loop, so it reads the same prompt as the others. Qwen2.5-VL's video input has one frame rate and no time stamps, so each
run of consecutive frames (an evidence range) is its own video, after its span as text (`<1537.5 - 1542.5 seconds>`).
The paid models (`gpt-4.1`, `gpt-5.1`, `gpt-5.5`, `gemini-3.6-flash`, `gemini-3.1-flash-lite`, `qwen35-flash`) get the
same 448 px frames as JPEG images, each after its time stamp (`<1537.5 seconds>`), between the same query text and final
line, through an OpenAI-compatible chat endpoint (the `openai` client of MERIT's m8 image: no new dependency, same image
tag): OpenAI's (`OPENAI_API_KEY`; `OPENAI_BASE_URL` a gateway; `DIRECT_GPT_PREFIX`, default `EGAGENT_GPT_PREFIX`, before
the GPT ids), Google's OpenAI-compatible endpoint (`GEMINI_API_KEY`) and OpenRouter (`OPENROUTER_API_KEY`,
`qwen/qwen3.5-flash-02-23`). Request settings are those of the board's other rows of the model (`API` in
`epic/direct_qa.py`). Each row keeps `api`: the model id and endpoint that answered, finish reason, tokens, and
OpenRouter's cost. Calls run 8 at a time (`DIRECT_API_THREADS`); a failed call or an empty reply is "Error", retried next run, except an empty one cut off at the output cap or refused by a content filter (`finish_reason` length / content_filter), which is final and wrong.

```bash
bash baselines/direct/run_direct.sh qa P01-20240203-123350 qwen3vl-8b-evclips batch10_agree2_end
GEMINI_API_KEY=... bash baselines/direct/run_direct.sh qa P01-20240203-123350 gemini-3.6-flash-evclips batch10_agree2_end
```

## Running

```bash
export EPIC_MORPH_DIR=<epic-morph checkout> PY=<python of the model's environment>
bash baselines/direct/run_direct.sh qa P05-20240425-201444 qwen3vl-8b-f128 batch10_agree2
DIRECT_LIMIT=6 bash baselines/direct/run_direct.sh qa P05-20240425-201444 qwen3vl-8b-f32 batch10_agree2   # smoke test
```

`qa` cuts the frames first when they are missing (`prep`, CPU, ffmpeg on `PATH`; by default into
`baselines/merit/work/`, the folder MERIT's prep uses, so a video is cut once for both). It skips questions
already answered with the same model and frame setting, so rerunning resumes; a question that ran out of GPU
memory is written as "Error" and retried on the next run.

## What comes out

`<DIRECT_OUT>/<arm>/<video>/` (default `baselines/direct/results/`, not committed):
- `answers_<bank>.json.gz`: one row per question with MERIT's fields (`ID`, `key`, `qhash`, `pos`, `type`,
  `question`, `choices`, `answer`, `response`, `parsed_answer`, `reasoning`, `evaluate`, `query_time`,
  `query_time_int`, empty `round_history`), so `baselines/merit/epic/collect_answers.py` and
  `compare_answers.py` read it, plus what the model was shown: `t_ask`, `frame_seconds`, `n_frames`,
  `video_tokens`, `prompt_tokens`, `seconds`, `setting`.
- `runs.jsonl`: one line per finished `qa` (commits, model, frame setting, torch and transformers versions, GPU, seconds).

`upload` puts the folder in the private dataset
[morph-graphs/epic-morph-results](https://huggingface.co/datasets/morph-graphs/epic-morph-results) at
`direct/<arm>/<video>/` (a member's token with write access: `HF_TOKEN`, or `hf auth login`).

## From an image, and on the job board

There is no Docker image of its own: the MERIT images (`baselines/merit/docker/`) hold the same two environments,
and `docker/run.sh` runs `run_direct.sh` in the one of the arm's model, with MERIT's `docker/run.sh` interface:

```bash
DIRECT_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/direct/docker/run.sh qa P05-20240425-201444 qwen3vl-8b-f128 batch10_agree2
DIRECT_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/direct/docker/run.sh upload P05-20240425-201444 qwen3vl-8b-f128
```

The image is `ghcr.io/morph-graphs/merit:$(bash baselines/direct/docker/image_tag.sh <arm>)`; with
`DIRECT_RUNTIME=apptainer` it is MERIT's `merit_m8.sif` / `merit_m27.sif` in `DIRECT_SIF_DIR` (default
`MERIT_SIF_DIR`). The checkout needs the MERIT submodule (`git submodule update --init baselines/merit/MERIT`).
The other settings are in the script's header; the `MERIT_*` ones the job board's worker sets are read too.

On the job board (`coord/master.py`, `DIRECT`) every arm above at N = 32, 128 and 768 has one `qa` task per bank and
video, with no build to wait for; a task is done when `direct/<arm>/<video>/answers_<bank>.json.gz` is in the
dataset. A worker takes them when started with `direct` in `--baselines`; it needs MERIT's images and nothing else.
