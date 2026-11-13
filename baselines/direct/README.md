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
| `env/setup_env.sh` | builds the model's MERIT environment (`baselines/merit/env/`): nothing extra is needed |

It uses, from `baselines/merit/` (the MERIT submodule must be checked out): the prep (`epic/prepare_epic.py`:
1 fps frames, question conversion, `match_answers`), MERIT's Qwen wrapper as the model loader, its final
answer prompt (`final_qa_egolife`), answer parser and scorer.

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
processor is given the exact pixel count, so it never rescales a frame (asserted per question). `-p<px>` in
the arm shrinks the frames first (224 px: 49 tokens per 2 frames), for large N.

**Prompt and decoding.** MERIT's final-answer step, word for word: its system prompt, `Query: ... Query Time:
... Choices: (A) ...`, `Context:`, then the video where MERIT has its retrieved captions and frames, then
"Please provide only the final answer from the choices given". Greedy decoding, up to 2,048 new tokens,
Qwen3.8 in instruct mode, MERIT's `ANSWER: X` parser and scorer. Differences from MERIT's step: the frames
are one video (time-stamped, 2 frames per token group) rather than separate images, and the system prompt
still speaks of "retrieved memory context", which here is the video.

**Arms.** `<model>-f<N>[-p<px>]`, e.g. `qwen3vl-8b-f128`. The model and frame setting are the arm, so each
setting has its own results folder and its own row in the results table ("N. Frames" = N).

| model | Hugging Face id | environment | GPU |
|---|---|---|---|
| `qwen3vl-8b` | Qwen/Qwen3-VL-8B-Instruct | MERIT m8 (transformers 4.57.3) | measured peak 17.5 GB (N = 32), 20.2 GB (128), 39.2 GB (768): a 24 GB GPU up to N = 128, a 48 GB GPU for 768 |
| `qwen38-27b` | Qwen/Qwen3.8-27B | MERIT m27 (transformers 5.8.0) | one 80 GB GPU, or 2x 48 GB (as MERIT m27; not run yet, N = 768 not sized) |

Time per question on an RTX A6000, Qwen3-VL-8B (P05-20240425-201444, 8 questions each): about 11 s (N = 32),
14 s (128), 55 s (768).

Other models need a name in MERIT's wrapper first (it picks the model class by name), then a line in
`run_direct.sh`.

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

In the results dataset these belong at `direct/<arm>/<video>/`. The `upload` stage is not wired yet:
`baselines/common/hf_results.py upload` refuses a folder without memory files and needs a no-memory mode
first. There is no Docker image yet either; the MERIT images hold the same environments.
