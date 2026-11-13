# baselines/vst: VST on EPIC-Morph

[VST](https://github.com/1ranGuan/VST) (Video Streaming Thinking; Guan et al., arXiv 2603.12262; Apache-2.0)
is a streaming video model: Qwen2.5-VL fine-tuned to write a short "streaming thought" for each stretch of
video as it arrives, and to answer a question from those thoughts plus the most recent frames. Its memory is
the text of its own thoughts. The method is the released checkpoint
[Catalan258/VST-7B](https://huggingface.co/Catalan258/VST-7B) (Qwen2.5-VL-7B weights, Apache-2.0) run by
upstream's own inference code; no other model, API or extra input is involved.

## Layout

| path | what |
|---|---|
| `VST/` | upstream [1ranGuan/VST](https://github.com/1ranGuan/VST) at `4a881b5` (2026-10-10), unpatched; to become a fork submodule |
| `epic/run_qa.py` | one video's bank questions: thoughts per question cutoff (cached), then the answers; `--blind` for the no-video floor |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset |
| `run_vst.sh` | one video, one arm, one stage (prep, build, qa, blind, upload, all) |
| `env/` | the environment, pinned: `wheels-vst7b.txt`, `requirements-vst7b.txt`, `install_requirements.sh`, `setup_env.sh` |

No Docker image and no job-board entry yet (`docker/` as in `baselines/merit`, an `EXPERIMENTS` row in
`coord/master.py`): until then this runs by hand on our clusters only.

## How VST runs here

Upstream evaluates through lmms-eval: its model class `Qwen2_5_VL_Stream_Think`
(`VST/eval/lmms-eval/lmms_eval/models/simple/qwen2_5_vl_stream_think.py`) takes a video file and a question.
`epic/run_qa.py` uses that class for the model, the video loader (`_read_video_decord_plus`) and the frame
budget (`_spatial_resize_video`), and repeats its `generate_until` loop outside the harness so that the video
can be cut at each question's time and the thoughts kept:

1. **Cut.** A question asked at `t_ask` gets the frames with timestamp <= `t_ask` (the loader's `video_end`);
   upstream's streaming benchmarks (OVO-Bench, StreamingBench) hand the model videos cut the same way.
2. **Sample.** 2 fps (qwen-vl-utils' default) up to `max_frames`; a longer cut is sampled uniformly over its
   whole length instead. With 384 frames that is everything up to 192 s, one frame every 4.7 s at 30 min and
   every 9.4 s at an hour: the cap trades time resolution for coverage, it never drops the start of the video.
3. **Budget.** The cut is split into equal segments (2 / 3 / 5 / 5 for <= 0.5 / < 4 / < 30 / >= 30 min) of
   `vid_tokens` visual tokens each. A frame takes at least 256 tokens, so 5 segments of 8192 tokens hold 320
   frames: a cut of 4 min or more is thinned from 384 to 320 frames (64 per segment).
4. **Think.** For every segment but the last the model writes a thought, reading the earlier thoughts as text.
5. **Answer.** From all thoughts + the last segment's frames + the question, at most 16 new tokens, greedy.

Settings are upstream's OVO-Bench run (`VST/eval/scripts/ovobench.sh`: think times `2-3-5-5`, 8192 tokens per
segment, 384 frames, flash-attention) with its prompt (question, lettered options, "Answer with the option's
letter from the given choices directly."). `VST_THINK_TIMES`, `VST_VID_TOKENS` and `VST_MAX_FRAMES` change
them. A results folder holds one setting (`<video>_settings.json`), so give such a run its own `VST_OUT`.

**The "768f" option.** Our group's earlier VST runs on other benchmarks (the results-sheet rows "VST,
Qwen2.5-VL-7B, 1 FPS (max 768 frames)") used `VST_THINK_TIMES=1-1-4-4 VST_VID_TOKENS=32768
VST_MAX_FRAMES=768`. Despite the label, those runs sampled at 2 fps: they passed no `fps`, so the loader used
qwen-vl-utils' default of 2 (up to 768 frames, i.e. everything up to 384 s, uniform beyond). The same holds
here. It has not been run on EPIC-Morph; with 32768 visual tokens per segment expect to need a 48 GB GPU.

One thing upstream does that we keep as it is, since changing it would change the method: the times written
into the prompt come from the frame count, so when step 3 thins the frames the model is told a shorter clock
than the real one (320/384 of it).

What we do not keep: on a loading or generation error upstream's harness fills in a random letter. Here the
question stays unanswered and is retried.

## Stages

| stage | what it does for VST |
|---|---|
| `prep` | nothing to make (the loader reads the source video); checks the video is there |
| `build` | writes `<video>_settings.json`. There is no question-independent memory to build: where the segments fall depends on where the video is cut, so thoughts belong to a cutoff |
| `qa` | per distinct `t_ask` of the bank: thoughts (cached in `thoughts/<t_ask>.json`, shared by every question and bank asking at that time), then one short generation per question |
| `blind` | the class's text-only path: system prompt, empty thoughts, the question |
| `upload` | the results folder to `vst/<arm>/<video>/` in morph-graphs/epic-morph-results (`hf_results.py --memory none`: there are no memory files) |

## Running

```bash
bash baselines/vst/env/setup_env.sh vst7b <venv dir>          # Python 3.11, CUDA 12.4 wheels
export PY=<venv dir>/bin/python EPIC_MORPH_DIR=<epic-morph checkout>
bash baselines/vst/run_vst.sh qa P05-20240425-201444 vst7b batch10_agree2      # one GPU with >= 24 GB, 64 GB of RAM
bash baselines/vst/run_vst.sh blind P05-20240425-201444 vst7b batch10_agree2
bash baselines/vst/run_vst.sh upload P05-20240425-201444 vst7b
```
The model comes from the Hugging Face cache (`Catalan258/VST-7B`, 16.6 GB), or from `VST_MODELS/VST-7B`.
`VST_LIMIT=2` answers only the first two cutoffs, into its own answers file (a smoke test). Every stage skips finished work: an answer is
kept while its question's wording and options, the settings and this adapter's `VERSION` are unchanged.

## What comes back

Per video, in `results/vst7b/<video>/` (not committed), and after `upload` in the dataset:
- `<video>_settings.json`: model and inference settings;
- `thoughts/<t_ask>.json`: the thoughts for that cutoff, with the frame count, frame size and timings;
- `answers_<bank>.json.gz`, `blind_<bank>.json.gz`: one row per question, keyed by the bank's `key` (`qhash`,
  response, `pred`, `evaluate`, the cutoff, frames, input tokens, seconds);
- `runs.jsonl`: one line per finished stage (commits, model, torch and transformers versions, GPU, seconds).

## Measured (one video, 2026-11-13)

`P05-20240425-201444` (27 min, 1408x1408), `batch10_agree2`, first 6 of its 7 cutoffs (18 questions) on one
RTX A5000 with 4 CPUs: every cutoff from 5 min on read 384 frames, kept 320 at 448x448 in 5 segments.
Per cutoff: decoding and resizing the frames 62 s at 5 min, 155-165 s at 16-17 min, 200 s at 27 min (CPU: decord
seeks in the full-resolution source); thinking 31-37 s. Per question 3.1 s (about 8,850 input tokens).
Peak GPU memory 18.7 GiB; peak RAM 50.6-54.4 GiB (the loader holds the 384 source frames as floats while resizing),
so ask for 64 GB. Decoding is most of the time: a prep stage that stores the 2 fps frames once would remove it,
at the price of not reading the source exactly as upstream does.

**Blind accuracy can sit below chance, by construction of the bank.** On this video the blind stage got 7 of
59 right where guessing expects 15.5. The option order and gold letters in the rows match the bank file (0
of 59 differ). EPIC-Morph's banks are filtered against exactly this: a question that two of three blind
models answer correctly is rewritten or dropped (epic-morph `v3/blind_panel.py`, `v3/refine.py`), which
leaves questions whose plausible-sounding option is wrong.
