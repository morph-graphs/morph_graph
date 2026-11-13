# baselines/api: closed-source API models on EPIC-Morph

The API rows of the results sheet have no memory: each question is answered from the video up to its `t_ask`.
That is what `epic_eval/` does (one ffmpeg-cut `[0, t_ask]` clip per distinct ask time, checked with ffprobe;
`epic_eval/README.md`), so these rows run epic_eval's steps and convert the answers into the baselines' result
layout. Rows prepared: `gemini-3.1-flash-lite`, `gemini-3.6-flash`, `gpt-5`. Gemini 3.1 Pro is dropped for now.

## Layout

| path | what |
|---|---|
| `configs/<name>.json` | one model setting (`media`, `model`, `runtime`; no paths). The name is the arm of the results |
| `run_api.sh` | one video, one config, one stage (prep, qa, upload, score, all) |
| `epic/make_run.py` | the epic_eval run folder for a config and a bank, without Slurm; keeps a copy of the bank it was asked |
| `epic/from_epic_eval.py` | a run folder -> `<arm>/<video>/answers_<bank>.json.gz` + `runs.jsonl` (no model call) |
| `../../epic_eval/backends/{gemini,openai}.py` | the model calls: video file (Gemini), sampled frames (OpenAI) |
| `env/requirements.txt`, `docker/` | the pinned clients; the image (CPU only, with ffmpeg), `build.sh`, `run.sh`, `image_tag.sh` |
| `work/`, `results/` | run folders and clips; results (neither committed) |

| config | model id | input | settings |
|---|---|---|---|
| `gemini-3.1-flash-lite-high` | `gemini-3.1-flash-lite` (stable) | the clip as a video file | 1 fps, `MEDIA_RESOLUTION_HIGH`, thinking HIGH, seed 0 |
| `gemini-3.6-flash-high` | `gemini-3.6-flash` (stable) | same | same |
| `gemini-3.8-flash-high` | `gemini-3.8-flash` | same | same: the settings of the run already made (below) |
| `gpt-5-f1024` | `gpt-5` (deprecated, **shut down 2026-12-11**) | frames of the clip | up to 1,024 frames, at most 1 a second, 512 px, `detail: high`, reasoning effort high |

Both new Gemini models accept thinking `high` (their documented levels are minimal, low, medium, high), so
the three Gemini configs differ in the model id only. Model ids, limits and prices were read from the
providers' pages on 2026-11-13 ([Gemini models](https://aistudio.google.com/docs/models),
[thinking](https://aistudio.google.com/docs/thinking), [pricing](https://aistudio.google.com/docs/pricing),
[GPT-5](https://developers.openai.com/docs/models/gpt-5),
[deprecations](https://developers.openai.com/docs/deprecations),
[images](https://developers.openai.com/docs/guides/images-vision)), not from a call with a key.

**GPT-5 takes no video**, so `backends/openai.py` sends frames of the same `upto_<t_ask>s.mp4` clip: evenly
spaced, ending on the clip's last frame, so none is from after `t_ask`. The frames come first in the request
(the provider's automatic prompt cache can then reuse them across a clip's questions), then the same question
prompt and system instruction as Gemini gets, preceded by one sentence saying the video is given as N frames
in order. Requests are sent with `store: false`. Count, size and detail are config fields (`max_frames`,
`max_fps`, `frame_side`, `detail`; `frame_timestamps: true` adds a `t=..s` label before each frame). 1,024
frames at 512 px are ~215k input tokens of the model's 272k; the provider's limits are 1,500 images and 512 MB
a request. Each answer row records `frame_times_s`, `last_frame_s`, `n_frames`, the token usage and the model
id the API returned (`model_returned`).

**The cutoff rule in the results.** The converter stops unless every streaming answer shows its input ended by
`t_ask`: the clip's length, and for frames the last frame time. It joins answers to the bank copy inside the
run folder only, and stops when two questions share an answer or a gold letter differs.

## For Max

Everything runs in one CPU image (no GPU, no model download). You need: this repo and epic-morph checked out
at `main`, the HD-EPIC videos (epic-morph `v3/DATA.md`), Docker, and your own **paid-tier** API keys (a
Gemini API key with billing enabled; an OpenAI API organisation key), exported in the shell, never written
into a file or a command.

```bash
docker pull ghcr.io/morph-graphs/api:$(bash baselines/api/docker/image_tag.sh)   # or: bash baselines/api/docker/build.sh
export EPIC_MORPH_DIR=<epic-morph checkout>     # videos: $EPIC_MORPH_DIR/data/HD-EPIC/videos, or set HDEPIC_VIDEOS
export GEMINI_API_KEY=...  OPENAI_API_KEY=...   # only the provider you are running is needed

# smoke test first: one video, two questions, then read baselines/api/work/runs/<config>_<bank>/answers/
bash baselines/api/docker/run.sh prep P07-20240531-201703 gemini-3.6-flash-high
API_LIMIT=2 bash baselines/api/docker/run.sh qa P07-20240531-201703 gemini-3.6-flash-high

# the run: every video of the bank, each config (prep is shared by all configs: the clips are the same)
for V in $(cd $EPIC_MORPH_DIR/outputs/v3/banks/batch10_agree2 && ls | sed 's/_questions.json//'); do
  bash baselines/api/docker/run.sh prep "$V" gemini-3.6-flash-high
  for C in gemini-3.1-flash-lite-high gemini-3.6-flash-high gpt-5-f1024; do
    bash baselines/api/docker/run.sh qa "$V" "$C" && bash baselines/api/docker/run.sh upload "$V" "$C"
  done
done
bash baselines/api/docker/run.sh score P07-20240531-201703 gemini-3.6-flash-high   # accuracy table of a whole run
```

- `prep` cuts the clips with ffmpeg (CPU; ~18 GB for the 10 batch10 videos, under `baselines/api/work/media/`).
- `qa` asks only the questions with no answer yet, so the same command resumes after a failure. It exits 2
  while questions are missing (their keys are printed; errors are in `work/runs/<config>_<bank>/errors/`).
  A reply cut off at the output cap is final: it is listed as missing and counts as wrong.
- `upload` needs a Hugging Face token with write access to the results dataset (`HF_TOKEN`, or `hf auth login`
  on the host). Results come back as `api/<config>/<video>/answers_<bank>.json.gz` and `runs.jsonl` in the
  private dataset [morph-graphs/epic-morph-results](https://huggingface.co/datasets/morph-graphs/epic-morph-results);
  nothing goes into git.
- On a cluster without Docker: `PY=<python with env/requirements.txt> bash baselines/api/run_api.sh ...`, with
  ffmpeg >= 5.1 on `PATH`.

Expected cost for the 598 batch10 questions, from the published prices and the token counts measured in the
3.8 Flash run (27.3M uncached input, 261.6M cache reads, 3.04M output incl. thinking; ~17M tokens of cached
clips, billed once at the input price if cache creation is billed, which is not confirmed):

| config | $/1M in, cache read, out | expected | if nothing is cached | cache storage on top |
|---|---|---|---|---|
| `gemini-3.1-flash-lite-high` | 0.25, 0.025, 1.50 | ~$18-22 | ~$77 | $1.00/1M/h: ~$3 at 10 min a clip, $34 at the 2 h TTL |
| `gemini-3.6-flash-high` (prices double on 2027-01-01) | 0.75, 0.075, 3.75 | ~$51-64 | ~$228 | $0.50/1M/h: ~$1 to $17 |
| `gpt-5-f1024` | 1.25, 0.125 cached, 10.00 | ~$170-200 | same (this is the uncached figure) | none |

The GPT-5 figure assumes ~5k reasoning tokens a question (~$30 of it); the Gemini figures assume these models
tokenize video like 3.8 Flash (256 tokens a second at these settings) and think about as much.

## What exists: Gemini 3.8 Flash on batch10

epic_eval run `27c6fddf_gemini38flash-high-v3-batch10_cfgb399d802`, made with the Slurm orchestrator from
`epic_eval/configs/gemini-3.8-flash-high-v3-batch10.json` on bank `v3-batch10-598` (`epic_eval/banks.json`:
epic-morph `0f30904`, where `outputs/v3/banks/batch10_agree2` was still committed; banks are now generated by
`v3/export_bank.py` and read from the epic-morph checkout, which is what `run_api.sh` does).

- 542/598 = 90.6%, 0 missing. **Inflated by selection**: the bank kept a question when 2+ of its 3 verifiers
  (Gemini 3.8 Flash, GPT-5.6-Sol, Qwen3.8-Max) answered it right, so a verifier's own score on it is biased up.
  The converter writes this as `note` in the `runs.jsonl` line of any result from one of those three models.
- Converted with `from_epic_eval.py --run <that folder> --out baselines/api/results`: 10 files, 598 rows, 542
  right, the same per video as `scores.json`; MERIT's `match_answers()` accepts all 598 rows.
- Usage: 288.8M prompt tokens (261.6M cache reads), 3.03M thinking tokens; 256 tokens per second of video;
  35k-840k prompt tokens a question (mean 483k); ask times 136-3,279 s; 132 distinct clips.

Limit to keep in mind: at 256 tokens/s a 1M-token window holds 68 min of video at these settings. Batch10's
longest clip (3,279 s, 840k tokens) fits; a later tranche with a video over ~65 min needs one lower setting
for the whole row, not per video. For GPT-5 the frame count is fixed, so only the spacing grows.

## Data handling

Paid-tier API routes only, never a consumer surface. Gemini API terms, paid services: "Google doesn't use
your prompts (including associated system instructions, cached content, and files ...) or responses to
improve our products"; prompts are logged "for a limited period of time, solely for detecting and preventing
violations of the Prohibited Use Policy" ([terms](https://aistudio.google.com/docs/terms)); on the unpaid tier
"human reviewers may read, annotate, and process your API input and output". OpenAI: "data sent to the OpenAI
API is not used to train or improve OpenAI models" unless opted in; abuse-monitoring logs are "retained for up
to 30 days" ([your data](https://developers.openai.com/docs/guides/your-data)). The Gemini backend deletes
its uploaded files and caches when a clip's questions are done.

## Open

1. `gpt-5` is shut down on 2026-12-11; a rerun after that needs another model id (a new config, a new row).
2. Like-for-like: Gemini sees the clip at 1 fps, GPT-5 at most 1,024 frames (one every ~3 s on a 55 min clip).
   A Gemini run on the same frames would separate the model from the sampling; not built.
3. Later tranches: the 68 min ceiling above; a new bank name is a new run folder (`<config>_<bank>`).
4. The job board (`coord/master.py` `EXPERIMENTS`) does not list these rows yet; they need no GPU pool.
