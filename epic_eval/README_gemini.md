# epic_eval with Gemini (API models)

The shared pipeline, run layout and scoring are in [README.md](README.md); this
page covers what is specific to `backends/gemini.py`. Configs:
`configs/gemini-3.8-flash-high.json` (video) and
`configs/gemini-3.8-flash-high-textonly.json` (the no-video control).

## Setup

- **Python**: the jobs' interpreter (`paths.python`) needs `google-genai`; so does
  the one running the orchestrator, which checks for a key before submitting eval.
  ffmpeg must be on `PATH` for prep (`paths.ffmpeg_module` is loaded when set).
- **Key**: `$GEMINI_API_KEY` (name set by `model.api_key_env`), or a file readable
  only by you at `model.api_key_file` (default `~/.config/epic_eval/gemini_api_key`).
  Never put it in a config or on the command line.
- **Code**: these configs set `code.repo_url`, so jobs run a mirror checkout of
  that repo and `--commit` must be pushed.

Cluster-specific paths (interpreter, `runs_root`, partitions, shared media) are in
`proj_context/proj_docs/{nexus,zaratan}.md`.

## Running

From the `morph_graph/` directory:

```bash
python -m epic_eval.orchestrator --config epic_eval/configs/gemini-3.8-flash-high.json \
    --commit <pushed sha> --dryrun                        # show scripts + task counts
python -m epic_eval.orchestrator --config epic_eval/configs/gemini-3.8-flash-high.json \
    --commit <pushed sha>                                 # submit prep -> eval -> score
```

Smoke test first: add `--videos P09-20240622-194642 --limit 2` and read
`answers/*.jsonl`. Rerunning the same command resumes the run.

Eval tasks need no GPU; one task per video, and within a task
`runtime.max_concurrent_groups` contexts are open at once with up to
`runtime.max_concurrent_requests` requests in flight. 429s and other retryable
errors are retried with exponential backoff (`model.max_retries`,
`model.retry_delay_s`).

## Choosing fps and resolution

Pick `model.media_resolution` / `model.fps` from measured token counts on the
longest video before the main run (the 61-min one can overflow 1M tokens):

```bash
python -m epic_eval.probe_tokens --run_dir <run> --video P09-20240623-120359 --fps 1 0.5
```

## Model keys

| key | meaning |
|---|---|
| `name`, `api_version` | model id and optional API version |
| `thinking_level`, `include_thoughts` | reasoning budget; thoughts are saved in each answer row |
| `temperature`, `seed`, `max_output_tokens` | generation settings |
| `fps`, `media_resolution` | how the uploaded clip is sampled |
| `cache_min_questions`, `cache_ttl_s` | context caching (below) |
| `max_retries`, `retry_delay_s`, `upload_timeout_s` | robustness |

## Context caching

Each group's video is uploaded once. Groups with at least
`model.cache_min_questions` questions get a context cache (system instruction +
video) that all their questions reuse; smaller groups, or groups whose cache
creation fails, are asked directly with the identical system instruction and
video part. `cache_used_frac` in `scores.json` reports how often the cache was hit.

A reply cut off by `max_output_tokens` (`MAX_TOKENS`, usually HIGH thinking on long
videos) is not retried by default; see README.md for `runtime.retry_truncated`.

## The text-only control

`configs/gemini-3.8-flash-high-textonly.json` is the same prompts and options with
`media.enabled: false`. The gap between the two runs is how much the video is
actually used — the options were written from narration transcripts, so text
priors may carry a lot.

## Reusing encoded clips

Clips depend only on `media` and `paths.video_dir`, so any run with the same media
settings can read an existing media folder in place: set `paths.media_root` to its
parent, and `run.runs_root` to a directory you can write. Prep then finds those
videos ready and submits nothing for them. `media_root` is not part of the run hash.
If the folder is read-only for you, a video not yet encoded there fails at prep;
use your own `media_root` until it is complete. Shared media folders on each
cluster are listed in the cluster docs.
