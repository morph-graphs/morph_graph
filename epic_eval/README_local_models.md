# epic_eval with local models

Models we host on our own GPUs, loaded once per eval task. The shared pipeline,
run layout and scoring are in [README.md](README.md). Today there is one local
backend, `backends/qwen38_vl.py` (Qwen3.8 through Hugging Face transformers),
with configs:

| config | mode | eval GPUs |
|---|---|---|
| `configs/qwen3.8-27b-thinking.json` | thinking | 1× H100 |
| `configs/qwen3.8-27b-instruct.json` | instruct | 1× H100 |
| `configs/qwen3.8-27b-thinking-a100x2.json` | thinking | 2× A100 40 GB (speed comparison) |

27B in bf16 is ~54 GB of weights: one 80 GB GPU, or two 40 GB GPUs
(`device_map="auto"` splits the layers across whatever the task was given).

## How a local run differs from an API run

- **Code comes from your checkout.** The local configs have no `code.repo_url`, so
  the orchestrator snapshots the checkout it runs from. `--commit` defaults to
  `HEAD`; the checkout must have no uncommitted changes to tracked files (jobs
  run a clean clone, so an uncommitted edit would silently not be in the run).
  Nothing has to be pushed first; an unpushed commit prints a note, since
  pushing is what keeps the run reproducible for others.
- **No dry run needed.** One command submits prep → eval → score, chained with
  `afterany`; `--dryrun` is there if you want to read the scripts first.
- **Questions are chunked.** With `slurm.eval.questions_per_task` (8 in these
  configs), each video's pending questions are split, in group order, into
  chunks; each chunk is one array task on one GPU allocation. The array is not
  throttled (add `slurm.eval.max_parallel` to cap how many run at once). A chunk's qids are frozen
  in `slurm/qids/`, and it writes its own `answers/<video>.<shard>.jsonl`, so
  tasks never share a file.
- **One model per task.** Worker threads share it: `runtime.max_concurrent_groups`
  lets the next group's video decode while the current one generates, but
  generation itself is serialized. Keep `runtime.max_concurrent_requests` at 1.

## Environment

Jobs call `py`, defined once at the top of every job script. Give it either:

- a **`container`** block (what Zaratan uses):
  ```json
  "container": {"sif": "<path>/morph_graphs.sif", "binds": ["/scratch"],
                "pythonpath": "<optional dir of extra pure-Python packages>"}
  ```
  → `py() { apptainer exec --nv --cleanenv --bind /scratch --env PYTHONPATH=… "$SIF" python "$@"; }`.
  `--cleanenv` keeps host module/compiler variables from breaking Triton inside
  the image, which is why extra paths go in through `--env`.
- or **`paths.python`**: any interpreter path — a conda env's `bin/python`, a
  venv, a system Python — with no `container` block.

Whichever it is needs torch, transformers, flash-attn, qwen-vl-utils and torchcodec
at the versions in `proj_context/setup/pinned.txt` / `morph_graphs.def`, plus ffmpeg on `PATH` for
prep. The orchestrator itself only needs a stock `python3` (≥ 3.10). How to get
each on a given cluster: `proj_context/proj_docs/{nexus,zaratan}.md`.

## Running

From the `morph_graph/` directory, after committing your changes:

```bash
# smoke test: one short video, two questions
python3 -m epic_eval.orchestrator --config epic_eval/configs/qwen3.8-27b-thinking.json \
    --fps 1 --max_frames 768 --videos P09-20240622-194642 --limit 2

# full run
python3 -m epic_eval.orchestrator --config epic_eval/configs/qwen3.8-27b-thinking.json \
    --fps 1 --max_frames 768 --note "why this run exists"
```

Output, one line per submitted step:

```
eval: 57 task(s) -> job 22610000 (gpu-h100, gpu:h100:1, 8 cpu, 64G, 4:00:00)
score: 1 task(s) -> job 22610001 (standard, no gpu, 1 cpu, 4G, 0:30:00)
```

Prep is skipped for videos whose clips already exist in the media folder.
Rerun the same command to resume; it submits only questions without a parsed
answer. Every submission is appended to `<run>/slurm/launches.jsonl` (time,
step, job id, commit, command line, note).

A generated eval script reads like this:

```bash
#!/bin/bash
#SBATCH --job-name=ee-eval-qwen38-27b-thinking-fps1-f768
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --array=1-57
#SBATCH --dependency=afterany:22609999
...
RUN=<runs_root>/<sha8>_qwen38-27b-thinking-fps1-f768_cfg<hash>
SIF=<path>/morph_graphs.sif
py() { apptainer exec --nv --cleanenv --bind /scratch --env PYTHONPATH=<pyextra> "$SIF" python "$@"; }
cd "$RUN/code_repo"

case "$SLURM_ARRAY_TASK_ID" in
  1) py -m epic_eval.run_eval --run_dir "$RUN" --video P09-20240622-194642 --qids_file "$RUN/slurm/qids/P09-20240622-194642.<shard>.txt" --shard <shard> ;;
  2) ...
esac
```

Logs are in `<run>/slurm/logs/<step>_<jobid>_<task>.{out,err}`; a failed
question is also recorded, with its traceback, in `errors/*.jsonl`.

## Frame sampling is a sweep axis

`--fps` and `--max_frames` are required for `qwen38_vl` and are appended to the
run tag (`…-fps1-f768`), so each setting gets its own run folder. The processor
samples `min(fps × duration, max_frames)` frames from each clip; clips longer
than `max_frames / fps` get `max_frames` frames spread over their length. Every
answer row records `fps`, `max_frames` and the frames actually sampled.

## `qwen38_vl` model keys

| key | meaning |
|---|---|
| `path` | weights directory |
| `attn_implementation` | default `flash_attention_2` |
| `mode` | `thinking` or `instruct`: the chat template's thinking switch plus Qwen's sampling preset |
| `max_new_tokens` | output cap, thinking included; hitting it is recorded as `MAX_TOKENS` (truncated) |
| `seed` | reseeded before every question |
| `fps`, `max_frames` | set per launch (above) |
| `processor_kwargs` | other video settings for the processor (default: the model's own video config) |
| `generation_kwargs` | extra `generate()` arguments, applied over the mode's preset |

Sampling presets (from the Qwen3.8 model card):

| mode | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| thinking | 1.0 | 0.95 | 20 | 0.0 |
| instruct | 0.7 | 0.80 | 20 | 1.5 |

Each group's video is decoded and patchified once; every question of the group
reuses those tensors and is prefilled from scratch (no KV reuse), so an answer
does not depend on which questions came before it. Answer rows keep the
thinking text in `thoughts` and the reply in `raw_text`.

## Adding a local model

Everything model-specific lives in one backend module; prep, grouping, prompts,
resume, scoring and the orchestrator are shared, and the `[0, t_ask]` clips are
already cut.

1. **`backends/<name>.py`** with the four functions in `backends/__init__.py`:

   ```python
   from ..common import parse_letter

   def make_client(mcfg):                      # once per eval task: load weights onto the GPU(s)
       ...
   def open_context(client, mcfg, video_path, system_instruction, n_questions):
       # video_path is full.mp4 or upto_<t_ask>s.mp4 (None in text-only runs).
       # Decode frames once per group; everything returned is handed back to ask().
       return {...}
   def ask(client, mcfg, ctx, prompt, n_options):
       text = ...                              # system + video + prompt through the chat template; generate
       return {"pred_letter": parse_letter(text, n_options), "raw_text": text,
               "usage": {...}, "finish_reason": "STOP"}   # any extra keys land in answers/*.jsonl
   def close_context(client, ctx):
       pass                                    # free frames / KV
   ```

   Return `"finish_reason": "MAX_TOKENS"` when generation hit the cap, so the
   question is recorded as truncated and reruns skip it instead of re-asking. If worker
   threads share the model, serialize `generate()` with a lock (as
   `qwen38_vl.py` does).
2. **Register it** in `backends/__init__.py`: `BACKENDS["<name>"] = "epic_eval.backends.<name>"`.
3. **Config**: copy a Qwen config; change `run.tag`, `model` (`backend` plus the
   keys your module reads), the environment (`container` or `paths.python`), and
   `slurm.eval` (partition, `gres`, `questions_per_task`).
4. **Smoke test** with `--videos P09-20240622-194642 --limit 2`, then read
   `answers/*.jsonl` and `errors/*.jsonl`.

`--fps` / `--max_frames` are currently enforced only for `qwen38_vl`; a new
backend reads whatever keys it needs from `model`.
