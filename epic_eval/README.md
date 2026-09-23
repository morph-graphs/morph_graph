# epic_eval — sandboxed MCQ evaluation on EPIC-Morph

Poses the EPIC-Morph question bank (in `vatsalag99/epic-morph`,
`outputs/questions/*_questions.json`) to a video model as 4-way multiple choice
and scores it. Modeled on VidGram's `pipeline_orchestrator.py`: every run
executes from fresh checkouts at pinned commits, never from a working tree.

## What the model sees

| question | video input | cached? |
|---|---|---|
| `streaming: false` (227) | the full recording | one cache per video, shared by all its full-video questions |
| `streaming: true` (222) | `[0, t_ask]` only — a physical ffmpeg cut, validated by ffprobe to end at or before `t_ask` | one cache per distinct `(video, t_ask)` (140 of them, 61 shared by 2–10 questions) |

Options are `gemini_mcq.options` in bank order (already shuffled on a
`video|question` seed; correct letter spread 100/116/112/121), answer is
`gemini_mcq.answer_index`. The generator's own `options` field is not used.
Groups with fewer than `model.cache_min_questions` questions, or whose cache
creation fails, are asked directly with the identical system instruction and
video part.

## Layout of a run

```
<runs_root>/
  _mirrors/                      bare mirrors of both repos (fetched every run)
  media/<media_hash>/<video>/    full.mp4, upto_<t_ask>s.mp4, _done.json   (shared across runs
                                 with the same media settings)
  <code sha8>_<tag>_cfg<hash>/   one run = one (code commit, config) pair
    code_repo/  questions_repo/  reproducibility_config.json
    answers/<video>.jsonl        one row per answered question (letter, raw text, thoughts, usage)
    errors/<video>.jsonl         failures; retried by the next run
    slurm/                       generated scripts, command lists, logs/%A_%a
    scores.json  scores.txt
```

The config hash covers everything except `slurm` and `runtime` (resources and
concurrency do not change answers), with both commits resolved to full SHAs first.

## Running

```bash
PY=/fs/nexus-scratch/anubhav/anubhav_python3.13.7/bin/python
cd <a morph_graph checkout>          # only the orchestrator runs from here
# key: $GEMINI_API_KEY, or a 600-mode file at ~/.config/epic_eval/gemini_api_key

$PY -m epic_eval.orchestrator --config epic_eval/configs/gemini-3.8-flash-high.json \
    --commit <pushed sha> --dryrun                        # show scripts + task counts
$PY -m epic_eval.orchestrator --config ... --commit <sha> # submit prep -> eval -> score
```

Rerunning the same command is the recovery path: prep skips videos whose
`_done.json` matches the files on disk, eval skips questions with a parsed
answer, and only failures are redone. Jobs are recorded with `nexrun launch`.
Smoke test: `--videos P09-20240622-194642 --limit 2`. Concurrency (`runtime.max_concurrent_groups`,
`runtime.max_concurrent_requests`, per eval task) is excluded from the run hash, so it can be
retuned without starting a new run; 429s are retried with exponential backoff.

Before the main run, pick `model.media_resolution` / `model.fps` from measured
token counts on the longest video (the 61-min one can overflow 1M tokens):

```bash
$PY -m epic_eval.probe_tokens --run_dir <run> --video P09-20240623-120359 --fps 1 0.5
```

`configs/gemini-3.8-flash-high-textonly.json` is the no-video control: same
prompts and options, `media.enabled: false`. The gap between the two runs is
how much the video is actually used — the options were written from narration
transcripts, so text priors may carry a lot.

## Reading the scores

`acc` divides by the whole bank (missing = wrong); `acc_answered` divides by
answered questions. Always quote `non_binary`. The 129 yes/no questions are
reported separately (`binary_yes_no`) because their option layout has been a
format cue in every bank version so far. At epic-morph `04fb5d5` each has one
option with the true verdict, two with the opposite verdict, and
"insufficient information", so "pick the verdict that appears once" is right
129/129 without the video. `chance` and the bank's `majority_baseline`
are printed beside every split.

## Adding a model (e.g. Qwen3-VL)

Everything model-specific lives in one backend module. Prep, grouping, prompts,
resume, scoring and the orchestrator are shared, and the `[0, t_ask]` clips are
already cut, so a new model inherits the no-leak guarantee without any work.

1. **`backends/qwen3_vl.py`** — four functions (contract in `backends/__init__.py`):

   ```python
   from ..common import parse_letter

   def make_client(mcfg):                      # once per eval task: load weights onto the GPU
       ...                                     # model_zoo/Qwen3-VL-8B-Instruct is on disk;
                                               # video_models/qwen3_vl.py shows the loading code
   def open_context(client, mcfg, video_path, system_instruction, n_questions):
       # video_path is full.mp4 or upto_<t_ask>s.mp4 (None in text-only runs).
       # Decode frames here, once per group (e.g. mcfg["fps"], mcfg["max_frames"]),
       # optionally prefill the KV cache; everything returned is handed back to ask().
       return {"frames": ..., "system_instruction": system_instruction}
   def ask(client, mcfg, ctx, prompt, n_options):
       text = ...                              # chat template: system + frames + prompt; generate
       return {"pred_letter": parse_letter(text, n_options), "raw_text": text,
               "usage": {...}, "cache_used": False}   # any extra keys land in answers/*.jsonl
   def close_context(client, ctx):
       pass                                    # free frames / KV
   ```

2. **Register it**: `BACKENDS["qwen3_vl"] = "epic_eval.backends.qwen3_vl"`.
3. **Config**: copy `configs/gemini-3.8-flash-high.json` and change
   - `run.tag`, and `model` to `{"backend": "qwen3_vl", ...}` with whatever
     keys the backend reads (the Gemini-only keys can go);
   - `paths.python` if the model needs a different environment;
   - `slurm.eval` to a GPU tier, e.g. `vulcan-scavenger` with `"gres": "gpu:rtxa6000:1"`;
   - `runtime.max_concurrent_groups` / `max_concurrent_requests` to `1` — the
     worker threads share one model, so either keep them at 1 or lock inside `ask`.
4. **Test** before a full run: `--videos P09-20240622-194642 --limit 2`, then
   open `answers/*.jsonl`.

Leave `media` as it is and set
`"paths": {"media_root": "/fs/cfar-projects/mem_open_vocab/morph_graphs/anubhav/epic_eval_runs/media", ...}`
to read the already-encoded clips in place (1024 px, 2 fps; group `openvocab` can
read them), with `run.runs_root` pointing at a directory you can write. Prep then
finds every video ready and submits nothing. `media_root` is not part of the run
hash. The shared folder is read-only for others, so a video not yet encoded there
fails at prep; use your own `media_root` until all 12 are in. If the model wants
different frames, change `media`: that gets its own media folder, and `prep`
re-encodes for it.

Nothing outside the backend is Gemini-specific except the API-key check in the
orchestrator, which only runs when `model.backend == "gemini"`, and
`probe_tokens.py`.

## Tests

```bash
module load ffmpeg/7.1 && $PY -m epic_eval.test_epic_eval
```

Checks the bank shape (449 / 222 streaming / unique ids / answer text matches
the scored letter), grouping (140 + 12 contexts), that each streaming group is
sent its own `[0, t_ask]` clip, resume, scoring denominators, and a real encode
of a 6-min video whose clips must end at or before `t_ask` with no audio.
