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

The config hash covers everything except `slurm` (resources do not change
results), with both commits resolved to full SHAs first.

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
Smoke test: `--videos P09-20240622-194642 --limit 2`.

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
answered questions. Always quote `non_binary`: the 129 yes/no questions carry
`caveat: binary_answer_format_cue` (the true answer is the only short option),
so they are reported separately. `chance` and the bank's `majority_baseline`
are printed beside every split.

## Adding a model

Write `backends/<name>.py` with `make_client`, `open_context`, `ask`,
`close_context` (see `backends/__init__.py`), register it in `BACKENDS`, and add
a config whose `slurm.eval` asks for a GPU if it needs one. Media prep is
model-independent and is reused.

## Tests

```bash
module load ffmpeg/7.1 && $PY -m epic_eval.test_epic_eval
```

Checks the bank shape (449 / 222 streaming / unique ids / answer text matches
the scored letter), grouping (140 + 12 contexts), that each streaming group is
sent its own `[0, t_ask]` clip, resume, scoring denominators, and a real encode
of a 6-min video whose clips must end at or before `t_ask` with no audio.
