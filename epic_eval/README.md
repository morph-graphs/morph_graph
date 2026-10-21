# epic_eval — sandboxed MCQ evaluation on EPIC-Morph

Poses the EPIC-Morph question bank (`vatsalag99/epic-morph`,
`outputs/questions/*_questions.json`) to a video model as multiple choice and
scores it. Modeled on VidGram's `pipeline_orchestrator.py`: every run executes
from fresh checkouts at pinned commits, never from a working tree.

This page covers what is shared by every model. Model-specific setup lives in:

| | |
|---|---|
| [README_gemini.md](README_gemini.md) | API models (Gemini): keys, context caching, token budgets |
| [README_local_models.md](README_local_models.md) | Models we host on our own GPUs (Qwen3.8): one-command launch, frame sampling, adding a backend |

How to get a Python environment and submit jobs on each cluster is in
[`proj_context/proj_docs/nexus.md`](../proj_context/proj_docs/nexus.md) and
[`proj_context/proj_docs/zaratan.md`](../proj_context/proj_docs/zaratan.md).

## Pipeline

```
orchestrator ──► prep ──► eval ──► score        (SLURM array jobs, chained with afterany)
                                    │
                                    └──► explorer  (run by hand, over any set of runs)
```

| step | module | one task = | writes |
|---|---|---|---|
| orchestrate | `orchestrator.py` | runs once, submits, exits | the run folder, code + question checkouts, job scripts |
| prep | `prep_media.py` | one video | `full.mp4` + one `upto_<t_ask>s.mp4` clip per streaming cutoff |
| eval | `run_eval.py` | one video, or one chunk of its questions | `answers/*.jsonl`, `errors/*.jsonl` |
| score | `score.py` | the whole run | `scores.json`, `scores.txt` |
| explore | `explorer.py` | — | one self-contained HTML page |

Every step decides its work from what is on disk, so rerunning the same
orchestrator command is also how a run is resumed after failures or preemption:
prep skips videos whose `_done.json` matches the clips on disk, eval skips
questions that already have a parsed answer, and only the rest is submitted.

## Directory

```
epic_eval/
  orchestrator.py        mirror/check out code + questions, build work lists, write + submit job scripts
  prep_media.py          ffmpeg: encode the full video and the [0, t_ask] clips (validated with ffprobe)
  run_eval.py            group questions by the video they see, open one context per group, ask, append answers
  score.py               accuracy splits over the whole bank
  explorer.py            HTML page over run folders (+ explorer_template.html)
  probe_tokens.py        Gemini only: measure video token counts before a run
  common.py              config/hash helpers, question loading, grouping, prompts, answer parsing
  backends/              one module per model family (contract in backends/__init__.py)
    gemini.py            API, with context caching
    qwen38_vl.py         local Hugging Face model
  configs/               one JSON per model setting
  test_epic_eval.py      invariants that decide whether the numbers mean anything
```

## Configs

A config is one JSON file. Its sections:

| section | what it sets | in the run hash? |
|---|---|---|
| `run` | `tag` (names the run folder), `runs_root` | tag only via the folder name |
| `code` | `repo_url` to mirror the eval code from; omit it to snapshot the local checkout (see README_local_models.md) | commit: yes |
| `questions` | `{"bank": "<name>"}` from `banks.json` (or an explicit `repo_url`, pinned `commit`, `glob`) | yes (the resolved commit) |
| `paths` | `video_dir`, `python` (interpreter for jobs), `ffmpeg_module`, optional `media_root` | yes, except `media_root` |
| `container` | Apptainer image jobs run in, instead of `paths.python` | yes |
| `media` | clip encoding: `max_side`, `encode_fps`, `crf`, `audio`, `enabled` (false = text-only) | yes |
| `model` | `backend` plus whatever that backend reads | yes |
| `runtime` | concurrency inside an eval task, `retry_truncated` | no |
| `slurm` | per-step account, partition, resources, optional `max_parallel` (array throttle) | no |

The run folder is `<code sha8>_<tag>_cfg<hash>`, with both commits resolved to
full SHAs before hashing. Resources and concurrency are left out of the hash, so
they can be retuned without starting a new run.

**Which Python runs what.** The orchestrator only needs a Python ≥ 3.10 with the
standard library (plus `google-genai` for Gemini's key check). The jobs run
`py -m epic_eval.<step>`, where `py` is defined once at the top of each job
script: `paths.python` if the config has no `container` block (a conda env, a
venv, or any interpreter path), otherwise `apptainer exec … <sif> python`.
Whichever it is needs the backend's packages, and ffmpeg on `PATH` for prep
(`paths.ffmpeg_module` is `module load`ed first when set).

## Question banks

`banks.json` names each benchmark bank: its repo, full commit SHA, glob and question
count (`v1-449`, `v2-150155-82`). Configs refer to a bank by name; the orchestrator
resolves it to that exact commit (so a bank name and the equivalent explicit section give
the same run folder) and stops if the bank's question count is not the one listed.
Moving a benchmark to a new bank commit is a one-line change in `banks.json` plus a new
name if the question set changes.

To see exactly what a bank holds (any epic-morph clone or the runs-root mirror works,
whatever branch it has checked out):

```bash
python3 -m epic_eval.bank --list
python3 -m epic_eval.bank v2-150155-82 --repo <epic-morph clone or mirror> [--csv questions.csv]
```

## Running

From the `morph_graph/` directory:

```bash
python3 -m epic_eval.orchestrator --config epic_eval/configs/<config>.json [--commit <ref>] \
    [--videos P09-20240622-150155 --limit 2]   # smoke test: one short video, two questions
    [--steps prep,eval,score] [--dryrun] [--note "why this run exists"]
```

`--dryrun` checks everything out and prints the job scripts without submitting.
Each submission is logged to `<run>/slurm/launches.jsonl`. The model READMEs
give the exact commands.

## Evidence-only mode

`media.mode: "evidence"` (default `"cutoff"`, everything above) shows each
question only its own evidence: the bank's `evidence.clips` (`[t0, t1]` per
event). `configs/gemini-3.8-flash-high-v2-evidence.json` is the V2 config with
this mode. The gap to the cutoff run separates finding the moments in a long
video from reasoning over them.

- **Media**: one `evidence_<qid>.mp4` per question. Clips are sorted, overlapping
  ones merged (nothing is shown twice), and cut at `t_ask` (some V2 checklist
  questions list later events of the object; those would show the future). Each
  clip is cut from the source with input-side `-ss/-t`, never from `full.mp4`,
  and all are concatenated in one encode with the same `max_side`/`encode_fps`/`crf`;
  the file's length is the summed clip length to within a frame. `_done.json`
  lists each file's clip bounds; a file is re-encoded when the bank's clips change.
  A question without evidence is an error. `mode` is part of the media hash, so
  these files get their own media folder.
- **Grouping**: one context per question, no caching (the files are seconds long).
- **Prompt**: the system instruction says the model sees a few short excerpts in
  chronological order and must answer from them; it does not mention `t_ask` or
  what was left out. Question, options and scoring are the cutoff mode's.

## Layout of a run

```
<runs_root>/
  _mirrors/                      bare mirrors of the repos (fetched unless the wanted full SHA is already there)
  media/<media_hash>/<video>/    full.mp4, upto_<t_ask>s.mp4, _done.json
                                 (shared by every run with the same media settings)
  <code sha8>_<tag>_cfg<hash>/   one run = one (code commit, config) pair
    code_repo/  questions_repo/  reproducibility_config.json
    answers/<video>.jsonl        one row per answered question (letter, raw text, thoughts, usage)
    errors/<video>.jsonl         failures; retried by the next launch
                                 (with slurm.eval.questions_per_task: answers|errors/<video>.<shard>.jsonl)
    slurm/                       <step>_<stamp>.slurm, launches.jsonl, qids/ chunk lists, logs/<step>_%A_%a
    scores.json  scores.txt
```

## What the model sees

| question | video input |
|---|---|
| `streaming: false` | the full recording |
| `streaming: true` | `[0, t_ask]` only — a physical ffmpeg cut, validated by ffprobe to end at or before `t_ask` |

Questions that see the same input form one group, and `run_eval` opens one
backend context per group (one per video for full-video questions, one per
distinct `(video, t_ask)` for streaming ones), so a new backend inherits the
no-leak guarantee without any work.

Options are `gemini_mcq.options` in bank order (already shuffled on a
`video|question` seed); the answer is `gemini_mcq.answer_index`. The
generator's own `options` field is not used.

A reply that hits the output cap before answering (`finish_reason:
MAX_TOKENS`, usually long thinking) is final: reruns skip it and `score` lists
it under `truncated_qids` (counted wrong in `acc`). Set
`runtime.retry_truncated: true` to re-ask them, e.g. after lowering the
thinking budget.

## Reading the scores

`acc` divides by the whole bank (missing = wrong); `acc_answered` divides by
answered questions. Always quote `non_binary`. The yes/no questions are
reported separately (`binary_yes_no`) because their option layout has been a
format cue in every bank version so far. At epic-morph `04fb5d5` each has one
option with the true verdict, two with the opposite verdict, and
"insufficient information", so "pick the verdict that appears once" is right
on all of them without the video. `chance` and the bank's `majority_baseline`
are printed beside every split.

## Looking at the answers: the run explorer

```bash
python -m epic_eval.explorer --runs <run_dir> [<run_dir> ...] \
    --labels "run A,run B" --out <file>.html [--verdicts verdicts.json]
```

Writes one self-contained HTML file from any set of run folders: every question
with its options, the correct answer, each run's pick and reasoning summary, and
the narration evidence with timestamps. Filter by question type, video,
streaming or full, yes/no, and outcome (e.g. "run A wrong, run B right");
accuracy tables follow the filters. Questions are joined across runs by qid; if a
later bank changed a question's options, each run keeps its own and the page
says so.

Each question takes a verdict (model wrong / bank answer wrong / question
ambiguous / options give it away / other) and a note:
- **Opened as a local file**: verdicts are kept in that browser; *Export
  verdicts* saves them as JSON, and `--verdicts <file>` embeds them in the next
  build, so they can travel through the run folder or git.
- **Published as a claude.ai artifact with the `db` capability**: verdicts are
  shared live between everyone the page is shared with (*Import verdicts* merges a
  local export into the shared set).

## Tests

```bash
EPIC_MORPH_DIR=<epic-morph checkout> python -m epic_eval.test_epic_eval   # ffmpeg on PATH for the media tests
```

Checks the bank shape (unique ids, streaming questions have a `t_ask`, the
scored letter's text matches the bank answer), grouping, that each streaming
group is sent its own `[0, t_ask]` clip, resume, concurrency, truncation
handling, scoring denominators, the explorer's join, and a real encode of a
6-min video whose clips must end at or before `t_ask` with no audio. Evidence
mode: on a synthetic black/white source, each evidence file's duration and frame
count match its clips (±0.3 s, ±1 frame) and it contains no white pixel (nothing
from outside the clips or after `t_ask`); one context per qid; the instruction never
mentions a cutoff.
