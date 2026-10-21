# How questions get into evals

A plain description of how EPIC-Morph questions go from being written to being
scored, and what to do when new questions arrive (e.g. V3).

## The short version

- **Questions are written and reviewed in the `epic-morph` repo.** For each video, the
  questions that get scored are in one file:
  `outputs/<version>/<video>/questions_eval/<video>_questions.json`.
- **Evals live in this repo (`morph_graph/epic_eval`).** They never read your working
  copy of epic-morph. They read epic-morph **as it was at one specific commit**.
- **`epic_eval/banks.json` says which commit.** Each entry is a named set of questions,
  a *bank*: "`v2-150155-82` = epic-morph at commit `c905841…`, these files, 82 questions".
- **Every eval config names a bank** (`"questions": {"bank": "v2-150155-82"}`), and every
  run records exactly which commit it used.

So new questions affect evals only when you (1) commit them in epic-morph and (2) add or
update a bank in `banks.json` that points at that commit.

```
 epic-morph                                    morph_graph
 ──────────                                    ───────────
 write / review questions
        │
 questions_eval/<video>_questions.json
        │  commit + push
        ▼
 commit  abc1234  ───────────────────────▶  epic_eval/banks.json
                                              "v3-...": {commit: abc1234..., glob, questions: N}
                                                        │
                                             epic_eval/configs/*.json
                                              "questions": {"bank": "v3-..."}
                                                        │
                                             orchestrator → run folder
                                              (records the commit it used)
```

## How a launch gets the questions

1. **The config names a bank:** `"questions": {"bank": "v2-150155-82"}`.
2. **`epic_eval/banks.json` turns the name into a commit and files:** epic-morph commit
   `c905841…`, files `outputs/v2/*/questions_eval/*_questions.json`, and the expected
   count (82), which is checked before anything is submitted.
3. **The launcher reads that commit from a copy of epic-morph on the cluster:**
   - your own epic-morph clone, if `EPIC_MORPH_DIR` is set to it (any branch can be
     checked out; it only has to contain the commit, so `git fetch` it if needed); or
   - otherwise the shared copy in the runs folder,
     `.../morph_graphs/epic_eval_runs/_mirrors/epic-morph.git`, which is updated from GitHub.

   It never reads the files you have open in epic-morph, only that exact commit.
4. **It copies the questions into the run folder** (`<run>/questions_repo/`). The jobs read
   them from there, and that copy stays as the run's record of what it was asked.

## Adding a new set of questions (e.g. V3)

1. **Finish the questions in epic-morph** and make sure each video has its scored file at
   `outputs/<version>/<video>/questions_eval/<video>_questions.json` (see "What an
   eval file must contain" below).
2. **Commit and push epic-morph** (from a login node). Note the full commit SHA:
   `git rev-parse HEAD`.
3. **Add a bank to `morph_graph/epic_eval/banks.json`.** Pick a name that says what it is,
   e.g. `"v3-150155-120"`:
   ```json
   "v3-150155-120": {
     "description": "V3 bank: 120 questions on P09-20240622-150155",
     "repo_url": "git@github.com:vatsalag99/epic-morph.git",
     "commit": "<full 40-character SHA>",
     "glob": "outputs/v3/*/questions_eval/*_questions.json",
     "questions": 120
   }
   ```
   `questions` is a safety check: a launch stops if the bank doesn't contain exactly
   that many.
4. **Check it holds what you expect:**
   ```bash
   python3 -m epic_eval.bank v3-150155-120 --repo <epic-morph clone> --csv v3.csv
   ```
   This prints the count, videos and question forms, and writes every scored question
   (options, key, ask time, evidence) to a CSV. The clone must contain that commit,
   so fetch it first if needed.
5. **Make configs for it:** copy an existing config, change `run.tag` and set
   `"questions": {"bank": "v3-150155-120"}`. Commit and push morph_graph.
6. **Launch** as usual. Prep encodes clips for any new videos or ask times; everything
   else is reused.

## Fixing questions in an existing set

A fixed question (a corrected answer, a reworded option) is a **new version of the
questions**, so it needs a new epic-morph commit:

1. Make the fix in epic-morph and commit it.
2. In `banks.json`, either:
   - **bump the existing bank's `commit`** to the new SHA, if you want every future run of
     that benchmark to use the fixed version; or
   - **add a new bank name** (e.g. `v2-150155-82-r2`), if you want to keep comparing
     against runs made with the old version.

   If the number of questions changed, update `questions` too.
3. Runs launched afterwards use the fix. **Runs that already finished keep the questions
   they were scored with.** They are not rescored automatically, because each answer
   records whether it was correct at the time it was asked.

## What an eval file must contain

One JSON file per video:
`{"video": ..., "questions": [ ... ]}`. Each question needs:

| field | meaning |
|---|---|
| `q` | the question text |
| `video` | the video id, e.g. `P09-20240622-150155` |
| `form` | the question type (used to break scores down) |
| `streaming` | `true` if asked partway through the video |
| `t_ask` | seconds into the video when it is asked (required when `streaming`) |
| `answer` | the correct answer, as text |
| `gemini_mcq.options` | the answer options shown to the model, in order |
| `gemini_mcq.answer_index` | which option is correct (0 = A) |
| `evidence.clips` | optional: `[{event_id, clip: [t0, t1]}]`, needed for evidence-only runs |

A question's identity is its video + text + ask time. Changing any of those makes it a
different question as far as results are concerned.

## Where things live

| what | where |
|---|---|
| questions being written | epic-morph, `outputs/<version>/<video>/` |
| the scored questions | epic-morph, `.../questions_eval/<video>_questions.json` |
| which commit each benchmark uses | morph_graph, `epic_eval/banks.json` |
| which benchmark a config scores | morph_graph, `epic_eval/configs/*.json` (`questions.bank`) |
| what a finished run used | the run folder's `reproducibility_config.json` and `questions_repo/` |
| a readable list of a bank's questions | `python3 -m epic_eval.bank <name> --repo <clone> --csv <file>` |

## Rules of thumb

- One bank name means one exact set of questions. Don't reuse a name for a different set.
- Always put the full 40-character SHA in `banks.json`. Short ones make launches try to
  reach GitHub, which fails from compute nodes.
- Never edit files inside a run folder. They are that run's record.
