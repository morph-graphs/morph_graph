# coord/sheet_update.py: the results sheet's `Auto: EPIC-Morph` tab

A plain script (cron, no LLM) that rewrites one tab of the results Google Sheet from the results dataset
(`morph-graphs/epic-morph-results`). It owns that tab and writes nowhere else: hand-maintained tabs refer
to it by formula.

## What the tab holds

Rows 1-9 are a header block: A1 the ownership sentinel, then `status` (`OK`, or `ERROR <time>: ...` when the
last run failed; the table below is then the last good run's), the last run time (UTC and US Eastern), the
results-dataset revision, the epic-morph commit the banks were generated from, the row count, the
conventions, and the taxonomy leaf names. Row 10 holds the column names, and the table starts at row 11:
one row per (baseline, arm, bank, tag) that has an answers file.

| columns | meaning |
|---|---|
| `row_key` | `<baseline>/<arm>/<bank>[/<tag>]`: the stable key to look a row up by (`XLOOKUP`, `VLOOKUP`) |
| `method`, `memory_model`, `reasoning_model`, `n_frames` | from `METHODS` in the script, mirroring the hand tab `Epic Morph Results`; an arm not listed there shows as `<baseline>/<arm>` |
| `baseline`, `arm`, `bank`, `tag` | as in the dataset path and file name (`answers_<bank>[_<tag>].json.gz`) |
| `videos_done`, `videos_started`, `videos_in_bank` | done: every question of the video has a counted answer; started: an answers file exists |
| `q_answered`, `q_in_bank`, `n_right` | counted answers, questions of the current bank, right ones |
| `acc_all_pct` | `n_right / q_in_bank`: an unanswered question is wrong |
| `acc_answered_pct`, `ci95_lo_pct`, `ci95_hi_pct`, `ci_method` | `n_right / q_answered` and its 95% interval: percentile bootstrap over videos (2,000 resamples, seed 0), Wilson when one video |
| `blind_videos`, `blind_answered`, `blind_right`, `blind_acc_all_pct`, `blind_acc_answered_pct` | the arm's no-memory floor on the bank (`blind_<bank>.json`), same rules (`blind_videos`: fully answered ones); empty when no blind file exists |
| `chance_pct` | mean of 1 / number of options over the bank |
| `answers_dropped` | rows of the answer files that were not counted: failed (empty or `Error`), stale, duplicate, or of a video outside the bank |
| `note` | `partial: ...` while videos are missing, then the `note` fields of the videos' `runs.jsonl` (e.g. inflated by selection) |
| `gpus`, `last_updated_utc` | GPU models and the latest `finished` time in `runs.jsonl` (the memory's stages and this bank's); empty when the lines carry none |
| `leaf_<code>_pct` | accuracy over the answered questions carrying that taxonomy leaf (epic-morph `v3/taxonomy.json`); a question with two leaves counts in both. Last, so a new leaf never moves the columns before it |

Percentages are plain percent values (36.29 means 36.29%), numbers are written as numbers (`RAW` input: no
text is ever taken for a formula), so the tab needs no number formats.

**Which answers count.** The banks are regenerated from the epic-morph checkout on every run
(`v3/export_bank.py`), and an answer counts only while its question's wording, options and gold letter are
those of the current bank: `prepare_epic.match_answers`, the rule the baselines use to reuse answers (by key;
by content for answer files from before keys, such as the batch10 MERIT runs). Whether an answer is right is
its recorded `evaluate`. A blind file with a tag (`blind_<bank>_<tag>.json`, MERIT answering from a given
context) is a row of its own, tag `blind-<tag>`, not a floor.

## Adding a method

Add a line to `METHODS` in `coord/sheet_update.py`: `(baseline, arm): (Method, Memory Model, Reasoning Model,
N. Frames)`, with the names used in the hand tab. A new bank is a name added to `BANKS` (it must be a bank
`v3/export_bank.py` knows). Nothing else: rows appear when the dataset holds answers.

## Safety

- One constant tab name (`TAB`); `--tab` (tests) accepts only names starting with `Auto: `.
- The tab is created by the script with a sentinel in A1, in one atomic request. An existing tab of that name
  without the sentinel is never written to.
- Every HTTP call goes through `send()`, which refuses any write that is not a values `PUT` into that tab or
  that creation request. There is no code that clears, inserts or deletes rows, or touches another tab. A
  shorter table is written padded with empty cells over the previous one, in the same single write.
- The scope is `https://www.googleapis.com/auth/spreadsheets` only; the service account key is read from
  `--sa-key` (default `~/.config/gsheet_sa.json`) and never logged.

## Running it

```bash
EPIC_MORPH_DIR=<epic-morph checkout at main> HF_TOKEN=<read token> \
    python coord/sheet_update.py --dry-run          # prints the tab's rows, contacts no sheet
python coord/sheet_update.py --cache <dir>          # writes the tab; <dir> keeps slim copies of the result files
```

Needs `huggingface_hub`, `google-auth` and `requests`. Result files are cached by blob hash, so a run with
nothing new downloads nothing (~10 s). Exit code 1 on failure, after writing the `ERROR` line into B2.

On Nexus it runs from cron on the login node every 30 minutes (`crontab -l | grep "epic-morph sheet updater"`),
through a wrapper next to the job board master's that refreshes its own clones of `main` of this repo and
epic-morph. On Nexus the wrapper is
`/fs/cfar-projects/mem_open_vocab/anubhav/sheet_updater/run_sheet.sh` and its log `logs/sheet.log` beside it (the
last 400 lines; one `=== <time>` block per run, ending in `exit=<code>`).
Stop it with:

```bash
crontab -l | grep -v "epic-morph sheet updater" | crontab -
```
