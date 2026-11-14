# coord/sheet_update.py: the results sheet's `Auto: EPIC-Morph` tab

A plain script (cron, no LLM) that rewrites one tab of the results Google Sheet from the results dataset
(`morph-graphs/epic-morph-results`). It owns that tab and writes nowhere else: hand-maintained tabs refer
to it by formula.

## What the tab holds

Rows 1-9 are a header block: A1 the ownership sentinel, then `status` (`OK`, or `ERROR <time>: ...` when the
last run failed; the table below is then the last good run's), the last run time (UTC and US Eastern), the
results-dataset revision, the epic-morph commit the banks were generated from, the row count, the
conventions, and the taxonomy leaf names. Row 10 holds the column names, and the table starts at row 11:
the rows of each (baseline, arm, bank, tag) that has an answers file, then those of each (baseline, arm, tag) with
bank `all`: the same columns over the questions of every bank (videos counted as `<bank>/<video>`; an unanswered
question is wrong there too, so a method that has answered one bank shows `partial`). The banks are those the job
board answers (`coord/master.py` EXPERIMENTS), so a new bank there needs no change here.

**End and streaming questions are scored apart.** Every question of a bank carries `ask_at` (epic-morph
`v3/export_bank.py`): `end`, asked at the video's end, or `t`, asked mid-video at its `t_ask` (streaming). Each
(baseline, arm, bank, tag) has three rows, in this order, told apart by the column `asked`:

| `asked` | questions | `row_key` |
|---|---|---|
| `end` | those asked at the video's end | `<baseline>/<arm>/<bank>[/<tag>]/end` |
| `streaming` | those asked at `t_ask` | `<baseline>/<arm>/<bank>[/<tag>]/streaming` |
| `both` | all of them: the row the run had before the split, same numbers | `<baseline>/<arm>/<bank>[/<tag>]` (unchanged) |

Each row counts only its own questions: `q_in_bank`, `q_answered`, `n_right`, the accuracies, the interval, the
videos (a video without a question of that kind is not in the row), the blind floor, chance and the leaf columns.
`end` + `streaming` add up to `both`. The answer files are matched against the whole bank and then cut, so
`answers_dropped` is the run's and the same on its three rows. A bank question without a known `ask_at` stops the
run (the `ERROR` line), rather than being put in either row.

**Streaming answers are provisional** (column `provisional`: `TRUE` on every row that holds streaming questions:
`streaming`, and `both` when the bank has any). Until streaming memories exist, every method answers a streaming question from its
full-video memory with retrieval cut at `t_ask`; those answers will be redone and overwrite these. When they are,
empty `PROVISIONAL` in the script.

| columns | meaning |
|---|---|
| `row_key` | `<baseline>/<arm>/<bank>[/<tag>][/end or /streaming]`: the stable key to look a row up by (`XLOOKUP`, `VLOOKUP`) |
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
| `asked`, `provisional` | `end`, `streaming` or `both` (above); `TRUE` when the row holds streaming answers. After `last_updated_utc`, so no column before them moved when they were added |
| `leaf_<code>_pct` | accuracy over the answered questions carrying that taxonomy leaf (epic-morph `v3/taxonomy.json`); a question with two leaves counts in both. Last, so a new leaf never moves the columns before it |

Percentages are plain percent values (36.29 means 36.29%), numbers are written as numbers (`RAW` input: no
text is ever taken for a formula), so the tab needs no number formats.

**Which answers count.** The banks are regenerated from the epic-morph checkout on every run
(`v3/export_bank.py`), and an answer counts only while its question's wording, options and gold letter are
those of the current bank: `prepare_epic.match_answers`, the rule the baselines use to reuse answers (by key;
by content for answer files from before keys, such as the batch10 MERIT runs). Whether an answer is right is
its recorded `evaluate`. The query time is not compared: an answer given before its question was moved to the
video's end (epic-morph #19) counts in the `end` row until the question is answered again at its new time. A blind
file with a tag (`blind_<bank>_<tag>.json`, MERIT answering from a given context) is a row of its own, tag
`blind-<tag>`, not a floor.

## The same scores as JSON (`--json`, for the job board)

`--json <file>` writes the scoring pass of that run to a file (one line, replaced whole; with `--dry-run` too, and
then nothing touches the sheet). The job board's Accuracy tab is drawn from it (`coord/collect_live.py` reads the
file, `coord/board_views.py` `accuracy()` turns it into the datasets `accuracy` and `accuracy_breakdown`), so the
board and the sheet cannot disagree: nothing is scored a second time.

- `meta`: `written_utc`, the results dataset and its revision, the epic-morph commit of the banks, the banks,
  `asked` (`["end", "streaming", "both"]`).
- `runs`: the table's rows under the column names above (an empty cell is `null`), without the leaf columns, plus
  `partial` (`videos_done < videos_in_bank`). The interval is the sheet's (`ci95_*`, `ci_method`). Three per
  scored (baseline, arm, bank, tag), with `asked` and `provisional` (`true` / `false`).
- `breakdown`: per run (`run` = its `row_key`; `asked` as in that row), `dimension` (`leaf`, `video`, `hops`, `form`) and `category`
  (`label`: the leaf's name): `n_answered`, `n_right`, `n_in_bank`. These are the counts the leaf columns are
  computed from. Within `video`, `hops` and `form` the categories of a run add up to its `q_answered` and
  `n_right`; within `leaf` they add up to more, because a question with two leaves counts in both. The `all` rows
  have no `video` breakdown (their videos are the banks' own rows'). No interval here: the reader computes one
  from the counts (the board page: Wilson).

After a failed run the file is left as it was, so its `written_utc` ages; the board marks scores older than 90
minutes as stale.

## Adding a method

Add a line to `METHODS` in `coord/sheet_update.py`: `(baseline, arm): (Method, Memory Model, Reasoning Model,
N. Frames)`, with the names used in the hand tab. A new bank comes from the master: once `coord/master.py` lists qa
for it, its rows and its questions in the `all` rows appear. Nothing else: rows appear when the dataset holds answers.

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
