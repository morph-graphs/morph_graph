# EPIC-Morph MCQ evaluation — session handoff (2026-09-22 → 09-23)

Everything a fresh session needs to pick this up. Written at the end of a long
Claude Code session; numbers are from files on disk, not memory.

## 1. What exists

| thing | where | state |
|---|---|---|
| Eval pipeline (sandboxed, SLURM) | `morph_graph` repo, `epic_eval/`, branch **`epic-eval`**, latest **`a504811`** (pushed to `vatsalag99/morph_graph`) | working, tested (11 tests: `module load ffmpeg/7.1 && python -m epic_eval.test_epic_eval`) |
| Question bank fixes | `epic-morph` repo, branch **`mcq-option-cues`**, commit **`a240b88`** (pushed to `vatsalag99/epic-morph`) | Anubhav to open a PR for Vatsal |
| Run outputs + shared encoded clips | `/fs/cfar-projects/mem_open_vocab/morph_graphs/anubhav/epic_eval_runs/` | see §3 |
| Meeting report (HTML, results on `04fb5d5` + analysis) | https://claude.ai/artifact/CAswKcU15D35md5Ajm17bw ; file `proj_context/benchmark_notes/epic_morph_gemini_report_2026-09-23.html` | private; share from the page's Share menu |
| This handoff | `proj_context/proj_notes/epic_eval_handoff_2026-09-23.md` | uncommitted |
| Run explorer (`python -m epic_eval.explorer --runs ...`, commit `fd34426`) | built: `epic_eval_runs/_explorer/gemini38flash_04fb5d5.html`; shared-verdict version: https://claude.ai/artifact/8Yb3A1UbXzepReGWg1cSgJ (db collection `verdicts`, one doc per qid: verdict, note, updated) | read verdicts back with ArtifactData `list` on `verdicts` |
| Design notes | `epic_eval/README.md` (pipeline); `epic-morph/design/07_option_cues.md` (bank fixes) | |

Pipeline in one paragraph: `python -m epic_eval.orchestrator --config epic_eval/configs/<cfg>.json --commit <sha> [--dryrun]`
mirrors both repos, checks out morph_graph at `--commit` and epic-morph at the config's
`questions.commit` into `<runs_root>/<sha8>_<tag>_cfg<hash>/`, then submits via `nexrun launch`:
**prep** (ffmpeg: full video + one `[0, t_ask]` clip per streaming cutoff, cut from the source with
input-side `-t`, pixel-tested for no post-`t_ask` content) → **eval** (one task per video; questions
grouped by cutoff; one Gemini upload + explicit cache per group; 8 contexts / 32 requests in flight)
→ **score** (`scores.json`/`scores.txt`). Rerunning the same command resumes. The config hash excludes
`slurm`, `runtime` and `paths.media_root`.

Adding a model (Qwen etc.): one file `epic_eval/backends/<name>.py` with `make_client`,
`open_context`, `ask`, `close_context`; register in `backends/__init__.py`; config with GPU `slurm.eval`
and `runtime` concurrency 1; `paths.media_root` → the shared media dir to reuse clips. README has a
Qwen3-VL walkthrough. Vatsal was given commit `a504811` for this.

## 2. Decisions made (by Anubhav)

- Gemini **Developer API** with key file `~/.config/epic_eval/gemini_api_key` (600). Model
  `gemini-3.8-flash`, `thinking_level=HIGH`, **`media_resolution=HIGH`**, 1 fps, temperature = model
  default, seed 0. Audio stripped. Video re-encoded 1024 px, 2 fps.
- Eval code in `morph_graph`, questions from `epic-morph` (pinned commit).
- CPU jobs on `vulcan-cpu` (see gotchas: 48 CPUs/node max, 4 jobs/user).
- Replies truncated at the output cap (`MAX_TOKENS`) are **final**, not retried; rerun later at a lower
  thinking level with `runtime.retry_truncated: true`. (None occurred in the full run.)
- Binary questions in the fixed bank: plain `yes`/`no` (the 2+2 layout was rejected, see §6).

## 3. Runs (all on questions `04fb5d5` unless noted)

| run dir (under `epic_eval_runs/`) | what | jobs |
|---|---|---|
| `a5048112_gemini38flash-high_cfgce0ebb2a` | **full video run**, 449/449 | prep 7625431, eval 7625435, score 7625436 |
| `a5048112_gemini38flash-high-textonly_cfg03234b40` | **text-only control** (question + options only), 449/449 | eval 7625437, score 7625438 |
| `media/c189f41a/` | encoded clips for all 12 videos (group `openvocab` read-only) | |
| `_verify_option_cues/` | text-only checks of the fixed bank (§6) | srun |
| older `bc2b4ca0_*`, `a818cfb3_*`, `6421c4af_*` | smoke tests (superseded) | |

Both full runs: 0 errors, 0 truncations, all `finish=STOP`, max thinking 62,915 of 65,536.
Cost ≈ $52 (video) + $7 (text-only) + ~$4 smoke + ~$4 bank-fix checks. Wall time 47 min.
Published price (gemini-3.8-flash, to 2026-12-31): input $0.75/M, output incl. thinking $3.75/M,
cached $0.075/M, storage $0.50/M/h; doubles 2027-01-01.

## 4. Results (video run vs text-only; single run each)

| split | n | video | text-only |
|---|---|---|---|
| all | 449 | 77.1% [73.0, 80.7] | 46.8% |
| non-yes/no (headline) | 320 | **75.0%** [70.0, 79.4] | 41.6% |
| non-yes/no streaming | 128 | 73.4% | 43.8% |
| non-yes/no full video | 192 | 76.0% | 40.1% |
| yes/no | 129 | 82.2% | 59.7% |

Video > text-only in 12/12 videos (sign test p≈0.0005) and 16/18 non-binary forms (2 ties).
Gemini `verify_claim` rated the headline "partially supported": CIs ignore clustering (12 videos,
22 templated forms, one participant P09), text-only prompt says "video unavailable" (not a pure
ablation), options written by the same model family, single run.

Per-form accuracy (video): count 50.0 (n=38), next_weighed 50 (2), streaming_last_added 65.5 (29),
first_use_after_retrieval 66.7 (12), washed_after_step 66.7 (6), checklist 67.9 (28), hob_duration 69.2
(26), what_else_in 70.0 (10), concurrent_activity 75 (4), composite 76.7 (43), streaming_placement 78.6
(14), activity_duration 80 (10), activity_order 80 (25), ingredient_flow_after 80 (5), have_i_added 82.1
(84), put_away_at_end 87.7 (65), washed_before 90 (10), streaming_flow 92.3 (13), contents_when_stored
100 (1), event_order 100 (11; text-only also 100), made_item_count 100 (2), same_unit 100 (11).

## 5. Analysis findings

- **Counting is the one clear weakness** (50%, CI 35–65): undercounts 15 vs over 4; 2→11/17, 3→5/10,
  4→2/5, 5→1/6; bowls 8 of 19 misses; ~29k thinking tokens/question.
- **Container state over time** (checklist, last-added, first-use, what-else: 66–70%): checklist errors
  are 9/9 supersets (claims a not-yet-added item — uses the whole recipe, not the cutoff); last-added /
  what-else confuse similar containers; first-use reasons from a container's purpose. **But at least 2 of 9
  checklist "failures" look like bank label errors** (frames at 13:10–13:13 in P09-20240624-160737 show
  salt added; 02:16–02:19 in P09-20240623-120359 show something red spooned in) — narration misses small
  additions. Suggested: audit all 26 container-state failures by frames (CPU only).
- **Hob duration** (69%): 7/8 errors overestimate, by up to 37 min (likely measuring from an earlier stint).
- Hops and video length do not predict difficulty here (1/2/3-hop 74/76/79%; ≥40-min videos best).
  `guessable=False` 70% vs `True` 83%.
- `composite` by `payoff_form`: put_away_at_end 86.7 (30), washed_after_step 60 (5), streaming_placement
  57.1 (7), contents_when_stored 0 (1). The indirect reference costs nothing on put_away.
- "Container state" is NOT a bank field; it groups 4 `form`s from 3 generators. `instances` (vessel ids)
  identifies container questions; `have_i_added` tests the same skill as yes/no.

## 6. Bank issues found and fixed (epic-morph `a240b88`)

Measured by `question_audit.py --bank` check D12 (blind rules per form) and by text-only re-runs.

| issue at 04fb5d5 | fix | text-only before → after |
|---|---|---|
| yes/no: true verdict was the only one of its kind (129/129) | options `yes`/`no` | 60% (chance 25) → 58% (chance 50) |
| event_order: only one named action offered (11/11) | options = the two named actions | 100% → 27% (chance 50) |
| checklist: middle option (23/28), chronological list prefix (28/28), recipe-consistent state | seeded list order + `universe`; distractors = other intake states | 57% → 46% (chance 25) |
| put_away_at_end: "left out" 76% base rate (also only "no", shortest) | **not changed — decision needed** | 72% blind |

Rejected along the way: 2+2 binary layout (same verdict + wrong reason: reasons not reliably wrong;
48% text-only vs 25% chance); uniform random checklist subsets (still 57% via recipe order); forbidding
the true option from being longest/shortest (reverse cue ~48%).

## 7. Open items

1. Anubhav: open PR `mcq-option-cues` → epic-morph main for Vatsal.
2. put_away_at_end base rate: more "put away" questions / cap left-out at 50% / report vs majority.
3. checklist end states: generator only asks between additions, so none/all never true (text-only 46%,
   ~40% expected from ruling out ends). Fix in `gen_cook_probes.gen_whats_left`.
4. Small forms flagged by D12: streaming_placement (majority 57%), what_else_in (longest 62%),
   washed_after_step (majority 50%).
5. Re-run Gemini video on the fixed bank once merged (168 changed questions; checklist qids change
   because question text changed) — ~$10–15. Update `questions.commit` in the configs.
6. Optional: narration-only control (transcript instead of video, ~$7); neutral-prompt text-only (~$7);
   repeat runs for variance (~$52 each); frame audit of container-state failures.
7. Qwen backend: Vatsal, from `a504811`.

## 8. Gotchas

- `vulcan-cpu` QoS says 1024 CPUs but the partition is two 48-CPU nodes: `-c64` is rejected; 4 jobs/user
  counts interactive sessions (Anubhav's long-running `bash` job 7607170 takes a slot).
- The scratchpad is `/tmp` on the login node: compute nodes cannot see it. Put run dirs on `/fs/...`.
- ffmpeg: `module load ffmpeg/7.1` works in `bash`, not in the zsh tool shell.
- Gemini MCP `review_diff` times out (504) on large payloads; send focused excerpts.
- HIGH thinking on the 61-min video used 49–63k of the 65,536 output cap; HIGH media on that video is
  950k tokens (fits under 1,048,576 only just).
- `--commit HEAD` in the orchestrator resolves to the remote default branch, not the local one: pass a SHA.
