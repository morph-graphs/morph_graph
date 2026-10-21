# AGENTS.md — running epic_eval on Zaratan as a collaborator

For coding agents (and people) running EPIC-Morph evals from their own Zaratan
account while the shared data, image and runs live in Vatsal's project scratch. The
repo-wide rules in [../AGENTS.md](../AGENTS.md) still apply (git flow, account policy,
monitoring); this file adds what differs when it isn't your scratch. How the
pipeline works: [README.md](README.md); local models: [README_local_models.md](README_local_models.md).

## Shared resources and your access

You need to be in the `zt-abhinav2-prj` group and on the `abhinav2-prj-cmsc` SLURM
account (check: `id -nG`, `sacctmgr -nP show assoc where user=$USER format=account`).
`S=/scratch/zt1/project/abhinav2-prj/user/vatsalag` below.

| what | path | access |
|---|---|---|
| runs root (run folders, `media/`, `_mirrors/`) | `$S/morph_graphs/epic_eval_runs` | read + write (group) |
| Apptainer image | `$S/morph_graphs/morph_graph/morph_graphs.sif` | read + run |
| extra pure-Python packages | `$S/pyextra_morph_graphs` | read |
| model weights | `$S/model_hub/` (Qwen3.8-27B, Qwen3.5-4B/9B, …) | read |
| HD-EPIC videos | `$S/datasets/HD-EPIC/videos/P0x/` | read |
| epic-morph mirror (question banks) | `$S/morph_graphs/epic_eval_runs/_mirrors/epic-morph.git` | read |

These paths are written into the Qwen configs; don't edit them to point elsewhere
unless you mean to (`run.runs_root`, `container.sif` and `model.path` are part of the run
hash, so a changed value means different run folders).

**Check before your first launch:** `test -w $S/morph_graphs/epic_eval_runs && echo writable`.
The runs root has been made group-writable; if that check fails, ask Vatsal to run
`chmod -R g+w $S/morph_graphs/epic_eval_runs`. Do not change permissions on someone
else's directories yourself.

## Setup (once)

1. **Your own clone of morph_graph, in your own scratch** (not in Vatsal's tree). The
   orchestrator snapshots the checkout it is run from, so jobs run your committed code:
   ```bash
   cd /scratch/zt1/project/abhinav2-prj/user/$USER
   git clone git@github.com:vatsalag99/morph_graph.git && cd morph_graph
   git checkout main
   ```
   Clone, pull and push on a **login node**; compute nodes have no route to GitHub.
2. **A Python ≥ 3.10 for the orchestrator** (it only needs the standard library; the jobs
   themselves run inside the image). The system `python3` is 3.6, too old.
   `module load python/3.10.10` works (no conda needed); confirm with
   `python3 -c "import epic_eval.orchestrator"` from the repo root.
3. **Point the launcher at your own epic-morph clone:** add
   `export EPIC_MORPH_DIR=<your epic-morph clone>` to `~/.bashrc`. Launches then read the
   questions from your clone (any branch checked out; it only has to contain the bank's
   commit, so `git fetch` it on a login node when a new bank is added), and
   `python3 -m epic_eval.bank` uses it by default. Without it, launches read the shared
   copy in the runs root, which is owned by Vatsal and which your git refuses.
   Resuming a run Vatsal created also needs that run's `<run>/code_repo` and
   `<run>/questions_repo` added with `git config --global --add safe.directory <path>`
   (Zaratan's git 2.43 takes exact paths only).
4. **`umask 002`** in every shell that launches (put it in `~/.bashrc`), so the files your
   runs create stay group-writable and both of you can resume each other's runs.

## Launching

From your clone's root, with a clean checkout (tracked files committed):

```bash
umask 002
python3 -m epic_eval.orchestrator --config epic_eval/configs/qwen3.8-27b-thinking-v2.json \
    --fps 1 --max_frames 2048 --note "<who and why>"
```

- **Configs that run on Zaratan:** `qwen3.8-27b-thinking-v2`, `qwen3.8-27b-instruct-v2`
  (2× A100). The `gemini-3.8-flash-high-v2*` configs still carry Nexus paths
  (`/fs/cfar-projects/…`); don't launch them on Zaratan until they are updated.
- **Which questions:** every config names a bank (`"questions": {"bank": "v2-150155-82"}`,
  82 streaming questions on P09-20240622-150155). To see them:
  `python3 -m epic_eval.bank v2-150155-82 --csv q.csv` (reads `$EPIC_MORPH_DIR`).
- **Account:** the configs use `abhinav2-prj-cmsc`. Never switch to `abhinav2-prj-aac`
  without the user's explicit permission. You share cmsc's fairshare and budget with
  Vatsal: check the queue and estimates (`sbatch --test-only`) before big launches, and
  say what a run will cost.
- **Encoded clips are shared:** runs with the same `media` settings reuse
  `$S/morph_graphs/epic_eval_runs/media/`, so prep usually submits nothing.

## Working alongside Vatsal

- **One person per run.** The same config at the same code commit maps to the same run
  folder, and relaunching resumes it. Before launching, check that nobody else has that
  run in flight: `squeue -u vatsalag,$USER -o "%.18i %.40j %.9T"` and the run folder's
  `slurm/launches.jsonl`. Two people resuming one run at once can answer the same
  questions twice.
- **Use your own commits for your experiments.** A run from your commit gets a folder
  named after it (`<sha8>_<tag>_cfg<hash>`), separate from runs made from other commits.
- **Never delete, move or hand-edit run folders, `media/` or `_mirrors/`** you didn't create.
  To redo a run, relaunch (resume) or launch from a new commit; ask before removing anything.
- **Question-bank changes go through epic-morph** (edit `questions_llm/bank.jsonl`,
  re-export, commit, push), then add or bump a named bank in `banks.json`. Don't edit
  the files inside a run folder's `questions_repo/`: that is the run's record.

## Updating the shared dashboard

The V2 scoreboard (https://claude.ai/artifact/MVRTeCNbD5psFx6M2dq7vX) is generated from
the repo: the run list is `epic_eval/dashboards/v2/runs.json`, the page layout is
`epic_eval/dashboard_template.html`, and the scores are recomputed from the run folders.

1. **Add the run** to `epic_eval/dashboards/v2/runs.json` (`id`, `label`, `run_dir`, `note`)
   and commit, so the other person's next rebuild includes it too.
2. **Rebuild** into a folder under your home (claude.ai can only publish from there):
   `python3 -m epic_eval.dashboard --registry epic_eval/dashboards/v2/runs.json --out ~/epic_eval_dashboard_v2`.
   It prints each run's answered count and the page size (limit 16 MB).
3. **Publish to the same URL** with the Artifact tool: `url` = the link above,
   `file_path` = `~/epic_eval_dashboard_v2/index.html` (read the artifact first if this
   session hasn't published it). You need edit access to the artifact; the owner grants it
   from the page's Share menu. If the publish is refused because someone published a
   newer version, rebuild from the latest `runs.json` and publish again; the page is
   fully generated, so nothing is lost.

Only runs on the same bank belong on one page (this one: `v2-150155-82`).

## Monitoring

As in [../AGENTS.md](../AGENTS.md#monitoring-jobs): notify on start and on real failures;
a task that FAILED only because a question hit the output cap (`finish_reason:
MAX_TOKENS` in `errors/*.jsonl`) is not a crash. Read the full `.err`/`.out` logs in
`<run>/slurm/logs/` before concluding anything. Use `squeue -u $USER`, not `-u vatsalag`.
