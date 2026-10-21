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
| runs root (run folders, `media/`, `_mirrors/`) | `$S/morph_graphs/epic_eval_runs` | read; **write needed to launch** (see prerequisite) |
| Apptainer image | `$S/morph_graphs/morph_graph/morph_graphs.sif` | read + run |
| extra pure-Python packages | `$S/pyextra_morph_graphs` | read |
| model weights | `$S/model_hub/` (Qwen3.8-27B, Qwen3.5-4B/9B, …) | read |
| HD-EPIC videos | `$S/datasets/HD-EPIC/videos/P0x/` | read |
| epic-morph mirror (question banks) | `$S/morph_graphs/epic_eval_runs/_mirrors/epic-morph.git` | read |

These paths are written into the Qwen configs; don't edit them to point elsewhere
unless you mean to (`run.runs_root`, `container.sif` and `model.path` are part of the run
hash, so a changed value means different run folders).

**Prerequisite (owner action, once):** the runs root is group read-only. Before your
first launch, check `test -w $S/morph_graphs/epic_eval_runs && echo writable`. If it is
not, ask Vatsal to run `chmod -R g+w $S/morph_graphs/epic_eval_runs`. Do not try to
change permissions on someone else's directories yourself.

## Setup (once)

1. **Your own clone of morph_graph, in your own scratch** (not in Vatsal's tree). The
   orchestrator snapshots the checkout it is run from, so jobs run your committed code:
   ```bash
   cd /scratch/zt1/project/abhinav2-prj/user/$USER
   git clone git@github.com:vatsalag99/morph_graph.git && cd morph_graph
   git checkout epic-eval          # or main, once epic-eval is merged
   ```
   Clone, pull and push on a **login node**; compute nodes have no route to GitHub.
2. **A Python ≥ 3.10 for the orchestrator** (it only needs the standard library; the jobs
   themselves run inside the image). The system `python3` is 3.6, too old. Use a module,
   e.g. `module load python/3.10.10`, or any interpreter you already have. Confirm with
   `python3 -c "import epic_eval.orchestrator"` from the repo root.
3. **`umask 002`** in every shell that launches (put it in `~/.bashrc`), so the files your
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
  `python3 -m epic_eval.bank v2-150155-82 --repo $S/morph_graphs/epic_eval_runs/_mirrors/epic-morph.git --csv q.csv`.
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

## Monitoring

As in [../AGENTS.md](../AGENTS.md#monitoring-jobs): notify on start and on real failures;
a task that FAILED only because a question hit the output cap (`finish_reason:
MAX_TOKENS` in `errors/*.jsonl`) is not a crash. Read the full `.err`/`.out` logs in
`<run>/slurm/logs/` before concluding anything. Use `squeue -u $USER`, not `-u vatsalag`.
