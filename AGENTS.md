# AGENTS.md — working in morph_graph

Notes for coding agents (and people) working in this repo: what we've settled
on, the cluster facts that keep biting, and how to launch and watch jobs. Project
background is in `proj_context/` (start with `proj_notes/proj_description.md`);
each package has its own README (`epic_eval/`, `event_eval/`).

## Repo map

| path | what it is |
|---|---|
| `graphrag/vl_graphrag/` | event-graph memory: `EventGraph`, `UnifiedVideoGraph`, QAFD retriever |
| `events/` | TSM event segmentation |
| `video_models/` | Qwen3-VL features/QA, SigLIP2 / PE-Core aligned encoders |
| `video_qa/` | end-to-end QA runner over benchmarks (EgoLongQA, VSI-Morph) |
| `event_eval/` | oracle event annotation (Gemini/Claude/GPT/Kimi), TSM comparison, viewer |
| `epic_eval/` | sandboxed EPIC-Morph MCQ eval on SLURM (Gemini API or local Qwen3.8) |
| `proj_context/` | design docs, notes; `setup/` holds the container definition + pinned versions |

The repo root is the package (`morph_graph.*` imports), except `epic_eval`, which
is run from the repo root as `python -m epic_eval.<module>`.

## Git

- Work directly on the branch the user names (currently `epic-eval`); don't create
  side branches or worktrees unless asked.
- Commit when asked, only the files that belong to the change; leave the user's
  other uncommitted work alone and say what's left.
- **Pushing happens from a login node.** Agent sessions usually run on a compute
  node with no route to GitHub (ssh 22, 443 and HTTPS all time out). Commit
  locally, then hand the user the exact command:
  `cd /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graph && git push origin <branch>`.
  The user may review first by pulling this checkout onto their own machine.
- A `git pull` here is a rebase: finish an interrupted one with `git rebase --continue`,
  not a plain `git commit`.

## Coding practices

- Match the surrounding code: comment density, naming, idiom. Small, direct changes.
- **No new tests unless asked.** Verify by running the real thing (a `--dryrun`, a
  real launch, reading the outputs) and report what you ran and what came back.
- Check claims against the code before writing them into docs.
- Docs describe the code, not one person's machine: no personal paths or
  personal tooling in READMEs. Environments are expressed through config
  (`paths.python` or a `container` block), because people differ — one of us
  uses conda, one doesn't, and Zaratan runs everything in a container.
  Cluster-specific setup belongs in `proj_context/proj_docs/{zaratan,nexus}.md`.
- Keep the run hash meaningful: settings that change answers go in hashed config
  sections; resources/concurrency (`slurm`, `runtime`) do not.

## Zaratan (UMD)

- **Network:** only login nodes reach the internet. Compute nodes (including the
  one an agent session usually runs on) cannot fetch/push git or download models.
- **Accounts:** default `--account=abhinav2-prj-cmsc`. `abhinav2-prj-aac` has
  higher priority but a small budget: last resort, only when a needed run is
  queue-blocked. The budget is limited; avoid oversized or duplicate runs.
- **QoS:** GPU jobs must use `--qos=normal`; `scavenger` is rejected for GPU
  (valid only for CPU jobs).
- **Partitions:** `standard` (CPU, default); `gpu-a100` (4× A100 40 GB per node);
  `gpu-h100` (4× H100 80 GB per node); `gpu` spans all GPU nodes and routes by
  `--gres` (`gpu:a100:N`, `gpu:h100:N`). Max time 7 days.
- **Queue waits:** fairshare on cmsc is low, and `gpu-h100` can be days out.
  Before committing to a GPU type, compare estimated starts without submitting:
  `sbatch --test-only -A abhinav2-prj-cmsc -p gpu --qos=normal --gres=gpu:a100:2 -c 8 --mem=64G -t 4:00:00 x.sh`.
- **Containers:** jobs run in Apptainer. `apptainer` is a module, not on `PATH`
  by default: scripts must `module load apptainer` (sbatch copies the submitting
  shell's environment, so a script can work from one shell and fail from another).
  Always `apptainer exec --nv --cleanenv --bind /scratch …`; `--cleanenv` keeps host
  compiler/module variables from breaking Triton, so pass variables with `--env`.
  Image: `/scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graph/morph_graphs.sif`,
  built from `proj_context/setup/morph_graphs.def` (how: `proj_context/setup/container_setup.md`).
  Pure-Python extras missing from the image live in
  `/scratch/zt1/project/abhinav2-prj/user/vatsalag/pyextra_morph_graphs` (on `PYTHONPATH` via `--env`).
- **ffmpeg** is inside the container; on the host it's `module load ffmpeg/6.0/gcc/11.3.0/`.
- **Paths** (`~/scratch` = `/scratch/zt1/project/abhinav2-prj/user/vatsalag`):
  models `~/scratch/model_hub/` (Qwen3.8-27B, Qwen3-VL-8B-Instruct, perception_models);
  HD-EPIC videos `~/scratch/datasets/HD-EPIC/videos/P0x/`;
  epic_eval runs `~/scratch/epic_eval_runs/` (encoded clips shared under `media/`).
- Heavy imports (torch) from scratch can take minutes; run them in the background
  rather than inside a 2-minute tool call.

## Nexus (UMIACS)

Anubhav's setup (the Gemini epic_eval runs): `/fs/...` paths, `vulcan-*`
partitions, `nexrun` for submissions, `module load ffmpeg/7.1`. None of it exists
on Zaratan. Details: `proj_context/proj_docs/nexus.md`.

## Running epic_eval

See `epic_eval/README.md` and `README_local_models.md`. The short version for a
local model on Zaratan, from a clean checkout at the repo root:

```bash
python -m epic_eval.orchestrator --config epic_eval/configs/qwen3.8-27b-thinking.json --fps 1 --max_frames 768
```

- One command submits prep → eval → score with `afterany` dependencies; no dry run
  needed. It refuses uncommitted changes to tracked files (jobs run a snapshot of `HEAD`).
- Rerunning the same command is the resume path. Changing `slurm`/`runtime` keeps
  the run folder; a new commit or other config change starts a new one.
- Lift array throttles on a queued job: `scontrol update JobId=<id> ArrayTaskThrottle=0`
  (check afterwards that `Dependency=` is intact with `scontrol show job`).
- Switching GPU type for a queued run: `scancel` its eval + score jobs, relaunch with
  the other config; encoded media is reused, so prep submits nothing.

## Monitoring jobs

What the user wants: **a push notification when a job starts, and when anything
fails after starting.** Routine progress stays in the session.

- Where things are, per run folder: `slurm/launches.jsonl` (what was submitted,
  job ids, commit), `slurm/logs/<step>_<jobid>_<task>.{out,err}`, `answers/*.jsonl`,
  `errors/*.jsonl` (stage `context` / `ask` / `parse`, with traceback; `parse` +
  `finish_reason: MAX_TOKENS` = truncated), `scores.txt`.
- Quick look: `squeue -u vatsalag -o "%.18i %.40j %.9T %.10M %.12P %R %S"` (`%S` =
  estimated start); outcomes: `sacct -j <ids> -X -n -P -o JobID,State,Elapsed,ExitCode,NodeList`.
- Watching: a poll loop every ~60 s over `sacct` that prints one line per event —
  first task RUNNING, every terminal state (`COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED`),
  answer counts every 25, new error rows — keeping its state in a file so a
  re-armed watcher does not repeat events, and exiting when score finishes.
  Filter for failures as well as successes: silence must mean "still running".
  While a job only sits in the queue, wait with one background
  `until sacct … | grep -v PENDING` loop instead of re-arming a watcher every 30 min.
- On a failure, read the **full** `.err` and `.out` logs and the error rows before
  concluding anything, and quote the actual error. Most of the real logging goes
  to stderr.
- Diagnose and report; don't rerun, cancel or change resources on your own unless
  the user has asked for that (fixing your own launcher bug and relaunching the
  same run the user asked for is fine — say so).
