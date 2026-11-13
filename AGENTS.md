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
| `baselines/merit/` | MERIT memory baseline on EPIC-Morph: run script, Docker images, MERIT fork as a submodule |
| `baselines/magic/` | MAGIC-Video memory baseline on EPIC-Morph: run script, Docker images, MAGIC-video fork as a submodule |
| `baselines/worldmm/` | WorldMM memory baseline on EPIC-Morph: run script, environments, WorldMM fork as a submodule |
| `baselines/m3agent/` | M3-Agent memory baseline on EPIC-Morph: run script, Docker image, m3-agent fork as a submodule |
| `baselines/vgent/` | Vgent graph-RAG baseline on EPIC-Morph, as released (Qwen2.5-VL-7B): run script, environment, upstream Vgent as a submodule |
| `baselines/api/` | closed-source API models (Gemini, GPT) on EPIC-Morph through `epic_eval`: run script, Docker image, converter to the baselines' result layout |
| `proj_context/` | design docs, notes; `setup/` holds the container definition + pinned versions |

The repo root is the package (`morph_graph.*` imports), except `epic_eval`, which
is run from the repo root as `python -m epic_eval.<module>`.

## Git

Two people (Anubhav, Vatsal) and their agents work here in parallel. The rules
that keep branches from drifting:

- Work on `<person>/<topic>` branches (`anubhav/...`, `vatsal/...`), on the branch
  the user names when there is one. One agent per branch, and one branch per git
  worktree when several agents run at once.
- **Never commit or push to `main`.** The pre-push hook refuses it, and force
  pushes too; enable it once per clone: `git config core.hooksPath .githooks`.
- **Stay close to main:** at the start of each working day,
  `git fetch origin && git merge origin/main`. Merge, do not rebase, a branch that
  is already pushed. **Never force-push**, `--force-with-lease` included.
- Anubhav squash-merges every PR in a daily batch prepared at 4 am. Open the PR
  with the template filled in, including evidence from one real sample.
- Before editing a file the other person is also changing (see open PRs and
  `scripts/sync_report.sh`), say so in your daily log first. Logs go in
  `notes/daily/YYYY-MM-DD_<person>.md`; read the other person's latest one first.
- Commit when asked, only the files that belong to the change; leave the user's
  other uncommitted work alone and say what's left.
- **Pushing happens from a login node.** Agent sessions usually run on a compute
  node with no route to GitHub (ssh 22, 443 and HTTPS all time out). Commit
  locally, then hand the user the exact command:
  `cd /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/morph_graph && git push origin <branch>`.
  The user may review first by pulling this checkout onto their own machine.
- If a clone has `pull.rebase` set, `git pull` rebases. Use
  `git fetch origin && git merge origin/main` on a pushed branch instead.

## Compute jobs for collaborators off our clusters (Max)

- **Every compute job handed to Max (or anyone running off the UMIACS/Zaratan clusters) runs in a Docker
  image built from this repo, from a commit on `main`.** Never send venv, conda or container-path
  instructions, or commands that name our clusters' paths.
- The images: `baselines/<baseline>/docker/` (Dockerfile, `build.sh`, `run.sh`, `image_tag.sh`; MERIT and
  MAGIC-Video so far), published by `.github/workflows/baseline-images.yml` as
  `ghcr.io/morph-graphs/<baseline>:<arm>-<env hash>`. The image holds
  the environment; the code is the mounted checkout. A new kind of job (another baseline, a new stage) first
  gets its dependencies into an image here, merged to main and built, before any note to Max is written.
- Publishing costs Actions minutes (2,000 a month, shared): run the workflow by hand on `main` after a merge
  that changes a baseline's `env/` or `docker/`; code-only merges keep the same image tag. A new baseline adds
  its (baseline, arm) rows to that workflow's matrix and its paths to its trigger.

## Compute jobs go through the job board (`coord/`)

- Baseline runs are tasks on the job board, not hand-written job lists: `coord/master.py publish` lists what
  should exist minus what the results dataset already holds, in the private dataset
  [morph-graphs/epic-morph-jobs](https://huggingface.co/datasets/morph-graphs/epic-morph-jobs), and each
  machine pool runs `coord/worker.py --pool <name>`, which claims, runs, uploads and reports. Design and
  reasons: `coord/DESIGN.md`.
- Don't run a listed task by hand without the board knowing: another pool may claim it too. New kinds of
  work are new entries in `EXPERIMENTS` in `coord/master.py`; a task that failed twice waits for a fix, then
  `coord/master.py reset --task <id>`.
- Workers are plain programs; an agent starts one, reads its log and handles failures, but never changes a
  task's settings to make it pass: environment fixes go into the baseline's `env/` files for everyone.

## Generated outputs go to the Hugging Face dataset, never into git

- Baseline outputs (memories, captions, answers, run logs) belong in the private dataset
  [morph-graphs/epic-morph-results](https://huggingface.co/datasets/morph-graphs/epic-morph-results), not in
  this repo or epic-morph. MERIT: `baselines/merit/run_merit.sh ... upload` (folder `merit/<arm>/<video>/`); a
  new baseline gets its own top-level folder there and the same kind of upload step.
- Reuse what is there: a memory depends only on the video and the model, so `build` takes it from the
  dataset before building, and every bank, probe set and variant (`MERIT_TAG`, `MERIT_NO_FRAMES`) answers from it.
- Banks are generated in epic-morph (`v3/export_bank.py <name>`), not committed; record the epic-morph commit
  with any result (run_merit.sh does, in `runs.jsonl`).

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

Full reference (hardware, QoS caps, billing, example submissions, refresh
commands): `proj_context/proj_docs/zaratan.md`. What to always follow:

- **Account:** `abhinav2-prj-aac` is approved for EPIC-Morph evaluation runs (Anubhav, 2026-11-12): CMSC
  is nearly used up (3.2 kSU of 100 left on 2026-11-13). `zr` defaults to CMSC; bill AAC with
  `ZR_ALLOW_AAC=1 ZR_ACCOUNT=abhinav2-prj-aac zr sbatch ...`. Other work: ask before choosing an account.
  Avoid oversized, duplicate or idle jobs either way.
- **QoS:** `normal` for every GPU job; `scavenger` is rejected for GPU (CPU only). For CPU jobs someone waits
  on, also `normal` (partition `standard`): `scavenger` queues behind everything; move a pending job with
  `scontrol update jobid=<id> partition=standard qos=normal`.
- **Always set `-t`** (GPU partitions default to 15 min). Debug/interactive jobs ≤ 4 h,
  and exit them when done.
- **Test new environments on a MIG slice first:** `-p gpu-a100_1g.5gb --gres=gpu:a100_1g.5gb:1`
  (7 SU/hr vs 48 for an A100, 144 for an H100).
- **GPU choice:** A100 = 40 GB, H100 = 80 GB. Over 40 GB, prefer **2+ A100s**
  (2× A100 = 96 SU/hr vs 1× H100 = 144). H100s only when A100s are scarce, and say
  that SU usage goes up. Check free GPUs, the pending queue, and estimated starts
  (`sbatch --test-only`) before recommending a type; commands in `zaratan.md`.
- **Billing is MAX_TRES:** stay within the free CPU/memory per GPU (A100: 32 cores,
  128 GB; H100: 24 cores, 128 GB) or those set the rate instead.
- **Per-user caps:** 38 A100s, 16 H100s, 10 MIG slices, 40 GPUs total, however many
  array tasks are queued. `-p gpu` is rerouted to the per-type partition.
- **Network:** only login nodes reach the internet. Compute nodes (including the
  one an agent session usually runs on) cannot fetch/push git, pip install or
  download models.
- **Containers:** `apptainer` is a module (`module load apptainer`; sbatch copies
  the submitting shell's environment, so never rely on it being loaded). Run with
  `apptainer exec --nv --cleanenv --bind /scratch …` and pass variables with `--env`.
  Image, extras and data paths: `zaratan.md` §6.
- Heavy imports (torch) from scratch can take minutes; run them in the background
  rather than inside a 2-minute tool call.

## Nexus (UMIACS)

Anubhav's setup (the Gemini epic_eval runs): `/fs/...` paths, `vulcan-*`
partitions, `nexrun` for submissions, `module load ffmpeg/7.1`. None of it exists
on Zaratan. Details: `proj_context/proj_docs/nexus.md`.

## Running epic_eval

Running from someone else's account on Zaratan (shared scratch, permissions, coordination):
[epic_eval/AGENTS.md](epic_eval/AGENTS.md).

Question banks are named in `epic_eval/banks.json` (`v1-449`, `v2-150155-82`) and configs
refer to them as `"questions": {"bank": "<name>"}`; inspect one with
`python3 -m epic_eval.bank <name> --repo <epic-morph clone or mirror> [--csv f]`.

See `epic_eval/README.md` and `README_local_models.md`. The short version for a
local model on Zaratan, from a clean checkout at the repo root:

```bash
python -m epic_eval.orchestrator --config epic_eval/configs/qwen3.8-27b-thinking-v2.json --fps 1 --max_frames 2048
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
