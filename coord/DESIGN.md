# coord: one job board for every baseline run, on every machine

Status: design, 2026-11-12. Nothing here runs yet.

## The problem

EPIC-Morph evaluation is ~1,850 GPU-hours on 48 GB cards plus ~800-1,800 H100-hours (estimate for ~3,000
questions, epic-morph `design/11`), spread over machines that belong to different people:

| pool | who runs it | GPUs | how a job starts | notes |
|---|---|---|---|---|
| `nexus` | Anubhav / his agent | A6000, L40S, A5000 pairs; H200 rarely | Slurm, fan-out over scavenger / cml-scavenger / vulcan-scavenger / tron high / vulcan-ampere (+ vulcan-exempt, sailon when granted) | preemptible tiers |
| `wriva` | friends, under their own accounts | H100, absolute priority | Slurm on the `wriva` partition | they run our code, not us |
| `arm00` | Anubhav / his agent | 1x GH200 96 GB, aarch64 | direct (no Slurm), one job at a time | shared: start only when idle |
| `zaratan` | Anubhav / his agent | H100, A100 | `zr sbatch`; compute nodes offline | costs SU (AAC needs Anubhav's yes) |
| `max` | Max / his agent | his GPUs | Docker images only | |

Work arrives in tranches (Max is scaling the benchmark), and every experiment must be repeated on each new
tranche. People must not hand-pick jobs, and nobody's GPUs should sit idle while work exists.

## Principles

1. **Desired state, not job lists.** The master never "launches experiment X". It compares what *should*
   exist (experiments x available videos and questions) with what *does* exist (outputs in the Hugging
   Face dataset) and creates tasks for the difference. A new tranche, a new probe set or a new experiment
   is just a larger desired state; the delta tasks appear on their own. Nothing is ever redone that exists.
2. **Workers are plain programs, not LLM agents.** Polling, claiming, launching, uploading and heartbeats are
   deterministic Python (`coord/worker.py --pool <name>`). A person's Claude agent starts the worker, reads
   its log, and is called in only for a failure. Polling with an LLM would burn tokens and behave
   differently on every machine.
3. **A store everyone can already reach, kept small.** The board is its own private dataset,
   `morph-graphs/epic-morph-jobs` (results stay in `epic-morph-results`, so result uploads never collide with
   claims). Participants already have Hugging Face accounts in the org for results; each uses their own token
   (no shared token). Only one writer per pool (the pool's worker) plus the master write there, and each
   batches its claims, lease renewals and done records into one commit per cycle: about 5-8 writers and well
   under 100 commits an hour, far from the Hub's rate limits. Tasks last hours, so minutes of latency are fine.
   If the Hub ever becomes the bottleneck, the same protocol moves to a small hosted Postgres.
4. **Claims are atomic and expire.** A claim is a commit made with `parent_commit` = the head the worker
   read; if anyone committed in between, the commit fails (HTTP 412) and the worker re-reads and retries.
   A claim carries a lease that the worker renews each cycle; a lapsed lease (preempted, machine gone) puts
   the task back. Outputs are idempotent and resumable, so a duplicate run wastes time but never corrupts.
5. **Every baseline has the same entry point** (as MERIT's `run_merit.sh` does today):
   `baselines/<b>/run.sh <stage> <video> <arm> [--questions <file>]`, stages `build`, `qa`, `blind`,
   `upload`, outputs to `<b>/<arm>/<video>/` in the dataset, `runs.jsonl` recording commits and GPU.
6. **Stable question identity.** Answers are keyed by the question's `key` (plus a hash of its text and
   options), never by position in a bank file. That is what lets a tranche add questions to a video and the
   board answer only the new ones. MERIT's `ID` is positional today: switch it to `key` first.
7. **Versioned outputs.** Each baseline adapter has a `VERSION` (bumped by hand when its semantics change,
   not on every edit), and each answer row records the hash of the memory it read. An output from another
   version, or an answer whose memory hash differs from the current memory, counts as not done.
8. **Comparable hardware.** Captions differ across GPU models (measured on MERIT), so each (baseline, arm)
   has a GPU family for its main table (e.g. 8B: 48 GB Ampere/Ada; 27B: Hopper = H100/H200/GH200). Other
   pools take that baseline's ablations, probes and no-context runs. Every output records its GPU.

## Units of work

A **task** is the smallest resumable unit:

    (baseline, arm, stage, video, condition, question set)   e.g.  merit/m27/build/P03-20240218-190556
                                                                   merit/m8/qa/P07-.../frames/{keys...}

- `build` tasks depend only on the video, so **all 156 videos' memories can be built now**, before their
  questions exist. MERIT, WorldMM (causal multiscale) and M3-Agent memorization are question-independent;
  MAGIC-Video's base graph is too, and its per-cutoff narrative chains move into its `qa` stage.
- `qa` / `blind` / probe tasks list the question keys they cover (the delta). A `qa` task depends on the
  video's `build` for the same baseline and arm.
- Each task carries requirements (GPUs, VRAM per GPU, arch, image or env, internet), an estimate in
  GPU-hours (measured rate x video minutes or question count), and a priority.

## Desired state

An **experiment** is one line in `coord/experiments.py` (a Python list, no config framework):

    ("merit", "m8",  "qa",    "frames",   "bank"),     # main result, 8B
    ("merit", "m8",  "qa",    "noframes", "probes"),   # M2 memory-only
    ("magic", "m27", "build", None,       None),
    ...

Inputs come from epic-morph at a pinned commit: the current bank (`export_bank.py`, tranche = bank name
plus commit) and probe sets. Outputs come from listing the dataset. `master.py reconcile` computes:

    desired = experiments x videos with questions (or all videos, for build)
    done    = answers/memories present in the dataset (per question key for qa)
    tasks   = desired - done - claimed

## The master: one priority list, pools pull from it

`coord/master.py` is deterministic Python run every ~20 min; a master lease in the store prevents two
masters. Each cycle it:

1. **Reconciles** desired against done (above) and writes the pending tasks as **one global priority list**:
   (a) work that unblocks the current tranche's main table (builds of videos that have questions, then
   their `qa`); (b) builds of videos Max's Stage A delivers next; (c) M2 probes; (d) ablations; (e) the rest.
   Within a level, longest first (keeps the tail short). Anubhav's pins and pauses from the dashboard apply.
2. **Does not pre-assign.** Each pool's worker, when it actually has a free slot, takes the highest-priority
   task its pool can run (arch, VRAM, GPU family for main-table work, internet, SU budget, the pool's own
   cap). Slurm queue waits are too unpredictable to plan around; pulling only on a real free slot avoids
   stranding tasks in a stalled queue. "Who does what next" on the dashboard is this rule applied to each
   pool: the top tasks it can run.
3. **Measures** from done records, per pool and stage: runtime per unit, queue wait, success rate,
   throughput. These drive ETAs and the per-person view, and refine the estimates, not the scheduling.
4. **Retries** a failed task once elsewhere, then turns it into a decision.

The Claude session (Anubhav's, or Fable's) wraps the master on a loop: runs it, reads failures and logs and
proposes fixes, and publishes the snapshot to the monitoring artifact. It never edits a running job's
settings to make it pass: an environment problem is fixed in the baseline's env files, for everyone.

## Workers

`coord/worker.py --pool <name>` loops every few minutes: heartbeat (free slots, running tasks, versions),
renew leases, claim the next tasks in its queue, launch, watch, and on finish upload outputs and write a
`done` record (runtime, GPU model, commits). Launchers per pool kind:

- `slurm` (nexus, wriva): one sbatch per task from a template; nexus fans out over its queues with the
  cancel-the-other-copies guard; wriva submits to its one partition under the friend's account.
  Jobs are `--requeue` and resume (every stage skips finished work: MERIT captions resume per clip).
- `direct` (arm00): runs the task in the foreground of a tmux session when `nvidia-smi` shows the GPU idle.
- `zaratan`: `zr sbatch` from the login node; models, images and inputs are staged to scratch before the
  job and outputs uploaded after it, from the login node (compute nodes are offline).
- `docker` (max): `docker/run.sh` per task, one task per GPU index.

## Environments

One definition per baseline and arm (`baselines/<b>/env/`), built three ways: the Docker image (Max), an
Apptainer image converted from it (Nexus, wriva, Zaratan: `apptainer build x.sif docker://ghcr.io/...`),
and a native aarch64 virtualenv on arm00 from the same requirement files (the images are x86 only).

## Data and secrets

- Videos: on Nexus (ours and wriva) from the shared `/fs` copy if the friends' accounts can read it, else a
  staged copy; on Zaratan a scratch copy; Max downloads per epic-morph `v3/DATA.md`.
- Tokens: each person's own Hugging Face token (write to the two datasets only); GitHub read for
  morph_graph and epic-morph; container images pulled from ghcr.io. Damage from a bad script is bounded:
  both datasets keep git history, and workers never delete outside their own task's folder.

## Monitoring artifact

A claude.ai page with a shared database that the master writes each cycle: overview numbers, per-person /
per-pool queues (what each agent picks next), pool health and speed, tranche x experiment progress,
estimates vs actuals, and decisions waiting for Anubhav. Anubhav can pin priorities and pause pools from
the page; the master reads those overrides.

## Order of work

1. Stable question keys in MERIT; `run.sh` interface check; the store (`jobs/` in the dataset).
2. `master.py reconcile` + `worker.py` with the `slurm`, `direct`, `zaratan` launchers, on MERIT only;
   dashboard live. Start the question-independent builds (all 156 videos) on arm00, Zaratan and Nexus.
3. Move MAGIC-Video and WorldMM (Vatsal's EPIC ports) and M3-Agent into `baselines/` with the same entry
   point and images; Qwen-MM-Plugins as an API arm.
4. Onboard wriva (Apptainer images, their worker) and Max (docker launcher).

## Answer (qa) tasks, per tranche

Added 2026-11-13 with the tranche-2 bank. `master.py` EXPERIMENTS holds a `qa` line per (baseline, arm, bank)
next to the builds; a tranche is one more pair of lines. The master lists a qa task for each video that has
questions in the bank and no `answers_<bank>.json.gz` at `<baseline>/<arm>/<video>/` in the results dataset;
while the video's build is still on the board the task depends on it (workers take it only once that build is
done), and it goes to the same pools as the arm's builds. `publish --dry-run` prints the counts and commits
nothing.

What the master needs: `EPIC_MORPH_DIR` set to its own epic-morph checkout that has the bank; it regenerates
`outputs/v3/banks/<bank>/` there each cycle. Without it `publish` stops (a tasks.json without the qa tasks
would drop them from the board), so set it in the cron's script before this reaches main.

What a worker needs for qa, beyond what builds need:
- `EPIC_MORPH_DIR`: an epic-morph checkout at a commit whose `v3/export_bank.py` BANKS has the bank, with the
  bank's B2 files (`outputs/v3/b2/<video>/questions_llm/{bank_checked,verify}.jsonl`). For `t2_agree2`: the 20
  videos of `v3/tranche2.txt`. `run_merit.sh qa` generates `outputs/v3/banks/<bank>/` there on first use and
  reuses it after, so after moving the checkout to a newer commit delete that folder to regenerate it.
- The video itself (as for build): qa cuts the clips and frames if this machine has not, and MERIT reads them.
- The memory comes from the results dataset (qa runs build, which fetches it; CPU only, seconds): a token that
  can read it, and HF_HUB_OFFLINE unset.
- No code change on the worker: `worker.py` already runs `run.sh qa <video> <arm> <bank> && run.sh upload ...`.

## Every ready baseline on the board

Added 2026-11-13. `EXPERIMENTS` lists build + qa (`t2_agree2`) for MERIT, MAGIC-Video and WorldMM, both arms each,
in that order (the order is the priority). Pools per (baseline, arm) are in `master.POOLS`: every m8 needs one GPU
of 40 GB or more (the Nexus A6000 pools, wriva, max, zaratan); every m27 needs an 80 GB GPU (wriva, max, zaratan), and only
MERIT's m27 also goes to arm00 (the images are x86; MERIT alone has a native aarch64 environment there). MAGIC's
build depends on MERIT's build of the same video and arm while that is listed, since it reuses MERIT's captions.
A memory counts as built when its folder has `runs.jsonl` and the baseline's last memory file (`MEMORY_DONE`).

A worker takes only the baselines it is started with: `worker.py --baselines merit magic worldmm` (default `merit`).
Name a baseline only once the machine has its images, or every task of it fails there and is held back after two
failures. All baselines' `docker/run.sh` read the worker's settings: `MERIT_GPU` and `MERIT_RUNTIME` (from
worker.py), `MERIT_SIF_DIR` (a folder of `<baseline>_<arm>.sif`), `MERIT_HF_CACHE` (the models) and `MERIT_WORK`
(the cluster's shared work folder; MAGIC and WorldMM work in its `magic/` and `worldmm/` subfolders). Each
baseline's own `MAGIC_*` / `WORLDMM_*` variable wins when set.

## One worker per person, many queues

Added 2026-11-13. A person on a Slurm cluster runs one worker, `worker.py --pool <person> --queues <file>`, not one
per queue. The file lists their queues, each with its sbatch arguments and the arms its GPU fits
(`coord/nexus_queues.json`: vulcan-sailon A6000, vulcan-high-h200 H200, scavenger A6000 and L40S, cml-scavenger A6000,
tron QoS high; `m8` everywhere, `m27` on the H200 only). The pool is listed in `master.POOLS` for both arms, and the
worker claims only the tasks one of its queues fits, so the master needs no per-person GPU knowledge.

Each claimed task is submitted to every queue that fits it at once, all copies under the job name `board-<task id>`.
The job's first line is `coord/slurm_guard.sh`: the first copy to start creates the marker `<task>.ran` in the
worker's state folder (noclobber, so two copies starting in the same second cannot both win) and cancels the
copies still pending; a copy that starts later exits 0 without work, and the winner restarting after a requeue
keeps the task. The winner writes the task's exit code to `<task>.rc`. The worker keeps the copies' ids in
`<task>.jobs` and runs one `squeue` per cycle: it cancels copies left over after a winner exists, records the
queue that ran the task in the claim (`queue`), marks it done or failed from `<task>.rc`, and failed when every copy
left the queue without an exit code (time limit, OOM, preemption without requeue). This is the house rule "submit
to two or three queues, the first to start cancels the rest", done per task by the worker.

`--slots` still caps tasks, not copies: 8 slots over 6 fitting queues can hold 48 pending jobs, well inside
Nexus's 500 per user and partition. A plain `--sbatch` worker (one queue, `sbatch --wait`) works as before.

Starting one on Nexus (`coord/nexus_worker.sbatch` has the details): a morph_graph checkout of main in a worker
folder on `/fs`, an HF token in a mode-600 file, then
`sbatch --output=$W/logs/%x-%j.out --export=ALL,POOL=<person>,W=$W,SLOTS=8 $W/mg/coord/nexus_worker.sbatch`.
