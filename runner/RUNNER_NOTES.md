# Runner notes (Zaratan side)

Facts the runner found on Zaratan. Overrides `NEXUS_AUTHORING.md` where they differ.
Started 2026-11-10T20:35:06Z. `S=/scratch/zt1/project/abhinav2-prj/user/vatsalag`.

## Runner session

- Claude Code on compute node `compute-a5-9` (SLURM job 23770390, `standard` partition,
  `srun --pty ... zsh`, no GPU). No internet, and **no ssh route to the login nodes either**
  (`ssh -o BatchMode=yes login-{1,2,3}.zaratan.umd.edu` times out). A command whose header says
  "run on a LOGIN node" is therefore claimed and published by the runner, but its body is
  handed verbatim to the user to run on a login node; the `.meta` records `executor=`.
- `tmux` is not installed on compute nodes. The sync daemon runs in tmux on `login-1`
  (`runner/sync.log`: loop started 2026-11-10T20:23Z, 120 s).
- Commits on this branch use the checkout's identity (Vatsal Agarwal <vatsalag99@gmail.com>).

## Research checkout

- `$S/morph_graphs/morph_graph`, branch `main` @ `593dcab` (2026-10-26, "Shared working
  setup for two people and their agents (#6)").
- Its `origin` is `git@github.com:vatsalag99/morph_graph.git` (the bridge's is
  `morph-graphs/morph_graph`). As of 2026-11-10T20:35:06Z its remote refs include `anubhav-gemini-v3`,
  `cleanup/untracked-files`, `salvage/event-eval-pilot`, `setup/agent-workflow` but **not**
  `vatsal/v3-evals`; commit `27c6fdd` is nevertheless present in its object store (and in the
  bridge checkout's). Whether `git fetch origin vatsal/v3-evals` works through that URL is what
  cmd_0001 step 1 will show.
- **Uncommitted local edits exist** in the research checkout from a 2026-11-10 session that
  predates the bridge: modified `epic_eval/common.py`, `epic_eval/explorer.py`,
  `epic_eval/score.py`; untracked `epic_eval/exchange.py`, `proj_context/proj_docs/exchange.md`.
  The orchestrator refuses to launch from a checkout with modified tracked files, so a command
  that runs it from `$S/morph_graphs/morph_graph` fails until the user stashes or commits them.
  Commands that use the `morph_graph_v3evals` worktree are unaffected. Not the runner's to touch.
- `git worktree list`: only the main checkout; `$S/morph_graphs/morph_graph_v3evals` does not
  exist yet (cmd_0001 creates it).

## Environment

- Orchestrator Python: `module load python/3.10.10` -> Python 3.10.10 (cvmfs spack view);
  `python3 -c "import epic_eval.orchestrator"` works from the research checkout. A conda
  Python 3.11.4 (`$S/miniconda3/bin`) is first on PATH by default; it also imports epic_eval.
- Container: `$S/morph_graphs/morph_graph/morph_graphs.sif` (7.8 GB, built 2026-09-24).
  `module load apptainer` -> apptainer 1.3.6.
- Host ffmpeg modules are 4.2.2 / 4.3.2 only (no `ffmpeg/7.1`); the Zaratan configs set
  `paths.ffmpeg_module: null` and use the ffmpeg inside the image.
- Runs root: `$S/morph_graphs/epic_eval_runs` (group-writable, setgid). Holds 8 V1/V2 Qwen run
  folders, `media/`, `_mirrors/{epic-morph.git,morph_graph.git}`. The epic-morph mirror does
  **not** have the v3 bank commit `0f30904` (login-node fetch needed; cmd_0001 step 2).
- Banks known to `main`'s `epic_eval/banks.json`: `v1-449`, `v2-150155-82`.
- Models: `$S/model_hub/{Qwen3.8-27B, Qwen3-VL-8B-Instruct, Qwen3.5-4B, Qwen3.5-9B,
  Qwen3.5-9B-Base, Qwen2.5-VL-7B-Instruct, Latent-VC-9B, perception_models}`.
- Videos: `$S/datasets/HD-EPIC/videos/P01..P09/`.

## SLURM facts (2026-11-10T20:35:06Z)

- Slurm 25.11.8. Associations: `abhinav2-prj-cmsc` and `abhinav2-prj-aac`, QoS `normal,scavenger`,
  default `normal`.
- **Account policy changed 2026-11-10: use `abhinav2-prj-aac`** (user's explicit instruction;
  RULES.md rule 4). The cmsc cap went 300 -> 100 kSU on 2026-11-09 10:38 ET
  (`sacctmgr show transactions`); the usage counter resets quarterly (last ~2026-10-01).
- **Budget: cmsc is almost exhausted.** `sbalance`: limit 100.00 kSU (GrpTRESMins
  billing=6,000,000; the docs' 300 kSU is out of date), used 95.09 kSU, **4.91 kSU unused**.
  aac: 548.82 kSU limit, 542.89 kSU unused (off limits without explicit permission).
  The 27B smoke draft (~100 SU) fits; a full batch10 run on cmsc does not. Nexus has to settle
  budget (raise, or an explicit aac permission in a command) before any full run.
- Lab queue on cmsc right now: user `anubhav` has 16 jobs of `gpu:a100:2` and 16 of
  `gpu:h100:1` PENDING, plus one CPU job running. Runner session job 23770390 is the only
  job of user `vatsalag`.
- GPUs free: 0/76 A100, 0/32 H100, 0/28 MIG. Pending per partition: 196 (a100), 104 (h100),
  64 (MIG). Expect long queue waits; queue age is precious (RULES.md).
- **Progress snapshots (user, 2026-11-11): `runner/snapshot_q27b.sh --loop 1800`** publishes `results/snap_<UTC>/` + `done/snap_<UTC>.meta` every 30 min (status, changed answer/error files, launches.jsonl; scores.txt when done), until scores.txt exists or `runner/STOP_SNAPSHOTS`. Runs in the runner session's background; after a session restart it has to be restarted.
