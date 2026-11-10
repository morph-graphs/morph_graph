# EPIC-Morph V3 evals: state at handoff (2026-11-10 14:56 ET)

Handed over from the question-generation session. Read this first; it is the single source of truth
for what is running and what is waiting. Update it as things change.

Goal: every baseline on the 598-question bank `v3-batch10-598` (epic-morph 0f30904), Anubhav's
protocol (code commit 27c6fdd, 1 fps, up to 768 frames, streaming at each question's t_ask).
The user said: run the three plain baselines first, then the memory methods.

## 1. Runs

| Run | Owner | Status | Where |
|---|---|---|---|
| Gemini 3.8 Flash | Anubhav | done, 542/598 = 90.6% | `/fs/cfar-projects/mem_open_vocab/morph_graphs/epic_eval_runs_v3/27c6fddf_gemini38flash-high-v3-batch10_cfgb399d802` (read-only) |
| MERIT · Qwen3-VL-8B | Anubhav | done, 217/598 = 36.3% | `anubhav/baselines/MERIT/output/v3/*/m8` (read-only) |
| MERIT · Qwen3.8-27B | Anubhav | running, 347/598 at 49.6% (14:00) | `anubhav/baselines/MERIT/output/v3/*/m27` (read-only) |
| **Gemini 3.1 Flash Lite** | Vatsal | **full run launched**: prep 7/9 videos done, eval + score waiting on prep | run dir `vatsal/epic_eval_runs_v3/27c6fddf_gemini31flashlite-high-v3-batch10_cfg6c81c204`; jobs prep 8342605, eval 8342608 (10 tasks, 2 at a time), score 8342609; CPU-only on cml-default |
| **Qwen3-VL-8B plain** | Vatsal | smoke passed (2/2 answered); **full run waits on user go-ahead** | dry run done: run dir `vatsal/epic_eval_runs_v3/27c6fddf_qwen3vl-8b-instruct-v3-batch10-fps1-f768_cfg6d508f62`, 78 eval tasks, 1× A6000 each, vulcan-scavenger. Launch after Gemini's prep finishes (shared clips). |
| **Qwen3.8-27B plain (Nexus)** | Vatsal | smoke #3 queued: eval 8343807 (2× A6000 scavenger), score 8343808 | run dir `vatsal/epic_eval_runs_v3/smoke_qwen38_27b_v3`. Smokes #1–2 failed (env, see §3). |
| **Qwen3.8-27B plain (Zaratan)** | Vatsal | bridge built, **not pushed** (see §4) | Zaratan config `epic_eval/configs/qwen3.8-27b-thinking-v3-batch10-zaratan.json` |
| MAGIC-Video · 8B | Vatsal | smoke 3/8; **full run waits on user go-ahead** (~60 A5000 GPU-h) | `/fs/cfar-projects/mem_open_vocab/morph_graphs/baselines/MAGIC-video` (adapter in `epic/`, README there). 27B arm needs MERIT's 27B captions (Anubhav's m27, now running). |
| WorldMM · 8B | Vatsal | smoke 4/10; next: check the 55/186 empty entity extractions, then full run (~25 GPU-h), go-ahead needed | `/fs/cfar-projects/mem_open_vocab/morph_graphs/baselines/WorldMM` (adapter in `epic/`, README there) |

Launch commands (from this worktree, with the env python that has google-genai, e.g.
`/fs/cfar-projects/mem_open_vocab/miniconda3/envs/morph_graphs_qwen3_cu126/bin/python`; plain `python3` lacks it
and dies at the Gemini key check):
```
PY=/fs/cfar-projects/mem_open_vocab/miniconda3/envs/morph_graphs_qwen3_cu126/bin/python
$PY -m epic_eval.orchestrator --config epic_eval/configs/<config>.json \
    --commit 27c6fddfa7d6518731de9869a99cd99d09d2cbf5 --fps 1 --max_frames 768 --note "<why>"
```
Add `--dryrun` first; `--videos P06-20240510-090328 --limit 2 --env_name <name>` for a smoke.
Rerunning the same command resumes. The orchestrator submits through `~/.claude/bin/nexrun`.

## 2. Configs (branch `vatsal/v3-evals`, local commits, NOT pushed)

| Config | Notes |
|---|---|
| `gemini-3.1-flash-lite-high-v3-batch10.json` | CPU-only, cml-default (no GPU needed: prep is ffmpeg, eval is API) |
| `qwen3-vl-8b-instruct-v3-batch10.json` | env `qwen38_vl_cu124` via `bin/python_ffmpeg`; 1× A6000 |
| `qwen3.8-27b-thinking-v3-batch10.json` | env `qwen38_vl_cu126` via `bin/python_ffmpeg`; 2× A6000 |
| `qwen3.8-27b-thinking-v3-batch10-zaratan.json` | Zaratan container, 2× A100, account abhinav2-prj-cmsc |

## 3. Environments (`/fs/cfar-projects/mem_open_vocab/miniconda3/envs/`)

- **`qwen38_vl_cu126`**: built to match the Zaratan container pins (`proj_context/setup/pinned.txt`):
  torch 2.8.0+cu126, transformers 5.8.0, flash-attn 2.8.3, fla 0.5.2, causal-conv1d 1.7.0,
  torchcodec 0.7.0 (+ nvidia-npp-cu12), google-genai. Use **`bin/python_ffmpeg`** (puts the env's ffmpeg and
  NPP on LD_LIBRARY_PATH for torchcodec). Imports and a P06 decode verified; 27B forward pass not yet (smoke #3).
- `qwen38_vl_cu124`: clone of videolvc + fixes (flash-attn 2.7.4, `bin/python_ffmpeg` for ffmpeg 7.1 libs).
  Fine for Qwen3-VL-8B. **27B crashes in it** (CUDA illegal memory access in the first forward pass; old fla 0.4.2).
- `morph_graphs_qwen3_cu126`: Gemini runs and the orchestrator itself.
- `module load ffmpeg/7.1` does NOT set LD_LIBRARY_PATH; torchcodec needs the wrappers above.
- Models: `/fs/cfar-projects/mem_open_vocab/model_hub/{Qwen3.8-27B,Qwen3-VL-8B-Instruct,...}`.

## 4. Zaratan bridge (git mailbox, same protocol as MemStream's)

- Branch `vatsal/zaratan-bridge` of `morph-graphs/morph_graph`, checkout `vatsal/zaratan_bridge/`.
  `README.md`, `runner/PROTOCOL.md`, `runner/RULES.md`, `runner/NEXUS_AUTHORING.md` there.
- `commands/cmd_0001.sh` (inventory + dry run, no spend) is queued locally. Draft for the next one:
  `nexus/drafts/smoke_q27b_v3.sh` (2-question 27B smoke, ~100 SU) — queue with `nexus/push_cmd.sh <draft>`.
- **Blocked:** neither branch is pushed (the agent's push was refused by the permission check). The user runs:
  ```
  git -C /fs/cfar-projects/mem_open_vocab/morph_graphs/vatsal/zaratan_bridge push origin vatsal/zaratan-bridge
  git -C /fs/cfar-projects/mem_open_vocab/morph_graphs/vatsal/morph_graph_evals push origin HEAD:refs/heads/vatsal/v3-evals
  ```
  Then the user's Zaratan agent gets the setup message (clone the bridge branch, start
  `runner/bridge_sync.sh --loop 120` in tmux on a login node, follow PROTOCOL.md / RULES.md, run cmd_0001).
- Status: `vatsal/zaratan_bridge/nexus/check_bridge.sh`; results: `nexus/pull_results.sh` (lands in
  `vatsal/epic_eval_runs_v3/_zaratan/<run>`; add that run dir to the scoreboard registry).
- Rules: never cancel a pending Zaratan job (queue age); cmsc account only; `--qos=normal` for GPU.

## 5. Scoreboard

- Page: https://claude.ai/artifact/CgqQ6sk7SGo838FVhvXh1f (private; share with Anubhav from its Share menu).
- Code: `epic_eval/scoreboard.py`, template `epic_eval/scoreboard_template.html`, registry
  `epic_eval/dashboards/v3/runs.json` (add or retarget runs here; `kind`: epic_eval / slices / planned).
- Hourly rebuild: `bash _scoreboard_v3/rebuild.sh`; on CHANGED, publish `_scoreboard_v3/index.html` to the URL
  above (from a new session: read the artifact first, then publish with `url`), then
  `cp _scoreboard_v3/data.json _scoreboard_v3/published.json`. Set up an hourly cron for it in the new session.
- The `q38_27b` entry is `kind: planned`; point it at the real run dir once a full 27B run exists.

## 6. Other channels

- vulcan-arm00 (GH200, not in SLURM): file queue at `vatsal/arm_queue/` (README, `q.py`, `worker.py`).
  The arm agent built env `/scratch0/vatsalag/miniconda3/envs/qwen38_vl_arm` but its worker never started
  (GPU was busy). Its last note: `arm_queue/msgs/to_nexus/20261110-035550-env-interim.md`.
- Old helper agents (WorldMM, MAGIC-Video) are finished; no jobs of theirs are queued.

## 7. Waiting on the user

1. Push the two branches (§4), then start the Zaratan runner.
2. Go-ahead for the Qwen3-VL-8B full run (dry run in §1).
3. MAGIC-Video 8B full run; WorldMM empty-extraction check then 8B full run.
4. MAGIC 27B captions: wait for Anubhav's m27 (recommended) or caption with 27B itself.

## 8. Working rules (from the user, this project)

- Dry run + explicit go-ahead before any full run; smoke tests are fine without asking.
- Record every sbatch via nexrun; artifacts (answers/, scores.txt) are the truth, not SLURM states.
- Report status as: running / waiting for GPU / waiting for approval. Keep reports short and organized.
- Never push to main; never force-push; ask before pushing. Never edit Anubhav's files, jobs or artifacts.
