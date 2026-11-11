# EPIC-Morph V3 evals: state (updated 2026-11-10 17:30 ET by the eval session)

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
| MERIT · Qwen3.8-27B | Anubhav | running, 517/598, 250 correct (16:27); its 30 s captions are complete for all 10 videos (unblocks MAGIC 27B) | `anubhav/baselines/MERIT/output/v3/*/m27` (read-only) |
| **Gemini 3.1 Flash Lite** | Vatsal | **done**, 364/598 = 60.9% (scores.txt) | run dir `vatsal/epic_eval_runs_v3/27c6fddf_gemini31flashlite-high-v3-batch10_cfg6c81c204` |
| **Qwen3-VL-8B plain** | Vatsal | **full run launched** 15:51: eval 8345340 (78 tasks, 1× A6000, vulcan-scavenger), score 8345341; waiting for GPUs | run dir `vatsal/epic_eval_runs_v3/27c6fddf_qwen3vl-8b-instruct-v3-batch10-fps1-f768_cfg6d508f62` |
| **Qwen3.8-27B plain (Nexus, on vulcan-arm)** | Vatsal | **full run on arm** (user go-ahead 21:15, 768 frames, FA2 in qwen38_vl_arm58): 78 tasks queued 21:18, NB the background scoring step died with the 06:50 session restart: re-arm scoring (q.py submit ... epic_eval.score) when all 78 tasks are answered. 00:27: 2/78 tasks done, 21/598 answered (8 correct); 1 question truncated at max_new_tokens 16384 (parse/MAX_TOKENS error row, protocol outcome, not retried). Old SLURM smoke #3 (8343807/8) still queued (cancel needs the user's OK). NB the same run is also queued on Zaratan (cmd_0003): duplicate, ask user whether to hold it ; **H200 (user asked 01:15)**: same eval script, tasks 40-78 as arrays 8350957 (40-45 vulcan-default-h200), 8350958 (46-51 vulcan-medium-h200), 8350959 (52-57 vulcan-high-h200), 8350960 (58-78 scavenger), gpu:h200-sxm:1, nexrun tag `q38-27b-h200`; quoted starts Nov 12-16. Arm works from task 1 up: when H200 tasks start, drop those task numbers from arm's pending (run_eval skips answered qids only at task start, so the same task must not run on both at once) ; **06:53 arm side held at the user's request**: 67 pending jobs moved to `arm_queue/held/` (resume: `mv held/*.json pending/`); current task finishes. H200 arrays for tasks 40-78 stay queued, so tasks ~10-39 have no runner until resumed ; 07:14 the running arm task (11) cancelled at the user's request for MAGIC m27 (its answers kept); arm finished tasks 1-10 | run dir `vatsal/epic_eval_runs_v3/27c6fddf_qwen38-27b-thinking-v3-batch10-fps1-f768_cfg8811a86e`, scoreboard `q38_27b` |
| **Qwen3.8-27B plain (Zaratan)** | Vatsal | **full run approved + queued** 20:20 as cmd_0003 (78 tasks × 2× A100, AAC, wait: none). cmd_0002 smoke eval 24001393 still pending on gpu-a100 (since 21:43Z); if it fails, send a hold for cmd_0003's eval. Results come back via nexus/pull_results.sh -> add the run dir to the scoreboard (`q38_27b` is `planned`) ; runner retargeted cmd_0002/cmd_0003 evals to gpu-h100 (1× H100) in place at the user's request (99688ae) ; **dedupe (user 07:15)**: cmd_0004 failed before holding (my bug: launches.jsonl key is 'job', not 'job_id'); runner confirmed the split matches (eval job 24004150; tasks 1-39 on gpu-h100, 40-78 on gpu-a100). Re-sent as cmd_0005 to hold 1-10 + 40-78; Zaratan keeps 11-39 | Zaratan config `epic_eval/configs/qwen3.8-27b-thinking-v3-batch10-zaratan.json` |
| MAGIC-Video · m8 (MERIT captions, local Qwen3-VL-8B everywhere) | Vatsal | **done** 00:20: 598/598, 196 correct (32.8%); controlled vs MERIT, not the published setup | `/fs/cfar-projects/mem_open_vocab/morph_graphs/baselines/MAGIC-video`, outputs `output/v3/nexus/magic_m8/` |
| **MAGIC-Video · mo (as published)** | Vatsal | **done** 22:10: 598/598, 0 failed answers (67 'Unable to generate answer' from the OpenRouter credit outage re-answered after the top-up, 49d02fa); OpenRouter $5.87 total | worktree `baselines/MAGIC-video/.claude/worktrees/mo` (branch `vatsal/mo-faithful`, local); models in `output/v3/nexus/magic_mo/run_config.json`; usage `output/v3/usage/mo_*.jsonl` |
| **MAGIC-Video · m27** | Vatsal | on `qwen38_vl_cu126` + FA2 (2b6891a). First 3 videos (P01-123350, P01-142301, P02) **failed in the graph stage: CUDA OOM** on 2× A6000 (27B server ~29 GB on each card + embeddings at batch 64) after ~45 min each; stages up to graph saved. Fixed 1c61fc1 (MAGIC_EMB_BATCH=16 for m27); all 10 videos resubmitted 22:40 (30 copies, 2× A6000, nexrun `magic-m27-v3`) ; 02:10 the 10 vulcan-scavenger copies moved in place to vulcan-ampere (P01×2, P02, P03, P05 vulcan-high 96G; P06–P09 vulcan-medium 64G, tight vs the 58 GB load peak) ; **on vulcan-arm** (user 06:50): client env `magic_arm` built by the arm agent pinned to the x86 client versions (openai 2.43.0, httpx 0.28.1, sklearn 1.6.1, scipy 1.15.2; pip check clean), `epic/arm_m27.sh` (3699178) runs one video per arm job; P01-123350 queued first, its SLURM copies held (scontrol hold) ; **07:40 shared arm + SLURM (user)**: `epic/m27_coord.py` (18063e6) runs in the eval session's background every 2 min: keeps one m27 video on arm, holds that video's SLURM copies, parks an arm job if SLURM starts the video first, releases a video's SLURM copies if arm fails it. Never cancels. Log: the eval job's tmp/m27_coord.log | same worktree |
| **MAGIC-Video · m27 on vulcan-arm** | Vatsal | user 06:50: arm's remaining Qwen3.8-27B plain jobs parked in `arm_queue/held/` (67; resume with `mv held/*.json pending/`); arm then runs env-magic-arm (client env: clone of qwen38_vl_arm + transformers 4.57.6, igraph, faiss ...) and m27 P01-20240203-123350 via `epic/arm_m27.sh` (3699178; server in qwen38_vl_arm58 + FA2). P01's 3 SLURM copies held (scontrol hold). Next: if P01 passes, queue more videos on arm and hold their SLURM copies | worktree `baselines/MAGIC-video/.claude/worktrees/mo` |
| **Qwen3-VL-8B plain** | Vatsal | **done** 21:05 on vulcan-arm: 598/598, **26.4% = chance** (scores.txt); SLURM 8345340/8345341 cancelled. Not an arm artifact (identical to x86 on overlap) nor a letter bug (same code gives Gemini 60.9/90.6); predictions carry no signal (B 49%). Likely cause: ~13k-token video budget -> ~30 tokens/frame at 400+ frames. Possible diagnostic: --max_frames 64/128 (new run folder, needs OK) | run dir `vatsal/epic_eval_runs_v3/27c6fddf_qwen3vl-8b-instruct-v3-batch10-fps1-f768_cfg6d508f62` |
| **Qwen3.8-27B on arm (env)** | Vatsal | `qwen38_vl_arm58` built (torch 2.7.1+cu128, transformers 5.8.0, flash_attn 2.8.3, fla 0.5.2, triton 3.3.1, pip check clean); arm_env_check.py with **FA2 passed** on the 768-frame P06 clip (13,517 tokens, STOP, coherent, 55 GB peak). arm is idle now | `vatsal/arm_queue/` |
| WorldMM · 8B (w8, all Qwen) | Vatsal | smoke 4/10; superseded by wg8 (user wants WorldMM's original GPT pipeline) | `/fs/cfar-projects/mem_open_vocab/morph_graphs/baselines/WorldMM` |
| **WorldMM · published** | Vatsal | **all 10 builds done** 00:25 (no 402s in the reruns; the 277 in the logs are the outage run 8348127). 9 QA jobs pending on A6000 (tron medium / cml-scavenger), P01-142301 QA done. QA is local Qwen (no API cost). OpenRouter $11.15 left; build spend ~$61 incl. the outage ; 3× A4000 QA test (01:00): server split OK, but the client card (Qwen3-Embedding + VLM2Vec) OOMs at 15.3 GB -> needs a 24 GB client card (e.g. 3× A5000); A6000 QA jobs stay | worktree `baselines/WorldMM/.claude/worktrees/published`, nexrun tag `worldmm-published-v3` |
| Qwen3-VL-8B plain (arm) note | | arm == x86 checked: the 2 P06 smoke questions give identical pred/frames/tokens on arm and x86; the ~27% accuracy is real (whole video packed into ~13k tokens: ~30 tokens per frame at 436 frames) | |

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
- Rules: never cancel a pending Zaratan job (queue age); account **`abhinav2-prj-aac`** from 2026-11-10 (user: CMSC credits cut, 4.9 kSU left; the 27B config now says aac); `--qos=normal` for GPU.

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

- GPT only through OpenRouter (`OPENROUTER_API_KEY`, `openai/<model>`); the OpenAI account has no credits.

- Dry run + explicit go-ahead before any full run; smoke tests are fine without asking.
- Record every sbatch via nexrun; artifacts (answers/, scores.txt) are the truth, not SLURM states.
- Report status as: running / waiting for GPU / waiting for approval. Keep reports short and organized.
- Never push to main; never force-push; ask before pushing. Never edit Anubhav's files, jobs or artifacts.
