#!/bin/bash
# id: (assigned by push_cmd.sh)
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: smoke_qwen38_27b_v3/answers/*.jsonl smoke_qwen38_27b_v3/errors/*.jsonl smoke_qwen38_27b_v3/scores.txt smoke_qwen38_27b_v3/slurm/launches.jsonl smoke_qwen38_27b_v3/slurm/logs/eval_*
# wait: jobs
# retry: 0
#
# SPEND: ~1 h of 2x A100 at most (~100 SU). Qwen3.8-27B thinking, 1 fps / 768 frames,
# 2 questions on P06-20240510-090328 (the same pair as the Nexus smoke tests).
# Needs the inventory command done (worktree $S/morph_graphs/morph_graph_v3evals).
# Prep of P06's clips runs first if the media isn't there (CPU, standard partition).
set -eu
S=/scratch/zt1/project/abhinav2-prj/user/vatsalag
WT=$S/morph_graphs/morph_graph_v3evals
module load python/3.10.10
cd $WT && umask 002 && python3 -m epic_eval.orchestrator \
  --config epic_eval/configs/qwen3.8-27b-thinking-v3-batch10-zaratan.json \
  --commit 27c6fddfa7d6518731de9869a99cd99d09d2cbf5 --fps 1 --max_frames 768 \
  --videos P06-20240510-090328 --limit 2 --env_name smoke_qwen38_27b_v3 \
  --note "bridge smoke: Qwen3.8-27B thinking v3, 2 q on P06"
