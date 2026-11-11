#!/bin/bash
# id: (assigned by push_cmd.sh)
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: 27c6fddf_qwen38-27b-thinking-v3-batch10*/answers/*.jsonl 27c6fddf_qwen38-27b-thinking-v3-batch10*/errors/*.jsonl 27c6fddf_qwen38-27b-thinking-v3-batch10*/scores.txt 27c6fddf_qwen38-27b-thinking-v3-batch10*/slurm/launches.jsonl
# wait: none
# retry: 0
#
# SPEND (user go-ahead 2026-11-10, account AAC): the FULL Qwen3.8-27B thinking plain run on the
# 598-question V3 bank, 1 fps / 768 frames: 78 eval tasks x gpu:a100:2, 8 h cap each (worst case
# ~60 kSU; expect much less), + prep + score. Same orchestrator command as cmd_0002 without
# --videos/--limit/--env_name. Run it from the same worktree cmd_0001 made (no fetch needed).
# The config still names abhinav2-prj-cmsc: move every job it submits to abhinav2-prj-aac in place
# (RULES.md rule 4). wait: none -> record the job ids and finish; later commands report on them.
# If cmd_0002 (the 2-question smoke) fails, Nexus will send a command to hold this run's eval
# job in place (scontrol hold, never cancel).
set -eu
S=/scratch/zt1/project/abhinav2-prj/user/vatsalag
WT=$S/morph_graphs/morph_graph_v3evals
module load python/3.10.10
cd $WT && umask 002 && python3 -m epic_eval.orchestrator \
  --config epic_eval/configs/qwen3.8-27b-thinking-v3-batch10-zaratan.json \
  --commit 27c6fddfa7d6518731de9869a99cd99d09d2cbf5 --fps 1 --max_frames 768 \
  --note "full V3 run: Qwen3.8-27B thinking plain, 598 q (user go-ahead 2026-11-10, AAC)"
