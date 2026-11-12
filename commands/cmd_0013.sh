#!/bin/bash
# id: cmd_0013
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: bridge_reports/hold_q27b/*
# wait: none
# retry: 0
#
# Split the Qwen3.8-27B tail (user go-ahead in the Nexus session 2026-11-11 22:3x: "release the nexus copies for 65-78").
# Nexus now runs eval tasks 65-78 on its own H200 copies, so HOLD (not cancel) 65-78 here to avoid duplicates.
# Only hold tasks that are still PENDING; leave anything already RUNNING alone and report it. 51-64 stay queued here.
set -eu
OUT=bridge_reports/hold_q27b; mkdir -p $OUT
R=$(ls -d 27c6fddf_qwen38-27b-thinking-v3-batch10*/ | head -1); R=${R%/}
J=$(python3 -c "
import json
rows=[json.loads(l) for l in open('$R/slurm/launches.jsonl')]
print([r for r in rows if r.get('step')=='eval'][-1]['job'])")
: > $OUT/hold.txt
for t in $(seq 65 78); do
  st=$(squeue -h -r -j "${J}_$t" -o '%T' 2>/dev/null || true)
  if [ "$st" = "PENDING" ]; then scontrol hold "${J}_$t" && echo "held ${J}_$t" >> $OUT/hold.txt
  else echo "left ${J}_$t state=${st:-gone}" >> $OUT/hold.txt; fi
done
cat $OUT/hold.txt
squeue -h -r -j $J -o '%i %T %r' | sort -V | tee $OUT/eval_state.txt
