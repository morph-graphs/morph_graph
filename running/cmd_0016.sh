#!/bin/bash
# id: cmd_0016
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: bridge_reports/cancel_q27b/*
# wait: none
# retry: 0
#
# Explicit user approval (RULES.md #3) in the Nexus session 2026-11-12 ~10:30 EST: "Cancel the leftover jobs".
# The full Qwen3.8-27B V3 run is closed (cmd_0015, option c: Nexus scores the merged answers). Cancel the 14 held eval
# tasks (1-10, 40-43; answered on Nexus) and the score job that waits on them. Touch nothing else. No spend.
set -eu
OUT=bridge_reports/cancel_q27b; mkdir -p $OUT
R=$(ls -d 27c6fddf_qwen38-27b-thinking-v3-batch10*/ | head -1); R=${R%/}
read J S < <(python3 -c "
import json
rows=[json.loads(l) for l in open('$R/slurm/launches.jsonl')]
print([r for r in rows if r.get('step')=='eval'][-1]['job'], [r for r in rows if r.get('step')=='score'][-1]['job'])")
echo "eval $J score $S" | tee $OUT/cancel.txt
squeue -h -r -j $J,$S -o '%i %T %r' | sort -V | tee $OUT/before.txt
for t in $(seq 1 10) $(seq 40 43); do st=$(squeue -h -r -j "${J}_$t" -o '%T' 2>/dev/null || true)
  if [ "$st" = "PENDING" ]; then scancel "${J}_$t" && echo "cancelled ${J}_$t" >> $OUT/cancel.txt; else echo "left ${J}_$t state=${st:-gone}" >> $OUT/cancel.txt; fi; done
st=$(squeue -h -j "$S" -o '%T' 2>/dev/null || true); [ "$st" = "PENDING" ] && scancel "$S" && echo "cancelled score $S" >> $OUT/cancel.txt || echo "left score $S state=${st:-gone}" >> $OUT/cancel.txt
cat $OUT/cancel.txt; sleep 2; squeue -h -r -j $J,$S -o '%i %T %r' 2>/dev/null | tee $OUT/after.txt || true
