#!/bin/bash
# id: (assigned by push_cmd.sh)
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: bridge_reports/dedupe_q27b/*
# wait: none
# retry: 0
#
# NO SPEND, holds only (user 2026-11-11: stop Zaratan duplicating the Nexus Qwen3.8-27B plain run).
# The Nexus run (same code 27c6fddf, same bank, 8 q/task) has eval tasks 1-10 answered and 40-78 queued
# on H200s; 11-39 have no Nexus runner. So this Zaratan run (cmd_0003) should run only tasks 11-39.
# 1. Verify the task split matches Nexus: 78 tasks and the same video + first qid for tasks 1, 11, 40, 78.
#    If anything differs, HOLD NOTHING and just report (Nexus will re-plan).
# 2. If it matches: hold the eval array tasks 1-10 and 40-78 IN PLACE (scontrol hold <job>_[...]).
#    Never scancel (queue age). Tasks 11-39 stay pending. Also hold nothing else (cmd_0002 smoke untouched).
set -eu
OUT=bridge_reports/dedupe_q27b; mkdir -p $OUT
R=$(ls -d 27c6fddf_qwen38-27b-thinking-v3-batch10*/ | head -1); R=${R%/}
E=$(ls -t $R/slurm/eval_*.slurm | head -1)
J=$(python3 -c "
import json
rows=[json.loads(l) for l in open('$R/slurm/launches.jsonl')]
print([r for r in rows if r.get('step')=='eval'][-1]['job_id'])")
declare -A WANT=([1]="P01-20240203-123350 P01-20240203-123350__175aaafcf9" [11]="P01-20240204-142301 P01-20240204-142301__d2f81af87a" \
                 [40]="P06-20240510-090328 P06-20240510-090328__8bf64dcf88" [78]="P09-20240624-160737 P09-20240624-160737__e7e48e4a34")
N=$(grep -cE '^\s+[0-9]+\) ' $E); ok=1
{ echo "run $R"; echo "eval script $E, eval job $J, tasks $N (want 78)"; [ "$N" = 78 ] || ok=0
  for t in 1 11 40 78; do
    l=$(grep -E "^\s+$t\) " $E); v=$(echo "$l" | grep -oE -- '--video [^ ]+' | cut -d' ' -f2)
    q=$(head -1 "$R/$(echo "$l" | grep -oE 'slurm/qids/[^"]+')")
    echo "task $t: $v $q | want ${WANT[$t]}"; [ "$v $q" = "${WANT[$t]}" ] || ok=0
  done; echo "split matches: $ok"; } | tee $OUT/split.txt
if [ $ok = 1 ]; then
  scontrol hold "${J}_[1-10]" && scontrol hold "${J}_[40-78]" && echo "held ${J}_[1-10] and ${J}_[40-78]" | tee -a $OUT/split.txt
fi
squeue -h -j $J -o '%i %T %r' | tee $OUT/eval_state.txt
