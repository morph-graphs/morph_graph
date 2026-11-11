#!/bin/bash
# id: cmd_0012
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: bridge_reports/status_q27b/* 27c6fddf_qwen38-27b-thinking-v3-batch10*/answers/*.jsonl 27c6fddf_qwen38-27b-thinking-v3-batch10*/errors/*.jsonl 27c6fddf_qwen38-27b-thinking-v3-batch10*/slurm/launches.jsonl
# wait: none
# retry: 0
#
# NO SPEND, read-only. Status of the full Qwen3.8-27B run (cmd_0003; tasks 11-39 active, 1-10 and 40-78 held
# by cmd_0005) and a copy of its answers/errors so Nexus can show them. Changes nothing.
set -eu
OUT=bridge_reports/status_q27b; mkdir -p $OUT
R=$(ls -d 27c6fddf_qwen38-27b-thinking-v3-batch10*/ | head -1); R=${R%/}
J=$(python3 -c "
import json
rows=[json.loads(l) for l in open('$R/slurm/launches.jsonl')]
print([r for r in rows if r.get('step')=='eval'][-1]['job'])")
{ date -u; echo "run $R eval job $J"
  squeue -h -j $J -o '%i %T %M %R' | sort -V
  echo "answers: $(cat $R/answers/*.jsonl 2>/dev/null | wc -l) rows in $(ls $R/answers 2>/dev/null | wc -l) files"
  echo "errors:  $(cat $R/errors/*.jsonl 2>/dev/null | wc -l) rows"
  sacct -j $J -n -P -o JobID,State,Elapsed,NodeList 2>/dev/null | grep -vE '\.(batch|extern)' | sort -V | tail -40
} | tee $OUT/status.txt
