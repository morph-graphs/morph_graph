#!/bin/bash
# id: cmd_0014
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: bridge_reports/release_q27b_65/*
# wait: none
# retry: 0
#
# User go-ahead in the Nexus session 2026-11-12 ~01:30 EST ("Yeah release them"). SPEND: 14 tasks x ~0.5 h x 144 SU/h on
# gpu-h100 = ~1-1.5 kSU on abhinav2-prj-aac. Nexus never got an H200 for its copies, and has now HELD 8350960_[65-78],
# so these run only here. Release eval tasks 65-78 of the full Qwen3.8-27B run; keep 1-10 and 40-43 held (done on Nexus).
set -eu
OUT=bridge_reports/release_q27b_65; mkdir -p $OUT
R=$(ls -d 27c6fddf_qwen38-27b-thinking-v3-batch10*/ | head -1); R=${R%/}
J=$(python3 -c "
import json
rows=[json.loads(l) for l in open('$R/slurm/launches.jsonl')]
print([r for r in rows if r.get('step')=='eval'][-1]['job'])")
for t in $(seq 65 78); do scontrol release "${J}_$t"; done; echo "released ${J}_65..78" | tee $OUT/release.txt
squeue -h -r -j $J -o '%i %T %r' | sort -V | tee $OUT/eval_state.txt
