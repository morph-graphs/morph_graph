#!/bin/bash
# id: cmd_0011
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: bridge_reports/release_q27b/*
# wait: none
# retry: 0
#
# Answer to req_20261111T2001Z. SPEND: ~35 tasks x ~0.5 h x 144 SU/h on gpu-h100 = ~2.5-3 kSU on abhinav2-prj-aac
# (user go-ahead in the Nexus session 2026-11-11: "yes, run 44-78 on Zaratan").
# Release eval tasks 44-78 of the full Qwen3.8-27B run in place. Keep 40-43 HELD: Nexus answered them on H200.
# Keep 1-10 held. Nexus has held its own H200 copies of 44-78 and vulcan-arm will not take them, so no duplicates.
set -eu
OUT=bridge_reports/release_q27b; mkdir -p $OUT
R=$(ls -d 27c6fddf_qwen38-27b-thinking-v3-batch10*/ | head -1); R=${R%/}
J=$(python3 -c "
import json
rows=[json.loads(l) for l in open('$R/slurm/launches.jsonl')]
print([r for r in rows if r.get('step')=='eval'][-1]['job'])")
scontrol release "${J}_[44-78]" 2>/dev/null && echo "released ${J}_[44-78]" | tee $OUT/release.txt || { for t in $(seq 44 78); do scontrol release "${J}_$t"; done; echo "released ${J}_44..78 one by one" | tee $OUT/release.txt; }
squeue -h -r -j $J -o '%i %T %r' | sort -V | tee $OUT/eval_state.txt
