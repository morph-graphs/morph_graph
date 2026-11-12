#!/bin/bash
# id: cmd_0015
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: bridge_reports/finish_q27b/*
# wait: none
# retry: 0
#
# Answer to req_20261112T1154Z: option (c). Nexus scores the merged answers (it already merges Zaratan's 486 with the
# 106 Nexus answers by qid: 592/598 answered, 6 MAX_TOKENS truncations). Zaratan does nothing more: do NOT cancel,
# release or score. The 14 held tasks (1-10, 40-43) and score job 24004151 stay pending; Vatsal may decide later to
# cancel them. No spend. This command only records the decision and the current queue state.
set -eu
OUT=bridge_reports/finish_q27b; mkdir -p $OUT
echo "decision: option (c), Nexus scores; nothing cancelled or released ($(date -u +%FT%TZ))" | tee $OUT/decision.txt
squeue -h -u $USER -o '%i %T %r' | grep -E '^2400415[01]' | sort -V | tee $OUT/queue_state.txt || true
