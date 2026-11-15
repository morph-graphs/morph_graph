#!/bin/bash
# id: cmd_0017
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: bridge_reports/board_inventory/*
# wait: none
# retry: 0
#
# READ-ONLY inventory for the morph_graphs job board (coord/worker.py) on Zaratan. Asked by Vatsal 2026-11-15,
# written by the Nexus board-worker session. Lists and counts only: no sbatch, no scancel, no git pull, no
# downloads, no deletes, no scontrol. Needs only /scratch and squeue; a minute on the runner node.
# Questions it answers: whose account runs the board's `zaratan` pool; does that checkout have its submodules
# (VST tasks died on "run: git submodule update --init baselines/vst/VST"); which of the board's SIF images,
# models and HD-EPIC videos already sit in our project scratch, so a `vatsal-zaratan` pool could reuse them.
set -u
OUT=/scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs/bridge_reports/board_inventory
mkdir -p "$OUT"; exec > >(tee "$OUT/inventory.txt") 2>&1
PRJ=/scratch/zt1/project/abhinav2-prj
echo "== who/where"; whoami; hostname; date
echo "== project scratch users (is it shared?)"; ls -la $PRJ/user/ 2>&1 | head -20
echo "== board jobs running under the project (whose account runs the zaratan pool?)"
squeue -A abhinav2-prj-cmsc,abhinav2-prj-aac -h -o "%u %j %T %P %M" 2>/dev/null | grep -i "board-" | awk '{print $1, $3}' | sort | uniq -c
echo "== board worker process / checkout in my folder"; pgrep -af "coord/worker.py" | sed 's/HF_TOKEN=[^ ]*/HF_TOKEN=***/' || echo "none running as me"
echo "== SU balance"; sbalance 2>/dev/null | head -12
for U in vatsalag $(ls $PRJ/user 2>/dev/null | grep -v vatsalag); do
  B=$PRJ/user/$U; [ -r "$B" ] || { echo "== $U: not readable"; continue; }
  echo "================ $U"
  echo "-- morph_graph checkouts"; find $B -maxdepth 4 -type d -name morph_graph 2>/dev/null | while read d; do echo "$d  $(git -C $d rev-parse --short HEAD 2>/dev/null)  uninit-submodules=$(git -C $d submodule status 2>/dev/null | grep -c '^-')"; done
  echo "-- SIF images (board wants <baseline>_<arm>.sif)"; find $B -maxdepth 4 -name "*.sif" 2>/dev/null | xargs -r ls -lh | awk '{print $5, $9}'
  echo "-- HF cache models"; for c in $B/hf_cache $B/model_hub; do [ -d $c ] && { echo "$c:"; ls $c $c/hub 2>/dev/null | grep -iE "models--|Qwen|Molmo|M3|whisper|Embedding" | head -20; }; done
  echo "-- HD-EPIC videos"; V=$B/datasets/HD-EPIC/videos; [ -d $V ] && echo "$(find $V -name '*.mp4' 2>/dev/null | wc -l) mp4 in $V ($(du -sh $V 2>/dev/null | cut -f1))" || echo "none at $V"
  echo "-- epic-morph banks"; find $B -maxdepth 3 -type d -name "epic-morph" 2>/dev/null | head -3
  echo "-- board worker state / outbox"; find $B -maxdepth 4 -type d \( -name "outbox" -o -name "state-*" \) 2>/dev/null | head -5
done
echo "== quota"; df -h $PRJ 2>/dev/null | tail -1
echo "== done: $OUT/inventory.txt"
