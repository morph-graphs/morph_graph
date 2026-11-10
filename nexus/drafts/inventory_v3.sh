#!/bin/bash
# id: (assigned by push_cmd.sh)
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs
# results: bridge_reports/inventory_v3/*
# wait: none
# retry: 0
#
# NO SPEND. Inventory + dry run for the EPIC-Morph v3 evals. Run on a LOGIN node
# (it fetches from GitHub). Takes a few minutes.
#   1. A separate worktree of morph_graph at origin/vatsal/v3-evals (the research
#      checkout itself is not touched).
#   2. The epic-morph mirror fetched so it has the v3 bank commit 0f30904.
#   3. Presence checks: protocol commit 27c6fdd, model, container, the 10 batch10 videos.
#   4. An orchestrator --dryrun of the Qwen3.8-27B v3 config (writes scripts, submits nothing).
set -u
S=/scratch/zt1/project/abhinav2-prj/user/vatsalag
MG=$S/morph_graphs/morph_graph
WT=$S/morph_graphs/morph_graph_v3evals
OUT=$S/morph_graphs/bridge_reports/inventory_v3; mkdir -p $OUT
exec > >(tee $OUT/report.txt) 2>&1
echo "host=$(hostname) $(date -u)"

echo "== 1. worktree at origin/vatsal/v3-evals"
git -C $MG fetch -q origin vatsal/v3-evals anubhav-gemini-v3 && echo fetched
[ -d $WT ] || git -C $MG worktree add -q --detach $WT origin/vatsal/v3-evals
git -C $WT checkout -q --detach origin/vatsal/v3-evals
git -C $WT log --oneline -3
git -C $WT cat-file -e 27c6fddfa7d6518731de9869a99cd99d09d2cbf5 && echo "protocol commit 27c6fdd: present"
ls -l $WT/epic_eval/configs/qwen3.8-27b-thinking-v3-batch10-zaratan.json

echo "== 2. epic-morph mirror"
M=$S/morph_graphs/epic_eval_runs/_mirrors/epic-morph.git
git -C $M fetch -q origin '+refs/heads/*:refs/heads/*' && echo fetched
git -C $M cat-file -e 0f30904df25370991554cf2fdd9c679f09eb05d2 && echo "bank commit 0f30904: present"

echo "== 3. model, container, videos"
ls $S/model_hub/Qwen3.8-27B | head -3; ls $S/model_hub/Qwen3.8-27B/*.safetensors | wc -l
ls -l $MG/morph_graphs.sif
for v in P01-20240203-123350 P01-20240204-142301 P02-20240211-175138 P03-20240217-131219 P05-20240425-201444 \
         P06-20240510-090328 P07-20240531-201703 P08-20240617-130401 P08-20240617-184909 P09-20240624-160737; do
  f=$(ls $S/datasets/HD-EPIC/videos/${v%%-*}/$v.* 2>/dev/null | head -1); echo "$v ${f:-MISSING}"; done

echo "== 4. orchestrator dry run"
module load python/3.10.10
cd $WT && umask 002 && python3 -m epic_eval.orchestrator \
  --config epic_eval/configs/qwen3.8-27b-thinking-v3-batch10-zaratan.json \
  --commit 27c6fddfa7d6518731de9869a99cd99d09d2cbf5 --fps 1 --max_frames 768 \
  --env_name dryrun_qwen38_27b_v3 --dryrun 2>&1 | grep -vE "^#SBATCH|^  [0-9]+\)" | tail -40

echo "== 5. queue"
squeue -u $USER -o "%.12i %.30j %.8T %.10M %R" | head -20
exit 0
