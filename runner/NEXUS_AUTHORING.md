# For the Nexus command author: commands the Zaratan runner can run

The runner executes `commands/cmd_NNNN.sh` exactly as written. Zaratan facts below;
the runner's discoveries go in `runner/RUNNER_NOTES.md`, which overrides this file.

## Paths

```
S=/scratch/zt1/project/abhinav2-prj/user/vatsalag
$S/morph_graphs/morph_graph                       research checkout (epic_eval lives here)
$S/morph_graphs/morph_graph/morph_graphs.sif      Apptainer image (configs' container.sif)
$S/morph_graphs/epic_eval_runs                    runs root (run folders, media/, _mirrors/)
$S/morph_graphs/epic_eval_runs/_mirrors/epic-morph.git   question-bank mirror
$S/model_hub/                                     Qwen3.8-27B, ...
$S/datasets/HD-EPIC/videos/P0x/                   videos
```

## Launching epic_eval

- From the research checkout root, with a clean tree (the orchestrator refuses
  uncommitted changes to tracked files and snapshots the commit it is given):
  ```
  module load python/3.10.10
  umask 002
  python3 -m epic_eval.orchestrator --config epic_eval/configs/<zaratan config>.json \
      --commit <sha> --fps 1 --max_frames 768 --note "<why>"
  ```
- Anything that fetches (`git fetch` of morph_graph or the epic-morph mirror) is a
  login-node step; say so in the command so the runner does it there.
- Jobs run inside the container; configs carry `container.sif`, `binds`, `pythonpath`.
  Account `abhinav2-prj-cmsc`, partition `gpu`, `--qos=normal`, `gpu:a100:2` for 27B.
- Without nexrun the orchestrator uses `sbatch --parsable`; job ids land in
  `<run>/slurm/launches.jsonl`.

## Results

Point `# results:` at the run folder's `answers/*.jsonl`, `errors/*.jsonl`,
`scores.txt` and `slurm/launches.jsonl`, relative to the runs root, so the
published tree is `results/cmd_NNNN/<run folder>/...` and `nexus/pull_results.sh`
recreates the run folder on Nexus for the scoreboard. Keep every file under 10 MB
(per-video answer files with thinking are a few MB).
