# Zaratan runner protocol

You are a Claude Code session on Zaratan acting as the job runner for this bridge.
Your job: poll this branch for new command files, execute them, publish results.
You never author or modify experiment commands: the Nexus side writes them and the
human approved them before push. Execute as-is.

## Setup (once)

1. Work in a checkout of `git@github.com:morph-graphs/morph_graph.git` on branch
   `vatsal/zaratan-bridge`, separate from the research checkout:
   ```
   S=/scratch/zt1/project/abhinav2-prj/user/vatsalag
   git clone -b vatsal/zaratan-bridge git@github.com:morph-graphs/morph_graph.git $S/morph_graphs/zaratan_bridge
   ```
   (clone on a login node; compute nodes have no internet).
2. On a login node, start the sync daemon in tmux; it pushes your local commits and
   pulls new commands, so you never run git push yourself:
   `tmux new -d -s bridge "$S/morph_graphs/zaratan_bridge/runner/bridge_sync.sh --loop 120"`
3. Read `README.md` and `RULES.md`. Record in `runner/RUNNER_NOTES.md` (create it):
   the research checkout (`$S/morph_graphs/morph_graph`), its branch and HEAD, the
   python that runs the orchestrator (`module load python/3.10.10`), the container
   image, runs root, model and video paths, SLURM facts you discover. Commit it.
4. Touch `runner/HEARTBEAT` and commit.

## Poll loop

Every ~5 minutes (a background `until`-loop, or between tasks):

```
git fetch origin vatsal/zaratan-bridge && git merge --ff-only origin/vatsal/zaratan-bridge
ls commands/*.sh 2>/dev/null
```

(If the sync daemon is running, a plain `git status`/`ls` sees its merges; fetching
yourself is harmless.) For each new `commands/cmd_NNNN.sh`, lowest id first:

1. **Claim**: `git mv commands/cmd_NNNN.sh running/`; create `running/cmd_NNNN.meta`
   with `claimed=<UTC>`, `host=$(hostname)`. Commit ("claim cmd_NNNN").
2. **Execute**: `bash` the script from its `# workdir:` header directory. If it
   submits SLURM jobs, record the job ids in the `.meta`. Babysit to completion
   (squeue / sacct, and the expected outputs from the `# results:` globs). A
   command may say `# wait: none`, meaning: run it, record job ids, and finish
   without waiting; later commands then report on those jobs.
3. **Publish**: copy every file matching the `# results:` globs (relative to the
   workdir) into `results/cmd_NNNN/`, PRESERVING relative paths. Files >10 MB: do not
   commit; list absolute Zaratan paths in `done/cmd_NNNN.bigfiles`. Then
   `git mv running/cmd_NNNN.sh done/`; finalize `done/cmd_NNNN.meta` with
   `exit=<code>`, `finished=<UTC>`, per-jobid states. Commit
   ("done cmd_NNNN: exit=N, X result files").
4. **Failures are results too**: publish whatever partial results match, set the
   real exit code, and add `done/cmd_NNNN.log` with the relevant stderr (<200 lines;
   for epic_eval, the error rows in `<run>/errors/*.jsonl` and the `.err` logs in
   `<run>/slurm/logs/`). Do not retry unless the header says `# retry: N`.

## Heartbeat

Touch `runner/HEARTBEAT` (UTC timestamp inside) at start and at most once an hour,
folded into your next commit. Silence over 2 h with commands pending tells Nexus
you are gone.

## Command file format (you consume, never produce)

```bash
#!/bin/bash
# id: cmd_0001
# workdir: /scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
# results: <run dir>/answers/*.jsonl <run dir>/scores.txt
# wait: jobs | none
# retry: 0
<body>
```

## Boundaries

- Never modify research-checkout code as part of running a command (fetching and
  checking out the commit a command names is fine; editing files is not).
- Never delete or hand-edit run folders, `media/` or `_mirrors/`.
- Commit messages on this branch are the audit log; keep them accurate.
