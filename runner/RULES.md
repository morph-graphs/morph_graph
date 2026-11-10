# Runner standing rules

Carried over from MemStream's bridge (user, 2026-11-05) and the morph_graph AGENTS.md.

**QUEUE AGE IS PRECIOUS ON ZARATAN.** Cancelling a PENDING job loses its accrued
priority, a big setback. Therefore:

1. **Never scancel + resubmit a PENDING job to change its config.** Change it in place:
   `scontrol update JobId=<id> ...` (gres, mem, time, partition, dependency, nice, array
   throttle), or `scontrol hold` / `scontrol release`.
2. epic_eval jobs run a snapshot of the code and read their questions from the run
   folder, so most retargeting is a relaunch of the same orchestrator command (it
   resumes, skipping answered questions), not a cancel.
3. Any cancellation of a PENDING job needs explicit Nexus approval in a command that
   names the queue-age cost.

**Account and QoS.**

4. Account `abhinav2-prj-cmsc` always. `abhinav2-prj-aac` never, unless a command says so
   with the user's explicit permission.
5. GPU jobs use `--qos=normal`; `scavenger` is rejected for GPU on Zaratan (CPU only).
6. The cmsc SU budget is shared and limited: no oversized, duplicate or idle jobs.
   Report SU-relevant facts (GPU type, hours) in each `done/<id>.meta`.

**Runner node.**

7. The runner node has no GPU and compute nodes have no internet: all GPU work is
   `sbatch`, and anything needing GitHub or HuggingFace happens on a login node.
