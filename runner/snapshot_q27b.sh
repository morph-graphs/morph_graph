#!/bin/bash
# snapshot_q27b.sh — publish a progress snapshot of the full Qwen3.8-27B V3 run (cmd_0003) to the bridge.
# User instruction (2026-11-11, Zaratan session): "publish snapshots to Nexus every 30 mins".
#
#   snapshot_q27b.sh            one snapshot
#   snapshot_q27b.sh --loop S   every S seconds (default 1800) until scores.txt exists or runner/STOP_SNAPSHOTS
#
# Each snapshot is a new id snap_<UTC>, so nexus/pull_results.sh treats it as new:
#   results/snap_<UTC>/bridge_reports/status_q27b/status.txt     (same content as the cmd_0006/0007 status)
#   results/snap_<UTC>/<run>/answers|errors/*.jsonl               only files changed since the previous snapshot
#   results/snap_<UTC>/<run>/slurm/launches.jsonl, scores.txt    (scores when the run is complete)
#   done/snap_<UTC>.meta                                          counts + task states
# Read-only on the run folder. Local commits only; the sync daemon pushes. Refreshes runner/HEARTBEAT
# when it is over an hour old.
set -u
B="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W=/scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graphs/epic_eval_runs
RUN=27c6fddf_qwen38-27b-thinking-v3-batch10-fps1-f768_cfg8b2c1f83
J=24004150
GITID=(-c user.name="Vatsal Agarwal" -c user.email=vatsalag99@gmail.com)

snapshot() {
    cd "$B" || return 1
    local id ts prev_meta ref n=0
    ts=$(date -u +%Y%m%dT%H%MZ); id="snap_$ts"
    prev_meta=$(ls done/snap_*.meta 2>/dev/null | sort | tail -1)
    ref=${prev_meta:-done/cmd_0003.meta}   # copy answer/error files newer than the previous snapshot
    local out="results/$id"; mkdir -p "$out/bridge_reports/status_q27b" "$out/$RUN/slurm"
    { date -u; echo "run $RUN eval job $J"
      squeue -h -j $J -o '%i %T %M %R' | sort -V
      echo "answers: $(cat $W/$RUN/answers/*.jsonl 2>/dev/null | wc -l) rows in $(ls $W/$RUN/answers 2>/dev/null | wc -l) files"
      echo "errors:  $(cat $W/$RUN/errors/*.jsonl 2>/dev/null | wc -l) rows"
      sacct -j $J -n -P -o JobID,State,Elapsed,NodeList 2>/dev/null | grep -vE '\.(batch|extern)' | sort -V | tail -40
    } > "$out/bridge_reports/status_q27b/status.txt"
    for f in "$W/$RUN"/answers/*.jsonl "$W/$RUN"/errors/*.jsonl; do
        [ -f "$f" ] || continue
        [ "$f" -nt "$ref" ] || continue
        [ "$(stat -c %s "$f")" -gt 10485760 ] && { echo "$f" >> "done/$id.bigfiles"; continue; }
        local rel=${f#$W/}; mkdir -p "$out/$(dirname "$rel")"; cp -p "$f" "$out/$rel"; n=$((n+1))
    done
    cp -p "$W/$RUN/slurm/launches.jsonl" "$out/$RUN/slurm/"
    local done_run=0
    [ -f "$W/$RUN/scores.txt" ] && { cp -p "$W/$RUN/scores.txt" "$out/$RUN/"; done_run=1; }
    local na ne nr nc nh
    na=$(cat $W/$RUN/answers/*.jsonl 2>/dev/null | wc -l); ne=$(cat $W/$RUN/errors/*.jsonl 2>/dev/null | wc -l)
    nr=$(squeue -h -j $J -r -t R 2>/dev/null | wc -l); nh=$(squeue -h -j $J -r -o %r 2>/dev/null | grep -c JobHeldUser)
    nc=$(sacct -j $J -X -n -P -o State 2>/dev/null | grep -c COMPLETED)
    printf 'snapshot=%s\nkind=progress snapshot of cmd_0003 run (user: every 30 min)\nanswers=%s\nerrors=%s\ntasks=%s running, %s completed, %s held\nfiles=%s changed answer/error files since previous snapshot, plus status.txt and launches.jsonl%s\n' \
        "$ts" "$na" "$ne" "$nr" "$nc" "$nh" "$n" "$([ $done_run = 1 ] && echo ', scores.txt (run complete)')" > "done/$id.meta"
    if [ $(( $(date -u +%s) - $(date -u -d "$(cat runner/HEARTBEAT)" +%s) )) -ge 3600 ]; then
        date -u +%Y-%m-%dT%H:%M:%SZ > runner/HEARTBEAT; git add runner/HEARTBEAT
    fi
    git add "results/$id" "done/$id.meta" $( [ -f "done/$id.bigfiles" ] && echo "done/$id.bigfiles" )
    git "${GITID[@]}" commit -q -m "snapshot $id: $na answers, $ne errors, $nr running / $nc completed / $nh held$([ $done_run = 1 ] && echo ', scores.txt')

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EzdPoKuZTVrgztfZPcKmoa"
    echo "$(date -u +%FT%TZ) $id: $na answers, $ne errors, $nr running, $nc completed, $n files" | tee -a runner/snapshot.log
    return $done_run
}

if [ "${1:-}" = "--loop" ]; then
    INTERVAL="${2:-1800}"
    while true; do
        [ -f "$B/runner/STOP_SNAPSHOTS" ] && { echo "STOP_SNAPSHOTS present; exiting" | tee -a "$B/runner/snapshot.log"; break; }
        snapshot && { sleep "$INTERVAL"; continue; }
        echo "run complete (scores.txt published); snapshot loop exiting" | tee -a "$B/runner/snapshot.log"; break
    done
else
    snapshot
fi
