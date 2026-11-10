#!/bin/bash
# bridge_sync.sh — run this ON A LOGIN NODE (compute nodes have no internet).
# It keeps the shared zaratan-bridge checkout in sync with GitHub so nobody has
# to run git by hand:
#   fetch -> rebase local commits onto remote (disjoint paths, no conflicts) ->
#   push. The compute-node runner only makes LOCAL commits; this pushes them and
#   pulls in new commands/.
#
# Two modes:
#   bridge_sync.sh            # one sync pass (ideal for cron every 1-2 min)
#   bridge_sync.sh --loop [S] # loop forever, S sec between passes (default 120)
#
# Safe to run from cron AND leave a --loop running; an flock lock prevents overlap.
# Stop it by creating runner/STOP_SYNC (or kill the process).

set -uo pipefail
BRIDGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE=origin
BRANCH=vatsal/zaratan-bridge
LOG="$BRIDGE/runner/sync.log"
LOCK="$BRIDGE/.sync.lock"
GIT_ID=(-c user.name=zaratan-runner -c user.email=vatsalag99@users.noreply.github.com)

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

sync_once() {
    cd "$BRIDGE" || { log "FATAL cd $BRIDGE"; return 2; }
    [ -f runner/STOP_SYNC ] && { log "STOP_SYNC present; skipping"; return 9; }

    # single-writer: if another pass holds the lock, bail quietly
    exec 9>"$LOCK"
    flock -n 9 || { log "another sync holds the lock; skip"; return 0; }

    if ! git fetch "$REMOTE" "$BRANCH" -q 2>>"$LOG"; then
        log "fetch failed (network?); will retry next pass"; return 1
    fi
    local local_head remote_head
    local_head=$(git rev-parse HEAD 2>/dev/null)
    remote_head=$(git rev-parse "$REMOTE/$BRANCH" 2>/dev/null)

    # Bring local up to date on top of remote. Autostash guards against a
    # publish that is mid-write; disjoint paths mean this never conflicts.
    if [ "$local_head" != "$remote_head" ]; then
        if ! git "${GIT_ID[@]}" rebase --autostash "$REMOTE/$BRANCH" -q 2>>"$LOG"; then
            log "rebase conflict — aborting, leaving tree clean for the runner"
            git rebase --abort 2>>"$LOG" || true
            return 1
        fi
    fi

    # Push anything local that the remote does not have yet.
    if [ -n "$(git rev-list "$REMOTE/$BRANCH..HEAD" 2>/dev/null)" ]; then
        if git push "$REMOTE" "$BRANCH" -q 2>>"$LOG"; then
            log "pushed $(git rev-list --count "$REMOTE/$BRANCH..HEAD@{1}" 2>/dev/null || echo '?') commit(s)"
        else
            log "push failed; will retry next pass"; return 1
        fi
    fi

    # Report if new commands arrived for the runner.
    local ncmd
    ncmd=$(ls commands/cmd_*.sh 2>/dev/null | wc -l)
    log "ok head=$(git rev-parse --short HEAD) pending_cmds=$ncmd"
    return 0
}

if [ "${1:-}" = "--loop" ]; then
    INTERVAL="${2:-120}"
    log "=== sync loop started (interval ${INTERVAL}s, host $(hostname)) ==="
    while true; do
        [ -f "$BRIDGE/runner/STOP_SYNC" ] && { log "STOP_SYNC present; loop exiting"; break; }
        sync_once
        sleep "$INTERVAL"
    done
else
    sync_once
fi
