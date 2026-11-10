#!/bin/bash
# Usage: check_bridge.sh
# Fetch the bridge branch and print a compact one-screen status: pending
# commands, claimed commands (with meta), finished ids with exit codes,
# heartbeat age, and any .bigfiles listings.
set -euo pipefail
BRIDGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BRIDGE"

if git fetch origin vatsal/zaratan-bridge 2>/dev/null; then
    git merge --ff-only origin/vatsal/zaratan-bridge >/dev/null 2>&1 \
        || echo "NOTE: local branch not ff-mergeable with remote (unpushed local commits?)"
else
    echo "NOTE: fetch failed — remote branch missing or GitHub unreachable; showing LOCAL state"
fi

shopt -s nullglob

echo "== pending (commands/) =="
pending=(commands/cmd_*.sh)
if [ "${#pending[@]}" -eq 0 ]; then echo "  (none)"; else
    for f in "${pending[@]}"; do echo "  $(basename "$f")"; done
fi

echo "== claimed (running/) =="
claimed=(running/cmd_*.sh)
if [ "${#claimed[@]}" -eq 0 ]; then echo "  (none)"; else
    for f in "${claimed[@]}"; do
        id="$(basename "$f" .sh)"
        meta="running/$id.meta"
        if [ -f "$meta" ]; then
            echo "  $id  $(tr '\n' ' ' < "$meta")"
        else
            echo "  $id  (no .meta!)"
        fi
    done
fi

echo "== finished (done/) =="
metas=(done/cmd_*.meta)
if [ "${#metas[@]}" -eq 0 ]; then echo "  (none)"; else
    for m in "${metas[@]}"; do
        id="$(basename "$m" .meta)"
        exit_line="$(grep -m1 '^exit=' "$m" || echo 'exit=?')"
        fin_line="$(grep -m1 '^finished=' "$m" || true)"
        echo "  $id  $exit_line  $fin_line"
    done
fi

echo "== heartbeat =="
if [ -f runner/HEARTBEAT ]; then
    hb_raw="$(head -1 runner/HEARTBEAT)"
    if hb_epoch=$(date -u -d "$hb_raw" +%s 2>/dev/null); then
        age_min=$(( ($(date -u +%s) - hb_epoch) / 60 ))
        echo "  $hb_raw  (${age_min} min ago)"
    else
        echo "  unparseable timestamp: $hb_raw"
    fi
else
    echo "  (no runner/HEARTBEAT — runner never started)"
fi

bigs=(done/cmd_*.bigfiles)
if [ "${#bigs[@]}" -gt 0 ]; then
    echo "== bigfiles (on Zaratan, not committed) =="
    for b in "${bigs[@]}"; do
        echo "  $(basename "$b" .bigfiles):"
        sed 's/^/    /' "$b"
    done
fi
