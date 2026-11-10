#!/bin/bash
# Usage: push_cmd.sh <script.sh>
# Assigns the next cmd id, copies the script into commands/, commits, pushes.
# Run from anywhere; operates on the bridge checkout this file lives in.
set -euo pipefail
BRIDGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?usage: push_cmd.sh <script.sh>}"
cd "$BRIDGE"
git fetch origin vatsal/zaratan-bridge 2>/dev/null && git merge --ff-only origin/vatsal/zaratan-bridge 2>/dev/null || true
last=$(ls commands/ running/ done/ 2>/dev/null | grep -oE 'cmd_[0-9]{4}' | sort -u | tail -1 | grep -oE '[0-9]+' || echo 0)
id=$(printf 'cmd_%04d' $((10#$last + 1)))
sed "s/^# id:.*/# id: $id/" "$SRC" > "commands/$id.sh"
grep -q "^# id: $id" "commands/$id.sh" || sed -i "1a # id: $id" "commands/$id.sh"
git add "commands/$id.sh"
git commit -q -m "queue $id: $(basename "$SRC")"
git push -q origin HEAD:vatsal/zaratan-bridge
echo "$id pushed"
