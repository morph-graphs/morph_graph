#!/bin/bash
# Usage: pull_results.sh [dest_root]
# Fetch + ff-merge the bridge branch, then for each results/<id>/ dir not yet
# recorded in nexus/.pulled, rsync its contents into dest_root (default:
# /fs/cfar-projects/mem_open_vocab/morph_graphs/vatsal/epic_eval_runs_v3/_zaratan), preserving the relative
# paths the runner published under results/<id>/. A destination file is
# replaced only by a newer copy (rsync --update): epic_eval answer files only
# grow, so a later command's copy is a superset; replaced files are listed. Pulled ids are appended to nexus/.pulled
# (local-only, gitignored). Prints done/<id>.meta for each newly pulled id.
set -euo pipefail
BRIDGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/fs/cfar-projects/mem_open_vocab/morph_graphs/vatsal/epic_eval_runs_v3/_zaratan}"
cd "$BRIDGE"

git fetch origin vatsal/zaratan-bridge
git merge --ff-only origin/vatsal/zaratan-bridge >/dev/null

PULLED="nexus/.pulled"
touch "$PULLED"

new_ids=()
for d in results/*/; do
    [ -d "$d" ] || continue
    id="$(basename "$d")"
    grep -qxF "$id" "$PULLED" || new_ids+=("$id")
done

if [ "${#new_ids[@]}" -eq 0 ]; then
    echo "pull_results: no new result dirs under results/ (dest: $DEST)"
    exit 0
fi

total_copied=0
skipped=()   # entries: "<id>: <relative path>"

for id in "${new_ids[@]}"; do
    echo "=== $id ==="

    # Pre-scan: note destination files that already exist; rsync --update
    # replaces them only with a newer copy.
    while IFS= read -r -d '' f; do
        rel="${f#results/$id/}"
        [ -e "$DEST/$rel" ] && skipped+=("$id: $rel")
    done < <(find "results/$id" -type f ! -name .gitkeep -print0)

    copied=$(rsync -a --update --exclude='.gitkeep' \
                   --itemize-changes "results/$id/" "$DEST/" \
             | grep -c '^>f' || true)
    total_copied=$((total_copied + copied))
    echo "$id" >> "$PULLED"
    echo "copied $copied file(s) into $DEST"

    if [ -f "done/$id.meta" ]; then
        echo "--- done/$id.meta ---"
        sed 's/^/    /' "done/$id.meta"
    else
        echo "WARNING: done/$id.meta not found — results published but command not finalized?"
    fi
    if [ -f "done/$id.bigfiles" ]; then
        echo "--- done/$id.bigfiles (NOT copied; fetch by scp from Zaratan) ---"
        sed 's/^/    /' "done/$id.bigfiles"
    fi
    echo
done

echo "=== summary ==="
echo "pulled ids : ${new_ids[*]}"
echo "files copied: $total_copied (dest root: $DEST)"

if [ "${#skipped[@]}" -gt 0 ]; then
    echo
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! NOTE: ${#skipped[@]} file(s) already existed; replaced if the new copy is newer !!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    printf '    %s\n' "${skipped[@]}"
fi
