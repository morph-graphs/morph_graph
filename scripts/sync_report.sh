#!/usr/bin/env bash
# Daily sync report: how far each branch is from main, whether it still merges
# cleanly, which files two branches both touch, and the open PRs.
# Read-only: fetches, never merges or pushes. Writes markdown to stdout.
#   scripts/sync_report.sh > notes/daily/sync_$(date +%F).md

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin --prune

main=origin/main
echo "# Sync report $(date '+%F %H:%M') — main at $(git log -1 --format='%h %s' $main)"
echo
echo "| branch | ahead | behind | merges into main | last commit |"
echo "|---|---|---|---|---|"

touched=$(mktemp); trap 'rm -f "$touched"' EXIT
branches=$(git for-each-ref --format='%(refname:short)' refs/remotes/origin | grep -vE '^origin(/HEAD|/main)?$' || true)
for b in $branches; do
    read -r behind ahead < <(git rev-list --left-right --count "$main...$b")
    if git merge-tree --write-tree "$main" "$b" >/dev/null 2>&1; then
        merge=clean
    else
        conflicted=$(git merge-tree --write-tree --name-only "$main" "$b" 2>/dev/null | sed -n '2,/^$/p' | grep . || true)
        n=$(printf '%s\n' "$conflicted" | grep -c . || true)
        more=""; [ "$n" -gt 5 ] && more="..."
        merge="**conflicts in $n files:** $(printf '%s\n' "$conflicted" | head -5 | tr '\n' ' ')$more"
    fi
    last=$(git log -1 --format='%ad %an: %s' --date=short "$b")
    echo "| \`${b#origin/}\` | $ahead | $behind | $merge | $last |"
    git diff --name-only "$(git merge-base "$main" "$b")" "$b" | sed "s|\$|	${b#origin/}|" >> "$touched"
done

echo
echo "## Files touched by more than one branch"
# One line per group of branches sharing files: the count and the first five.
overlap=$(sort "$touched" | awk -F'\t' '{b[$1]=b[$1] (b[$1] ? " + " : "") $2; n[$1]++}
        END {for (f in b) if (n[f] > 1) print b[f] "\t" f}' | sort \
    | awk -F'\t' '$1 != g {if (g) print "- " g ": " c " files (" s (c > 5 ? ", ..." : "") ")"; g=$1; c=0; s=""}
        {c++; if (c <= 5) s = s (c > 1 ? ", " : "") "`" $2 "`"}
        END {if (g) print "- " g ": " c " files (" s (c > 5 ? ", ..." : "") ")"}')
echo "${overlap:-none}"

echo
echo "## Open PRs"
if command -v gh >/dev/null; then
    prs=$(gh pr list --state open --json number,title,headRefName,author,isDraft,updatedAt \
        --template '{{range .}}- #{{.number}} {{if .isDraft}}(draft) {{end}}`{{.headRefName}}` {{.title}} ({{.author.login}}, updated {{timeago .updatedAt}}){{"\n"}}{{end}}' 2>/dev/null) \
        || prs="(gh could not list PRs for this remote)"
    echo "${prs:-none}"
else
    echo "(gh not installed)"
fi
