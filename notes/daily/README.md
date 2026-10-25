# Daily logs

One file per person per day: `YYYY-MM-DD_<person>.md` (`2026-10-25_anubhav.md`).
Agents append to it at the end of each session; read the other person's latest
file before starting. Commit logs on your own branch with the rest of your work.

```markdown
## <time> · <branch> @ <short sha>
Changed: what was done, in a line or two.
Evidence: the one-sample check (shape, loss, count) or "none yet".
Next: what this branch does next.
Heads-up: anything the other person needs to know (a shared file you are
about to edit, an interface change, a run occupying GPUs). Omit if none.
```

`scripts/sync_report.sh > notes/daily/sync_YYYY-MM-DD.md` writes the branch
report (ahead/behind, clean merge or not, overlapping files, open PRs).
