# zaratan-bridge (morph_graph / EPIC-Morph)

Command/result mailbox between the UMIACS side (Nexus) and Zaratan, carried as the
orphan branch `vatsal/zaratan-bridge` of `git@github.com:morph-graphs/morph_graph.git`.
Neither cluster can ssh to the other non-interactively; both reach GitHub. Nexus
pushes command files; a Claude Code session on Zaratan polls, executes and pushes
small results back. Bulk artifacts stay on Zaratan and are fetched by interactive
scp only when needed. Same design as MemStream's bridge (MemStream_private,
branch `zaratan-bridge`).

Layout:

- `commands/` new job requests from Nexus (`cmd_NNNN.sh`, self-describing header)
- `running/`  commands the runner has claimed (moved here with `.meta` stamped)
- `done/`     finished commands: the script, `.meta` with exit code, `.bigfiles` list
- `results/`  small result files, one subdir per command id
- `runner/`   the Zaratan-side protocol (PROTOCOL.md, RULES.md) and helpers
- `nexus/`    Nexus-side helpers (push a command, check status, pull results)

Rules that keep this sane:

- One writer per directory: Nexus writes only `commands/` (and `nexus/`); the
  Zaratan runner writes `running/`, `done/`, `results/`, `runner/`.
- Nothing over 10 MB is ever committed; big artifacts are listed in
  `done/<id>.bigfiles` with their Zaratan paths instead.
- Command files are executed exactly as written: all human approval happens on the
  Nexus side before push.
