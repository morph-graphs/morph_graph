# vendor/MemStream: where these files come from

Copied unmodified on 2026-11-13 from Vatsal Agarwal's ReKV fork (MemStream), working tree
`/fs/cfar-projects/mem_open_vocab/ReKV_qwen3_wt` on Nexus:

- repository `github.com/vatsalag99/MemStream` (and its private mirror `MemStream_private`), branch
  `dev/qwen3-vl-integration`, commit `0efaa6e1aea6ce37c9c227ce12563a680d237b93` (2026-11-04), **plus uncommitted
  changes** in that working tree (`SOURCE.git_status.txt`, `SOURCE.git_diffstat.txt`; 14 tracked files modified,
  among them `model/attention/kv_cache_manager.py` and `model/qwen3_vl_rekv.py`). The files here are the working
  tree's, so they are not exactly that commit. `SOURCE.sha256` has the hash of every copied file.
- MemStream is a fork of ReKV (github.com/Becomebright/ReKV; Di et al., ICLR 2025). Neither repository has a
  LICENSE file; `pyproject.toml` (kept here) carries the classifier "Apache Software License". Authorship: the
  ReKV authors and Vatsal Agarwal; MemStream is unpublished research code, committed here with his agreement (not for use outside the
  team without asking him).
- Copied: `model/**.py`, `video_qa/**.py`, the Qwen3-VL notes in `docs/`, three of the `slurm/` run scripts,
  `pyproject.toml`, `prepare.sh`, `README.md`, `MEMSTREAM_CODEBASE_OVERVIEW.md`. Not copied: `analysis/`, data,
  results, notebooks.
- Used by `../epic/run_memstream.py`: `model/qwen3_vl_rekv.py` (`load_model`, `Qwen3_VL_ReKV`) and what it
  imports (`model/patch.py`, `model/patch_qwen3_vl.py`, `model/qwen3_utils.py`, `model/abstract_rekv.py`,
  `model/attention/`). `video_qa/` is here for reference only (the prompt format and frame sampling that
  `run_memstream.py` reproduces).

Nothing in this folder is edited. To update, copy again and regenerate `SOURCE.sha256` (answers record a hash
of it, so answers made with other files count as not done).
