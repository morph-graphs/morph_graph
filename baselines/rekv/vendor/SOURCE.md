# vendor/MemStream: where these files come from

A **subset**, copied on 2026-11-13, of Vatsal Agarwal's ReKV fork (MemStream):

- working tree `/fs/cfar-projects/mem_open_vocab/ReKV_qwen3_wt` on Nexus; repository `vatsalag99/MemStream` (the branch
  is in his private mirror), branch `dev/qwen3-vl-integration`, commit `0efaa6e1aea6ce37c9c227ce12563a680d237b93`
  (2026-11-04) **plus uncommitted changes** in that tree (14 tracked files modified, +1,779 / -107 lines, among them
  `model/attention/kv_cache_manager.py` and `model/qwen3_vl_rekv.py`). The files here are the working tree's.
- MemStream is a fork of ReKV (github.com/Becomebright/ReKV; Di et al., ICLR 2025). Neither repository has a
  LICENSE file; both `pyproject.toml` carry the classifier "Apache Software License". Authorship: the ReKV
  authors and Vatsal Agarwal; unpublished research code, committed here with his agreement.
- What is kept: 15 files (9,664 lines). First the import closure of a real run of
  `../epic/run_memstream.py` was measured (the vendored entries of `sys.modules` after encoding one video and
  answering 8 questions: 21 files). Six of those were loaded only through module-level imports of code for other
  models and research features (`patch_qwen2_vl.py`, `patch_internvl3.py`, `attention/rope.py`,
  `attention/rekv_attention.py`, `attention/qwenvl/rope.py`, `attention/utils.py`) and are dropped.
  `compression_methods.py` and `diagnostics.py` stay because the plain path calls into them
  (`no_compression`, `RecencyProfile`, `apply_env_overrides`).
- **4 files differ from the source, by `MemStream.patch` only** (a unified diff from his files to ours, 22 changed
  lines: the imports of the dropped files are removed, and each branch that would have used them raises
  `RuntimeError("not vendored: ...")`, so a configuration other than plain ReKV on Qwen3-VL fails loudly). All
  other files are byte-identical. `patch -R -p2 < MemStream.patch` in this folder gives his text back.
- `SOURCE.sha256`: the hashes of the kept files as they are in the source tree (to check a later re-sync), then
  the hashes of the 4 edited files as vendored. Answers record a hash of that file.

| file | lines | state |
|---|---|---|
| `model/abstract_rekv.py` | 147 | byte-identical; used by the Qwen3-VL path |
| `model/attention/__init__.py` | 6 | edited (imports only): imports of `rope.py`, `rekv_attention.py`, `qwenvl/rope.py` removed |
| `model/attention/compression_methods.py` | 1342 | edited (imports only): import of `utils.get_bipartite_soft_matching` removed; the two ToMe call sites raise "not vendored" |
| `model/attention/diagnostics.py` | 915 | byte-identical; kept because the plain path calls into it |
| `model/attention/dot_production_attention/__init__.py` | 28 | byte-identical; used by the Qwen3-VL path |
| `model/attention/dot_production_attention/base.py` | 30 | byte-identical; used by the Qwen3-VL path |
| `model/attention/dot_production_attention/triton_impl.py` | 665 | byte-identical; used by the Qwen3-VL path |
| `model/attention/kv_cache_manager.py` | 3522 | edited (imports only): import of `utils.select_topk_watershed` removed; the `topk_watershed_rekv` branch raises "not vendored" |
| `model/attention/memory_units.py` | 284 | byte-identical; used by the Qwen3-VL path |
| `model/attention/qwenvl/rekv_attention.py` | 222 | byte-identical; used by the Qwen3-VL path |
| `model/attention/qwenvl/rope_qwen3.py` | 926 | byte-identical; used by the Qwen3-VL path |
| `model/patch.py` | 471 | edited (imports only): imports of the Qwen2-VL, InternVL and 1-D RoPE code removed; those three branches raise "not vendored" |
| `model/patch_qwen3_vl.py` | 263 | byte-identical; used by the Qwen3-VL path |
| `model/qwen3_utils.py` | 141 | byte-identical; used by the Qwen3-VL path |
| `model/qwen3_vl_rekv.py` | 702 | byte-identical; used by the Qwen3-VL path |
