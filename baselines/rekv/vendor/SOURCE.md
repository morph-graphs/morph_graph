# vendor/MemStream: where these files come from

A **subset**, copied unmodified on 2026-11-13, of Vatsal Agarwal's ReKV fork (MemStream):

- working tree `/fs/cfar-projects/mem_open_vocab/ReKV_qwen3_wt` on Nexus; repository `vatsalag99/MemStream` (the branch
  is in his private mirror), branch `dev/qwen3-vl-integration`, commit `0efaa6e1aea6ce37c9c227ce12563a680d237b93`
  (2026-11-04) **plus uncommitted changes** in that tree (14 tracked files modified, +1,779 / -107 lines, among them
  `model/attention/kv_cache_manager.py` and `model/qwen3_vl_rekv.py`). The files here are the working tree's.
- MemStream is a fork of ReKV (github.com/Becomebright/ReKV; Di et al., ICLR 2025). Neither repository has a
  LICENSE file; both `pyproject.toml` carry the classifier "Apache Software License". Authorship: the ReKV
  authors and Vatsal Agarwal; unpublished research code, committed here with his agreement.
- What is kept: exactly the 21 files (11,592 lines) that a real run of `../epic/run_memstream.py` imports
  (measured: the vendored entries of `sys.modules` after encoding one video and answering 8 questions, Nexus job
  8399115; the same job then ran from a copy holding only these files and gave the same answers). 4,177 of
  those lines are files that a used file imports at module level although their features are off in plain ReKV;
  they are kept so that no vendored file is edited.
- `SOURCE.sha256`: the hash of every kept file (answers record a hash of it). Nothing here is edited; to update,
  copy the same files again and regenerate it (`cd MemStream && find . -name "*.py" | sort | xargs sha256sum`).

| file | lines | why |
|---|---|---|
| `model/abstract_rekv.py` | 147 | used by the Qwen3-VL path |
| `model/attention/__init__.py` | 11 | used by the Qwen3-VL path |
| `model/attention/compression_methods.py` | 1341 | imported at module level by a used file; its features are off |
| `model/attention/diagnostics.py` | 915 | imported at module level by a used file; its features are off |
| `model/attention/dot_production_attention/__init__.py` | 28 | used by the Qwen3-VL path |
| `model/attention/dot_production_attention/base.py` | 30 | used by the Qwen3-VL path |
| `model/attention/dot_production_attention/triton_impl.py` | 665 | used by the Qwen3-VL path |
| `model/attention/kv_cache_manager.py` | 3524 | used by the Qwen3-VL path |
| `model/attention/memory_units.py` | 284 | used by the Qwen3-VL path |
| `model/attention/qwenvl/rekv_attention.py` | 222 | used by the Qwen3-VL path |
| `model/attention/qwenvl/rope.py` | 626 | imported at module level by a used file; its features are off |
| `model/attention/qwenvl/rope_qwen3.py` | 926 | used by the Qwen3-VL path |
| `model/attention/rekv_attention.py` | 187 | imported at module level by a used file; its features are off |
| `model/attention/rope.py` | 116 | imported at module level by a used file; its features are off |
| `model/attention/utils.py` | 485 | imported at module level by a used file; its features are off |
| `model/patch.py` | 472 | used by the Qwen3-VL path |
| `model/patch_internvl3.py` | 236 | imported at module level by a used file; its features are off |
| `model/patch_qwen2_vl.py` | 271 | imported at module level by a used file; its features are off |
| `model/patch_qwen3_vl.py` | 263 | used by the Qwen3-VL path |
| `model/qwen3_utils.py` | 141 | used by the Qwen3-VL path |
| `model/qwen3_vl_rekv.py` | 702 | used by the Qwen3-VL path |
