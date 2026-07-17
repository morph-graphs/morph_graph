# Known bugs and fixes

## `illegal hardware instruction` on eval import

**Error:** `illegal hardware instruction (core dumped)` when importing/running
GraphRAG-based QA — often before any eval logs appear.

**Cause:** Importing `morph_graph.graphrag.*` ran `graphrag/__init__.py`, which eagerly loaded `graphrag.py` → `_storage` → **`hnswlib`**. The native `hnswlib` wheel can SIGILL on some cluster CPUs. Unrelated to `--key_source video`, torchcodec, or Qwen3-VL.

**Fix:** Lazy imports in `graphrag/__init__.py` (don't load `GraphRAG` at package init) and `_storage/__init__.py` (only eager-load `JsonKVStorage`, `NanoVectorDBStorage`, `NetworkXStorage`; lazy-load `HNSWVectorStorage` / `Neo4jStorage`).

**Verify:**
```bash
python -m morph_graph.video_qa.cli \
  --benchmark vsi_morph \
  --output-json outputs/smoke.json \
  --cache-dir outputs/smoke_cache \
  --max-samples 1
```
