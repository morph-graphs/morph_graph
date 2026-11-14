# Qwen2.5-VL Compatibility Fix for RVS-Ego and RVS-Movie Datasets

## Problem Summary

Qwen2.5-VL and Qwen2-VL models were incompatible with the RVS-Ego and RVS-Movie datasets due to how these datasets use **streaming video encoding**. 

### Root Cause

RVS datasets use `rekv_stream_vqa.py` which processes videos **incrementally** as temporal windows arrive:
- Video frames are encoded in segments as questions come in
- `encode_video()` is called multiple times for the same video
- Each call should **append** to the existing cache, not reinitialize it

However, Qwen models called `initialize_cache_for_video()` on every `encode_video()` call, which:
1. **Resets the global_buffer to zeros** (line 568-571 in `kv_cache_manager.py`)
2. **Wipes out previously encoded video data**
3. Makes streaming impossible - each new segment overwrites the previous one

### Why Other Datasets Worked

Other datasets (egoschema, mlvu, videomme, etc.) use `rekv_offline_vqa.py`:
- Process the **entire video at once** before any questions
- Call `encode_video()` only **once per video**
- Cache reinitialization happens only once, so no data loss

## Solution

Added a **streaming mode flag** to the `encode_video()` method that controls cache initialization behavior:

### Changes Made

#### 1. **Qwen Models** (`qwen2_5_vl_rekv.py` and `qwen2_vl_rekv.py`)

**Modified `initialize_cache_for_video()`:**
```python
def initialize_cache_for_video(self, new_n_frame_tokens, cache_position_ids, force_reinit=False):
    # Only initialize if not already initialized (for streaming) or if forced
    if force_reinit or not getattr(self, '_cache_initialized_for_video', False):
        self._initialize_cache_for_video(new_n_frame_tokens, cache_position_ids)
        self._cache_initialized_for_video = True
```

**Modified `encode_video()` signature:**
```python
def encode_video(self, video, target_frames=None, streaming=False):
    # ...
    # For streaming mode, only initialize cache on first call
    # For non-streaming mode, always reinitialize (force_reinit=True)
    self.initialize_cache_for_video(new_n_frame_tokens, cache_position_ids, force_reinit=not streaming)
```

**Modified `clear_cache()`:**
```python
def clear_cache(self): 
    super().clear_cache()
    # Reset streaming video cache initialization flag
    self._cache_initialized_for_video = False
    # ... rest of method
```

#### 2. **Streaming VQA** (`rekv_stream_vqa.py`)

Updated to pass `streaming=True` when encoding incremental video segments:
```python
# Pass streaming=True for incremental encoding (important for Qwen models)
self.qa_model.encode_video(video_tensor[int(video_start_idx):int(video_end_idx)], streaming=True)
```

#### 3. **Base Class** (`abstract_rekv.py`)

Updated signature for backward compatibility:
```python
def encode_video(self, video, **kwargs):
    # Accept kwargs for compatibility (e.g., streaming, target_frames)
```

## Behavior

### Streaming Mode (`streaming=True`)
- **First call**: Initializes cache normally
- **Subsequent calls**: Skips cache initialization, appends to existing cache
- **Used by**: RVS-Ego, RVS-Movie (via `rekv_stream_vqa.py`)

### Non-Streaming Mode (`streaming=False`, default)
- **Every call**: Reinitializes cache (existing behavior)
- **Used by**: All other datasets (via `rekv_offline_vqa.py`)

### Cache Reset
- `clear_cache()` resets the initialization flag
- Each new video starts fresh with proper initialization

## Testing

### Test RVS-Ego
```bash
python video_qa/run_eval.py \
    --model qwen2_5_vl_7b \
    --dataset rvs_ego \
    --num_chunks 1 \
    --sample_fps 0.5 \
    --retrieve_size 64 \
    --encode_chunk_size 64 \
    --n_local 15000 \
    --prefill_method swa \
    --retrieval_method rekv
```

### Test RVS-Movie
```bash
python video_qa/run_eval.py \
    --model qwen2_5_vl_7b \
    --dataset rvs_movie \
    --num_chunks 1 \
    --sample_fps 0.5 \
    --retrieve_size 64 \
    --encode_chunk_size 64 \
    --n_local 15000 \
    --prefill_method swa \
    --retrieval_method rekv
```

### Test Non-Streaming Datasets (Regression Check)
```bash
python video_qa/run_eval.py \
    --model qwen2_5_vl_7b \
    --dataset egoschema \
    --num_chunks 1 \
    --sample_fps 0.5 \
    --retrieve_size 64 \
    --encode_chunk_size 64
```

## Backward Compatibility

✅ **Other models** (LLaVA-OneVision, Video-LLaVA, LongVA):
- Base class accepts `**kwargs`, so `streaming` parameter is ignored
- No changes needed to these models

✅ **Offline VQA** (all non-RVS datasets):
- Default `streaming=False` preserves existing behavior
- Cache always reinitializes (as before)

✅ **External models** (CLIP, SigLIP, PECore):
- These don't have streaming issues (simpler encoding)
- Unaffected by changes

## Files Modified

1. `model/qwen2_5_vl_rekv.py` - Added streaming support
2. `model/qwen2_vl_rekv.py` - Added streaming support  
3. `video_qa/rekv_stream_vqa.py` - Pass streaming flag
4. `model/abstract_rekv.py` - Accept kwargs for compatibility

## Key Points

- **Minimal changes**: Only affects Qwen models and streaming VQA
- **No breaking changes**: Default behavior unchanged for all existing use cases
- **Fixes streaming**: RVS datasets now work with Qwen models
- **Maintains performance**: No overhead for non-streaming datasets

## Expected Results

After these changes, you should be able to:
- ✅ Run Qwen2.5-VL on RVS-Ego without cache reinitialization issues
- ✅ Run Qwen2.5-VL on RVS-Movie without cache reinitialization issues
- ✅ Run Qwen2.5-VL on all other datasets (offline VQA) as before
- ✅ Run all other models on all datasets as before

## Questions?

If you encounter issues:
1. Check that `clear_cache()` is called at the start of each new video
2. Verify that streaming mode is only used for RVS datasets
3. Ensure the initialization flag resets between videos
