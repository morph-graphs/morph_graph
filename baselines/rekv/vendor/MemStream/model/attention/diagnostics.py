"""Env-gated diagnostics and analysis-storage code for ContextManager.

Split out of kv_cache_manager.py. Every function takes the manager as its first
argument and is inert unless its env flag / store_* toggle is set. Moving code
here must never change the numerics of a normal (flags-off) run.
"""
import csv
import math
import os

import torch
import torch.nn.functional as F
from ezcolorlog import root_logger as logger


def apply_env_overrides(store_full, store_adjacent_similarities, store_attn_scores):
    """Apply analysis env toggles to the ContextManager constructor flags.

    STORAGE_ON gates store_full/store_adjacent_similarities (never KV offload).
    STORE_ATTN_SCORES and the MEMSTREAM_*_DIR diagnostics force the torch
    attention impl via store_attn_scores="prefill" so _append sees full
    [B, H, Q, K] local-stage weights. All are inert in normal runs (env unset).
    """
    if os.environ.get('STORAGE_ON') == 'True':
        logger.info("Storage is on!")
        store_full = True
        store_adjacent_similarities = True

    if os.environ.get("STORE_ATTN_SCORES") == 'True':
        store_attn_scores = "prefill"

    # MEMSTREAM_TEXTMASS_DIR -> record_text_visual_attention_mass
    # MEMSTREAM_MASKNULL_DIR -> record_masked_renorm_perplexity (swa only)
    # MEMSTREAM_PROXYFID_DIR -> record_proxy_fidelity (swa only)
    for _env in ("MEMSTREAM_TEXTMASS_DIR", "MEMSTREAM_MASKNULL_DIR", "MEMSTREAM_PROXYFID_DIR"):
        if os.environ.get(_env):
            store_attn_scores = "prefill"

    return store_full, store_adjacent_similarities, store_attn_scores


# ---------------------------------------------------------------------------
# store_full analysis storage
# ---------------------------------------------------------------------------

def init_full_storage(mgr, global_k):
    """Allocate the full-KV debug buffers (store_full only)."""
    mgr.full_gpu_k = torch.empty((mgr.num_units, mgr.unit_size_kv, 0, mgr.dim_head), dtype=global_k.dtype, device=global_k.device)
    mgr.full_gpu_v = torch.empty((mgr.num_units, mgr.unit_size_kv, 0, mgr.dim_head), dtype=global_k.dtype, device=global_k.device)
    mgr.full_visual_mask_buffer = torch.empty((mgr.num_units, 0), dtype=torch.bool, device=global_k.device)


def append_full_storage(mgr, global_k, global_v, visual_pos_masks):
    """Append this chunk's KV to the full-KV debug buffers (store_full only)."""
    mgr.full_gpu_k = torch.cat([mgr.full_gpu_k, global_k], dim=-2)
    mgr.full_gpu_v = torch.cat([mgr.full_gpu_v, global_v], dim=-2)
    if visual_pos_masks is not None:
        mgr.full_visual_mask_buffer = torch.cat([mgr.full_visual_mask_buffer, visual_pos_masks.to(global_k.device)], dim=1)


def record_full_similarity_prefill(mgr, global_k_full, global_h_q_full):
    """Token-wise similarity on the GPU (pre-offload) retrieval path (store_full only)."""
    if mgr.num_blocks_seen < 250:
        full_similarity = torch.einsum("bhnd, bhqd -> bhqn", global_k_full.float(), global_h_q_full.float())  # (batch_size, num_qtokens, block_num)
        full_similarity = full_similarity / math.sqrt(mgr.dim_head)  # (batch_size, n_heads,num_qtokens, block_num)
        full_similarity = F.softmax(full_similarity, dim=-1)  # (batch_size, n_heads,num_qtokens, block_num)
        mgr.full_similarity = full_similarity.mean(dim=1, keepdim=False)  # (batch_size, num_qtokens, block_num)
        logger.info(f"Full Similarity Shape: {mgr.full_similarity.shape}")
    else:
        logger.info(f"Full Similarity is not stored because num_blocks_seen >= 250 -- too expensive to store!")
        mgr.full_similarity = None


def record_full_similarity_offloaded(mgr, global_h_q_full):
    """Token-wise similarity on the CPU (offloaded) retrieval path (store_full only)."""
    if mgr.num_blocks_seen < 250:
        mgr.full_similarity = get_all_token_similarity(mgr, global_h_q_full, mode="qk").float()
    else:
        logger.info(f"Full Similarity is not stored because num_blocks_seen >= 250 -- too expensive to store!")
        mgr.full_similarity = None


def get_all_token_similarity(mgr, query, mode=None):
    """
    Calculates token-level attention weights (softmax style) between a query
    and ALL stored global keys. This method uses the `mgr.full_gpu_k` cache.
    This implements Softmax( (Q @ K.T) / sqrt(d_k) ).

    Args:
        query: (batch_size, num_heads, n_query, dim_head)
               The query tokens (e.g., global_q after RoPE).

    Returns:
        attention_weights: (batch_size, num_heads, n_query, n_all_keys)
                           The matrix of softmax attention weights.
    """
    if not mgr.initialized:
        logger.error("ContextManager not initialized. Call append() first.")
        raise RuntimeError("ContextManager not initialized. Call append() first.")

    if mgr.full_gpu_k.shape[-2] == 0:
        logger.warning("Full GPU K-cache is empty. Returning empty tensor.")
        return torch.empty(
            (query.shape[0], query.shape[1], query.shape[2], 0),
            device=query.device,
            dtype=query.dtype
        )

    if mode == "qk":

        # mgr.full_gpu_k is (batch_size, n_head_kv, n_keys, dim_head)
        # query is (batch_size, n_head, n_query, dim_head)

        # Expand K to match Q's heads (handles GQA)
        # all_keys shape: (batch_size, n_head, n_keys, dim_head)
        global_k = mgr.full_gpu_k
        all_keys = mgr._from_group_kv(global_k)

        n_batch, n_head, n_query, n_dim = query.shape
        _n_batch, _n_head, n_keys, _n_dim = all_keys.shape

        assert n_batch == _n_batch, "Batch size mismatch"
        assert n_head == _n_head, "Head count mismatch after GQA expansion"
        assert n_dim == _n_dim, "Head dimension mismatch"

        # Use float32 for stable calculation
        q_float = query.float()   # (B, H, N_q, D)
        k_float = all_keys.float()  # (B, H, N_k, D)

        # 1. Calculate raw dot product
        # (B, H, N_q, D) @ (B, H, D, N_k) -> (B, H, N_q, N_k)
        k_float_t = k_float.transpose(-1, -2)
        attn_scores = torch.matmul(q_float, k_float_t)

        # 2. Scale by sqrt(d_k)
        scale_factor = math.sqrt(n_dim)
        attn_scores = attn_scores / scale_factor

        # 3. Apply softmax
        # Softmax is applied on the last dimension (the keys)
        attention_weights = F.softmax(attn_scores, dim=-1)
        attention_weights = attention_weights[:, :, :, mgr.n_init:]

        # Return matrix in the original query dtype
        return attention_weights.to(query.dtype)

    elif mode == "kk":

        global_k = mgr.full_gpu_k[:, :, mgr.n_init:, :].cpu()  # (batch_size, n_head_kv, n_keys, dim_head)

        # extract the frame features from global_k using visual_pos_masks
        if mgr.has_adaptive_block_size:

            assert mgr.full_visual_mask_buffer.size(-1) == sum(mgr.adaptive_block_sizes)
            assert global_k.size(2) == sum(mgr.adaptive_block_sizes)

            visual_pos_masks = mgr.full_visual_mask_buffer.squeeze(0).to(global_k.device)
            global_k = global_k[:, :, visual_pos_masks]

        k = torch.nn.functional.normalize(global_k.float(), dim=-1)

        b, h, m, d = k.shape
        assert global_k.size(-2) % mgr.block_size == 0, f"{global_k.size(-2)} - {mgr.block_size}"
        block_num = global_k.size(-2) // mgr.block_size

        logger.info(f"Computing block similarity for {m} keys -> {block_num} blocks")

        # Pool to per-block mean keys *before* the matmul: building the full (m, m)
        # token-similarity matrix first costs O(m^2) host RAM (49 GB for a 16-minute
        # video at 0.5 FPS) and was silently OOM-killed. Pooling first is exactly
        # equivalent (dot product is bilinear, index sets independent); the head mean
        # stays outside the dot product, matching the original ordering.
        k_blocks = k.view(b, h, block_num, mgr.block_size, d).mean(dim=3)  # (b, h, block_num, d)
        block_sims = torch.matmul(k_blocks, k_blocks.transpose(-1, -2))     # (b, h, bn, bn)
        block_sims = block_sims.mean(dim=1).cpu()                           # (b, bn, bn)

        mgr.block_sims = block_sims
        return block_sims


# ---------------------------------------------------------------------------
# adjacent-key similarities (store_adjacent_similarities)
# ---------------------------------------------------------------------------

def record_adjacent_similarity(mgr, local_k, len_q):
    """Frame-level cosine similarity with the previous frame, for event chunking."""
    # Compute frame-level representation by averaging all keys in the frame
    # current_frame_k: (batch, n_head_kv, len_k, dim_head)
    current_frame_repr = local_k[:, :, -len_q:].mean(dim=-2, keepdim=True)  # (batch, n_head_kv, 1, dim_head)
    # Compute similarity with previous frame if available
    if mgr.last_block_key is not None:
        # Previous frame representation: (batch, n_head_kv, 1, dim_head)
        prev_frame_repr = mgr.last_block_key
        # Compute cosine similarity between frame representations
        prev_frame_norm = F.normalize(prev_frame_repr.float(), p=2, dim=-1)
        curr_frame_norm = F.normalize(current_frame_repr.float(), p=2, dim=-1)
        cos_sim = torch.sum(prev_frame_norm * curr_frame_norm, dim=-1)
        # Average across batch and heads
        avg_cos_sim = cos_sim.mean().item()
        mgr.adjacent_key_similarities.append(avg_cos_sim)

    # Store current frame representation for next iteration
    mgr.last_block_key = current_frame_repr.clone()


def get_adjacent_key_similarities(mgr):
    """
    Get the list of adjacent key cosine similarities.
    Only available when store_adjacent_similarities=True.

    Returns:
        list: List of cosine similarity values between adjacent keys
    """
    if mgr.adjacent_key_similarities is None:
        logger.warning("Adjacent key similarities not available. Set store_adjacent_similarities=True to enable.")
        return []
    return mgr.adjacent_key_similarities


def get_similarity_statistics(mgr):
    """
    Get statistics about the adjacent key similarities.

    Returns:
        dict: Dictionary containing mean, std, min, max of similarities
    """
    if mgr.adjacent_key_similarities is None or len(mgr.adjacent_key_similarities) == 0:
        logger.warning("No adjacent key similarities available.")
        return {}

    sims = torch.tensor(mgr.adjacent_key_similarities)
    stats = {
        'mean': sims.mean().item(),
        'std': sims.std().item(),
        'min': sims.min().item(),
        'max': sims.max().item(),
        'count': len(mgr.adjacent_key_similarities)
    }
    return stats


def find_event_boundaries(mgr, threshold=None, method='percentile', percentile=25):
    """
    Find potential event boundaries based on drops in adjacent key similarity.
    Low similarity between adjacent keys suggests a scene/event change.

    Args:
        threshold: float, absolute threshold for similarity (if method='absolute')
        method: str, 'absolute' or 'percentile' or 'std'
        percentile: float, percentile threshold (if method='percentile', lower = more boundaries)

    Returns:
        list: Indices where event boundaries are detected (0-indexed positions between keys)
    """
    if mgr.adjacent_key_similarities is None or len(mgr.adjacent_key_similarities) == 0:
        logger.warning("No adjacent key similarities available for event boundary detection.")
        return []

    sims = torch.tensor(mgr.adjacent_key_similarities)

    if method == 'absolute':
        if threshold is None:
            threshold = 0.5
        boundaries = (sims < threshold).nonzero(as_tuple=True)[0].tolist()

    elif method == 'percentile':
        threshold_val = torch.quantile(sims, percentile / 100.0).item()
        boundaries = (sims < threshold_val).nonzero(as_tuple=True)[0].tolist()
        logger.info(f"Event boundary threshold (percentile {percentile}): {threshold_val:.4f}")

    elif method == 'std':
        # Boundaries where similarity is more than N std below mean
        if threshold is None:
            threshold = 1.0  # N standard deviations
        mean_sim = sims.mean()
        std_sim = sims.std()
        threshold_val = mean_sim - threshold * std_sim
        boundaries = (sims < threshold_val).nonzero(as_tuple=True)[0].tolist()
        logger.info(f"Event boundary threshold (mean - {threshold}*std): {threshold_val:.4f}")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'absolute', 'percentile', or 'std'")

    logger.info(f"Found {len(boundaries)} event boundaries using method '{method}'")
    return boundaries


# ---------------------------------------------------------------------------
# prefill attention-score diagnostics (store_attn_scores / MEMSTREAM_*_DIR)
# ---------------------------------------------------------------------------

def _get_frame_key_layout(mgr, num_keys):
    """Map the local-window key axis to frames, supporting adaptive tokens per frame.

    Returns (frame_key_counts, n_init_keys) where frame_key_counts[i] is the
    number of keys of the i-th frame in key order (oldest first, current frame
    last) and n_init_keys is the number of leading sink keys. Returns (None, 0)
    if the layout can't be tiled exactly (e.g. subsampled/compressed keys).
    """
    # Adaptive per-frame token counts (Qwen3-VL: vision + timestamp/prefix tokens
    # vary per frame). Find the suffix of adaptive_block_sizes that tiles the keys.
    if getattr(mgr, 'has_adaptive_block_size', False) and getattr(mgr, 'adaptive_block_sizes', None):
        n_init_keys = 0 if mgr.init_exc else mgr.n_init
        target = num_keys - n_init_keys
        sizes = mgr.adaptive_block_sizes
        acc, cnt = 0, 0
        for s in reversed(sizes):
            acc += s
            cnt += 1
            if acc >= target:
                break
        if acc != target:
            return None, 0
        return list(sizes[len(sizes) - cnt:]), n_init_keys

    # Uniform tokens per frame: infer leading init keys from the remainder
    patches_per_frame = mgr.grid_hw[0] * mgr.grid_hw[1]
    n_init_keys = num_keys % patches_per_frame
    num_frames = (num_keys - n_init_keys) // patches_per_frame
    return [patches_per_frame] * num_frames, n_init_keys


def _compute_attention_entropy(attn_weights, n_init_keys):
    """
    Computes per-head entropy in raw bits (mean over queries only).
    attn_weights: [Batch, Heads, Queries, Keys]
    Returns: (entropy_bits [num_heads], num_keys)
    """
    # Slice out the init tokens to focus on spatial/temporal keys
    # If you keep them, the entropy shows how much the model 'escapes' the sink
    if n_init_keys > 0:
        attn_weights = attn_weights[:, :, :, n_init_keys:]
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

    num_keys = attn_weights.shape[-1]

    # Raw entropy in bits: -sum(p * log2(p)) -> [B, H, Q]
    entropy = -torch.sum(attn_weights * torch.log2(attn_weights + 1e-9), dim=-1)

    return entropy.mean(dim=-1).squeeze(0).float(), num_keys


def _compute_past_frame_mass(attn_weights, frame_key_counts, n_init_keys):
    """Per-frame attention mass over past frames: [B, H, F] (F excludes current).

    Handles variable tokens per frame via scatter-add instead of a uniform
    reshape. Mass is renormalized over past frames (init + current excluded).
    """
    device = attn_weights.device
    counts = torch.as_tensor(frame_key_counts, device=device)
    num_frames = counts.numel()
    # Frame id per key, oldest=0 ... current=num_frames-1
    frame_ids = torch.repeat_interleave(torch.arange(num_frames, device=device), counts)

    spatial_attn = attn_weights[:, :, :, n_init_keys:].clone()  # [B, H, Q, K]
    # Mask the current frame (last in key order), renormalize over past frames
    spatial_attn[:, :, :, frame_ids == num_frames - 1] = 0.0
    spatial_attn = spatial_attn / (spatial_attn.sum(dim=-1, keepdim=True) + 1e-8)

    b, h, q, _ = spatial_attn.shape
    per_frame = spatial_attn.new_zeros(b, h, q, num_frames)
    per_frame.scatter_add_(-1, frame_ids.expand(b, h, q, -1), spatial_attn)
    return per_frame[..., :-1], num_frames  # drop the (zeroed) current-frame slot


def _compute_temporal_span(per_frame_mass, num_frames):
    """Mean temporal distance attended (1=previous frame, larger=older)."""
    n_past = per_frame_mass.shape[-1]
    # Distance of past frame i (oldest=0) from the current frame
    distances = torch.arange(n_past, 0, -1, device=per_frame_mass.device).float()
    mean_frame_per_query = torch.sum(per_frame_mass * distances, dim=-1)  # [B, H, Q]
    mean_frame_per_head = mean_frame_per_query.mean(dim=-1)
    std_frame_per_head = mean_frame_per_query.std(dim=-1)
    return mean_frame_per_head, std_frame_per_head


def _compute_frame_coverage(per_frame_mass, thresholds=(0.5, 0.8, 0.9)):
    """How many past frames hold the majority of attention mass.

    For each head: sort past frames by mass and count how many are needed to
    reach each coverage threshold. Small counts => attention is concentrated
    on a few frames => per-frame budgets from query-frame attention are
    informative.

    Returns: dict threshold -> [num_heads] tensor of frame counts.
    """
    n_past = per_frame_mass.shape[-1]
    # Average over queries -> [B, H, F], renormalize
    per_frame = per_frame_mass.mean(dim=-2)
    per_frame = per_frame / (per_frame.sum(dim=-1, keepdim=True) + 1e-8)

    sorted_mass, _ = per_frame.sort(dim=-1, descending=True)
    cum_mass = sorted_mass.cumsum(dim=-1)  # [B, H, F]

    counts = {}
    for t in thresholds:
        # Smallest n with cumulative mass >= t; clamp handles degenerate heads
        n_needed = (cum_mass < t).sum(dim=-1) + 1
        counts[t] = n_needed.clamp(max=n_past).squeeze(0)  # [num_heads]
    return counts


def _record_attention_stats(mgr, score_list, frame_key_counts, n_init_keys):
    """Temporal-span / entropy / coverage stats per head, appended to
    ATTN_SCORES_DIR/<video>_attention_stats.csv."""
    per_frame_mass, num_frames = _compute_past_frame_mass(score_list[0], frame_key_counts, n_init_keys)
    mean_frame_per_head, std_frame_per_head = _compute_temporal_span(per_frame_mass, num_frames)
    entropy_bits, entropy_num_keys = _compute_attention_entropy(score_list[0], n_init_keys)
    coverage_counts = _compute_frame_coverage(per_frame_mass)

    # Save attention statistics incrementally to CSV
    video_id = os.environ.get('CURRENT_VIDEO_ID', 'BV1TG411U7zi')
    save_dir = os.environ.get('ATTN_SCORES_DIR', '/nfshomes/vatsalag/mem_open_vocab/ReKV/attn_scores')
    csv_path = f"{save_dir}/{video_id}_attention_stats.csv"

    os.makedirs(save_dir, exist_ok=True)

    # Squeeze batch dimension: [1, num_heads] -> [num_heads]
    mean_frame_per_head = mean_frame_per_head.squeeze(0)
    std_frame_per_head = std_frame_per_head.squeeze(0)
    num_heads = mean_frame_per_head.shape[0]

    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            writer.writerow(['video_id', 'layer_idx', 'window_idx', 'head_idx',
                           'mean_frame_attended', 'std_frame_attended', 'num_frames',
                           'entropy_bits', 'num_keys', 'perplexity',
                           'frames_at_50', 'frames_at_80', 'frames_at_90'])

        # Write statistics for each head (average temporal distance attended)
        for head_idx in range(num_heads):
            h_bits = float(entropy_bits[head_idx].cpu())
            writer.writerow([
                video_id,
                mgr.layer_idx,
                mgr.num_blocks_seen,
                head_idx,
                float(mean_frame_per_head[head_idx].cpu()),
                float(std_frame_per_head[head_idx].cpu()),
                float(num_frames),
                h_bits,
                int(entropy_num_keys),
                float(2.0 ** h_bits),
                int(coverage_counts[0.5][head_idx].cpu()),
                int(coverage_counts[0.8][head_idx].cpu()),
                int(coverage_counts[0.9][head_idx].cpu()),
            ])

    print(f"[Layer {mgr.layer_idx}, Window {mgr.num_blocks_seen}] Saved temporal attention stats for {num_heads} heads", flush=True)


def record_prefill_attention_diagnostics(mgr, score_list, local_q, local_k, visual_pos_masks, subsample_mask):
    """All prefill-time attention-score diagnostics.

    Caller gates on `get_score and start_compression`; the per-diagnostic env
    gates are checked here. Preserves the original ordering: attention stats,
    then textmass, masknull, proxyfid.
    """
    frame_key_counts, n_init_keys = _get_frame_key_layout(mgr, score_list[0].size(-1))
    if frame_key_counts is None or len(frame_key_counts) < 2:
        # Compressed windows don't tile into uniform dense frames; entropy needs no
        # tiling, so record it alone (frame-coverage fields left blank).
        _record_entropy_only(mgr, score_list)
    else:
        _record_attention_stats(mgr, score_list, frame_key_counts, n_init_keys)

    # Text-vs-visual attention-mass diagnostic. Inert unless MEMSTREAM_TEXTMASS_DIR is set.
    textmass_dir = os.environ.get('MEMSTREAM_TEXTMASS_DIR')
    if textmass_dir and visual_pos_masks is not None:
        record_text_visual_attention_mass(
            mgr, score_list, visual_pos_masks, subsample_mask,
            n_kv_heads=local_k.size(1), save_dir=textmass_dir,
        )

    # Masked-renormalization null diagnostic. Inert unless MEMSTREAM_MASKNULL_DIR is set.
    masknull_dir = os.environ.get('MEMSTREAM_MASKNULL_DIR')
    if masknull_dir and visual_pos_masks is not None:
        record_masked_renorm_perplexity(
            mgr, score_list, visual_pos_masks, subsample_mask,
            save_dir=masknull_dir,
        )

    # akv2 proxy-fidelity diagnostic. Inert unless MEMSTREAM_PROXYFID_DIR is set.
    # Requires the frame tiling computed above (same guard, so it is always
    # assigned when this condition is reachable).
    proxyfid_dir = os.environ.get('MEMSTREAM_PROXYFID_DIR')
    if proxyfid_dir and frame_key_counts is not None and len(frame_key_counts) >= 3:
        record_proxy_fidelity(
            mgr, score_list, local_q, local_k, frame_key_counts, n_init_keys,
            save_dir=proxyfid_dir,
        )


def record_proxy_fidelity(
    mgr, score_list, local_q, local_k, frame_key_counts, n_init_keys, save_dir,
):
    """Diagnostic (gated on MEMSTREAM_PROXYFID_DIR): how faithfully the akv2
    query-affinity proxy ranks past frames/tokens relative to TRUE attention.

    The akv2 affinity channel scores token k as (qbar . k)/sqrt(d) with qbar the
    pre-RoPE current-chunk queries mean-pooled per kv-head group. This records,
    per (layer, window, kv-head), the Spearman rank correlation between that
    proxy and the true (post-RoPE, post-softmax) attention mass, at two levels:
      - frame level (what akv2's BUDGET allocation consumes): true mass summed
        per past frame vs proxy averaged per past frame
      - token level (what akv2's within-frame SELECTION consumes)
    plus the top-5 frame-set overlap. Init keys and the current frame are
    excluded from both sides.

    swa arm only: asserts the score key axis equals the uncompressed local
    window, so keys align 1:1 with local_k.
    """
    score = score_list[0]
    if score is None:
        raise AssertionError("proxyfid diagnostic: local-stage score is None despite get_score")
    if score.dim() != 4 or score.size(0) != 1:
        raise AssertionError(f"proxyfid diagnostic expects [1, H, Q, K] local scores, got {tuple(score.shape)}")

    video_id = os.environ.get('CURRENT_VIDEO_ID', 'unknown_video')
    window_idx = mgr.num_blocks_seen
    if not hasattr(mgr, '_proxyfid_windows_seen'):
        mgr._proxyfid_windows_seen = set()
    dedupe_key = (video_id, window_idx)
    if dedupe_key in mgr._proxyfid_windows_seen:
        return
    mgr._proxyfid_windows_seen.add(dedupe_key)

    device = score.device
    n_kv_heads = local_k.size(1)
    w = score[0].float()                      # (Hq, Q, K)
    n_q_heads, _, n_keys = w.shape
    if n_keys != local_k.size(-2):
        raise AssertionError(
            f"proxyfid diagnostic is swa-only: score key axis ({n_keys}) != "
            f"local_k length ({local_k.size(-2)}); run with prefill_method=swa"
        )
    if n_q_heads % n_kv_heads != 0:
        raise AssertionError(f"query heads ({n_q_heads}) not a multiple of kv heads ({n_kv_heads})")
    group = n_q_heads // n_kv_heads

    counts = torch.as_tensor(frame_key_counts, device=device)
    num_frames = counts.numel()
    frame_ids = torch.repeat_interleave(torch.arange(num_frames, device=device), counts)
    # Past tokens: exclude init keys and the current (last) frame
    past_tok = torch.zeros(n_keys, dtype=torch.bool, device=device)
    past_tok[n_init_keys:] = frame_ids != (num_frames - 1)
    n_past_frames = num_frames - 1
    past_frame_ids = frame_ids[frame_ids != (num_frames - 1)]  # aligned to past tokens

    # --- true mass per token (query-head mean -> kv-head group mean) ---------
    true_tok = w.mean(dim=1)                                   # (Hq, K)
    true_tok = true_tok.view(n_kv_heads, group, n_keys).mean(dim=1)  # (nh_kv, K)
    true_tok = true_tok[:, past_tok]                           # (nh_kv, P)

    # --- proxy per token: exactly the akv2 affinity channel ------------------
    qbar = local_q[0].float().view(n_kv_heads, group, local_q.size(-2), -1).mean(dim=(1, 2))  # (nh_kv, d)
    k_all = local_k[0].float()                                 # (nh_kv, K, d)
    proxy_tok = torch.einsum('hd,hkd->hk', qbar, k_all) / math.sqrt(k_all.size(-1))
    proxy_tok = proxy_tok[:, past_tok]                         # (nh_kv, P)

    # --- frame level: true mass summed, proxy averaged, per past frame -------
    fid = past_frame_ids.expand(n_kv_heads, -1)
    true_frame = true_tok.new_zeros(n_kv_heads, n_past_frames).scatter_add_(-1, fid, true_tok)
    proxy_sum = proxy_tok.new_zeros(n_kv_heads, n_past_frames).scatter_add_(-1, fid, proxy_tok)
    frame_sizes = counts[:-1].to(proxy_sum.dtype).unsqueeze(0)
    proxy_frame = proxy_sum / frame_sizes

    def spearman(a, b):
        """Rowwise Spearman via Pearson of ranks. a, b: (H, N)."""
        ra = torch.argsort(torch.argsort(a, dim=-1), dim=-1).float()
        rb = torch.argsort(torch.argsort(b, dim=-1), dim=-1).float()
        ra = ra - ra.mean(dim=-1, keepdim=True)
        rb = rb - rb.mean(dim=-1, keepdim=True)
        num = (ra * rb).sum(dim=-1)
        den = ra.norm(dim=-1) * rb.norm(dim=-1) + 1e-8
        return num / den

    sp_frame = spearman(true_frame, proxy_frame)               # (nh_kv,)
    sp_token = spearman(true_tok, proxy_tok)                   # (nh_kv,)

    k_top = min(5, n_past_frames)
    top_true = true_frame.topk(k_top, dim=-1).indices
    top_proxy = proxy_frame.topk(k_top, dim=-1).indices
    overlap = torch.stack([
        torch.isin(top_true[h_i], top_proxy[h_i]).float().mean()
        for h_i in range(n_kv_heads)
    ])

    csv_path = f"{save_dir}/{video_id}_proxy_fidelity.csv"
    os.makedirs(save_dir, exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['video_id', 'layer_idx', 'window_idx', 'kv_head',
                             'n_past_frames', 'spearman_frame', 'spearman_token',
                             'top5_frame_overlap'])
        for h_i in range(n_kv_heads):
            writer.writerow([
                video_id, mgr.layer_idx, window_idx, h_i, int(n_past_frames),
                float(sp_frame[h_i].cpu()), float(sp_token[h_i].cpu()),
                float(overlap[h_i].cpu()),
            ])


def record_text_visual_attention_mass(
    mgr, score_list, visual_pos_masks, subsample_mask, n_kv_heads, save_dir,
):
    """Diagnostic (gated on MEMSTREAM_TEXTMASS_DIR): fraction of local-window
    attention MASS landing on text keys (timestamp + vision_start/end) vs
    visual keys, per (layer, window, query-head).

    Hypothesis under test: AKS (swa_kk_adjacent_*) deletes only visual
    tokens, raising the *token-count* share of text in the window ~10x --
    but that has only ever been computed from counts. Each row therefore
    records BOTH the attention-mass fraction and the token-count share
    (n_text_keys / n_keys) so mass-vs-share can be compared directly.

    Alignment contract (asserted, never guessed):
      - swa (subsample_mask is None): keys are the uncompressed window;
        the key axis must equal visual_pos_masks' length exactly.
      - compressed (kk/dilated): keys are the kept tokens; subsample_mask
        must be per-KV-head integer indices (1, n_kv, L) into the
        uncompressed window and the key axis must equal L.
      - GQA: the score head axis is QUERY heads; kv-head masks are
        expanded with repeat_interleave (matching torch_impl's k.expand
        ordering) and the ratio must divide evenly.
      - Local-stage rows do NOT sum to 1 (softmax is joint with the
        global/sink stage), so masses are normalized by the actual row
        sum, which is also recorded (row_sum_mean).
    """
    score = score_list[0]
    if score is None:
        raise AssertionError("textmass diagnostic: local-stage score is None despite get_score")
    if score.dim() != 4 or score.size(0) != 1:
        raise AssertionError(f"textmass diagnostic expects [1, H, Q, K] local scores, got {tuple(score.shape)}")

    video_id = os.environ.get('CURRENT_VIDEO_ID', 'unknown_video')
    window_idx = mgr.num_blocks_seen
    # Record each (video, window) once per layer: QA/decode-time appends
    # re-enter _append with the same num_blocks_seen and must not add rows.
    if not hasattr(mgr, '_textmass_windows_seen'):
        mgr._textmass_windows_seen = set()
    dedupe_key = (video_id, window_idx)
    if dedupe_key in mgr._textmass_windows_seen:
        return
    mgr._textmass_windows_seen.add(dedupe_key)

    w = score[0].float()  # (Hq, Q, K)
    n_q_heads, _, n_keys = w.shape
    if n_q_heads % n_kv_heads != 0:
        raise AssertionError(
            f"textmass diagnostic: query heads ({n_q_heads}) not divisible by kv heads ({n_kv_heads})")
    group = n_q_heads // n_kv_heads

    vmask = visual_pos_masks[0].to(device=w.device).bool()  # (N,) over the UNCOMPRESSED window
    n_uncomp = vmask.numel()
    # Same prefix rule as _frame_geometry: pre-offload windows start with
    # n_init non-visual sink tokens; once init_exc is set they live in the
    # global stage instead. Sink text is NOT timestamp text, so it gets its
    # own category rather than polluting text_mass_frac.
    prefix_len = mgr.n_init if (
        not mgr.init_exc
        and n_uncomp >= mgr.n_init
        and not vmask[:mgr.n_init].any()
    ) else 0
    prefix_mask = torch.arange(n_uncomp, device=w.device) < prefix_len

    if subsample_mask is None:
        # swa: keys are the uncompressed window, one shared mask for all heads
        if n_keys != n_uncomp:
            raise AssertionError(
                "textmass diagnostic (swa path): key axis "
                f"({n_keys}) != visual mask length ({n_uncomp}); refusing to slice by guesswork")
        key_visual = vmask.unsqueeze(0).expand(n_q_heads, n_keys)
        key_prefix = prefix_mask.unsqueeze(0).expand(n_q_heads, n_keys)
    else:
        if subsample_mask.dtype not in (torch.int32, torch.int64):
            raise AssertionError(
                "textmass diagnostic: expected integer kept-token indices for the compressed "
                f"path, got dtype={subsample_mask.dtype} shape={tuple(subsample_mask.shape)} "
                "(boolean subsample masks, e.g. tome, are not supported)")
        if subsample_mask.dim() != 3 or subsample_mask.size(0) != 1 or subsample_mask.size(1) != n_kv_heads:
            raise AssertionError(
                f"textmass diagnostic: kept-token indices shaped {tuple(subsample_mask.shape)}, "
                f"expected (1, n_kv_heads={n_kv_heads}, L)")
        if n_keys != subsample_mask.size(-1):
            raise AssertionError(
                f"textmass diagnostic (compressed path): key axis ({n_keys}) != "
                f"kept-token count ({subsample_mask.size(-1)})")
        kept = subsample_mask[0].to(device=w.device).long()  # (n_kv, K)
        if int(kept.min()) < 0 or int(kept.max()) >= n_uncomp:
            raise AssertionError(
                f"textmass diagnostic: kept indices out of range [0, {n_uncomp}): "
                f"min={int(kept.min())}, max={int(kept.max())}")
        # kv head g owns query heads [g*group, (g+1)*group) -- same ordering
        # torch_impl uses when it expands k/v for GQA.
        kept_q = kept.repeat_interleave(group, dim=0)  # (Hq, K)
        key_visual = vmask[kept_q]
        key_prefix = prefix_mask[kept_q]

    key_text = (~key_visual) & (~key_prefix)

    row_sum = w.sum(dim=-1)  # (Hq, Q)
    denom = row_sum.clamp_min(1e-9)

    def _mass_frac(mask):
        mass = torch.einsum('hqk,hk->hq', w, mask.float())
        return (mass / denom).mean(dim=-1)  # (Hq,)

    text_frac = _mass_frac(key_text).cpu()
    visual_frac = _mass_frac(key_visual).cpu()
    prefix_frac = _mass_frac(key_prefix).cpu()
    row_sum_mean = row_sum.mean(dim=-1).cpu()
    n_text = key_text.sum(dim=-1).cpu()
    n_visual = key_visual.sum(dim=-1).cpu()
    n_prefix = key_prefix.sum(dim=-1).cpu()

    os.makedirs(save_dir, exist_ok=True)
    arm = mgr.prefill_method
    csv_path = os.path.join(save_dir, f"{video_id}__{arm}__textmass.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'video_id', 'arm', 'layer_idx', 'window_idx', 'head_idx',
                'text_mass_frac', 'visual_mass_frac', 'prefix_mass_frac', 'row_sum_mean',
                'n_text_keys', 'n_visual_keys', 'n_prefix_keys', 'n_keys',
            ])
        for h in range(n_q_heads):
            writer.writerow([
                video_id, arm, mgr.layer_idx, window_idx, h,
                float(text_frac[h]), float(visual_frac[h]), float(prefix_frac[h]),
                float(row_sum_mean[h]),
                int(n_text[h]), int(n_visual[h]), int(n_prefix[h]), int(n_keys),
            ])


def record_masked_renorm_perplexity(
    mgr, score_list, visual_pos_masks, subsample_mask, save_dir,
):
    """Diagnostic (gated on MEMSTREAM_MASKNULL_DIR): counterfactual
    masked-and-renormalized attention perplexity, swa arm ONLY.

    Question under test: AKS (swa_kk_adjacent_4) narrows the per-head
    perplexity distribution (raises p10, cuts p90) relative to swa. Is
    that a representational effect, or a mechanical consequence of
    deleting keys and renormalizing? Here, on an swa run where ALL keys
    are present, each (window, layer, head) row records the TRUE
    perplexity plus the perplexity the same score matrix would have under
    two content-blind masks that copy AKS's composition (keep all prefix +
    all text keys, 1/16 of visual keys):

      - structured: every 16th visual key, fixed stride from the first
        visual position (AKS `_4` = dilation 4 per spatial axis = 1/16 of
        visual patches; no content criterion here).
      - random: a uniformly random 1/16 subset of visual keys, drawn
        FRESH PER ROW (i.e. per window/layer/head; deterministic seed per
        (layer, window) so re-runs reproduce).

    For each mask the score matrix is sliced to the kept columns, each
    query row is renormalized by its own new row sum, and entropy is
    H = -sum p log2 p in bits, averaged over queries, with perplexity
    2**H -- the same convention the true path uses (exponentiate the
    query-mean entropy).

    Deliberate composition gap, NOT mimicked: kk_adjacent additionally
    keeps the most recent nd*chunk keys dense (its subsample_window
    excludes the tail), so AKS's n_keys runs higher than these masks at
    the same stride. n_keys are recorded so the gap is visible.

    Alignment contract (asserted, never guessed): this is an swa-only
    counterfactual -- subsample_mask must be None and the key axis must
    equal visual_pos_masks' length exactly. Prefix classification reuses
    the same rule as record_text_visual_attention_mass.
    """
    score = score_list[0]
    if score is None:
        raise AssertionError("masknull diagnostic: local-stage score is None despite get_score")
    if score.dim() != 4 or score.size(0) != 1:
        raise AssertionError(
            f"masknull diagnostic expects [1, H, Q, K] local scores, got {tuple(score.shape)}")
    if subsample_mask is not None:
        raise AssertionError(
            "masknull diagnostic is an swa-only counterfactual (all keys must be present); "
            f"got a compressed arm ({mgr.prefill_method}) with subsample_mask set")

    video_id = os.environ.get('CURRENT_VIDEO_ID', 'unknown_video')
    window_idx = mgr.num_blocks_seen
    # Record each (video, window) once per layer: QA/decode-time appends
    # re-enter _append with the same num_blocks_seen and must not add rows.
    if not hasattr(mgr, '_masknull_windows_seen'):
        mgr._masknull_windows_seen = set()
    dedupe_key = (video_id, window_idx)
    if dedupe_key in mgr._masknull_windows_seen:
        return
    mgr._masknull_windows_seen.add(dedupe_key)

    w = score[0].float()  # (Hq, Q, K)
    n_q_heads, n_queries, n_keys = w.shape

    vmask = visual_pos_masks[0].to(device=w.device).bool()  # (N,) uncompressed window
    n_uncomp = vmask.numel()
    if n_keys != n_uncomp:
        raise AssertionError(
            "masknull diagnostic: key axis "
            f"({n_keys}) != visual mask length ({n_uncomp}); refusing to slice by guesswork")

    # Same prefix rule as record_text_visual_attention_mass.
    prefix_len = mgr.n_init if (
        not mgr.init_exc
        and n_uncomp >= mgr.n_init
        and not vmask[:mgr.n_init].any()
    ) else 0
    prefix_mask = torch.arange(n_uncomp, device=w.device) < prefix_len

    visual_idx = vmask.nonzero(as_tuple=True)[0]          # (n_vis,)
    nonvisual_idx = (~vmask).nonzero(as_tuple=True)[0]    # prefix + text
    n_vis = visual_idx.numel()
    n_prefix = int(prefix_mask.sum())
    n_text = n_uncomp - n_vis - n_prefix
    if n_vis == 0:
        raise AssertionError(
            f"masknull diagnostic: window {window_idx} has no visual keys; nothing to mask")

    STRIDE = 16  # AKS `_4` keeps 1/(4*4) of visual patches

    # structured mask: shared across heads
    keep_struct = torch.cat([nonvisual_idx, visual_idx[::STRIDE]]).sort().values
    n_keep_vis = visual_idx[::STRIDE].numel()

    # random mask: fresh per head, deterministic per (layer, window)
    gen = torch.Generator()
    gen.manual_seed((mgr.layer_idx * 1000003 + window_idx * 1009 + 271828) % (2 ** 31))
    rand_rows = []
    for _ in range(n_q_heads):
        perm = torch.randperm(n_vis, generator=gen)[:n_keep_vis].to(w.device)
        rand_rows.append(torch.cat([nonvisual_idx, visual_idx[perm]]).sort().values)
    idx_rand = torch.stack(rand_rows)  # (Hq, Kk)

    def _entropy_bits(sub):
        # sub: (Hq, Q, Kk) non-negative scores. Local-stage rows do NOT
        # sum to 1 (softmax is joint with the global/sink stage), so every
        # variant -- including the true one -- renormalizes each query row
        # by its own row sum before the entropy.
        rs = sub.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        p = sub / rs
        h = -(p * torch.log2(p.clamp_min(1e-12))).sum(dim=-1)  # (Hq, Q)
        return h.mean(dim=-1)  # (Hq,) query-mean entropy in bits

    h_true = _entropy_bits(w)
    h_struct = _entropy_bits(w[:, :, keep_struct])
    sub_rand = w.gather(2, idx_rand.unsqueeze(1).expand(n_q_heads, n_queries, idx_rand.size(1)))
    h_rand = _entropy_bits(sub_rand)

    h_true_c = h_true.cpu(); h_struct_c = h_struct.cpu(); h_rand_c = h_rand.cpu()
    n_keys_masked = int(keep_struct.numel())  # identical for both masks by construction

    os.makedirs(save_dir, exist_ok=True)
    arm = mgr.prefill_method
    csv_path = os.path.join(save_dir, f"{video_id}__{arm}__masknull.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'video_id', 'arm', 'layer_idx', 'window_idx', 'head_idx',
                'H_true', 'perp_true',
                'H_masked_structured', 'perp_masked_structured',
                'H_masked_random', 'perp_masked_random',
                'n_keys_true', 'n_keys_masked',
                'n_visual_keys', 'n_visual_keys_kept', 'n_text_keys', 'n_prefix_keys',
                'n_queries',
            ])
        for h in range(n_q_heads):
            writer.writerow([
                video_id, arm, mgr.layer_idx, window_idx, h,
                float(h_true_c[h]), float(2.0 ** float(h_true_c[h])),
                float(h_struct_c[h]), float(2.0 ** float(h_struct_c[h])),
                float(h_rand_c[h]), float(2.0 ** float(h_rand_c[h])),
                int(n_keys), n_keys_masked,
                n_vis, n_keep_vis, n_text, n_prefix,
                int(n_queries),
            ])


def _record_entropy_only(mgr, score_list):
    """Entropy/perplexity rows for windows that don't tile into dense frames
    (compressed methods). n_init_keys=0: post-init_exc local windows hold no sink keys."""
    entropy_bits, num_keys = _compute_attention_entropy(score_list[0], 0)
    video_id = os.environ.get('CURRENT_VIDEO_ID', 'unknown')
    save_dir = os.environ.get('ATTN_SCORES_DIR')
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    csv_path = f"{save_dir}/{video_id}_attention_stats.csv"
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(['video_id', 'layer_idx', 'window_idx', 'head_idx',
                        'mean_frame_attended', 'std_frame_attended', 'num_frames',
                        'entropy_bits', 'num_keys', 'perplexity',
                        'frames_at_50', 'frames_at_80', 'frames_at_90'])
        for h in range(entropy_bits.shape[0]):
            b = float(entropy_bits[h].cpu())
            w.writerow([video_id, mgr.layer_idx, mgr.num_blocks_seen, h,
                        '', '', '', b, num_keys, 2.0 ** b, '', '', ''])
