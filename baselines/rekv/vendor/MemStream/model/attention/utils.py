import torch
from einops import rearrange

import numpy as np
from scipy import ndimage 
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

# ============================================================================
# Helper functions for prominence strategy
# ============================================================================

def smooth_gaussian(x, sigma=1):
    # Apply gaussian smoothing FIRST, then min-max normalization (matching notebook)
    x = np.asarray(x, float)
    
    if sigma == 0:
        # If no smoothing, just normalize
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-6:
            return (x - x_min) / (x_max - x_min)
        else:
            return x
    
    # Smooth first (using gaussian_filter to match notebook exactly)
    smoothed = gaussian_filter(x, sigma=sigma)
    
    # Then normalize
    x_min, x_max = smoothed.min(), smoothed.max()
    if x_max - x_min > 1e-6:
        return (smoothed - x_min) / (x_max - x_min)
    else:
        return smoothed


def watershed_1d(x, min_dist=5):
    x = np.asarray(x, float)
    mins, _ = find_peaks(-x, distance=min_dist)
    if len(mins) == 0:
        return np.zeros(len(x), dtype=int)
    
    markers = np.zeros(len(x), dtype=int)
    markers[mins] = np.arange(1, len(mins) + 1)
    
    x_norm = x - x.min()
    if x_norm.max() > 0:
        x_norm /= x_norm.max()
    x_uint = (x_norm * 65535).astype(np.uint16)
    
    return ndimage.watershed_ift(x_uint, markers)


def extract_basins(sims, labels):
    basins = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        
        basins.append({
            "idx": idx,
            "score": sims[idx].max(),
        })
    return basins

def select_topk_watershed(avg_sims, frame_budget, **kwargs):
    """Top-k: Select top-k frames by similarity score.
    
    Returns:
        dict: {
            'frame_indices': list of selected frame indices,
            'basins': list of selected basin dicts (sorted by score),
            'all_basins': list of all basins (sorted by score)
        }
    """
    sigma = kwargs.get('sigma', 1)
    min_dist = kwargs.get('min_dist', 5)
    sampling_ratio = kwargs.get('sampling_ratio', 2)
    adaptive = kwargs.get('adaptive', False)
    min_ratio = kwargs.get('min_ratio', 1)
    max_ratio = kwargs.get('max_ratio', 4)
    
    smoothed = smooth_gaussian(avg_sims, sigma=sigma)
    labels = watershed_1d(smoothed, min_dist=min_dist)
    basins = extract_basins(smoothed, labels)
    
    # Sort basins by score descending
    sorted_basins = sorted(basins, key=lambda x: x['score'], reverse=True)
    
    # Select top-k basins that fit within frame budget
    selected_basins = []
    selected_indices = []
    total_frames = 0
    
    for basin in sorted_basins:
        if total_frames + basin['length'] <= frame_budget:
            selected_basins.append(basin)
            selected_indices.extend(basin['idx'])
            total_frames += basin['length']
        else:
            break
    
    return {
        'frame_indices': np.sort(np.array(selected_indices)).tolist(),
        'basins': selected_basins,
        'all_basins': sorted_basins
    }


def select_topk_watershed_mmr(avg_sims, frame_budget, **kwargs):
    """MMR-based: Select basins using Maximal Marginal Relevance for diversity.
    
    Returns:
        dict: {
            'frame_indices': list of selected frame indices,
            'basins': list of selected basin dicts (in MMR selection order),
            'all_basins': list of all basins (sorted by score)
        }
    """
    sigma = kwargs.get('sigma', 1)
    min_dist = kwargs.get('min_dist', 5)
    alpha = kwargs.get('alpha', 0.5)
    min_ratio = kwargs.get('min_ratio', 1)
    max_ratio = kwargs.get('max_ratio', 2)
    redundancy_sigma = kwargs.get('redundancy_sigma', 20)
    
    smoothed = smooth_gaussian(avg_sims, sigma=sigma)
    labels = watershed_1d(smoothed, min_dist=min_dist)
    basins = extract_basins(smoothed, labels)
    
    # Sort all basins by score for reference
    sorted_basins = sorted(basins, key=lambda x: x['score'], reverse=True)
    
    # Select basins using MMR (returns selected frame indices)
    if kwargs.get('adaptive', False):
        frame_indices, total_frames, selected_scores, selected_centers = select_top_k_basins_mmr_adaptive(smoothed, basins, frame_budget, alpha=alpha,
        redundancy_sigma=redundancy_sigma,
        min_ratio=min_ratio,
        max_ratio=max_ratio)
    else:
        frame_indices = select_top_k_basins_mmr(basins, frame_budget, alpha=alpha,
        redundancy_sigma=redundancy_sigma)
    
    # Reconstruct which basins were selected (in order they were picked)
    frame_indices_set = set(frame_indices.tolist())
    selected_basins = []
    for basin in basins:
        if any(idx in frame_indices_set for idx in basin['idx']):
            selected_basins.append(basin)
    
    return {
        'frame_indices': frame_indices.tolist(),
        'basins': selected_basins,
        'all_basins': sorted_basins
    }

def watershed_1d(x, min_dist=5):
    x = np.asarray(x, float)

    mins, _ = find_peaks(-x, distance=min_dist)
    if len(mins) == 0:
        return np.zeros(len(x), dtype=int)

    markers = np.zeros(len(x), dtype=int)
    markers[mins] = np.arange(1, len(mins) + 1)

    x_norm = x - x.min()
    if x_norm.max() > 0:
        x_norm /= x_norm.max()
    x_uint = (x_norm * 65535).astype(np.uint16)

    return ndimage.watershed_ift(x_uint, markers)

def extract_basins(sims, labels):
    basins = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue

        basins.append({
            "idx": idx,
            "center": idx.mean(),
            "length": len(idx),
            "score": sims[idx].max(),
        })
    return basins

def select_top_k_basins(basins, frame_budget):
    # Sort basins by score descending (highest similarity first)
    sorted_basins = sorted(basins, key=lambda x: x['score'], reverse=True)
    
    selected_indices = []
    total_frames = 0
    
    for basin in sorted_basins:
        # Check if adding the whole basin exceeds our frame budget
        if total_frames + basin['length'] <= frame_budget:
            selected_indices.extend(basin['idx'])
            total_frames += basin['length']
        else:
            # Stop once the next best segment would put us over the budget
            break
            
    return np.sort(np.array(selected_indices))


def select_top_k_basins_mmr(basins, frame_budget, alpha=0.5, redundancy_sigma=20):
    """
    MMR-based basin selection for diversity and coverage (optimized).
    
    alpha: 1.0 = Pure Top-K (Recall focus), 0.0 = Pure Diversity (Coverage focus)
    redundancy_sigma: Gaussian kernel sigma for temporal redundancy. Larger values penalize wider areas.
    """
    if not basins:
        return np.array([])
    
    # Pre-compute basin properties as numpy arrays for vectorization
    n_basins = len(basins)
    scores = np.array([b['score'] for b in basins])
    centers = np.array([b['center'] for b in basins])
    lengths = np.array([b['length'] for b in basins])
    
    # Track which basins are still available
    available = np.ones(n_basins, dtype=bool)
    selected_basin_indices = []
    selected_indices = []
    total_frames = 0
    
    # Pre-compute the Gaussian kernel denominator
    gauss_denom = 2 * redundancy_sigma ** 2
    
    while total_frames < frame_budget and np.any(available):
        # Get indices of available basins
        avail_idx = np.where(available)[0]
        
        # Compute relevance scores (already normalized)
        relevance = scores[avail_idx]
        
        # Compute redundancy
        if len(selected_basin_indices) == 0:
            redundancy = np.zeros(len(avail_idx))
        else:
            # Vectorized distance calculation
            # Shape: (n_available, n_selected)
            selected_centers = centers[selected_basin_indices]
            dists = np.abs(centers[avail_idx, np.newaxis] - selected_centers[np.newaxis, :])
            
            # Gaussian kernel for redundancy, take max across all selected
            redundancy_matrix = np.exp(-(dists ** 2) / gauss_denom)
            redundancy = np.max(redundancy_matrix, axis=1)
        
        # MMR Score Equation (vectorized)
        mmr_scores = alpha * relevance - (1 - alpha) * redundancy
        
        # Pick the winner
        best_local_idx = np.argmax(mmr_scores)
        best_basin_idx = avail_idx[best_local_idx]
        
        # Budget check
        if total_frames + lengths[best_basin_idx] <= frame_budget:
            selected_indices.extend(basins[best_basin_idx]['idx'])
            selected_basin_indices.append(best_basin_idx)
            total_frames += lengths[best_basin_idx]
            available[best_basin_idx] = False
        else:
            # Mark as unavailable and continue
            available[best_basin_idx] = False
            
    return np.sort(np.array(selected_indices))


def select_top_k_basins_mmr_adaptive(layer_similarity, basins, frame_budget, 
                                     alpha=0.5, redundancy_sigma=20, 
                                     min_ratio=1, max_ratio=4):
    """
    MMR-based basin selection with adaptive sampling based on basin contrast (optimized).
    
    alpha: 1.0 = Pure Top-K, 0.0 = Pure Diversity
    redundancy_sigma: Gaussian kernel sigma for temporal redundancy
    min_ratio: Dense sampling for high-contrast peaks (e.g., 1)
    max_ratio: Sparse sampling for low-contrast/flat peaks (e.g., 4)
    """
    if not basins:
        return np.array([]), 0, [], []
    
    sig = layer_similarity
    n_basins = len(basins)
    
    # Pre-compute basin properties as numpy arrays for vectorization
    scores = np.array([b['score'] for b in basins])
    centers = np.array([b['center'] for b in basins])
    
    # Pre-calculate dynamic range (contrast) for each basin - vectorized
    contrasts = np.array([np.max(sig[b['idx']]) - np.min(sig[b['idx']]) for b in basins])
    
    # Normalize contrasts for the whole layer to scale ratios between min/max
    c_min, c_max = contrasts.min(), contrasts.max()
    norm_contrasts = (contrasts - c_min) / (c_max - c_min + 1e-6)
    
    # Pre-compute adaptive ratios for all basins
    # High contrast -> norm_c = 1.0 -> ratio = min_ratio (Dense)
    # Low contrast  -> norm_c = 0.0 -> ratio = max_ratio (Sparse)
    adaptive_ratios = np.round(max_ratio - (norm_contrasts * (max_ratio - min_ratio))).astype(int)
    
    # Track which basins are still available
    available = np.ones(n_basins, dtype=bool)
    selected_basin_indices = []
    selected_indices = []
    selected_scores = []
    total_frames = 0
    
    # Pre-compute the Gaussian kernel denominator
    gauss_denom = 2 * redundancy_sigma ** 2
    
    while total_frames < frame_budget and np.any(available):
        # Get indices of available basins
        avail_idx = np.where(available)[0]
        
        # Compute relevance scores (already normalized)
        relevance = scores[avail_idx]
        
        # Compute redundancy
        if len(selected_basin_indices) == 0:
            redundancy = np.zeros(len(avail_idx))
        else:
            # Vectorized distance calculation
            # Shape: (n_available, n_selected)
            selected_centers = centers[selected_basin_indices]
            dists = np.abs(centers[avail_idx, np.newaxis] - selected_centers[np.newaxis, :])
            
            # Gaussian kernel for redundancy, take max across all selected
            redundancy_matrix = np.exp(-(dists ** 2) / gauss_denom)
            redundancy = np.max(redundancy_matrix, axis=1)
        
        # MMR Score Equation (vectorized)
        mmr_scores = alpha * relevance - (1 - alpha) * redundancy
        
        # Pick the winner
        best_local_idx = np.argmax(mmr_scores)
        best_basin_idx = avail_idx[best_local_idx]
        
        # Get adaptive ratio for this basin
        adaptive_ratio = adaptive_ratios[best_basin_idx]
        
        # Sample the indices
        basin_idx = basins[best_basin_idx]['idx']
        sampled_idx = basin_idx[::adaptive_ratio]
        num_sampled = len(sampled_idx)
        
        # Budget check
        if total_frames + num_sampled <= frame_budget:
            selected_indices.extend(sampled_idx)
            selected_basin_indices.append(best_basin_idx)
            selected_scores.append(mmr_scores[best_local_idx])
            total_frames += num_sampled
            available[best_basin_idx] = False
        else:
            # Mark as unavailable and continue
            available[best_basin_idx] = False
            
    selected_centers = centers[selected_basin_indices].tolist()
    return np.sort(np.array(selected_indices)), total_frames, selected_scores, selected_centers

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def bipartite_soft_matching(k: torch.Tensor, r: int) -> torch.Tensor:
    """
    Input is keys from attention, size (b*nh*m, h*w, d)
    r is the reduction ratio
    """
    batch_size, num_tokens, d = k.shape

    k = k / k.norm(dim=-1, keepdim=True)

    # Create index mapping for even and odd tokens
    even_indices = list(range(0, num_tokens, 2))
    odd_indices = list(range(1, num_tokens, 2))

    num_even = len(even_indices)  # Handle odd num_tokens correctly
    num_odd = len(odd_indices)

    # Split tokens while keeping track of original indices
    # a, b = (b*nh, m, h*w / 2, d), (b*nh, m, h*w / 2, d)
    a, b = k[..., even_indices, :], k[..., odd_indices, :]

    scores = torch.matmul(a, b.transpose(-1, -2)) # (b*nh*m, num_even, num_odd)

    node_max, node_idx = scores.max(dim=-1) # (b*nh*m, num_even)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # (b*nh*m, num_even, 1)

    unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens # (b*nh*m, num_even - r, 1)
    src_idx = edge_idx[..., :r, :]  # Merged Tokens # (b*nh*m, r, 1)
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # (b*nh*m, r, 1)

    def merge(x: torch.Tensor) -> torch.Tensor:
        """Input is of shape - (b, n, d) ~ (b*nh*m, h*w, d)"""

        # Split using the same even/odd indices
        src, dst = x[..., even_indices, :], x[..., odd_indices, :] # (b*nh*m, num_even, d_x), (b*nh*m, num_odd, d_x)
        
        d_x = x.shape[-1]  # Get actual feature dimension from input

        unm = src.gather(dim=-2, index=unm_idx.expand(batch_size, num_even - r, d_x)) # (b*nh*m, num_even - r, d_x)
        src = src.gather(dim=-2, index=src_idx.expand(batch_size, r, d_x)) # (b*nh*m, r, d_x)

        # Create a mask for positions that will be merged
        merge_mask = torch.zeros_like(dst, dtype=dst.dtype)
        merge_mask.scatter_(-2, dst_idx.expand(batch_size, r, d_x), 1.0)

        # Count how many tokens are merged into each position
        merge_counts = torch.zeros_like(dst[..., 0], dtype=torch.float32)
        merge_counts.scatter_add_(-1, dst_idx[..., 0],
                        torch.ones_like(src_idx[..., 0], dtype=torch.float32))
        merge_counts = merge_counts.unsqueeze(-1) + 1.0  # Add 1 for the destination token itself
        # Add the features
        dst = dst.scatter_add(-2, dst_idx.expand(batch_size, r, d_x), src)
        # Divide by the actual number of merged tokens
        dst = dst * torch.where(merge_mask > 0, 1.0/merge_counts, 1.0)

        return torch.cat([unm, dst], dim=-2)

    return merge

def get_bipartite_soft_matching(keys, values, merge_ratio, num_iters=1):
    """
    Apply iterative bipartite soft matching to merge tokens.
    
    Args:
        keys: (b, nh, m, h, w, d) - key features
        values: (b, nh, m, h, w, d) - value features
        merge_ratio: percentage of tokens to merge per iteration
        num_iters: number of merge iterations
    
    Returns:
        merged_keys: (b, nh, m, new_hw, d) - merged keys across frames 
        merged_values: (b, nh, m, new_hw, d) - merged values across frames 
    """
    b, nh, m, h, w, d = keys.shape

    keys = rearrange(keys, "b nh m h w d -> (b nh m) (h w) d") # (b*nh*m, h*w, d)
    values = rearrange(values, "b nh m h w d -> (b nh m) (h w) d") # (b*nh*m, h*w, d)

    # float32, NOT keys.dtype: the grid holds patch INDICES and is averaged through
    # the merges, so it must represent them exactly. bf16 has an 8-bit mantissa and is
    # exact only up to 256 -- fine at Qwen2.5's <=256 patches/frame, but at Qwen3's 510
    # (17x30) every odd index above 256 rounds to even, quantizing merged-token RoPE
    # positions over the back half of each frame. fp32 is exact to 2^24. Qwen2.5 results
    # are unchanged (bf16 was already exact in that range).
    h_coords = torch.arange(h, device=keys.device, dtype=torch.float32)
    w_coords = torch.arange(w, device=keys.device, dtype=torch.float32)
    h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij') # (h, w), (h, w)
    hw_grid = h_grid.flatten() * w + w_grid.flatten()  # (h*w,)
    hw_grid = hw_grid[None, :].expand(b*nh*m, -1).unsqueeze(-1) # (b*nh*m, h*w, 1)

    for _ in range(num_iters):
        num_tokens = keys.shape[1]
        num_merge_tokens = int(num_tokens * merge_ratio)
        print(f"num_merge_tokens: {num_merge_tokens}")
        merge_fn = bipartite_soft_matching(keys.clone(), num_merge_tokens)
        keys = merge_fn(keys)
        values = merge_fn(values)
        hw_grid = merge_fn(hw_grid)

    # merged k/v frame-wise features 
    keys = rearrange(keys, '(b nh m) n d -> b nh m n d', b=b, nh=nh, m=m) # (b, nh, m, n, d)
    values = rearrange(values, '(b nh m) n d -> b nh m n d', b=b, nh=nh, m=m) # (b, nh, m, n, d)

    # represents the position ids of the merged tokens in the original frame-wise features 
    hw_grid = rearrange(hw_grid, '(b nh m) n d -> b nh m n d', b=b, nh=nh, m=m) # (b, nh, m, n, 1)
    return keys, values, hw_grid