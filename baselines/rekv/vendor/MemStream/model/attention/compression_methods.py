"""
Attention compression methods for efficient video processing.

Each compression method must follow this signature:
    Args:
        local_k: (batch, n_head_kv, total_len, dim_head) - key cache
        local_v: (batch, n_head_kv, total_len, dim_head) - value cache
        local_q: (batch, n_head, len_q, dim_head) - current query
        n_init: int - number of initial sink tokens
        init_exc: bool - whether init cache is full
        grid_hw: tuple(int, int) - (height, width) of spatial grid
        **kwargs: method-specific parameters
    
    Returns:
        tuple: (compressed_k, compressed_v, subsample_mask, sliding_window)
            - compressed_k: (batch, n_head_kv, compressed_len, dim_head)
            - compressed_v: (batch, n_head_kv, compressed_len, dim_head)
            - subsample_mask: (m, h, w) bool tensor or None
            - sliding_window: int - effective window size

How to add a new compression method:
    1. Define a function here with the above signature
    2. Import it in kv_cache_manager.py
    3. Add a case in ContextManager._apply_compression() to call your function
    4. Use it by setting prefill_method="swa_<your_method>_<params>"

Example:
    def apply_my_compression(local_k, local_v, local_q, n_init, init_exc, grid_hw, my_param):
        # Your compression logic here
        ...
        return compressed_k, compressed_v, subsample_mask, sliding_window
"""
import torch
import torch.nn.functional as F
from einops import rearrange
import math
from ezcolorlog import root_logger as logger

from model.attention.utils import get_bipartite_soft_matching

def apply_clustered_compression(
    local_k, local_v, local_q, 
    n_init, init_exc, grid_hw, 
    n_clusters, n_iters, nd=1
): 
    """ Apply clustered compression to previous frames. Specifically, 
        cluster key features frame-wise and get associated value features. 
    
    Args: 
        local_k: (batch, n_head_kv, total_len, dim_head)
        local_v: (batch, n_head_kv, total_len, dim_head)
        local_q: (batch, n_head, len_q, dim_head)
        n_init: number of initial sink tokens
        init_exc: whether init cache is full
        grid_hw: (height, width) of spatial grid
        n_clusters: number of clusters 
        n_iters: number of iterations 

    Returns: 
        tuple: (compressed_k, compressed_v, subsample_mask, sliding_window)
            - compressed_k: (batch, n_head_kv, compressed_len, dim_head)
            - compressed_v: (batch, n_head_kv, compressed_len, dim_head)
            - subsample_mask: (batch, n_head_kv, M, H, W) - per-head mask for RoPE
            - sliding_window: int - length of compressed sequence
    """

    def calculate_pooled_dist(X, centroids, grid_hw, d, pool_factor='full'): 

        """ compute distance on reduced dimension of X and centroids """

        B, m, D = X.shape 
        B, nc, D = centroids.shape 

        X_ = rearrange(X, "B m (h w d) -> (B m) d h w", h=grid_hw[0], w=grid_hw[1], d=d)
        C_ = rearrange(centroids, "B nc (h w d) -> (B nc) d h w", h=grid_hw[0], w=grid_hw[1], d=d)

        if pool_factor == "full":
            X_ = rearrange(X_.mean(dim=(-2, -1)), "(B m) d -> B m d", B=B, m=m)
            C_ = rearrange(C_.mean(dim=(-2, -1)), "(B nc) d -> B nc d", B=B, nc=nc)

        else: 
            r = int(math.sqrt(pool_factor))  # Convert ratio to stride
            X_ = F.avg_pool2d(X_, kernel_size=r, stride=r)
            C_ = F.avg_pool2d(C_, kernel_size=r, stride=r)

            X_ = rearrange(X_, "(B m) d h w -> B m (h w d)", B=B, m=m)
            C_ = rearrange(C_, "(B nc) d h w -> B nc (h w d)", B=B, nc=nc)

        dists = torch.cdist(X_.float(), C_.float())  # (B, m, nc)
        return dists

    len_q = local_q.size(-2) * nd

    st = 0 if init_exc else n_init
    pre_k, sparse_k, dense_k = local_k[:, :, :st, :], local_k[:, :, st:-len_q, :], local_k[:, :, -len_q:, :]
    pre_v, sparse_v, dense_v = local_v[:, :, :st, :], local_v[:, :, st:-len_q, :], local_v[:, :, -len_q:, :]

    # reshape sparse_k/sparse_v into frame-wise features 
    sparse_k = rearrange(sparse_k, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])
    sparse_v = rearrange(sparse_v, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])
    b, nh, m, h, w, d = sparse_k.shape
    D = h * w * d

    X = rearrange(sparse_k, "b nh m h w d -> b nh m (h w d)")    # (b, nh, m, D)
    Xv =  rearrange(sparse_v, "b nh m h w d -> b nh m (h w d)")  # (b, nh, m, D)

    # init centroids 
    rand_idx = torch.randperm(m, device=X.device)[:n_clusters]
    centroids = X[:, :, rand_idx, :]  # (b, nh, n_clusters, D)
    assoc = torch.zeros(b * nh, n_clusters, m, device=X.device, dtype=X.dtype) # (b*nh, n_clusters, m)

    for _ in range(n_iters):

        # reshape for cdist
        X_ = X.reshape(b * nh, m, D) # (b*nh, m, D)
        C_ = centroids.reshape(b * nh, n_clusters, D) # (b*nh, n_clusters, D)
        
        dists = calculate_pooled_dist(X_, C_, grid_hw, d, pool_factor='full')  # (b*nh, m, n_clusters)
        labels = dists.argmin(dim=-1)  # (b*nh, m) int64

        # Ensure no empty clusters by splitting the largest cluster when needed
        # This keeps k-means stable without leaving any cluster without members.
        counts_tmp = torch.zeros(b * nh, n_clusters, device=X.device, dtype=torch.int64)
        for k in range(n_clusters):
            counts_tmp[:, k] = (labels == k).sum(dim=1)

        for idx in range(b * nh):
            empty_clusters = (counts_tmp[idx] == 0).nonzero(as_tuple=False).flatten()
            if empty_clusters.numel() == 0:
                continue
            for empty_k in empty_clusters:
                largest_k = counts_tmp[idx].argmax()
                largest_members = (labels[idx] == largest_k).nonzero(as_tuple=False).flatten()
                if largest_members.numel() <= 1:
                    # Cannot split further; leave as is
                    continue
                split_size = max(1, largest_members.numel() // 2)
                perm = torch.randperm(largest_members.numel(), device=X.device)
                move_indices = largest_members[perm[:split_size]]
                labels[idx, move_indices] = empty_k
                counts_tmp[idx, largest_k] -= split_size
                counts_tmp[idx, empty_k] += split_size
        
        # update centroids 
        new_centroids = torch.zeros_like(C_)
        counts = torch.zeros(b * nh, n_clusters, device=X.device) # (b*nh, n_clusters) 

        for k in range(n_clusters):
            mask = labels == k  # (b*nh, m)
            counts[:, k] = mask.sum(dim=1)
            new_centroids[:, k] = (X_ * mask.unsqueeze(-1)).sum(dim=1)
            assoc[:, k] = mask 

        # Avoid division by zero for empty clusters
        counts = counts.clamp_min(1.0)
        new_centroids = new_centroids / counts.unsqueeze(-1)
        centroids = new_centroids.reshape(b, nh, n_clusters, D)

    # assert no empty clusters
    assert counts.min() > 0, "Empty clusters found"

    # calculate sparse_v_clustered 
    Xv_ = Xv.reshape(b * nh, m, D)
    weights = assoc / counts.unsqueeze(-1) 
    weights = weights.to(Xv_.device).to(Xv_.dtype)

    # (b*nh, n_clusters, m) @ (b*nh, m, D) -> (b*nh, n_clusters, D)
    sparse_v_clustered = torch.einsum("bcm,bmd->bcd", weights, Xv_)  # (b*nh, n_clusters, D)
    sparse_v_clustered = rearrange(
        sparse_v_clustered,
        "(b nh) nc (h w d) -> b nh nc h w d",
        b=b, nh=nh, h=h, w=w, d=d
    )

    # Derive effective frame indices for each cluster by averaging frame indices
    # under the soft assignment weights, per head (each head has its own clusters)
    frame_ids = torch.arange(m, device=assoc.device, dtype=torch.float32)  # (m,)
    eff_frame_idx = (weights * frame_ids.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (b*nh, n_clusters)
    eff_frame_idx = eff_frame_idx.reshape(b, nh, n_clusters)  # (b, nh, n_clusters)

    # Sort clusters by temporal position for each head independently
    sort_idx = eff_frame_idx.argsort(dim=-1)  # (b, nh, n_clusters)

    # Reorder clustered k/v along the cluster dimension per head
    # Use gather to sort each head independently
    sort_idx_expanded = sort_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (b, nh, n_clusters, D)
    centroids = torch.gather(centroids, dim=2, index=sort_idx_expanded)  # (b, nh, n_clusters, D)
    sparse_k_clustered = rearrange(centroids, "b nh nc (h w d) -> b nh nc h w d", h=h, w=w, d=d)
    
    sort_idx_expanded_v = sort_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w, d)  # (b, nh, n_clusters, h, w, d)
    sparse_v_clustered = torch.gather(sparse_v_clustered, dim=2, index=sort_idx_expanded_v)  # (b, nh, n_clusters, h, w, d)

    # create compressed_k/v (clusters are now temporally ordered)
    sparse_k_clustered = rearrange(sparse_k_clustered, "b nh m h w d -> b nh (m h w) d")
    sparse_v_clustered = rearrange(sparse_v_clustered, "b nh m h w d -> b nh (m h w) d")
    compressed_local_k = torch.cat([pre_k, sparse_k_clustered, dense_k], dim=-2).to(local_k.dtype)
    compressed_local_v = torch.cat([pre_v, sparse_v_clustered, dense_v], dim=-2).to(local_v.dtype)

    # Build per-head RoPE subsample masks
    # Each head gets its own mask based on its cluster's effective frame indices
    rep_idx, idx = torch.sort(eff_frame_idx.floor().long(), dim=-1)
    k = rep_idx.size(-1)

    rep_idx = torch.gather(
        torch.cummax(
            rep_idx - torch.arange(k, device=rep_idx.device),
            dim=-1
        ).values
        + torch.arange(k, device=rep_idx.device),
        -1,
        idx.argsort(dim=-1),
    ).clamp(0, m-1)

    for h in range(nh):
        assert torch.unique(rep_idx[:, h, :]).numel() == n_clusters, f"Head {h} has {torch.unique(rep_idx[:, h, :]).numel()} unique clusters, expected {n_clusters}"
    
    subsample_mask = torch.zeros(b, nh, m, grid_hw[0], grid_hw[1], dtype=torch.bool, device=local_k.device)
    
    # Set mask for each head independently
    for batch_i in range(b):
        for head_i in range(nh):
            subsample_mask[batch_i, head_i, rep_idx[batch_i, head_i], :, :] = True

    sliding_window = compressed_local_k.shape[-2]

    return compressed_local_k, compressed_local_v, subsample_mask, sliding_window


def apply_pooled_compression(
    local_k, local_v, local_q,
    n_init, init_exc, grid_hw, 
    pool_ratio, use_max_pool=False, nd=1
):
    """
    Apply average or max pooling compression to previous frames.
    
    Args:
        local_k: (batch, n_head_kv, total_len, dim_head)
        local_v: (batch, n_head_kv, total_len, dim_head)
        local_q: (batch, n_head, len_q, dim_head)
        n_init: number of initial sink tokens
        init_exc: whether init cache is full
        grid_hw: (height, width) of spatial grid
        pool_ratio: pooling ratio (e.g., 4 means 2x2 pooling)
        use_max_pool: if True, use max pooling; otherwise use average pooling
        nd: multiplier for determining number of dense frames (default: 1)
    
    Returns:
        tuple: (compressed_k, compressed_v, subsample_mask, sliding_window)
    """
    len_q = local_q.size(-2) * nd
    r = int(math.sqrt(pool_ratio))  # Convert ratio to stride
    
    st = 0 if init_exc else n_init
    pre_k, sparse_k, dense_k = local_k[:, :, :st, :], local_k[:, :, st:-len_q, :], local_k[:, :, -len_q:, :]
    pre_v, sparse_v, dense_v = local_v[:, :, :st, :], local_v[:, :, st:-len_q, :], local_v[:, :, -len_q:, :]

    # Pool sparse k/v
    sparse_k = rearrange(sparse_k, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])
    b, nh, m, h, w, d = sparse_k.shape
    pool_fn = F.max_pool2d if use_max_pool else F.avg_pool2d
    
    def pool_tensor(t):
        t = rearrange(t, "b nh m h w d -> b (nh m d) h w")
        t = pool_fn(t, kernel_size=r, stride=r)
        return rearrange(t, "b (nh m d) h w -> b nh (m h w) d", nh=nh, m=m, d=d)
    
    sparse_k = pool_tensor(sparse_k)
    sparse_v = rearrange(sparse_v, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])
    sparse_v = pool_tensor(sparse_v)

    compressed_local_k = torch.cat([pre_k, sparse_k, dense_k], dim=-2)
    compressed_local_v = torch.cat([pre_v, sparse_v, dense_v], dim=-2)

    # Create stride-based spatial mask
    Hf, Wf = (h // r) * r, (w // r) * r
    logger.info(f"Hf: {Hf}, Wf: {Wf}, h: {h}, w: {w}, r: {r}")
    spatial_mask = torch.zeros(h, w, dtype=torch.bool, device=local_k.device)
    spatial_mask[:Hf, :Wf] = (torch.arange(Hf, device=local_k.device) % r == 0)[:, None] & (torch.arange(Wf, device=local_k.device) % r == 0)[None, :]
    subsample_mask = spatial_mask.unsqueeze(0).expand(m, -1, -1)

    sliding_window = compressed_local_k.shape[-2]
    
    return compressed_local_k, compressed_local_v, subsample_mask, sliding_window

def apply_kk_guided_compression(
    local_k, local_v, local_q,
    n_init, init_exc, grid_hw,
    method, **kwargs
):
    """
    Apply KK-guided compression to previous frames.
    Notes 
    - local_q is not actually used, but rather used for truncating current frame via len_q
    
    Args:
        local_k: (batch, n_head_kv, total_len, dim_head)
        local_v: (batch, n_head_kv, total_len, dim_head)
        local_q: (batch, n_head, len_q, dim_head) 
        n_init: number of initial sink tokens
        init_exc: whether init cache is full
        grid_hw: (height, width) of spatial grid
        method: method for selecting KK, supports full or adjacent 
        kwargs: additional arguments for the method
            dilation_rate: dilation rate for reduction/subsampling
            nd: multiplier for determining number of dense frames (default: 1)
            dense_len: explicit trailing dense-token length. If omitted, use
                local_q.size(-2) * nd for backwards compatibility.
            top_p: bool for top-p sampling
            keep_k: explicit per-frame keep-count, overriding the stride-derived budget
                (used by swa_kk_keep_<s>). None keeps the stride behaviour unchanged.
    """
    nd = kwargs.get('nd', 1)
    len_q = kwargs.get('dense_len', local_q.size(-2) * nd)

    st = 0 if init_exc else n_init
    pre_k, sparse_k, dense_k = local_k[:, :, :st, :], local_k[:, :, st:-len_q, :], local_k[:, :, -len_q:, :]
    pre_v, sparse_v, dense_v = local_v[:, :, :st, :], local_v[:, :, st:-len_q, :], local_v[:, :, -len_q:, :]

    # rearrange sparse_k/v into frame-wise features 
    sparse_k = rearrange(sparse_k, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])
    b, nh, m, h, w, d = sparse_k.shape
    sparse_v = rearrange(sparse_v, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])

    if method == "full":
        raise NotImplementedError("Full method not implemented")

    elif method == "vdiv":
        # VALUE-DIVERSITY SELECTION.
        # eff_distinct = 1 / || sum_i vhat_i ||^2, so maximising the effective number
        # of independent directions means minimising the resultant's norm. A token's
        # projection onto that resultant is EXACTLY its total cosine similarity to
        # every other token in the frame:
        #     vhat_i . (sum_j vhat_j) = sum_j cos(v_i, v_j)
        # so "drop the most redundant" needs one matmul, not a greedy O(m*n) loop.
        # This is the criterion kk_adjacent should have used: it scores VALUES, which
        # is what the attention output sum(p*v) is actually built from, rather than
        # key novelty against the previous frame.
        dilation_rate = kwargs['dilation_rate']
        sparse_k = rearrange(sparse_k, "b nh m h w d -> b nh m (h w) d")
        sparse_v = rearrange(sparse_v, "b nh m h w d -> b nh m (h w) d")

        keep_k = kwargs.get('keep_k')
        k = int(keep_k) if keep_k is not None else max(1, (h * w) // (dilation_rate ** 2))
        k = max(1, min(k, h * w))

        vn = F.normalize(sparse_v.float(), dim=-1)                  # (b,nh,m,hw,d)
        resultant = vn.sum(dim=3, keepdim=True)                     # (b,nh,m,1,d)
        centrality = (vn * resultant).sum(-1)                       # (b,nh,m,hw)
        # subtract self-similarity (=1) so a token is not penalised for itself
        centrality = centrality - 1.0
        # ROUNDS>1 recomputes the resultant after each partial drop, so the criterion
        # is not blinded by whatever direction dominated the uncompressed frame.
        rounds = int(kwargs.get('vdiv_rounds', 1))
        if rounds > 1 and k < h * w:
            alive = torch.ones_like(centrality, dtype=torch.bool)
            n_now = h * w
            for _r in range(rounds):
                tgt = int(round(n_now - (n_now - k) * (_r + 1) / rounds))
                tgt = max(k, min(tgt, n_now))
                cur = vn * alive.unsqueeze(-1).to(vn.dtype)
                res = cur.sum(dim=3, keepdim=True)
                cen = (cur * res).sum(-1) - alive.float()
                # +inf, not -inf: selection below keeps the LOWEST centrality, so a
                # dead token marked -inf would be resurrected every round.
                cen = torch.where(alive, cen, torch.full_like(cen, float('inf')))
                keep_idx = cen.topk(tgt, dim=-1, largest=False).indices
                alive = torch.zeros_like(alive).scatter_(3, keep_idx, True)
            centrality = torch.where(alive, centrality, torch.full_like(centrality, float('inf')))

        sampled_indices = centrality.topk(k, dim=-1, largest=False).indices.long()
        # raster order, same rationale as the kk keep_k path: makes the k==h*w gate
        # bit-exact against the uncompressed window.
        sampled_indices = sampled_indices.sort(dim=-1).values

        idx = sampled_indices[..., None].expand(-1, -1, -1, -1, d)
        sparse_k = torch.gather(sparse_k, dim=3, index=idx)
        sparse_v = torch.gather(sparse_v, dim=3, index=idx)
        sparse_k = rearrange(sparse_k, "b nh m n d -> b nh (m n) d")
        sparse_v = rearrange(sparse_v, "b nh m n d -> b nh (m n) d")

        subsample_mask = torch.zeros(b, nh, m, h*w, dtype=torch.bool, device=local_k.device)
        subsample_mask.scatter_(dim=3, index=sampled_indices, value=True)
        subsample_mask = subsample_mask.view(b, nh, m, h, w)

    elif method == "adjacent":

        # subsampling scale factor - e.g., r=4 --> 1/16 of tokens attended to 
        dilation_rate = kwargs['dilation_rate']

        sparse_k = rearrange(sparse_k, "b nh m h w d -> b nh m (h w) d")
        sparse_v = rearrange(sparse_v, "b nh m h w d -> b nh m (h w) d")

        # WHICH TENSOR IS SCORED. score_on='v' makes this vv_adj: the exact same
        # adjacent-frame criterion, but measuring novelty on VALUES rather than keys.
        # That isolates key-vs-value as the only difference from swa_kk_adjacent_<r>.
        # Values are never RoPE'd, so the postrope correction is meaningless for them
        # and is refused rather than silently ignored.
        _score_on = kwargs.get('score_on', 'k')
        _src = sparse_v if _score_on == 'v' else sparse_k

        # compute adjacent KK cosine-similarity. With postrope_cos/sin set, the later
        # frame is first rotated by the constant inter-block offset R(delta), so novelty
        # is measured in the rope space attention actually operates in.
        _prc, _prs = kwargs.get('postrope_cos'), kwargs.get('postrope_sin')
        if _prc is not None and _score_on == 'v':
            raise ValueError("postrope correction is key-space only; values carry no RoPE")
        if _prc is not None:
            _c = _prc.to(sparse_k.device, sparse_k.dtype).view(1, 1, 1, 1, -1)
            _s = _prs.to(sparse_k.device, sparse_k.dtype).view(1, 1, 1, 1, -1)
            _kn = sparse_k[:, :, 1:, :, :]
            _k1, _k2 = _kn.chunk(2, dim=-1)
            _kn_rot = _kn * _c + torch.cat((-_k2, _k1), dim=-1) * _s
            sparse_kk_adjacent_sim = (F.normalize(sparse_k[:, :, :-1, :, :], dim=-1) *
                                      F.normalize(_kn_rot, dim=-1)).sum(dim=-1)
        else:
            sparse_k_norm = F.normalize(_src, dim=-1)
            sparse_kk_adjacent_sim = (sparse_k_norm[:, :, 1:, :, :] * sparse_k_norm[:, :, :-1, :, :]).sum(dim=-1) # (b, nh, m-1, h*w)

        keep_k = kwargs.get('keep_k', None)

        if keep_k is None:
            # for the first frame - sample via dilation
            dilated_h_indices = torch.arange(0, h, dilation_rate, device=local_k.device)
            dilated_w_indices = torch.arange(0, w, dilation_rate, device=local_k.device)
            h_grid, w_grid = torch.meshgrid(dilated_h_indices, dilated_w_indices, indexing='ij')
            frame_0_indices = h_grid.flatten()[None, None, :] * w + w_grid.flatten()[None, None, :] # (1, 1, h*w)
            frame_0_indices = frame_0_indices.expand(b, nh, -1).unsqueeze(-2) # (b, nh, 1, h*w)

            # create sample indices
            k = len(dilated_h_indices) * len(dilated_w_indices)
        else:
            # Explicit keep-count (swa_kk_keep_<s>). The selection criterion is unchanged --
            # only the budget handed to topk differs. Frame 0 has no predecessor and so no
            # similarity scores, and needs a fixed index set of exactly k. A separable stride
            # cannot supply one for arbitrary k (17x30 reaches 510/135/60/40, never 255), and
            # an anisotropic stride that does (r_h=1, r_w=2 -> 17*15 = 255) would halve
            # resolution along one axis -- injecting exactly the aspect distortion these arms
            # hold fixed. Use a diagonal-stripe ordering instead: grouping by (row+col) % s
            # and taking the first k is isotropic at every ratio (on 17x30 at s=2 it yields
            # 15 per row and an even 9/8 per column).
            k = int(keep_k)
            if not (1 <= k <= h * w):
                raise ValueError(f"keep_k={k} out of range for grid {h}x{w} ({h * w} patches)")
            rows = torch.arange(h, device=local_k.device).view(h, 1).expand(h, w)
            cols = torch.arange(w, device=local_k.device).view(1, w).expand(h, w)
            stripe = max(1, int(round((h * w) / k)))
            order = torch.argsort(((rows + cols) % stripe).flatten(), stable=True)
            frame_0_indices = order[:k][None, None, :].expand(b, nh, -1).unsqueeze(-2)

        if kwargs.get('top_p', False) and sparse_kk_adjacent_sim.shape[-2] > 0:
            p = kwargs.get('p', 0.5)

            # Use top-p to choose one fixed per-frame budget, keeping tensor shapes simple.
            sim_avg = sparse_kk_adjacent_sim.mean(dim=1)
            dist_avg = (1.0 - sim_avg.clamp(-1.0, 1.0)).float()
            dist_prob = dist_avg / (dist_avg.sum(dim=-1, keepdim=True) + 1e-8)
            sorted_probs, _ = dist_prob.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            nucleus = (cumsum - sorted_probs) < p
            k = int(nucleus.sum(dim=-1).amax().item())
            k = max(1, min(k, h * w))

        if sparse_kk_adjacent_sim.shape[-2] == 0:
            sampled_indices = frame_0_indices.long()
        else:
            sampled_indices = sparse_kk_adjacent_sim.topk(k, dim=-1, largest=False).indices
            sampled_indices = torch.cat([frame_0_indices, sampled_indices], dim=2).long()

        if keep_k is not None:
            # topk returns indices ordered by similarity, not by position, so the kept
            # patches come out permuted within each frame. Positions survive (the Qwen3
            # path carries absolute kept_indices into RoPE), but raster order is the
            # canonical layout every other method produces, and it makes the keep_k=h*w
            # no-op gate bit-exact against the uncompressed window -- a far stronger
            # regression test than "same token count". Left off the stride path so
            # existing swa_kk_adjacent_<r> arms stay byte-for-byte reproducible.
            sampled_indices = sampled_indices.sort(dim=-1).values

        idx = sampled_indices[..., None].expand(-1, -1, -1, -1, d)
        sparse_k = torch.gather(sparse_k, dim=3, index=idx)
        sparse_v = torch.gather(sparse_v, dim=3, index=idx)
        sparse_k = rearrange(sparse_k, "b nh m n d -> b nh (m n) d")
        sparse_v = rearrange(sparse_v, "b nh m n d -> b nh (m n) d")

        subsample_mask = torch.zeros(b, nh, m, h*w, dtype=torch.bool, device=local_k.device)
        subsample_mask.scatter_(dim=3, index=sampled_indices, value=True)
        subsample_mask = subsample_mask.view(b, nh, m, h, w)

    compressed_local_k = torch.cat([pre_k, sparse_k, dense_k], dim=-2)
    compressed_local_v = torch.cat([pre_v, sparse_v, dense_v], dim=-2)

    sliding_window = compressed_local_k.shape[-2]

    return compressed_local_k, compressed_local_v, subsample_mask, sliding_window

def apply_patch_sample_compression(
    local_k, local_v, local_q,
    n_init, init_exc, grid_hw,
    method='dilated', 
    **kwargs,
):
    """
    Apply dilated attention compression to previous frames.
    
    Args:
        local_k: (batch, n_head_kv, total_len, dim_head)
        local_v: (batch, n_head_kv, total_len, dim_head)
        local_q: (batch, n_head, len_q, dim_head)
        n_init: number of initial sink tokens
        init_exc: whether init cache is full
        grid_hw: (height, width) of spatial grid
        method: method for sampling patches, supports dilated or random 
    
    Returns:
        tuple: (compressed_k, compressed_v, subsample_mask, sliding_window)
    """
    nd = kwargs.get('nd', 1)
    len_q = local_q.size(-2) * nd
    
    st = 0 if init_exc else n_init
    pre_k, sparse_k, dense_k = local_k[:, :, :st, :], local_k[:, :, st:-len_q, :], local_k[:, :, -len_q:, :]
    pre_v, sparse_v, dense_v = local_v[:, :, :st, :], local_v[:, :, st:-len_q, :], local_v[:, :, -len_q:, :]

    # Apply dilation to sparse k/v
    sparse_k = rearrange(sparse_k, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])
    b, nh, m, h, w, d = sparse_k.shape
    sparse_v = rearrange(sparse_v, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])
    
    if method == "dilated": 
        # Select tokens with dilation (every r-th token in both h and w)
        r = kwargs['dilation_rate']
        sparse_k, sparse_v, spatial_mask = apply_spatial_dilation(
            sparse_k, sparse_v, h, w, r
        )
        subsample_mask = spatial_mask.unsqueeze(0).expand(m, -1, -1)

        # Reshape back to sequence
        sparse_k = rearrange(sparse_k, "b nh m h w d -> b nh (m h w) d")
        sparse_v = rearrange(sparse_v, "b nh m h w d -> b nh (m h w) d")
        
    elif method == "random": 

        sparse_k = rearrange(sparse_k, "b nh m h w d -> b nh m (h w) d")
        sparse_v = rearrange(sparse_v, "b nh m h w d -> b nh m (h w) d")

        scale_factor = kwargs['scale_factor'] # scale factor for downsampling (e.g., 4 means sample 1/4 of the tokens)
        n_patches_to_sample = h * w // scale_factor 

        sample_indices = [torch.randperm(h*w, device=local_k.device, dtype=torch.long)[:n_patches_to_sample] for _ in range(m)]
        sample_indices = torch.stack(sample_indices, dim=0) # (m, n_patches_to_sample)

        idx = sample_indices[None, None, :, :, None] # (1, 1, m, h*w, 1)
        idx = idx.expand(sparse_k.shape[0], sparse_k.shape[1], -1, -1, sparse_k.shape[-1]) # (b, nh, m, h*w, d)

        sparse_k = torch.gather(sparse_k, dim=3, index=idx) # (b, nh, m, n_patches_to_sample, d)
        sparse_v = torch.gather(sparse_v, dim=3, index=idx) # (b, nh, m, n_patches_to_sample, d)

        # Reshape back to sequence
        sparse_k = rearrange(sparse_k, "b nh m nps d -> b nh (m nps) d")
        sparse_v = rearrange(sparse_v, "b nh m nps d -> b nh (m nps) d")

        # create spatial mask 
        spatial_mask = torch.zeros(m, h*w, dtype=torch.bool, device=local_k.device) 
        subsample_mask = spatial_mask.scatter(dim=1, index=sample_indices, value=True)
        subsample_mask = rearrange(subsample_mask, "m (h w) -> m h w", h=grid_hw[0], w=grid_hw[1])

    elif method == "tome": 
    
        # apply token merging across framewise K features 
        merge_ratio = kwargs['merge_ratio']
        
        sparse_k, sparse_v, hw_grid = get_bipartite_soft_matching(sparse_k, sparse_v, merge_ratio, num_iters=5)
        sparse_k = rearrange(sparse_k, "b nh m n d -> b nh (m n) d").to(local_k.dtype)
        sparse_v = rearrange(sparse_v, "b nh m n d -> b nh (m n) d").to(local_v.dtype)

        n_tokens_subsampled = hw_grid.shape[-2] 

        # create position_mapping - hw_grid: (b, nh, m, n, 1)
        sampled_indices = hw_grid.squeeze(-1).long() # (b, nh, m, n)
        h_indices = sampled_indices // w # (b, nh, m, n) 
        w_indices = sampled_indices % w  # (b, nh, m, n)

        frame_indices = torch.arange(m, device=local_k.device) # (m,)
        frame_indices = frame_indices[None, None, :, None].expand(b, nh, -1, n_tokens_subsampled ) # (b, nh, m, n_tokens_subsampled)

        position_mapping = torch.stack([frame_indices, h_indices, w_indices], dim=-1) # (b, nh, m, n_tokens_subsampled, 3)
        position_mapping = rearrange(position_mapping, "b nh m nts c ->  b nh (m nts) c") # (b, nh, m*nts, 3)

        subsample_mask = position_mapping

    compressed_local_k = torch.cat([pre_k, sparse_k, dense_k], dim=-2)
    compressed_local_v = torch.cat([pre_v, sparse_v, dense_v], dim=-2)

    sliding_window = compressed_local_k.shape[-2]
    
    return compressed_local_k, compressed_local_v, subsample_mask, sliding_window

def apply_spatial_dilation(sparse_k, sparse_v, h, w, dilation_rate):
    """
    Apply spatial dilation to select every r-th token in height and width dimensions.
    
    Args:
        sparse_k: (batch, n_head_kv, n_frames, h, w, dim_head) - key tensor
        sparse_v: (batch, n_head_kv, n_frames, h, w, dim_head) - value tensor
        h: int - height dimension size
        w: int - width dimension size
        dilation_rate: int - dilation rate (e.g., 2 means every 2nd token)
    
    Returns:
        tuple: (dilated_k, dilated_v, spatial_mask)
            - dilated_k: (batch, n_head_kv, n_frames, h//r, w//r, dim_head)
            - dilated_v: (batch, n_head_kv, n_frames, h//r, w//r, dim_head)
            - spatial_mask: (h, w) bool tensor indicating selected positions
    """
    r = dilation_rate
    device = sparse_k.device
    
    # Generate dilated indices for h and w dimensions
    dilated_h_indices = torch.arange(0, h, r, device=device)
    dilated_w_indices = torch.arange(0, w, r, device=device)
    
    # Index the spatial dimensions
    dilated_k = sparse_k[:, :, :, dilated_h_indices, :, :]
    dilated_k = dilated_k[:, :, :, :, dilated_w_indices, :]
    dilated_v = sparse_v[:, :, :, dilated_h_indices, :, :]
    dilated_v = dilated_v[:, :, :, :, dilated_w_indices, :]
    
    # Create spatial mask
    spatial_mask = torch.zeros(h, w, dtype=torch.bool, device=device)
    spatial_mask[dilated_h_indices[:, None], dilated_w_indices[None, :]] = True
    
    return dilated_k, dilated_v, spatial_mask

def apply_frame_sample_compression(
    local_k, local_v, local_q,
    n_init, init_exc, grid_hw,
    n_frames_to_sample, 
    method='uniform', 
    dilation_rate=None,
    nd=1
):
    """
    Apply frame sampling compression to previous frames.
    
    Args:
        local_k: (batch, n_head_kv, total_len, dim_head)
        local_v: (batch, n_head_kv, total_len, dim_head)
        local_q: (batch, n_head, len_q, dim_head)
        n_init: number of initial sink tokens
        init_exc: whether init cache is full
        grid_hw: (height, width) of spatial grid
        n_frames_to_sample: number of frames to uniformly sample
        method: method for sampling frames 
        nd: multiplier for determining number of dense frames (default: 1)
    
    Returns:
        tuple: (compressed_k, compressed_v, subsample_mask, sliding_window)
    """

    assert method in ['uniform', 'random', 'recent', 'event'], f"Invalid sample method: {method}"
    len_q = local_q.size(-2) * nd
    
    st = 0 if init_exc else n_init
    pre_k, sparse_k, dense_k = local_k[:, :, :st, :], local_k[:, :, st:-len_q, :], local_k[:, :, -len_q:, :]
    pre_v, sparse_v, dense_v = local_v[:, :, :st, :], local_v[:, :, st:-len_q, :], local_v[:, :, -len_q:, :]

    sparse_k = rearrange(sparse_k, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])
    b, nh, m, h, w, d = sparse_k.shape
    sparse_v = rearrange(sparse_v, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])

    # generate sample indices 
    n_frames_to_sample = min(m, n_frames_to_sample)  # Handle case where m < n_frames_to_sample
        
    if method == "uniform":
        sampled_indices = torch.linspace(0, m - 1, n_frames_to_sample, device=local_k.device).round().long()
    elif method == "random":
        sampled_indices = torch.randperm(m, device=local_k.device)[:n_frames_to_sample]
    elif method == "recent":
        # The newest n frames -- the "is this just a shorter sliding window?" control.
        sampled_indices = torch.arange(
            m - n_frames_to_sample, m, device=local_k.device)
    elif method == "event":
        # compute distance between adjacent frames 
        sparse_k_for_norm = rearrange(sparse_k, "b nh m h w d -> b nh m (h w) d")
        sparse_k_norm = F.normalize(sparse_k_for_norm, dim=-1) 
        sparse_kk_adjacent_sim = (sparse_k_norm[:, :, 1:, :, :] * sparse_k_norm[:, :, :-1, :, :]).sum(dim=-1) # (b, nh, m-1, h*w) 
        sparse_kk_adjacent_sim = sparse_kk_adjacent_sim.sum(dim=-1) # (b, nh, m-1) 
        # always include the first frame of the window 
        sparse_kk_adjacent_sim = torch.cat([torch.zeros(b, nh, 1, dtype=torch.float32, device=local_k.device), sparse_kk_adjacent_sim], dim=2) # (b, nh, m)
        sampled_indices_per_head = sparse_kk_adjacent_sim.topk(n_frames_to_sample, dim=-1, largest=False).indices # (b, nh, n_frames_to_sample)
    else:
        raise ValueError(f"Invalid sample method: {method}")

    if method == "event": 
        # sampled_indices_per_head: (b, nh, n_frames_to_sample) -> (b, nh, n_frames_to_sample, h, w, d)
        idx = sampled_indices_per_head[..., None, None, None].expand(-1, -1, -1, h, w, d) 
        sparse_k = torch.gather(sparse_k, dim=2, index=idx) # (b, nh, n_frames_to_sample, h, w, d)
        sparse_v = torch.gather(sparse_v, dim=2, index=idx) # (b, nh, n_frames_to_sample, h, w, d)
        sparse_k = rearrange(sparse_k, "b nh m h w d -> b nh (m h w) d")
        sparse_v = rearrange(sparse_v, "b nh m h w d -> b nh (m h w) d")

        compressed_local_k = torch.cat([pre_k, sparse_k, dense_k], dim=-2)
        compressed_local_v = torch.cat([pre_v, sparse_v, dense_v], dim=-2)

        subsample_mask = torch.zeros(b, nh, m, h, w, dtype=torch.bool, device=local_k.device)
        idx = sampled_indices_per_head[..., None, None].expand(-1, -1, -1, h, w) # (b, nh, n_frames_to_sample, h, w)
        subsample_mask.scatter_(dim=2, index=idx, value=True)

    else: 
        # Select sampled frames
        sparse_k_sampled = sparse_k[:, :, sampled_indices, :, :, :]  # (b, nh, n_frames_to_sample, h, w, d)
        sparse_v_sampled = sparse_v[:, :, sampled_indices, :, :, :]
        
        # Apply spatial dilation if specified (every r-th token in h and w)
        if dilation_rate is not None and dilation_rate > 1:
            sparse_k_sampled, sparse_v_sampled, spatial_mask = apply_spatial_dilation(
                sparse_k_sampled, sparse_v_sampled, h, w, dilation_rate
            )
            # Create combined temporal + spatial mask
            subsample_mask = torch.zeros(m, grid_hw[0], grid_hw[1], dtype=torch.bool, device=local_k.device)
            subsample_mask[sampled_indices] = spatial_mask.unsqueeze(0)
        else:
            subsample_mask = torch.zeros(m, grid_hw[0], grid_hw[1], dtype=torch.bool, device=local_k.device)
            subsample_mask[sampled_indices] = True
        
        sparse_k = rearrange(sparse_k_sampled, "b nh m h w d -> b nh (m h w) d")
        sparse_v = rearrange(sparse_v_sampled, "b nh m h w d -> b nh (m h w) d")

        compressed_local_k = torch.cat([pre_k, sparse_k, dense_k], dim=-2)
        compressed_local_v = torch.cat([pre_v, sparse_v, dense_v], dim=-2)

    sliding_window = compressed_local_k.shape[-2]
    return compressed_local_k, compressed_local_v, subsample_mask, sliding_window

def apply_tome_with_frame_sampling(
    local_k, local_v, local_q,
    n_init, init_exc, grid_hw,
    n_frames_to_sample,
    merge_ratio,
    nd=1,
    num_iters=2
):
    """
    Hybrid compression: First sample frames uniformly, then apply ToMe within those frames.
    
    This combines temporal (frame-level) and spatial (patch-level) compression:
    1. Uniformly sample n_frames_to_sample frames from all available frames
    2. Apply Token Merging (ToMe) to merge similar patches within selected frames
    
    Args:
        local_k: (batch, n_head_kv, total_len, dim_head)
        local_v: (batch, n_head_kv, total_len, dim_head)
        local_q: (batch, n_head, len_q, dim_head)
        n_init: number of initial sink tokens
        init_exc: whether init cache is full
        grid_hw: (height, width) of spatial grid
        n_frames_to_sample: number of frames to uniformly sample
        merge_ratio: ratio of tokens to merge (0.5 = merge 50% of tokens)
        nd: multiplier for determining number of dense frames (default: 1)
        num_iters: number of iterations for ToMe bipartite matching (default: 2)
    
    Returns:
        tuple: (compressed_k, compressed_v, subsample_mask, sliding_window)
    
    Example:
        If you have 16 frames and specify n_frames_to_sample=8, merge_ratio=0.5:
        - Step 1: Uniformly sample 8 frames from 16 (e.g., frames [0,2,4,6,8,10,12,14])
        - Step 2: Within each of those 8 frames, merge 50% of patches using ToMe
    """
    len_q = local_q.size(-2) * nd
    
    st = 0 if init_exc else n_init
    pre_k, sparse_k, dense_k = local_k[:, :, :st, :], local_k[:, :, st:-len_q, :], local_k[:, :, -len_q:, :]
    pre_v, sparse_v, dense_v = local_v[:, :, :st, :], local_v[:, :, st:-len_q, :], local_v[:, :, -len_q:, :]

    # Reshape to separate frames
    sparse_k = rearrange(sparse_k, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])
    b, nh, m, h, w, d = sparse_k.shape
    sparse_v = rearrange(sparse_v, "b nh (m h w) d -> b nh m h w d", h=grid_hw[0], w=grid_hw[1])

    # Step 1: Uniformly sample frames.
    # <= 1 is a FRACTION of this window's frames (1.0 = all of them, the no-sampling
    # ablation); > 1 is an absolute count. Window-size-invariant across tpf/fps.
    if 0 < n_frames_to_sample <= 1:
        n_frames_to_sample = max(1, round(m * n_frames_to_sample))
    n_frames_to_sample = min(m, int(n_frames_to_sample))  # Handle case where m < n_frames_to_sample
    sampled_indices = torch.linspace(0, m - 1, n_frames_to_sample, device=local_k.device).round().long()

    # Select sampled frames
    sparse_k_sampled = sparse_k[:, :, sampled_indices, :, :, :]  # (b, nh, n_frames_to_sample, h, w, d)
    sparse_v_sampled = sparse_v[:, :, sampled_indices, :, :, :]
    
    # Step 2: Apply ToMe to the sampled frames
    sparse_k_merged, sparse_v_merged, hw_grid = get_bipartite_soft_matching(
        sparse_k_sampled, sparse_v_sampled, merge_ratio, num_iters=num_iters
    )
    
    # Reshape to sequence format
    sparse_k_merged = rearrange(sparse_k_merged, "b nh m n d -> b nh (m n) d").to(local_k.dtype)
    sparse_v_merged = rearrange(sparse_v_merged, "b nh m n d -> b nh (m n) d").to(local_v.dtype)
    
    # Concatenate with init and dense tokens
    compressed_local_k = torch.cat([pre_k, sparse_k_merged, dense_k], dim=-2)
    compressed_local_v = torch.cat([pre_v, sparse_v_merged, dense_v], dim=-2)
    
    # Create position mapping for RoPE
    # hw_grid: (b, nh, n_frames_to_sample, n_tokens_merged, 1) - contains original spatial indices
    n_tokens_merged = hw_grid.shape[-2]
    sampled_indices_grid = hw_grid.squeeze(-1).long()  # (b, nh, n_frames_to_sample, n_tokens_merged)
    h_indices = sampled_indices_grid // w
    w_indices = sampled_indices_grid % w
    
    # Map back to original frame indices (not sampled indices)
    frame_indices = sampled_indices[None, None, :, None].expand(b, nh, -1, n_tokens_merged)  # (b, nh, n_frames_to_sample, n_tokens_merged)
    
    position_mapping = torch.stack([frame_indices, h_indices, w_indices], dim=-1)  # (b, nh, n_frames_to_sample, n_tokens_merged, 3)
    position_mapping = rearrange(position_mapping, "b nh m nts c -> b nh (m nts) c")  # (b, nh, total_merged_tokens, 3)
    
    subsample_mask = position_mapping
    sliding_window = compressed_local_k.shape[-2]
    
    return compressed_local_k, compressed_local_v, subsample_mask, sliding_window


def rebuild_mixed_window(
    local_k, local_v, frame_geometry, sparse_indices, replacements=None,
):
    """Shared KK/ToMe rebuild of text, sparse visual tokens, and dense frames."""
    prefix_len, visual_starts, _, boundary_indices = frame_geometry
    n_frames, sparse_frames = visual_starts.numel(), len(sparse_indices)
    nh = local_k.size(1)
    kept_parts, kv_parts = [], [] if replacements is not None else None

    def keep(indices, replacement=None):
        if indices.numel() == 0:
            return
        indices = indices.to(local_k.device)
        if indices.dim() == 1:
            indices = indices.expand(nh, -1)
        kept_parts.append(indices)
        if kv_parts is not None:
            if replacement is None:
                gather = indices[None, ..., None].expand(
                    local_k.size(0), -1, -1, local_k.size(-1))
                replacement = (
                    torch.gather(local_k, 2, gather),
                    torch.gather(local_v, 2, gather),
                )
            kv_parts.append(replacement)

    keep(torch.arange(prefix_len, device=local_k.device))
    previous_end = prefix_len - 1
    for frame_idx in range(n_frames):
        end = int(boundary_indices[2 * frame_idx + 1].item())
        if frame_idx >= sparse_frames:
            keep(torch.arange(previous_end + 1, end + 1, device=local_k.device))
        elif sparse_indices[frame_idx] is not None:
            start = int(boundary_indices[2 * frame_idx].item())
            keep(torch.arange(previous_end + 1, start + 1, device=local_k.device))
            replacement = None if replacements is None else replacements[frame_idx]
            keep(sparse_indices[frame_idx], replacement)
            keep(torch.tensor([end], device=local_k.device))
        previous_end = end
    keep(torch.arange(previous_end + 1, local_k.size(-2), device=local_k.device))

    kept_indices = torch.cat(kept_parts, -1)[None].long()
    if kv_parts is None:
        gather = kept_indices[..., None].expand(
            local_k.size(0), -1, -1, local_k.size(-1))
        compressed_k = torch.gather(local_k, 2, gather)
        compressed_v = torch.gather(local_v, 2, gather)
    else:
        compressed_k = torch.cat([part[0] for part in kv_parts], 2)
        compressed_v = torch.cat([part[1] for part in kv_parts], 2)
    assert kept_indices.size(-1) == compressed_k.size(-2)
    return compressed_k, compressed_v, kept_indices, compressed_k.size(-2)


def apply_mixed_tome_compression(
    local_k, local_v, local_q, visual_pos_masks, frame_geometry,
    grid_hw, nd, merge_ratio, n_frames_to_sample=None, num_iters=5,
):
    """Apply existing ToMe to visual tokens, then use the shared mixed rebuild."""
    _, visual_starts, visual_ends, _ = frame_geometry
    hw, n_frames = grid_hw[0] * grid_hw[1], visual_starts.numel()
    sparse_frames = n_frames - min(nd, n_frames)
    if not bool(torch.all((visual_ends - visual_starts) == hw)):
        raise AssertionError(
            "mixed window has invalid ToMe frame widths: "
            f"expected={hw}, actual={(visual_ends - visual_starts).tolist()}")
    if sparse_frames == 0:
        return local_k, local_v, None, local_k.size(-2)

    # local_q[:, :, :hw] is a stand-in for the dense-frame length: the callees compute
    # len_q = local_q.size(-2) * nd and we strip min(nd, n_frames) * hw below, so the two
    # only agree while local_q is at least one full frame wide. True today (blocks are
    # per-frame: hw + timestamp + 2 boundary tokens), but fail loudly if that changes --
    # the kk path sidesteps this by passing dense_len explicitly.
    if local_q.size(-2) < hw:
        raise AssertionError(
            f"ToMe dense-frame proxy invalid: local_q len={local_q.size(-2)} < hw={hw}; "
            "pass dense_len explicitly instead of slicing local_q")

    visual = torch.nonzero(visual_pos_masks[0], as_tuple=False).squeeze(-1).to(local_k.device)
    visual_k, visual_v = local_k[:, :, visual], local_v[:, :, visual]
    args = (visual_k, visual_v, local_q[:, :, :hw], 0, True, grid_hw)
    if n_frames_to_sample is None:
        compressed_k, compressed_v, positions, _ = apply_patch_sample_compression(
            *args, method='tome', merge_ratio=merge_ratio, nd=nd)
    else:
        compressed_k, compressed_v, positions, _ = apply_tome_with_frame_sampling(
            *args, n_frames_to_sample=n_frames_to_sample,
            merge_ratio=merge_ratio, nd=nd, num_iters=num_iters)

    selected = torch.unique_consecutive(positions[0, 0, :, 0])
    n_per_frame = positions.size(2) // selected.numel()
    dense_len = min(nd, n_frames) * hw
    merged_k = compressed_k[:, :, :-dense_len].reshape(
        local_k.size(0), local_k.size(1), -1, n_per_frame, local_k.size(-1))
    merged_v = compressed_v[:, :, :-dense_len].reshape_as(merged_k)
    sparse_indices, replacements = [None] * sparse_frames, [None] * sparse_frames
    for offset, frame_idx in enumerate(selected.tolist()):
        frame_pos = positions[0, :, offset * n_per_frame:(offset + 1) * n_per_frame]
        patch = frame_pos[..., 1] * grid_hw[1] + frame_pos[..., 2]
        sparse_indices[frame_idx] = visual_starts[frame_idx].to(local_k.device) + patch.long()
        replacements[frame_idx] = (
            merged_k[:, :, offset], merged_v[:, :, offset])
    return rebuild_mixed_window(
        local_k, local_v, frame_geometry, sparse_indices, replacements)


def frame_keep_to_sparse_indices(keep_sparse, visual_starts, visual_ends, nh, device):
    """
    Turn a per-sparse-frame keep/drop decision into the `sparse_indices` list that
    rebuild_mixed_window() consumes.

    A kept frame contributes ALL of its visual token indices (expanded across kv heads);
    a dropped frame contributes None. The None is load-bearing and is NOT
    interchangeable with an empty tensor: rebuild_mixed_window only vanishes the whole
    frame -- timestamp text, <|vision_start|>, patches, <|vision_end|> -- on None. An
    empty tensor leaves the frame behind as text-only boundary tokens, which inflates the
    window's timestamp:visual ratio, i.e. the exact thing frame-level methods exist to
    avoid.

    Shared by every frame-level arm (uniform / random / recent / qframe) so that
    "same budget, different selection" is true by construction rather than by assumption.
    """
    return [
        torch.arange(visual_starts[i], visual_ends[i], device=device).expand(nh, -1)
        if bool(keep_sparse[i]) else None
        for i in range(len(keep_sparse))
    ]


class RecencyProfile:
    """
    Running mean of the head-pooled per-frame query affinity, indexed by RECENCY
    (frames_ago = n_frames - 1 - frame_idx). One instance per decoder layer per video.

    Why recency and not absolute position: early in a video the sliding window is still
    filling, so a frame's absolute index in the window is not comparable across calls
    while its distance from the incoming frame is. Indexing by frames_ago means every
    observation lands in a slot that means the same thing for the whole video.

    Only the profile's SHAPE matters -- top-k selection is invariant to adding a constant
    to every frame's score -- so a plain cumulative mean needs no bias correction.
    """

    def __init__(self):
        self._sum = None   # (N,) float32
        self._cnt = None   # (N,) float32

    def _grow(self, n, device):
        if self._sum is None:
            self._sum = torch.zeros(n, dtype=torch.float32, device=device)
            self._cnt = torch.zeros(n, dtype=torch.float32, device=device)
        elif self._sum.numel() < n:
            new = torch.zeros(n, dtype=torch.float32, device=self._sum.device)
            new[: self._sum.numel()] = self._sum
            self._sum = new
            new = torch.zeros(n, dtype=torch.float32, device=self._cnt.device)
            new[: self._cnt.numel()] = self._cnt
            self._cnt = new

    def update(self, frames_ago, values):
        """Accumulate one window's observations. Call AFTER scoring, never before, so a
        window never de-means itself with its own contribution."""
        if frames_ago.numel() == 0:
            return
        self._grow(int(frames_ago.max()) + 1, values.device)
        idx = frames_ago.to(self._sum.device).long()
        self._sum.index_add_(0, idx, values.detach().float().to(self._sum.device))
        self._cnt.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    def lookup(self, frames_ago):
        """
        Mean affinity for each requested recency slot, forward-filled from the
        youngest-seen neighbour for slots not yet observed. Returns (means, n_unseen).

        Increasing index = older, so forward-filling propagates from younger to older --
        the correct direction, because the window grows by one frame at a time and so at
        most the single oldest slot is ever unseen. With nothing observed at all the
        result is zeros, i.e. the score falls back to raw affinity for that one window.
        """
        idx = frames_ago.long()
        if self._sum is None:
            return torch.zeros(idx.shape, dtype=torch.float32, device=idx.device), int(idx.numel())
        n = self._sum.numel()
        mean = self._sum / self._cnt.clamp(min=1.0)
        seen = self._cnt > 0
        # index of the nearest younger (lower-index) seen slot, or -1 if none
        ar = torch.arange(n, device=mean.device)
        last = torch.where(seen, ar, torch.full_like(ar, -1)).cummax(0).values
        filled = torch.where(last >= 0, mean[last.clamp(min=0)], torch.zeros_like(mean))
        safe = idx.clamp(max=n - 1).to(mean.device)
        out = torch.where(
            (idx.to(mean.device) < n) & seen[safe].logical_or(last[safe] >= 0),
            filled[safe],
            torch.zeros_like(filled[safe]),
        )
        n_unseen = int((idx.to(mean.device) >= n).sum()) + int((~seen[safe]).sum())
        return out.to(idx.device), n_unseen


def frame_query_affinity(local_k, local_q, visual_starts, visual_ends, n_sparse):
    """
    Pre-RoPE pooled Q.K affinity of each sparse frame to the INCOMING frame.

    At encode time the incoming frame is the query attending to the sliding window, so
    "how much does the new frame need this window frame" is directly computable -- the
    asymmetry that makes a query-relevance term unavailable during streaming encode does
    not actually apply here.

    Returns (per_head, pooled):
        per_head: (nh_kv, n_sparse)  -- for diagnostics and the z-normalised variant
        pooled:   (n_sparse,)        -- mean over kv heads, what selection uses

    Two deliberate choices, both matching existing code:
      * Query side pools over the GQA group's heads AND over all query positions in the
        chunk, identical to apply_budgeted_kk_compression's `q_pool`, so the two arms
        share one definition of Q.K.
      * Key side uses VISUAL tokens only (matching the tail probe's `vidx`), excluding
        the timestamp text whose much larger per-token mass would otherwise dominate a
        frame's score.

    Frames are reduced with a ragged segment mean over visual_starts/visual_ends rather
    than by reshaping to (m, h, w). That reshape assumes a purely visual window and is
    precisely why swa_event_sample_<n> is unsupported on mixed text/visual windows; the
    segment form has no uniform-per-frame-width requirement.
    """
    b, nh, _, d = local_k.shape
    if b != 1:
        raise ValueError(f"frame_query_affinity requires batch size 1, got {b}")
    nh_q = local_q.size(1)
    if nh_q % nh != 0:
        raise ValueError(f"query heads ({nh_q}) not a multiple of kv heads ({nh})")
    group = nh_q // nh
    q_pool = local_q.float().view(b, nh, group, local_q.size(-2), d).mean(dim=(2, 3))  # (b, nh, d)

    tok = (local_k.float() * q_pool[:, :, None, :]).sum(dim=-1) / math.sqrt(d)  # (b, nh, L)
    tok = tok[0]                                                               # (nh, L)

    starts = visual_starts[:n_sparse].to(tok.device).long()
    widths = (visual_ends[:n_sparse].to(tok.device).long() - starts)
    if int(widths.min()) <= 0:
        raise ValueError("frame_query_affinity: non-positive visual frame width")
    fid = torch.repeat_interleave(torch.arange(n_sparse, device=tok.device), widths)
    offs = torch.arange(int(widths.sum()), device=tok.device) - torch.repeat_interleave(
        torch.cumsum(widths, 0) - widths, widths)
    pos = torch.repeat_interleave(starts, widths) + offs

    per_head = torch.zeros(nh, n_sparse, dtype=torch.float32, device=tok.device)
    per_head.index_add_(1, fid, tok[:, pos])
    per_head = per_head / widths.to(per_head.dtype)[None, :]
    return per_head, per_head.mean(dim=0)


def no_compression(local_k, local_v, n_local):
    """
    Standard path with no compression.
    
    Args:
        local_k: (batch, n_head_kv, total_len, dim_head)
        local_v: (batch, n_head_kv, total_len, dim_head)
        n_local: local window size
    
    Returns:
        tuple: (local_k, local_v, None, sliding_window)
    """
    return local_k, local_v, None, n_local

def apply_budgeted_kk_compression(
    local_k, local_v, local_q,
    n_init, init_exc, grid_hw,
    avg_keep, alpha=0.5, floor_frac=0.25,
    nd=1, dense_len=None,
):
    """AKS-2 (akv2): count-targeted, budget-reallocated, query-guided key selection.

    Three deliberate differences from apply_kk_guided_compression(method="adjacent"):

    1. Count-targeted. The retained budget is ``avg_keep`` visual tokens per sparse
       frame on average, independent of the input tokens-per-frame. Scaling the input
       budget therefore enlarges the candidate pool without growing the attended set
       (the fixed-ratio path retains r^2-fold more tokens as TPF grows, which is what
       degrades it at high budgets).
    2. Budget reallocation. The window's total budget (avg_keep * m) is spent where
       score mass is, subject to a per-frame floor of ``floor_frac * avg_keep`` tokens:
       static frames give up budget to high-novelty / high-attention frames, but no
       frame is ever evicted outright.
    3. Query guidance. The selection score mixes the AKS novelty signal with the
       current query frame's pre-RoPE QK affinity (a proxy for the attention mass the
       query will place on each past token):

           score = alpha * z(1 - adjacent-frame key cosine) + (1 - alpha) * z(q_pool . k)

       both channels z-normalised per (batch, kv-head) over all sparse tokens.
       alpha=1 is pure novelty (AKS criterion under a budget); alpha=0 is pure
       query-guided selection.

    Per-frame counts are computed once from the head-mean score and SHARED across
    heads; each head then selects its own top-count patches. Shared counts keep
    per-head totals equal (required by the 5-dim RoPE mask path) and satisfy the
    Qwen3 mixed-window rebuild, which reshapes each frame's kept patches to (nh, -1).

    Kept indices are sorted ascending within the window so the gathered sequence is
    in raster order and matches the mask's nonzero order exactly (the stride path's
    topk-order permutation is a known position-assignment hazard on the pure path).

    Frame 0 has no predecessor: its novelty channel gets the window-mean novelty as a
    neutral prior, plus a tiny diagonal-stripe tie-break so the pure-novelty case
    degrades to an isotropic pattern instead of a raster prefix.

    Args (beyond the shared signature):
        avg_keep: target retained visual tokens per sparse frame (a count, not a ratio)
        alpha: weight on novelty vs query affinity, in [0, 1]
        floor_frac: per-frame floor as a fraction of avg_keep
        dense_len: explicit trailing dense-token length (the Qwen3 caller passes
            exc_block_size * nd); defaults to local_q.size(-2) * nd

    Returns:
        (compressed_k, compressed_v, subsample_mask(b, nh, m, h, w), sliding_window)
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not (0.0 <= floor_frac <= 1.0):
        raise ValueError(f"floor_frac must be in [0, 1], got {floor_frac}")

    h, w = grid_hw
    hw = h * w
    len_q = dense_len if dense_len is not None else local_q.size(-2) * nd

    st = 0 if init_exc else n_init
    pre_k, sparse_k, dense_k = local_k[:, :, :st, :], local_k[:, :, st:-len_q, :], local_k[:, :, -len_q:, :]
    pre_v, sparse_v, dense_v = local_v[:, :, :st, :], local_v[:, :, st:-len_q, :], local_v[:, :, -len_q:, :]

    b, nh, n_sparse_tokens, d = sparse_k.shape
    assert b == 1, "akv2 assumes batch size 1 (as does the per-head RoPE mask path)"
    if n_sparse_tokens % hw != 0:
        raise ValueError(f"sparse region ({n_sparse_tokens} tokens) does not tile the {h}x{w} grid")
    m = n_sparse_tokens // hw

    if m == 0:
        subsample_mask = torch.zeros(b, nh, 0, h, w, dtype=torch.bool, device=local_k.device)
        return local_k, local_v, subsample_mask, local_k.shape[-2]

    sparse_k = sparse_k.view(b, nh, m, hw, d)
    sparse_v = sparse_v.view(b, nh, m, hw, d)

    avg_keep = int(round(avg_keep))
    if not (1 <= avg_keep <= hw):
        raise ValueError(f"avg_keep={avg_keep} out of range for grid {h}x{w} ({hw} patches)")

    def _z(x):
        mu = x.mean(dim=(-2, -1), keepdim=True)
        sd = x.std(dim=(-2, -1), keepdim=True)
        return (x - mu) / (sd + 1e-6)

    score = torch.zeros(b, nh, m, hw, dtype=torch.float32, device=local_k.device)

    if alpha > 0.0:
        k_norm = F.normalize(sparse_k.float(), dim=-1)
        if m > 1:
            adjacent_sim = (k_norm[:, :, 1:] * k_norm[:, :, :-1]).sum(dim=-1)  # (b, nh, m-1, hw)
            novelty = 1.0 - adjacent_sim.clamp(-1.0, 1.0)
            neutral = novelty.mean(dim=(-2, -1), keepdim=True).expand(b, nh, 1, hw)
            novelty = torch.cat([neutral, novelty], dim=2)  # (b, nh, m, hw)
            score = score + alpha * _z(novelty)
        # m == 1: only frame 0 exists and novelty is undefined -- the stripe
        # tie-break below decides (isotropic pattern), matching the keep_k path.

    if alpha < 1.0:
        nh_q = local_q.size(1)
        if nh_q % nh != 0:
            raise ValueError(f"query heads ({nh_q}) not a multiple of kv heads ({nh})")
        group = nh_q // nh
        # Pre-RoPE pooled query per kv-head group: mean over the group's heads and
        # over the current chunk's query tokens.
        q_pool = local_q.float().view(b, nh, group, local_q.size(-2), d).mean(dim=(2, 3))  # (b, nh, d)
        qk = (sparse_k.float() * q_pool[:, :, None, None, :]).sum(dim=-1) / math.sqrt(d)  # (b, nh, m, hw)
        score = score + (1.0 - alpha) * _z(qk)

    # Frame-0 tie-break: diagonal-stripe ordering (isotropic at any keep ratio).
    rows = torch.arange(h, device=local_k.device).view(h, 1).expand(h, w)
    cols = torch.arange(w, device=local_k.device).view(1, w).expand(h, w)
    stripe = max(1, int(round(hw / avg_keep)))
    stripe_order = torch.argsort(((rows + cols) % stripe).flatten(), stable=True)
    stripe_rank = torch.argsort(stripe_order)
    score[:, :, 0, :] = score[:, :, 0, :] - stripe_rank.float() * 1e-5

    # --- Allocation: per-frame counts from the head-mean score ---------------------
    budget = min(avg_keep * m, m * hw)
    floor = max(1, int(round(floor_frac * avg_keep)))
    floor = min(floor, hw)
    if floor * m > budget:
        floor = budget // m

    mean_score = score.mean(dim=1)  # (b, m, hw)
    within_frame_rank = torch.argsort(torch.argsort(mean_score, dim=-1, descending=True), dim=-1)
    boosted = mean_score + (within_frame_rank < floor).float() * 1e6
    chosen = boosted.view(b, m * hw).topk(budget, dim=-1).indices  # (b, budget)
    counts = torch.zeros(b, m, dtype=torch.long, device=local_k.device)
    counts.scatter_add_(1, chosen // hw, torch.ones_like(chosen))

    # --- Selection: each head keeps its own top-count patches per frame ------------
    head_rank = torch.argsort(torch.argsort(score, dim=-1, descending=True), dim=-1)  # (b, nh, m, hw)
    sel_mask = head_rank < counts[:, None, :, None]  # (b, nh, m, hw)

    # --- Gather in ascending (raster) order so sequence order == mask order --------
    flat_mask = sel_mask.view(b, nh, m * hw)
    nz = flat_mask.nonzero(as_tuple=False)  # rows sorted by (b, nh, position)
    flat_idx = nz[:, 2].view(b, nh, budget)
    gather_idx = flat_idx[..., None].expand(-1, -1, -1, d)
    sel_k = torch.gather(sparse_k.reshape(b, nh, m * hw, d), 2, gather_idx)
    sel_v = torch.gather(sparse_v.reshape(b, nh, m * hw, d), 2, gather_idx)

    compressed_local_k = torch.cat([pre_k, sel_k.to(local_k.dtype), dense_k], dim=-2)
    compressed_local_v = torch.cat([pre_v, sel_v.to(local_v.dtype), dense_v], dim=-2)
    subsample_mask = sel_mask.view(b, nh, m, h, w)
    sliding_window = compressed_local_k.shape[-2]

    return compressed_local_k, compressed_local_v, subsample_mask, sliding_window


def apply_gt_alloc_compression(
    local_k, local_v, local_q,
    n_init, init_exc, grid_hw,
    dense_frame_mask, floor_keep,
    nd=1, dense_len=None, dense_keep=None,
    keep_vector=None,
):
    """GT-allocation ORACLE (swa_gtalloc): frames flagged in dense_frame_mask keep ALL
    patches; every other frame keeps `floor_keep` patches chosen by the v1 criterion
    (per-head adjacent-frame dissimilarity; diagonal stripe for frame 0).

    This is deliberately NOT a deployable method: dense_frame_mask comes from the
    per-video union of ground-truth clue windows, injected at launch time. It measures
    the ceiling of the allocation axis — what perfect knowledge of evidence location
    buys over uniform allocation — with v1's selection semantics everywhere else
    (raw bf16 similarities, per-head choice, no normalization, no query term).

    dense_frame_mask: 1-D bool over this window's SPARSE frames (oldest first);
    caller derives it from global frame indices. Counts per frame are shared across
    heads (hw or floor_keep), satisfying the Qwen3 rebuild contract.

    Returns (compressed_k, compressed_v, mask(b, nh, m, h, w), sliding_window).
    """
    h, w = grid_hw
    hw = h * w
    len_q = dense_len if dense_len is not None else local_q.size(-2) * nd
    st = 0 if init_exc else n_init
    pre_k, sparse_k, dense_k = local_k[:, :, :st, :], local_k[:, :, st:-len_q, :], local_k[:, :, -len_q:, :]
    pre_v, sparse_v, dense_v = local_v[:, :, :st, :], local_v[:, :, st:-len_q, :], local_v[:, :, -len_q:, :]

    b, nh, n_sparse, d = sparse_k.shape
    assert b == 1, "gtalloc assumes batch size 1"
    if n_sparse % hw != 0:
        raise ValueError(f"sparse region ({n_sparse}) does not tile the {h}x{w} grid")
    m = n_sparse // hw
    if m == 0:
        return local_k, local_v, torch.zeros(b, nh, 0, h, w, dtype=torch.bool, device=local_k.device), local_k.shape[-2]

    if keep_vector is not None:
        # allocation-vector mode: arbitrary per-frame counts, same v1 selection
        counts = [int(x) for x in keep_vector]
        if len(counts) != m:
            raise ValueError(f"keep_vector has {len(counts)} entries for {m} sparse frames")
        for x in counts:
            if not (1 <= x <= hw):
                raise ValueError(f"keep_vector count {x} out of range for grid {h}x{w}")
        dm = None
    else:
        dm = torch.as_tensor(dense_frame_mask, dtype=torch.bool, device=local_k.device)
        if dm.numel() != m:
            raise ValueError(f"dense_frame_mask has {dm.numel()} entries for {m} sparse frames")
        floor_keep = int(floor_keep)
        if not (1 <= floor_keep <= hw):
            raise ValueError(f"floor_keep={floor_keep} out of range for grid {h}x{w}")
        dense_keep = hw if dense_keep is None else int(dense_keep)
        if not (1 <= dense_keep <= hw):
            raise ValueError(f"dense_keep={dense_keep} out of range")
        counts = [int(dense_keep if bool(dm[i]) else floor_keep) for i in range(m)]

    sparse_k = sparse_k.view(b, nh, m, hw, d)
    sparse_v = sparse_v.view(b, nh, m, hw, d)

    sel = torch.zeros(b, nh, m, hw, dtype=torch.bool, device=local_k.device)
    if dm is not None and dense_keep == hw:
        sel[:, :, dm, :] = True  # GT frames: keep everything

    # v1 criterion for floored frames: per-head adjacent-frame similarity, keep least similar
    if m > 1:
        k_norm = F.normalize(sparse_k, dim=-1)
        adj = (k_norm[:, :, 1:] * k_norm[:, :, :-1]).sum(dim=-1)  # (b, nh, m-1, hw), bf16 like v1
    rows = torch.arange(h, device=local_k.device).view(h, 1).expand(h, w)
    cols = torch.arange(w, device=local_k.device).view(1, w).expand(h, w)
    stripe = max(1, int(round(hw / counts[0])))
    stripe_order = torch.argsort(((rows + cols) % stripe).flatten(), stable=True)
    for i in range(m):
        keep_i = counts[i]
        if dm is not None and bool(dm[i]) and dense_keep == hw:
            continue  # already set fully above
        if i == 0:
            sel[:, :, 0, stripe_order[:keep_i]] = True
        else:
            idxs = adj[:, :, i - 1, :].topk(keep_i, dim=-1, largest=False).indices
            sel[:, :, i, :].scatter_(-1, idxs, True)

    # gather in ascending (raster) order; per-head totals equal by construction
    flat = sel.view(b, nh, m * hw)
    total = int(flat[0, 0].sum())
    nz = flat.nonzero(as_tuple=False)
    flat_idx = nz[:, 2].view(b, nh, total)
    gidx = flat_idx[..., None].expand(-1, -1, -1, d)
    sel_k = torch.gather(sparse_k.reshape(b, nh, m * hw, d), 2, gidx)
    sel_v = torch.gather(sparse_v.reshape(b, nh, m * hw, d), 2, gidx)

    compressed_k = torch.cat([pre_k, sel_k.to(local_k.dtype), dense_k], dim=-2)
    compressed_v = torch.cat([pre_v, sel_v.to(local_v.dtype), dense_v], dim=-2)
    return compressed_k, compressed_v, sel.view(b, nh, m, h, w), compressed_k.shape[-2]
