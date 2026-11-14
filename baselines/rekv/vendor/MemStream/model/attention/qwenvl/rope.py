import torch
from typing import Union, Tuple, List
import os

from ezcolorlog import root_logger as logger

if not os.environ.get('ENABLE_LOGGING'):
    logger.disabled = True 

class RotaryEmbeddingQwenVL(torch.nn.Module):
    """
    Rotary position embeddings based on those in RoFormer.
    Query and keys are rotated with position-dependent complex-like rotations.
    """

    def __init__(
        self, 
        dim: int,                       # per-head hidden size (must be even)
        base: Union[int, int] = 10000, # RoPE frequency base (θ scaling)
        distance_scale: Union[int, float] = 1, # scales positions (e.g., NTK / rope-θ scaling)
        mrope_section: List[int] = [16, 24, 24], # default for QwenVL, 
    ):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale

        # Build inverse frequencies for even indices [0,2,4,...] in the head dim.
        # arange(..., device="cuda") pre-allocates on GPU; dtype float32.
        inv_freq = 1.0 / (
            float(base) ** (torch.arange(0, dim, 2, device=torch.device("cuda"), dtype=torch.float32) / dim)
        )
        # Save inv_freq as a non-trainable buffer (not persisted in state_dict by default).
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Caches for last computed tables to avoid recomputation per forward.
        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None

        # mRoPE settings 
        self.mrope_section = [int(x) for x in mrope_section]
        assert 2 * sum(self.mrope_section) == dim, (
            f"mRoPE sections {self.mrope_section} must satisfy 2*sum == dim ({dim})."
        )

        # store position ids for video 
        self.n_init = 0 
        self.position_ids = None 
        self._global_buffer_size = -1  # n_init + n_video_tokens

        # flags 
        self.initialized_for_qa = False 

        # device 
        self.device = None

        # Saved video RoPE cache for streaming mode
        self._video_cos_cached = None
        self._video_sin_cached = None
        self._video_seq_len_cached = None
        self._video_n_init = None
        self._video_position_ids = None 

    def _update_cos_sin_tables_interleaved(self, global_buffer_size, query_size=2000): 
        """ creates cos/sin tables with this format 
            |---- n_init (1D) ---- | ----- video (3D)----- | ---- query_text (1D) ---- | 
            |---- n_init (1D) ---- | ----- query text (1D)----- | ---- video (3D) ---- | 

        Args 
            - global_buffer_size (int): n_init + n_video_tokens 
            - query_size (int): default num of query tokens 
            - adaptive (bool): if True, then adjust the cos and sin tables based on the global buffer size each time 
        """ 

        n_init = self.n_init 

        assert hasattr(self, "n_init"), "n_init not initialized!"
        assert hasattr(self, "position_ids"), "position ids not initialized!" 

        # generate the position ids for the initial cache  
        init_position_ids = (
            torch.arange(n_init, device=self.device).type_as(self.inv_freq)
            .view(1, 1, -1)
            .expand(3, 1, -1) # (3, input_ids.shape[0], -1) 
        )

        # generate the position ids for the query text 
        query_position_ids = (
            torch.arange(query_size, device=self.device).type_as(self.inv_freq)
            .view(1, 1, -1)
            .expand(3, 1, -1) # (3, input_ids.shape[0], -1) 
        ).clone().to(self.device)        

        # generate the position ids for the video cache 
        L = ((self.position_ids[1].max() + 1) * (self.position_ids[2].max() + 1)).item()
        assert (global_buffer_size - n_init) % L == 0, f"{global_buffer_size} - {n_init} does not fit all frames of token-length {L}"
        # global_buffer_frames = (global_buffer_size - n_init) // L 
        video_position_ids = self.position_ids.clone().to(self.device) # (3, 1, L) 
        video_position_ids = video_position_ids[:, :, :(global_buffer_size - n_init)]

         # shift the t-indices in the cache by n_init (init_position_ids is between 0 and 13)
        pos_shift = n_init 
        video_position_ids += pos_shift

        # shift the query position ids in the cache by incrementing 
        # the MAXIMUM position ID of the preceding modality by one
        if video_position_ids.numel() == 0:
            raise RuntimeError(
                f"empty video position ids: global_buffer_size={global_buffer_size}, n_init={n_init}; "
                f"cannot anchor the query positions after the video segment"
            )
        pos_shift = video_position_ids.max().item() + 1
        query_position_ids = query_position_ids.clone() + pos_shift

         # create the final set of position ids 
        position_ids = torch.cat([init_position_ids, 
                                  video_position_ids.to(init_position_ids.device), 
                                  query_position_ids.to(init_position_ids.device)], dim=-1)

        # compute the cosine and sine tables based on position ids 
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.distance_scale
        sin = emb.sin() * self.distance_scale

        # (3, 1, N, dim)
        self._cos_cached = cos 
        self._sin_cached = sin 
        self._seq_len_cached = cos.size(-2) 
        self._global_buffer_size = global_buffer_size

        # done initializing cos/sin tables for question answering 
        self.initialized_for_qa = True 

        return self._cos_cached, self._sin_cached
    
    def _update_cos_sin_tables_pos(self, device, video_position_ids, n_init): 
        """ M-RoPE - Given video_grid_thw - generate the full position ids and then compute cos/sin tables 
            Args: 
                - device: torch.device - cuda or cpu 
                - video_cache_position_ids: (3, 1, n_video_tokens) 
        """

        # generate the position ids for the input text 
        init_position_ids = (
            torch.arange(n_init, device=device).type_as(self.inv_freq)
            .view(1, 1, -1)
            .expand(3, 1, -1) # (3, input_ids.shape[0], -1) 
        )

        # logger.info(f"INV FREQ - {self.inv_freq.device} - {device} - {init_position_ids.device}")
        self.n_init = n_init
        self.position_ids = video_position_ids 

        # shift the t,h,w-indices in the cache by n_init (init_position_ids is between 0 and 13)
        t_shift = n_init 
        position_ids = video_position_ids.clone().to(device) 
        position_ids += t_shift 

        # create the final set of position ids 
        position_ids = torch.cat([init_position_ids, position_ids.to(init_position_ids.device)], dim=-1)

        # compute the cosine and sine tables based on position ids 
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

        print(f"[use_cache=True] inv_freq_expanded shape: {inv_freq_expanded.shape}, mean: {inv_freq_expanded.mean():.6f}, std: {inv_freq_expanded.std():.6f}")
        print(f"[use_cache=True] position_ids shape: {position_ids.shape}, mean: {position_ids.float().mean():.6f}, std: {position_ids.float().std():.6f}")
        print(f"[use_cache=True] freqs shape: {freqs.shape}, mean: {freqs.float().mean():.6f}, std: {freqs.float().std():.6f}")

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.distance_scale
        sin = emb.sin() * self.distance_scale

        # (3, 1, N, dim)
        self._cos_cached = cos 
        self._sin_cached = sin 
        self._seq_len_cached = cos.size(-2) 

        # reset for the next video 
        self.initialized_for_qa = False 

        logger.info(f"updated cos-sin based on pos {self._cos_cached.shape}, {self._sin_cached.shape} - {video_position_ids.shape}")

    def save_video_rope_cache(self):
        """Save the current video RoPE cache for later restoration (streaming mode)"""
        self._video_cos_cached = self._cos_cached.clone() if self._cos_cached is not None else None
        self._video_sin_cached = self._sin_cached.clone() if self._sin_cached is not None else None
        self._video_seq_len_cached = self._seq_len_cached
        self._video_n_init = self.n_init
        self._video_position_ids = self.position_ids.clone() if self.position_ids is not None else None

    def restore_video_rope_cache(self):
        """Restore the saved video RoPE cache (for streaming mode when switching back from QA)"""
        if self._video_cos_cached is not None:
            self._cos_cached = self._video_cos_cached
            self._sin_cached = self._video_sin_cached
            self._seq_len_cached = self._video_seq_len_cached
            self.n_init = self._video_n_init
            self.position_ids = self._video_position_ids
            self.initialized_for_qa = False

        return self._cos_cached, self._sin_cached

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _interleave_axes(self, cos_or_sin_3: torch.Tensor, mrope_section: int, unsqueeze_dim: int):
        """
        cos_or_sin_3: (3, 1, L, D)  # axis 0: [T,H,W]
        Returns: (1, 1, L, D) after cycling T,H,W per chunk along -1.
        """
        # match your snippet exactly
        chunk = mrope_section * 2                          # account for rotate_half pairing
        pieces = cos_or_sin_3.split(chunk, dim=-1)         # each: (3,1,L,chunk)
        out = torch.cat([p[i % 3] for i, p in enumerate(pieces)], dim=-1)  # (1,L,D)
        return out.unsqueeze(unsqueeze_dim)                # -> (1,1,L,D) for broadcast

    @torch.no_grad()
    def apply_rotary_pos_emb_one_angle_mrope(self, x: torch.Tensor, index, **kwargs): 
        """
        Use cached tables (3,1,N,D) to build per-axis cos/sin.
        - Temporal axis uses the single row at `index-1` (anchored across the whole chunk).
        - Height/Width use their true positions from the current slice [right-length:right].
        """
        assert self._cos_cached is not None and self._sin_cached is not None
        assert self._cos_cached.dim() == 4 and self._cos_cached.size(0) == 3, \
            "Expected cached mRoPE tables of shape (3,1,N,D)"
        
        # move to proper device 
        self._cos_cached = self._cos_cached.to(self.inv_freq.device)
        self._sin_cached = self._sin_cached.to(self.inv_freq.device)
        
        dtype = x.dtype 

        # pass length for 1D input or compute based on video position ids  
        L = kwargs.get("L", ((self.position_ids[1].max() + 1) * (self.position_ids[2].max() + 1)).item())
        logger.info(f"L: {L}")

        right = index 
        input_length = x.size(-2) 

        assert input_length >= L, f"{L} > {input_length}"
        n_repeats = input_length // L 
        
        # Slice H/W (and T for shape) for current span
        cos_3 = self._cos_cached[:, :, right - L:right, :].to(x.dtype).repeat(1, 1, n_repeats, 1)   # (3,1,L,D)
        sin_3 = self._sin_cached[:, :, right - L:right, :].to(x.dtype).repeat(1, 1, n_repeats, 1)   # (3,1,L,D)

        # Interleave T/H/W per your mrope_section cycling
        cos = self._interleave_axes(cos_3, self.mrope_section, unsqueeze_dim=1)  # (1,1,L,D)
        sin = self._interleave_axes(sin_3, self.mrope_section, unsqueeze_dim=1)  # (1,1,L,D)

        # Apply to queries (global_q). H/W are now properly positional; T is anchored.
        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)
    
    def apply_rotary_pos_emb(self, x, length, right, cos, sin):

        if cos.device != x.device:
            cos = cos.to(x.device)
            sin = sin.to(x.device)

        # x: tensor to rotate (e.g., q or k). length = current subsequence length to index.
        # right = total current sequence length (end index). We slice cos/sin to the last `length` steps.
        logger.info(f"x: {x.shape}, length: {length}, right: {right}, cos: {cos.shape}, sin: {sin.shape}")
        dtype = x.dtype
        if cos.dim() == 2:
            # shape: [T, D]
            cos = cos[right-length:right, :]
            sin = sin[right-length:right, :]
        elif cos.dim() == 3:
            # shape: [1, T, D] (broadcastable over batch)
            cos = cos[:, right-length:right, :]
            sin = sin[:, right-length:right, :]
        elif  cos.dim() == 4:
            # shape: [1, 1, T, D] (broadcastable over batch & heads)
            cos = cos[:, :, right-length:right, :]
            sin = sin[:, :, right-length:right, :]

        # Interleave T/H/W per mrope_section cycling
        if cos.size(0) == 3: 
            cos = self._interleave_axes(cos, self.mrope_section, unsqueeze_dim=1)  # (1,1,L,D)
            sin = self._interleave_axes(sin, self.mrope_section, unsqueeze_dim=1)  # (1,1,L,D)            

        # Apply rotation: x*cos + rotate_half(x)*sin
        # Cast to float for numerical stability, then back to original dtype (e.g. fp16/bf16).
        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)

    def _update_cos_sin_tables(self, x, seq_dim):
        # Ensure cos/sin caches cover sequence length used by x along seq_dim.
        seq_len = x.size(seq_dim)
        if seq_len > self._seq_len_cached:
            # Need larger tables → (re)compute
            self._seq_len_cached = seq_len
            # Positions t = [0..T-1] on same device/dtype family as inv_freq
            t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
            # Outer product: [T] x [D/2] → [T, D/2]; multiply positions by inv_freq (scaled by distance_scale)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            # Duplicate along last dim so we have [T, D] (pairing cos/sin for each 2-dim block)
            emb = torch.cat((freqs, freqs), dim = -1)
            # Precompute cos/sin with broadcast-friendly leading dims depending on x.ndim.
            if x.dim() == 2:
                # [T, D]
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
            elif x.dim() == 3:
                # [1, T, D]
                self._cos_cached = emb.cos()[None, :, :]
                self._sin_cached = emb.sin()[None, :, :]
            elif x.dim() == 4:
                # [1, 1, T, D]
                self._cos_cached = emb.cos()[None, None, :, :]
                self._sin_cached = emb.sin()[None, None, :, :]
        # Return the (possibly updated) caches.
        return self._cos_cached, self._sin_cached

    def _update_cos_sin_tables_len(self, seq_len, device=None, dim = None):
        """ 1D RoPE - update caches to cover an explicit seq_len, without an input tensor. """
        if dim is None:
            # If dim not given, infer from existing cache
            assert self._cos_cached is not None
            dim = self._cos_cached.dim()

        if device is None: 
            device = self.device 

        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.outer(t * self.distance_scale, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)
        # Match previously chosen broadcast rank.
        if dim == 2:
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        elif dim == 3:
            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]
        elif dim == 4:
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]  

        self.device = device 

        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb_one_angle(
        self, x: torch.Tensor, index, **kwargs
    ):
        """ Apply rotary embeddings for one angle across all features """

        logger.info(f"Apply Rotary Pos Emb: {x.shape}")
        logger.info(f"INV FREQ - {self.inv_freq.device} - {x.device}")

        """ apply M-RoPE """
        if self._cos_cached.size(0) == 3: 
            return self.apply_rotary_pos_emb_one_angle_mrope(x, index, **kwargs)

        """ apply 1D-RoPE """
        # Apply RoPE for a single position `index` (1-based slice end) to x.
        dtype = x.dtype
        cos, sin = self._update_cos_sin_tables_len(index, x.device)
        # Narrow cos/sin to just the last position index-1:index along time.
        if cos.dim() == 2:
            cos = cos[index-1:index, :]
            sin = sin[index-1:index, :]
        elif cos.dim() == 3:
            cos = cos[:, index-1:index, :]
            sin = sin[:, index-1:index, :]
        elif  cos.dim() == 4:
            cos = cos[:, :, index-1:index, :]
            sin = sin[:, :, index-1:index, :]

        # Rotate the vector at this single time step.
        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)

    def _select_from_mask(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Select embeddings based on a 3D boolean mask.
        
        Args:
            x: Embeddings (3, 1, M*H*W, D) or (3, 1, M, H, W, D)
            m: Boolean mask (M, H, W)
        Returns:
            Selected embeddings (3, 1, N, D) where N = number of True values in mask
        """
        m = m.to(self.device)
        M, H, W = m.shape
        
        # Ensure x is in spatial shape
        if x.dim() == 4:
            x = x.reshape(3, 1, M, H, W, x.size(-1))
        
        # Get indices of True values
        indices = torch.nonzero(m, as_tuple=False)  # (N, 3) [frame_idx, h_idx, w_idx]
        return x[:, :, indices[:, 0], indices[:, 1], indices[:, 2], :]  # (3, 1, N, D)
    
    def _select_from_positions(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Select embeddings based on explicit position indices.
        
        Args:
            x: Embeddings (3, 1, M, H, W, D)
            positions: Position indices (n_tokens, 3) where last dim is [frame, h, w]
        Returns:
            Selected embeddings (3, 1, n_tokens, D)
        """
        frame_idx = positions[:, 0].long()
        h_idx = positions[:, 1].long()
        w_idx = positions[:, 2].long()
        return x[:, :, frame_idx, h_idx, w_idx, :]
    
    def _concat_cos_sin_parts(self, init_part, sparse_part, dense_part):
        """Concatenate init, sparse, and dense parts of cos/sin embeddings."""
        return torch.cat([init_part, sparse_part, dense_part], dim=2)
    
    def _process_per_head(self, q, k, nh_q, nh_kv, seq_dim, 
                          cos_init, sin_init, cos_sparse, sin_sparse, cos_dense, sin_dense,
                          sparse_selector_fn):
        """Process RoPE per-head with different sparse selections per head.
        
        Args:
            sparse_selector_fn: Function that takes head_idx and returns (cos_selected, sin_selected)
        """
        q_heads_lst, k_heads_lst = [], []
        
        for head_idx in range(nh_kv):
            q_head_idx = head_idx * nh_q // nh_kv
            
            # Get sparse cos/sin for this head
            cos_selected, sin_selected = sparse_selector_fn(head_idx)
            
            # Concatenate parts
            cos_head = self._concat_cos_sin_parts(cos_init, cos_selected, cos_dense)
            sin_head = self._concat_cos_sin_parts(sin_init, sin_selected, sin_dense)
            
            # Extract head slices
            q_head = q[:, q_head_idx:q_head_idx+nh_q//nh_kv, :, :]
            k_head = k[:, head_idx:head_idx+1, :, :]
            
            # Apply RoPE
            q_head_rope = self.apply_rotary_pos_emb(q_head, q_head.size(seq_dim), cos_head.size(2), cos_head, sin_head)
            k_head_rope = self.apply_rotary_pos_emb(k_head, k_head.size(seq_dim), cos_head.size(2), cos_head, sin_head)
            
            q_heads_lst.append(q_head_rope)
            k_heads_lst.append(k_head_rope)
        
        q_out = torch.cat(q_heads_lst, dim=1).to(q.dtype).to(q.device)
        k_out = torch.cat(k_heads_lst, dim=1).to(k.dtype).to(k.device)

        return q_out, k_out

    def forward_subsampled(self, q: torch.Tensor, k: torch.Tensor, 
                seq_dim=-2, right=None,
                subsample_window=None, subsample_mask=None, 
                **kwargs):
        """ Apply RoPE with subsampled embeddings based on mask or position mapping.
        
        Args: 
            q: (batch, n_head, len_q, dim_head)
            k: (batch, n_head_kv, len_kv, dim_head)
            seq_dim: dimension of the sequence
            right: right index for the sequence
            subsample_window: (init_exc, length)
            subsample_mask: Can be one of:
                - (M, H, W): 3D static mask for all heads (dilated, random methods)
                - (b, nh, M, H, W): 5D per-head mask (clustering method)
                - (b, nh, n_tokens, 3): 4D position mapping where last dim is [frame, h, w] (ToMe method)
            kwargs: Additional keyword arguments
        Returns: 
            q: (batch, n_head, len_q, dim_head)
            k: (batch, n_head_kv, len_kv, dim_head)
        """
        if not subsample_window:
            return q, k
            
        assert subsample_mask is not None, "subsample_mask must be provided when subsample_window is set"
        init_exc, length = subsample_window

        # Crop cos/sin tables to relevant range
        cos = self._cos_cached[:, :, self.n_init:right]
        sin = self._sin_cached[:, :, self.n_init:right]
        
        # === 3D MASK: Static mask for all heads (dilated, random) ===
        if subsample_mask.dim() == 3:
            M, H, W = subsample_mask.shape
            
            # Reshape to spatial dimensions
            cos = cos.reshape(3, 1, -1, H, W, cos.size(-1))
            sin = sin.reshape(3, 1, -1, H, W, sin.size(-1))

            n_frames_to_subsample = (length - self.n_init) // (H * W) if not init_exc else length // (H * W)
            assert n_frames_to_subsample <= M, f"n_frames_to_subsample ({n_frames_to_subsample}) > mask frames ({M})"
            
            # Select based on mask and concatenate parts
            cos_sparse = self._select_from_mask(cos[:, :, :n_frames_to_subsample], subsample_mask[:n_frames_to_subsample])
            sin_sparse = self._select_from_mask(sin[:, :, :n_frames_to_subsample], subsample_mask[:n_frames_to_subsample])
            
            cos_final = self._concat_cos_sin_parts(
                self._cos_cached[:, :, :self.n_init],
                cos_sparse,
                cos[:, :, n_frames_to_subsample:].reshape(3, 1, -1, cos.size(-1))
            )
            sin_final = self._concat_cos_sin_parts(
                self._sin_cached[:, :, :self.n_init],
                sin_sparse,
                sin[:, :, n_frames_to_subsample:].reshape(3, 1, -1, sin.size(-1))
            )
            
            return (
                self.apply_rotary_pos_emb(q, q.size(seq_dim), cos_final.size(2), cos_final, sin_final),
                self.apply_rotary_pos_emb(k, k.size(seq_dim), cos_final.size(2), cos_final, sin_final),
            )
        
        # === 5D MASK: Per-head mask (clustering) ===
        elif subsample_mask.dim() == 5:
            nh_q = q.shape[1]
            b, nh_kv, M, H, W = subsample_mask.shape
            assert b == 1, "batch size must be 1 for per-head masks"
            assert k.shape[1] == nh_kv, f"Number of KV heads in k ({k.shape[1]}) must match mask heads ({nh_kv})"
            
            # Reshape to spatial dimensions
            cos = cos.reshape(3, 1, -1, H, W, cos.size(-1))
            sin = sin.reshape(3, 1, -1, H, W, sin.size(-1))
            
            n_frames_to_subsample = (length - self.n_init) // (H * W) if not init_exc else length // (H * W)
            assert n_frames_to_subsample <= M, f"n_frames_to_subsample ({n_frames_to_subsample}) > mask frames ({M})"
            
            # Prepare common parts
            cos_init = self._cos_cached[:, :, :self.n_init]
            sin_init = self._sin_cached[:, :, :self.n_init]
            cos_dense = cos[:, :, n_frames_to_subsample:].reshape(3, 1, -1, cos.size(-1))
            sin_dense = sin[:, :, n_frames_to_subsample:].reshape(3, 1, -1, sin.size(-1))
            
            # Define selector function for each head
            def select_for_head(head_idx):
                head_mask = subsample_mask[0, head_idx, :n_frames_to_subsample]  # (n_frames, H, W)
                cos_selected = self._select_from_mask(cos[:, :, :n_frames_to_subsample], head_mask)
                sin_selected = self._select_from_mask(sin[:, :, :n_frames_to_subsample], head_mask)
                return cos_selected, sin_selected
            
            return self._process_per_head(q, k, nh_q, nh_kv, seq_dim,
                                         cos_init, sin_init, cos, sin, cos_dense, sin_dense,
                                         select_for_head)
        
        # === 4D POSITION MAPPING: Explicit positions for ToMe ===
        elif subsample_mask.dim() == 4 and subsample_mask.shape[-1] == 3:
            nh_q = q.shape[1]
            b, nh_kv, n_sparse_tokens, _ = subsample_mask.shape
            position_mapping = subsample_mask

            assert b == 1, "batch size must be 1 for position mapping"
            assert k.shape[1] == nh_kv, f"Number of KV heads in k ({k.shape[1]}) must match mapping heads ({nh_kv})"
            
            # Extract grid dimensions from position mapping
            H, W = kwargs.get("height_width") 
            assert H is not None and W is not None, "height_width must be provided when subsample_mask is a position mapping"

            # Reshape to spatial dimensions
            cos = cos.reshape(3, 1, -1, H, W, cos.size(-1))
            sin = sin.reshape(3, 1, -1, H, W, sin.size(-1))
            
            n_frames_to_subsample = (length - self.n_init) // (H * W) if not init_exc else length // (H * W)
            assert n_frames_to_subsample <= cos.shape[2], f"n_frames_to_subsample ({n_frames_to_subsample}) > available frames ({cos.shape[2]})"
            
            # Prepare common parts
            cos_init = self._cos_cached[:, :, :self.n_init]
            sin_init = self._sin_cached[:, :, :self.n_init]
            cos_dense = cos[:, :, n_frames_to_subsample:].reshape(3, 1, -1, cos.size(-1))
            sin_dense = sin[:, :, n_frames_to_subsample:].reshape(3, 1, -1, sin.size(-1))
            
            # Define selector function for each head
            def select_for_head(head_idx):
                head_positions = position_mapping[0, head_idx]  # (n_sparse_tokens, 3)
                cos_selected = self._select_from_positions(cos, head_positions)
                sin_selected = self._select_from_positions(sin, head_positions)
                return cos_selected, sin_selected
            
            return self._process_per_head(q, k, nh_q, nh_kv, seq_dim,
                                         cos_init, sin_init, cos, sin, cos_dense, sin_dense,
                                         select_for_head)


    def forward(self, q: torch.Tensor, k: torch.Tensor, 
                seq_dim= -2, global_buffer_size=None, 
                init_exc=False, right=None, 
                update_global_buffer_size=False, 
                subsample_window=None, subsample_mask=None,
                **forward_subsampled_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # If subsampling is requested, use the specialized method
        if subsample_window is not None:
            assert right is not None
            right_qk = right + self.n_init if init_exc else right
            return self.forward_subsampled(q, k, seq_dim, right_qk, subsample_window, subsample_mask, **forward_subsampled_kwargs)
        
        """ apply 1D-RoPE - update cos/sin tables """
        if self._cos_cached.size(0) != 3: 
            # Ensure cos/sin cover the k-sequence length (k typically defines context length).
            self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim=seq_dim)

        """ apply RoPE (both 1D and M-RoPE) """
        # update the cos/sin tables for question answering
        if global_buffer_size and not self.initialized_for_qa: 
            self._update_cos_sin_tables_interleaved(global_buffer_size)
        elif update_global_buffer_size: 
            assert global_buffer_size is not None 
            self._update_cos_sin_tables_interleaved(global_buffer_size)

        query_cos = self._cos_cached
        query_sin = self._sin_cached

        # right index for k - offset by n_init tokens if init_exc is True 
        right_k = k.size(seq_dim) + self.n_init if init_exc else k.size(seq_dim)
        # query right index is same as k if not specified
        right = right_k if right is None else right 

        # Rotate q and k with appropriate slices from the cached tables.
        return (
            self.apply_rotary_pos_emb(q, q.size(seq_dim), right, query_cos, query_sin),
            self.apply_rotary_pos_emb(k, k.size(seq_dim), right_k, self._cos_cached, self._sin_cached),
        )