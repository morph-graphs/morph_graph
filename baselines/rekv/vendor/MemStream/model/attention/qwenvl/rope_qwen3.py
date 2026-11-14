import torch
from typing import Union, Tuple, List
import os
from itertools import accumulate 

from ezcolorlog import root_logger as logger

if not os.environ.get('ENABLE_LOGGING'):
    logger.disabled = True 

class RotaryEmbeddingQwen3VL(torch.nn.Module):
    """
    Rotary position embeddings based on those in RoFormer.
    Query and keys are rotated with position-dependent complex-like rotations.
    """

    def __init__(
        self, 
        dim: int,                       # per-head hidden size (must be even)
        base: Union[int, int] = 10000, # RoPE frequency base (θ scaling)
        distance_scale: Union[int, float] = 1, # scales positions (e.g., NTK / rope-θ scaling)
        trope_section: List[int] = [24, 20, 20], # default for QwenVL, 
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

        # T-RoPE section settings (temporal, height, width frequency allocation)
        self.trope_section = [int(x) for x in trope_section]
        assert 2 * sum(self.trope_section) == dim, (
            f"T-RoPE sections {self.trope_section} must satisfy 2*sum == dim ({dim})."
        )

        # store position ids for video 
        self.n_init = 0 
        self.position_ids = None 
        self._global_buffer_size = -1  # n_init + n_video_tokens

        # flags 
        self._has_trope_tables = False  # True once video-specific T-RoPE tables are built

        # device 
        self.device = None

        # Saved video RoPE cache for streaming mode
        self._video_cos_cached = None
        self._video_sin_cached = None
        self._video_seq_len_cached = None
        self._video_n_init = None
        self._video_position_ids = None

        # Position ids the current _cos_cached/_sin_cached window table was built from.
        # This is the invalidation key for the skip in _recompute_cos_sin_tables_for_window:
        # the table is reused only when a fresh build would use EXACTLY these ids. Set to
        # None by every other builder that overwrites _cos_cached, so the skip can never
        # pair a stale key with a foreign table.
        self._built_window_pos_ids = None
        self._video_built_window_pos_ids = None

    # ---- REKV_ROPE_TRACE audit helpers (instrumentation only; no behaviour change) ----
    def _rt_fresh_window_pos_ids(self, device, video_position_ids, n_init, max_pos):
        """Mirror the position-id math of _recompute_cos_sin_tables_for_window without
        touching any shared state. Returns the right-justified (3,1,N) position ids a
        fresh rebuild for this request would use."""
        init_position_ids = (
            torch.arange(n_init, device=device).type_as(self.inv_freq)
            .view(1, 1, -1)
            .expand(3, 1, -1)
        )
        position_ids = video_position_ids.clone().to(device) + n_init
        position_ids = torch.cat([init_position_ids, position_ids.to(init_position_ids.device)], dim=-1)
        return position_ids + (max_pos - position_ids.max())

    def _rt_cos_sin_from_pos(self, position_ids):
        """Compute interleaved-trope cos/sin tables from explicit position ids (audit only)."""
        position_ids = position_ids.to(self.device)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_trope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.distance_scale, emb.sin() * self.distance_scale

    def _update_cos_sin_tables_interleaved(self, retrieved_position_ids, query_size=2000):
        """ creates cos/sin tables with this format 
            |---- n_init (1D) ---- | ----- video (3D)----- | ---- query_text (1D) ---- | 

        Args 
            - retrieved_position_ids: explicit packed retrieved-video position ids
            - query_size (int): default num of query tokens 
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

        assert retrieved_position_ids is not None, "retrieved_position_ids are required for Qwen3 retrieved RoPE"
        retrieved_position_ids = retrieved_position_ids.to(self.device).type_as(self.inv_freq)
        position_ids = torch.cat([init_position_ids, retrieved_position_ids], dim=-1)
        global_buffer_size = position_ids.size(-1)

        # shift the query position ids in the cache by incrementing 
        # the MAXIMUM position ID of the preceding modality by one
        pos_shift = position_ids.max().item() + 1
        query_position_ids = query_position_ids.clone() + pos_shift

        position_ids = torch.cat([position_ids, query_position_ids.to(position_ids.device)], dim=-1)

        # compute the cosine and sine tables based on position ids 
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_trope(freqs)

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.distance_scale
        sin = emb.sin() * self.distance_scale

        # (1, N, dim) — pre-interleaved T-RoPE
        self._cos_cached = cos 
        self._sin_cached = sin 
        self._seq_len_cached = cos.size(-2) 
        self._global_buffer_size = global_buffer_size
        self._has_trope_tables = True

        # keep track of current block-sizes
        self.curr_block_sizes = []
        # this table is the QA (interleaved) table, not a window table
        self._built_window_pos_ids = None

        if os.environ.get("REKV_ROPE_TRACE"):
            self._rt_kind = "interleaved"
            self._rt_pos_ids = position_ids.clone()
            self._rt_builder = getattr(self, "_rt_caller", None)
            self._rt_n_interleaved = getattr(self, "_rt_n_interleaved", 0) + 1
            if self._rt_n_interleaved <= 80 or self._rt_n_interleaved % 5000 == 0:
                print(
                    f"[RTRACE] INTERLEAVED_BUILD n={self._rt_n_interleaved} "
                    f"table={cos.size(-2)} retr_len={retrieved_position_ids.size(-1)}",
                    flush=True,
                )

        return self._cos_cached, self._sin_cached

    def _update_cos_sin_tables_pos(self, device, video_position_ids, n_init, block_sizes, video_block_size): 
        """ M-RoPE - Given video_grid_thw - generate the full position ids and then compute cos/sin tables 
            Args: 
                - device: torch.device - cuda or cpu 
                - video_cache_position_ids: (3, 1, n_video_tokens) 
                - n_init: int - number of init tokens
                - block_sizes: list[int] - per-frame token counts for frames in this window (including timestamp and <st> and <ed> tokens)
                - video_block_size: int - number of video tokens per frame
        """

        # generate the position ids for the input text 
        init_position_ids = (
            torch.arange(n_init, device=device).type_as(self.inv_freq)
            .view(1, 1, -1)
            .expand(3, 1, -1) # (3, input_ids.shape[0], -1) 
        )

        self.n_init = n_init
        self.position_ids = video_position_ids 
        self.video_block_size = video_block_size

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
        freqs = self.apply_interleaved_trope(freqs)

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.distance_scale
        sin = emb.sin() * self.distance_scale

        # (1, N, dim) — pre-interleaved T-RoPE
        self._cos_cached = cos 
        self._sin_cached = sin 
        self._seq_len_cached = cos.size(-2) 
        self._has_trope_tables = True

        # update curr_block_sizes
        self.curr_block_sizes = block_sizes
        # this is the full-video table (not right-justified to a window); force the first
        # window recompute to rebuild rather than pair its key with this table
        self._built_window_pos_ids = None

        if os.environ.get("REKV_ROPE_TRACE"):
            self._rt_kind = "full"
            self._rt_pos_ids = position_ids.clone()
            self._rt_full_pos_ids = position_ids.clone()
            self._rt_builder = getattr(self, "_rt_caller", None)
            print(
                f"[RTRACE] FULL_BUILD table={cos.size(-2)} n_init={n_init} "
                f"builder={self._rt_builder}",
                flush=True,
            )

        logger.info(f"updated cos-sin based on pos {self._cos_cached.shape}, {self._sin_cached.shape} - {video_position_ids.shape}")

    def _recompute_cos_sin_tables_for_window(self, device, video_position_ids, n_init, block_sizes, max_pos):
        """Recompute cos/sin tables for a shifted sliding window WITHOUT overwriting _seq_len_cached.
        
        Right-justifies position_ids so that the max position equals the original _seq_len_cached
        (the LM-Infinite cap set during the first call to _update_cos_sin_tables_pos).
        
        Args:
            device: torch.device
            video_position_ids: (3, 1, n_window_tokens) - position ids for the current window's video tokens
            n_init: int - number of init/sink tokens
            block_sizes: list[int] - per-frame token counts for frames in this window
        """
        assert self._seq_len_cached is not None, "Must call _update_cos_sin_tables_pos first to set the anchor"

        # Compute the position ids a fresh build would use. This is cheap (index arithmetic,
        # a concat, and a shift on a (3, 1, N) tensor); the expensive part of a rebuild is
        # the frequency matmul + cos/sin below, which stays behind the skip.
        init_position_ids = (
            torch.arange(n_init, device=device).type_as(self.inv_freq)
            .view(1, 1, -1)
            .expand(3, 1, -1)
        )

        # shift video positions by n_init
        position_ids = video_position_ids.clone().to(device)
        position_ids += n_init

        # concatenate init + video position ids
        position_ids = torch.cat([init_position_ids, position_ids.to(init_position_ids.device)], dim=-1)

        # right-justify: shift so max position
        t_shift = max_pos - position_ids.max()
        position_ids = position_ids + t_shift

        # Skip the rebuild only when the cached table was built from EXACTLY these position
        # ids, i.e. reuse is numerically exact (same window POSITION, not just same shape).
        # Keying on block_sizes alone was a bug: shapes never change at a fixed token budget,
        # so the skip fired on 100% of sliding reads against a stale-anchored table.
        # AUDIT: REKV_ROPE_FORCE_REBUILD=1 disables the skip (experimental A/B arm).
        _prev_pos_ids = getattr(self, "_built_window_pos_ids", None)
        _skip = (
            _prev_pos_ids is not None
            and _prev_pos_ids.shape == position_ids.shape
            and bool(torch.equal(_prev_pos_ids.to(position_ids.device), position_ids))
            and not os.environ.get("REKV_ROPE_FORCE_REBUILD")
        )
        _trace = bool(os.environ.get("REKV_ROPE_TRACE"))
        if _trace and _skip:
            # AUDIT: the requesting layer will reuse whatever table is currently cached.
            # Compare the position ids that table was built from against the ids a fresh
            # rebuild for THIS request would use. Identical => reuse is numerically exact.
            caller = getattr(self, "_rt_caller", None)
            fresh = self._rt_fresh_window_pos_ids(device, video_position_ids, n_init, max_pos)
            prev = getattr(self, "_rt_pos_ids", None)
            self._rt_n_skip = getattr(self, "_rt_n_skip", 0) + 1
            same = (
                prev is not None
                and prev.shape == fresh.shape
                and bool(torch.equal(prev.to(fresh.device), fresh))
            )
            if same:
                self._rt_n_skip_safe = getattr(self, "_rt_n_skip_safe", 0) + 1
            else:
                self._rt_n_skip_stale = getattr(self, "_rt_n_skip_stale", 0) + 1
                if self._rt_n_skip_stale <= 200:
                    if prev is None or prev.shape != fresh.shape:
                        print(
                            f"[RTRACE] STALE_SKIP caller={caller} "
                            f"builder={getattr(self, '_rt_builder', None)} "
                            f"kind={getattr(self, '_rt_kind', None)} "
                            f"prev_shape={None if prev is None else tuple(prev.shape)} "
                            f"fresh_shape={tuple(fresh.shape)}",
                            flush=True,
                        )
                    else:
                        prev_d = prev.to(fresh.device)
                        rows_diff = int((prev_d != fresh).any(dim=0).any(dim=0).sum())
                        max_pos_delta = float((prev_d.float() - fresh.float()).abs().max())
                        cos_f, sin_f = self._rt_cos_sin_from_pos(fresh)
                        max_dcos = float((cos_f - self._cos_cached.to(cos_f.device)).abs().max())
                        max_dsin = float((sin_f - self._sin_cached.to(sin_f.device)).abs().max())
                        # Delta structure: a row-uniform position shift cancels in attention
                        # (q and k share the table; logits see only position differences).
                        # frac_qshift = fraction of rows whose delta equals the LAST row's
                        # delta (the q anchor). 1.0 => pure global shift, attention-invariant.
                        rel = ""
                        for _nm, _dim in (("t", 0), ("h", 1), ("w", 2)):
                            _d = (fresh.float() - prev_d.float())[_dim, 0]
                            _uni = int(torch.unique(_d).numel())
                            _frac = float((_d == _d[-1]).float().mean())
                            rel += f" {_nm}_uniq={_uni} {_nm}_frac_qshift={_frac:.4f}"
                        print(
                            f"[RTRACE] STALE_SKIP caller={caller} "
                            f"builder={getattr(self, '_rt_builder', None)} "
                            f"kind={getattr(self, '_rt_kind', None)} "
                            f"rows_diff={rows_diff}/{fresh.size(-1)} "
                            f"max_pos_delta={max_pos_delta} "
                            f"max_dcos={max_dcos:.6f} max_dsin={max_dsin:.6f}{rel}",
                            flush=True,
                        )
            if self._rt_n_skip % 5000 == 0:
                print(
                    f"[RTRACE] SKIP_SUMMARY total={self._rt_n_skip} "
                    f"safe={getattr(self, '_rt_n_skip_safe', 0)} "
                    f"stale={getattr(self, '_rt_n_skip_stale', 0)} "
                    f"rebuilds={getattr(self, '_rt_n_rebuild', 0)}",
                    flush=True,
                )
        if _skip:
            return

        self.n_init = n_init
        self.position_ids = video_position_ids

        # compute cos/sin tables (position_ids were computed above, before the skip test)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_trope(freqs)

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.distance_scale
        sin = emb.sin() * self.distance_scale

        # update cos/sin but NOT _seq_len_cached
        self._cos_cached = cos
        self._sin_cached = sin

        # record the position ids this table was built from (the skip's invalidation key).
        # position_ids is a fresh tensor (cat + add), so no aliasing with caller state.
        self._built_window_pos_ids = position_ids

        # update block sizes for downstream RoPE application
        self.curr_block_sizes = block_sizes

        if _trace:
            self._rt_kind = "window"
            self._rt_pos_ids = position_ids.clone()
            self._rt_builder = getattr(self, "_rt_caller", None)
            self._rt_n_rebuild = getattr(self, "_rt_n_rebuild", 0) + 1
            # ADDITIVE trigger: post-fix, rebuilds dominate and skips are rare, so the
            # skip-side SKIP_SUMMARY (every 5000 skips) may never fire. Emit the same
            # summary from the rebuild side as well.
            if self._rt_n_rebuild % 5000 == 0:
                print(
                    f"[RTRACE] SKIP_SUMMARY total={getattr(self, '_rt_n_skip', 0)} "
                    f"safe={getattr(self, '_rt_n_skip_safe', 0)} "
                    f"stale={getattr(self, '_rt_n_skip_stale', 0)} "
                    f"rebuilds={self._rt_n_rebuild}",
                    flush=True,
                )

    def save_video_rope_cache(self):
        """Save the current video RoPE cache for later restoration (streaming mode)"""
        self._video_cos_cached = self._cos_cached.clone() if self._cos_cached is not None else None
        self._video_sin_cached = self._sin_cached.clone() if self._sin_cached is not None else None
        self._video_seq_len_cached = self._seq_len_cached
        self._video_n_init = self.n_init
        self._video_position_ids = self.position_ids.clone() if self.position_ids is not None else None
        self._video_built_window_pos_ids = (
            self._built_window_pos_ids.clone() if self._built_window_pos_ids is not None else None
        )
        if os.environ.get("REKV_ROPE_TRACE"):
            self._video_rt_state = (
                getattr(self, "_rt_kind", None),
                getattr(self, "_rt_pos_ids", None),
                getattr(self, "_rt_builder", None),
            )

    def restore_video_rope_cache(self):
        """Restore the saved video RoPE cache (for streaming mode when switching back from QA)"""
        if self._video_cos_cached is not None:
            self._cos_cached = self._video_cos_cached
            self._sin_cached = self._video_sin_cached
            self._seq_len_cached = self._video_seq_len_cached
            self.n_init = self._video_n_init
            self.position_ids = self._video_position_ids
            # restore the skip key alongside the table it describes
            self._built_window_pos_ids = self._video_built_window_pos_ids
            if os.environ.get("REKV_ROPE_TRACE") and getattr(self, "_video_rt_state", None) is not None:
                self._rt_kind, self._rt_pos_ids, self._rt_builder = self._video_rt_state
        return self._cos_cached, self._sin_cached

    def apply_interleaved_trope(self, freqs):
        """Apply interleaved T-RoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            freqs: (3, bs, seq_len, head_dim // 2)
        returns:
            freqs_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.trope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)      

    def _generate_cos_sin_tables_pos(self, video_position_ids, right_justify=False, max_pos=None): 
        """Helper to build cos/sin tables from position_ids with optional right-justification."""
        position_ids = video_position_ids.to(self.device)
        if right_justify:
            # clone before shifting: `.to()` is a no-op when the tensor is already on `self.device`,
            # so an in-place `+=` here would corrupt the caller's position ids (which are shared
            # across all decoder layers).
            position_ids = position_ids.clone()
            t_shift = max_pos - position_ids.max()
            position_ids += t_shift
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_trope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.distance_scale
        sin = emb.sin() * self.distance_scale
        return cos, sin       

    @torch.no_grad()
    def apply_rotary_pos_emb_one_angle_trope(self, x: torch.Tensor, index, **kwargs): 
        """
        Apply T-RoPE using pre-interleaved cached tables (1, N, D).
        Uses per-frame right-justification for variable frame sizes (Qwen3-VL).
        Each frame is right-justified to max position (_seq_len_cached) for sink token attention.
        """
        assert "block_sizes" in kwargs, "block_sizes must be provided for Qwen3-VL T-RoPE"
        assert "curr_position_ids" in kwargs, "curr_position_ids must be provided for Qwen3-VL T-RoPE"
        block_sizes = kwargs["block_sizes"]
        curr_position_ids = kwargs["curr_position_ids"]
        assert sum(block_sizes) == x.size(-2), f"Sum of block_sizes ({sum(block_sizes)}) != x seq_len ({x.size(-2)})"
        assert self.position_ids is not None, "position_ids must be initialized"
        
        dtype = x.dtype
        device = x.device
        max_pos = index 

        ends = list(accumulate(block_sizes))
        starts = [0] + ends[:-1]

        # Rebase each block's positions against its predecessor. Work on a copy: the caller's
        # position ids are built once per encode chunk and reused by every decoder layer, so
        # mutating them in place would give layers 1..N-1 progressively wrong angles (and would
        # also poison `sliding_window_pos_ids`, which is appended from the same tensor).
        curr_position_ids = curr_position_ids.clone()
        for i in range(len(ends) - 1, 0, -1):
            curr_position_ids[:, :, ends[i-1]:ends[i]] -= (
                curr_position_ids[:, :, starts[i-1]:ends[i-1]].max() + 1
            )

        # Generate cos/sin tables for each block with right_justify=True
        cos_list = []
        sin_list = []
        for start, end in zip(starts, ends):
            block_pos_ids = curr_position_ids[:, :, start:end]  # (3, 1, block_size)
            cos_block, sin_block = self._generate_cos_sin_tables_pos(block_pos_ids, True, max_pos)
            cos_list.append(cos_block)  # (1, block_size, D)
            sin_list.append(sin_block)  # (1, block_size, D)

        # Concatenate all cos/sin tables
        cos_all = torch.cat(cos_list, dim=1)  # (1, total_seq_len, D)
        sin_all = torch.cat(sin_list, dim=1)  # (1, total_seq_len, D)
        
        # Apply RoPE: x*cos + rotate_half(x)*sin
        # cos_all: (1, total_seq_len, D) -> (1, 1, total_seq_len, D) for broadcasting
        cos_all = cos_all.unsqueeze(0).to(dtype)  # (1, 1, total_seq_len, D)
        sin_all = sin_all.unsqueeze(0).to(dtype)  # (1, 1, total_seq_len, D)
        
        return ((x.float() * cos_all) + (self.rotate_half(x).float() * sin_all)).to(dtype)

    def apply_rotary_pos_emb(self, x, length, right, cos, sin):

        if cos.device != x.device:
            cos = cos.to(x.device)
            sin = sin.to(x.device)

        # x: tensor to rotate (e.g., q or k). length = current subsequence length to index.
        # right = total current sequence length (end index). We slice cos/sin to the last `length` steps.
        if os.environ.get("REKV_ROPE_TRACE"):
            _T = cos.size(-2)
            if right > _T or right - length < 0:
                print(
                    f"[RTRACE] SHORT-SLICE x={tuple(x.shape)} length={length} right={right} "
                    f"table={_T} (slice yields {max(0, min(right, _T) - max(right - length, 0))} rows)",
                    flush=True,
                )
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

        out = ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)
        return out

    def _update_cos_sin_tables(self, x, seq_dim):
        # Ensure cos/sin caches cover sequence length used by x along seq_dim.
        seq_len = x.size(seq_dim)
        if seq_len > self._seq_len_cached:
            # Need larger tables → (re)compute
            self._seq_len_cached = seq_len
            # a 1D table is about to overwrite _cos_cached; drop the window skip key
            self._built_window_pos_ids = None
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
        # a 1D table is about to overwrite _cos_cached; drop the window skip key
        self._built_window_pos_ids = None
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

        """ apply T-RoPE (video tables) """
        # Encoding: the caller supplies the chunk's per-frame block sizes and position ids, so each
        # frame gets its own right-justified table.
        if self._has_trope_tables and kwargs.get("block_sizes") is not None:
            return self.apply_rotary_pos_emb_one_angle_trope(x, index, **kwargs)

        # QA / attention-sink anchoring: no block structure to honour, just one anchored position
        # broadcast across every query token (LM-Infinite style). This mirrors the Qwen2.5-VL
        # `L=1` case in RotaryEmbeddingQwenVL.apply_rotary_pos_emb_one_angle_mrope; the Qwen3
        # tables are already interleaved to (1, N, D), so no per-axis merge is needed here.
        if self._has_trope_tables:
            dtype = x.dtype
            cos = self._cos_cached.to(x.device)
            sin = self._sin_cached.to(x.device)
            right = min(int(index), cos.size(-2))
            assert right >= 1, f"anchor index {index} is out of range for tables of length {cos.size(-2)}"
            cos = cos[..., right - 1:right, :]
            sin = sin[..., right - 1:right, :]
            while cos.dim() < x.dim():
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)

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
            x: Embeddings (1, M*H*W, D) or (1, M, H, W, D)
            m: Boolean mask (M, H, W)
        Returns:
            Selected embeddings (1, N, D) where N = number of True values in mask
        """
        m = m.to(self.device)
        M, H, W = m.shape
        
        # Ensure x is in spatial shape: (1, M, H, W, D)
        if x.dim() == 3:
            x = x.reshape(1, M, H, W, x.size(-1))
        
        # Get indices of True values
        indices = torch.nonzero(m, as_tuple=False)  # (N, 3) [frame_idx, h_idx, w_idx]
        return x[:, indices[:, 0], indices[:, 1], indices[:, 2], :]  # (1, N, D)
    
    def _select_from_positions(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Select embeddings based on explicit position indices.
        
        Args:
            x: Embeddings (1, M, H, W, D)
            positions: Position indices (n_tokens, 3) where last dim is [frame, h, w]
        Returns:
            Selected embeddings (1, n_tokens, D)
        """
        frame_idx = positions[:, 0].long()
        h_idx = positions[:, 1].long()
        w_idx = positions[:, 2].long()
        return x[:, frame_idx, h_idx, w_idx, :]
    
    def _concat_cos_sin_parts(self, init_part, sparse_part, dense_part):
        """Concatenate init, sparse, and dense parts of cos/sin embeddings along seq dim.
        All parts have shape (1, N_i, D)."""
        return torch.cat([init_part, sparse_part, dense_part], dim=1)
    
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
            
            # Apply RoPE — cos_head is (1, N, D), seq dim is 1
            q_head_rope = self.apply_rotary_pos_emb(q_head, q_head.size(seq_dim), cos_head.size(1), cos_head, sin_head)
            k_head_rope = self.apply_rotary_pos_emb(k_head, k_head.size(seq_dim), cos_head.size(1), cos_head, sin_head)
            
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

        # Crop cos/sin tables to relevant range — tables are (1, N, D)
        cos = self._cos_cached[:, self.n_init:right, :]
        sin = self._sin_cached[:, self.n_init:right, :]

        # --- compressed mask --- (keep-token indices)
        if subsample_mask.dim() == 3 and subsample_mask.dtype != torch.bool:
            nh_q = q.shape[1]
            b, nh_kv, _ = subsample_mask.shape
            assert b == 1, "batch size must be 1 for explicit kept-token indices"
            assert k.shape[1] == nh_kv, f"Number of KV heads in k ({k.shape[1]}) must match index heads ({nh_kv})"

            # token_indices index into local_k: with init_exc=True the sink tokens are excluded,
            # so window index i maps to table row n_init + i (right = len_k + n_init, see
            # forward()); with init_exc=False the origin is row 0. Slice the table at that
            # origin before gathering — gathering from [:right] instead rotated every kept key
            # (and q) with positions shifted by -n_init against the appended sink KV.
            origin = self.n_init if init_exc else 0
            cos_win = self._cos_cached[:, origin:right, :]
            sin_win = self._sin_cached[:, origin:right, :]
            empty_cos, empty_sin = cos_win[:, :0, :], sin_win[:, :0, :]

            def select_for_head(head_idx):
                token_indices = subsample_mask[0, head_idx].long().to(cos_win.device)
                assert token_indices.min() >= 0 and token_indices.max() < cos_win.size(1), (
                    token_indices.min(), token_indices.max(), cos_win.size(1), token_indices.shape,
                )
                return cos_win[:, token_indices, :], sin_win[:, token_indices, :]

            return self._process_per_head(q, k, nh_q, nh_kv, seq_dim,
                                          empty_cos, empty_sin, cos_win, sin_win,
                                          empty_cos, empty_sin, select_for_head)

        # === 3D MASK: Static mask for all heads (dilated, random) ===
        if subsample_mask.dim() == 3:
            M, H, W = subsample_mask.shape
            
            # Reshape to spatial dimensions: (1, frames, H, W, D)
            cos = cos.reshape(1, -1, H, W, cos.size(-1))
            sin = sin.reshape(1, -1, H, W, sin.size(-1))

            n_frames_to_subsample = (length - self.n_init) // (H * W) if not init_exc else length // (H * W)
            assert n_frames_to_subsample <= M, f"n_frames_to_subsample ({n_frames_to_subsample}) > mask frames ({M})"
            
            # Select based on mask and concatenate parts
            cos_sparse = self._select_from_mask(cos[:, :n_frames_to_subsample], subsample_mask[:n_frames_to_subsample])
            sin_sparse = self._select_from_mask(sin[:, :n_frames_to_subsample], subsample_mask[:n_frames_to_subsample])
            
            cos_final = self._concat_cos_sin_parts(
                self._cos_cached[:, :self.n_init, :],
                cos_sparse,
                cos[:, n_frames_to_subsample:].reshape(1, -1, cos.size(-1))
            )
            sin_final = self._concat_cos_sin_parts(
                self._sin_cached[:, :self.n_init, :],
                sin_sparse,
                sin[:, n_frames_to_subsample:].reshape(1, -1, sin.size(-1))
            )
            
            return (
                self.apply_rotary_pos_emb(q, q.size(seq_dim), cos_final.size(1), cos_final, sin_final),
                self.apply_rotary_pos_emb(k, k.size(seq_dim), cos_final.size(1), cos_final, sin_final),
            )
        
        # === 5D MASK: Per-head mask (clustering) ===
        elif subsample_mask.dim() == 5:
            nh_q = q.shape[1]
            b, nh_kv, M, H, W = subsample_mask.shape
            assert b == 1, "batch size must be 1 for per-head masks"
            assert k.shape[1] == nh_kv, f"Number of KV heads in k ({k.shape[1]}) must match mask heads ({nh_kv})"
            
            # Reshape to spatial dimensions: (1, frames, H, W, D)
            cos = cos.reshape(1, -1, H, W, cos.size(-1))
            sin = sin.reshape(1, -1, H, W, sin.size(-1))
            
            n_frames_to_subsample = (length - self.n_init) // (H * W) if not init_exc else length // (H * W)
            assert n_frames_to_subsample <= M, f"n_frames_to_subsample ({n_frames_to_subsample}) > mask frames ({M})"
            
            # Prepare common parts
            cos_init = self._cos_cached[:, :self.n_init, :]
            sin_init = self._sin_cached[:, :self.n_init, :]
            cos_dense = cos[:, n_frames_to_subsample:].reshape(1, -1, cos.size(-1))
            sin_dense = sin[:, n_frames_to_subsample:].reshape(1, -1, sin.size(-1))
            
            # Define selector function for each head
            def select_for_head(head_idx):
                head_mask = subsample_mask[0, head_idx, :n_frames_to_subsample]  # (n_frames, H, W)
                cos_selected = self._select_from_mask(cos[:, :n_frames_to_subsample], head_mask)
                sin_selected = self._select_from_mask(sin[:, :n_frames_to_subsample], head_mask)
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

            # Reshape to spatial dimensions: (1, frames, H, W, D)
            cos = cos.reshape(1, -1, H, W, cos.size(-1))
            sin = sin.reshape(1, -1, H, W, sin.size(-1))
            
            n_frames_to_subsample = (length - self.n_init) // (H * W) if not init_exc else length // (H * W)
            assert n_frames_to_subsample <= cos.shape[1], f"n_frames_to_subsample ({n_frames_to_subsample}) > available frames ({cos.shape[1]})"
            
            # Prepare common parts
            cos_init = self._cos_cached[:, :self.n_init, :]
            sin_init = self._sin_cached[:, :self.n_init, :]
            cos_dense = cos[:, n_frames_to_subsample:].reshape(1, -1, cos.size(-1))
            sin_dense = sin[:, n_frames_to_subsample:].reshape(1, -1, sin.size(-1))
            
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
                retrieved_position_ids=None,
                **forward_subsampled_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # If subsampling is requested, use the specialized method
        if subsample_window is not None:
            assert right is not None
            right_qk = right + self.n_init if init_exc else right
            return self.forward_subsampled(q, k, seq_dim, right_qk, subsample_window, subsample_mask, **forward_subsampled_kwargs)
        
        """ apply 1D-RoPE - update cos/sin tables (only when no video T-RoPE tables exist) """
        if not self._has_trope_tables: 
            # Ensure cos/sin cover the k-sequence length (k typically defines context length).
            self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim=seq_dim)

        """ apply RoPE (both 1D and M-RoPE) """
        # update the cos/sin tables for question answering
        if retrieved_position_ids is not None:
            self._update_cos_sin_tables_interleaved(retrieved_position_ids)
        elif global_buffer_size or update_global_buffer_size:
            raise AssertionError("retrieved_position_ids are required when global_buffer_size is provided for Qwen3 RoPE")

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