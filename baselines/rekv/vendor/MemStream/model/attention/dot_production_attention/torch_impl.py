import math
import os
import torch

from .base import MultiStageDotProductionAttention

# MEMSTREAM_TOPK_ATTN=<k>: each query attends only its top-k keys (exact top-k,
# applied to the concatenated logits pre-softmax). Research knob; requires the
# torch attention path (fattn off). Unset => bit-identical to stock behavior.
_TOPK_ATTN = int(os.environ.get("MEMSTREAM_TOPK_ATTN", "0"))

# MEMSTREAM_TOPP_ATTN=<p>: nucleus variant of the above. Instead of a fixed key
# COUNT, each (query, head) keeps the smallest set of keys whose attention mass
# reaches p, so the kept count adapts to how peaked that head is. This equalizes
# the mass discarded rather than the keys kept -- the axis top-k cannot vary.
#
# The nucleus is located inside a top-K shortlist (MEMSTREAM_TOPP_CAP) so the
# full ~39k-key axis is never sorted, which would OOM alongside the KV cache.
# Probabilities are still exact: exp(logit - logsumexp(ALL logits)), not a
# renormalization within the shortlist. If a row's nucleus does not fit in the
# shortlist that row is left UNMASKED -- conservative, preserving mass rather
# than silently truncating to the cap -- and counted in _TOPP_STATS.
_TOPP_ATTN = float(os.environ.get("MEMSTREAM_TOPP_ATTN", "0") or 0.0)
_TOPP_CAP = int(os.environ.get("MEMSTREAM_TOPP_CAP", "4096"))
# Realized nucleus size is a free diagnostic: "how many keys does this head
# actually need", measured rather than inferred from perplexity.
_TOPP_STATS = {"rows": 0, "overflow": 0, "kept": 0}
# MEMSTREAM_ATTN_SPLIT=1: record the true local-window vs global/sink mass split.
_ATTN_SPLIT = bool(os.environ.get("MEMSTREAM_ATTN_SPLIT"))
_ATTN_SPLIT_BUF = []
# Same counters keyed by layer. The aggregate mean_kept averages over layers
# whose effective key counts differ by ~200x, so it hides the distribution that
# actually matters; this is the per-layer breakdown.
_TOPP_BY_LAYER = {}

if _TOPP_ATTN or _TOPK_ATTN:
    # Printed once at import. A run with the knob set but flash attention still
    # active would be a SILENT no-op -- a dense arm under a masked arm's name --
    # so make the active configuration visible in every log.
    print(f"[MEMSTREAM-KNOB] torch_impl imported: TOPK={_TOPK_ATTN} TOPP={_TOPP_ATTN} "
          f"CAP={_TOPP_CAP} -- masking IS active", flush=True)



def topp_stats(by_layer=False):
    s = dict(_TOPP_STATS)
    s["mean_kept"] = (s["kept"] / s["rows"]) if s["rows"] else 0.0
    s["overflow_frac"] = (s["overflow"] / s["rows"]) if s["rows"] else 0.0
    if by_layer:
        s["layers"] = {
            li: {
                "mean_kept": d["kept"] / d["rows"] if d["rows"] else 0.0,
                "overflow_frac": d["overflow"] / d["rows"] if d["rows"] else 0.0,
                "frac_of_window": (d["kept"] / d["rows"] / d["n_keys"]) if d["rows"] and d["n_keys"] else 0.0,
                "n_keys": d["n_keys"], "rows": d["rows"],
            } for li, d in sorted(_TOPP_BY_LAYER.items())
        }
    return s


def _apply_topp(logits, p_thresh, cap, layer_idx=None):
    """Mask each row to its nucleus. Returns logits with sub-nucleus keys -inf."""
    n = logits.size(-1)
    K = min(cap, n)
    lse = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    vals, _ = logits.topk(K, dim=-1)                      # descending
    probs = torch.exp(vals.float() - lse)                 # exact probabilities
    cum = probs.cumsum(-1)
    # keep key j iff the mass strictly BEFORE it has not yet reached p, so the
    # element that crosses the threshold is retained.
    keep = (cum - probs) < p_thresh
    fits = cum[..., -1:] >= p_thresh                      # nucleus inside shortlist?
    # smallest kept logit == nucleus boundary
    thresh = torch.where(keep, vals, vals.new_full((1,), float('inf'))).min(-1, keepdim=True).values
    _TOPP_STATS["rows"] += fits.numel()
    _TOPP_STATS["overflow"] += int((~fits).sum().item())
    # counts must be built off the int64 sum, NOT off `keep` (bool) -- new_full on
    # a bool tensor would cast n to True==1 and undercount every overflow row.
    counts = keep.sum(-1, keepdim=True)
    n_rows = fits.numel()
    n_over = int((~fits).sum().item())
    n_kept = int(torch.where(fits, counts, torch.full_like(counts, n)).sum().item())
    _TOPP_STATS["kept"] += n_kept
    if layer_idx is not None:
        d = _TOPP_BY_LAYER.setdefault(int(layer_idx), {"rows": 0, "overflow": 0, "kept": 0, "n_keys": 0})
        d["rows"] += n_rows; d["overflow"] += n_over; d["kept"] += n_kept
        d["n_keys"] = max(d["n_keys"], n)   # window size this layer saw
    # rows whose nucleus overflowed the shortlist keep everything
    thresh = torch.where(fits, thresh, thresh.new_full((1,), float('-inf')))
    return logits.masked_fill(logits < thresh, float('-inf'))


class TorchMultiStageDotProductionAttention(MultiStageDotProductionAttention):
    def __init__(self, q_shape, dtype, device):
        super().__init__(q_shape, dtype, device)
        self.logits_list = []
        self.v_list = []
        self.mask_list = []
        self.get_score_list = []
        self.kv_len_list = []

    def finalize(self):
        logits = torch.cat(self.logits_list, dim=-1)
        if _TOPK_ATTN and logits.size(-1) > _TOPK_ATTN:
            thresh = logits.topk(_TOPK_ATTN, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < thresh, float('-inf'))
        if _TOPP_ATTN:
            logits = _apply_topp(logits, _TOPP_ATTN, _TOPP_CAP, getattr(self, "layer_idx", None))
        p = torch.softmax(logits, dim=-1)
        # TRUE local-vs-global split. finalize() softmaxes over the CONCATENATION of
        # the local sliding window and the global/sink KV, so this is the only place
        # the real distribution exists. A probe that softmaxes the local window alone
        # renormalises local mass to 1 and never sees the sinks at all.
        if _ATTN_SPLIT and self.kv_len_list:
            _loc = self.kv_len_list[0]
            _lm = float(p[..., :_loc].sum())
            _tot = float(p.sum()) or 1.0
            _ATTN_SPLIT_BUF.append({"layer": int(getattr(self, "layer_idx", -1)),
                                    "local_mass": _lm / _tot,
                                    "global_mass": 1.0 - _lm / _tot,
                                    "n_local_keys": int(_loc),
                                    "n_global_keys": int(logits.size(-1) - _loc)})
            _out = os.environ.get("ATTN_SPLIT_OUT")
            if _out:
                import json as _json
                with open(_out, "a") as _f:
                    _f.write(_json.dumps(_ATTN_SPLIT_BUF[-1]) + "\n")
                _ATTN_SPLIT_BUF.clear()
        st = 0
        for kv_len, mask, get_score, v in zip(self.kv_len_list, self.mask_list, self.get_score_list, self.v_list):
            ed = st + kv_len
            tmp = p[:, :, :, st: ed]
            tmp = torch.masked_fill(
                tmp,
                mask==False,
                0
            )
            if get_score:
                # Store full attention matrix (batch, heads, num_queries, num_keys)
                # Don't sum over queries - we want the full 2D matrix
                self.score_list.append(tmp)
            else:
                self.score_list.append(None)

            self.ret.add_(
                torch.matmul(tmp, v)
            )

            st = ed

    def append(
            self, 
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
            sliding_window = None,
            complement_sliding_window:bool = False,
            end=False, get_score=False,
            *args, **kwargs
        ):

        len_q = q.size(-2)
        len_k = k.size(-2)

        num_heads = q.size(1)
        num_heads_kv = k.size(1)
        if num_heads != num_heads_kv:
            shape = list(k.shape)
            num_group = num_heads // num_heads_kv
            k = k[:, :, None, :, :].expand(shape[0], shape[1], num_group, shape[2], shape[3])
            k = k.reshape(shape[0], num_heads, shape[2], shape[3])
            v = v[:, :, None, :, :].expand(shape[0], shape[1], num_group, shape[2], shape[3])
            v = v.reshape(shape[0], num_heads, shape[2], shape[3])

        if sliding_window is None:
            mask = torch.ones(
                (len_q, len_k),
                dtype=torch.bool,
                device=q.device
            )
        else:
            if isinstance(sliding_window, int):
                sliding_window = (len_k - len_q, sliding_window)

            dist = torch.arange(
                len_q, dtype=torch.int64, device=q.device
            )[:, None] - torch.arange(
                len_k, dtype=torch.int64, device=k.device
            )[None, :] + sliding_window[0]
            if complement_sliding_window:
                mask = dist >= sliding_window[1]
            else:
                mask = (dist < sliding_window[1]) & (dist >= 0)

        m_shape = [1] * (4-mask.dim()) + list(mask.shape)
        mask = mask.view(m_shape)
        self.v_list.append(v)
        self.mask_list.append(mask)
        self.get_score_list.append(get_score)
        self.kv_len_list.append(k.size(-2))
        logits = torch.matmul(q, k.transpose(-1, -2))
        logits = torch.masked_fill(
            logits,
            mask==False,
            float("-inf")
        )
        logits.mul_(1/math.sqrt(q.size(-1)))
        self.logits_list.append(logits)

        if end:
            self.finalize()
