import re
import math
from itertools import accumulate
import torch
import os
from typing import Optional
from collections import defaultdict
from .dot_production_attention import get_multi_stage_dot_production_attention
from ezcolorlog import root_logger as logger

# Retrieval-time AKS: applies the adjacent-novelty selection to the KEYS used to
# score blocks at retrieval, leaving the encoded KV dense. Unset => no-op.
_RETRIEVAL_AKS = int(os.environ.get("MEMSTREAM_RETRIEVAL_AKS", "0") or 0)
_RETRIEVAL_AKS_PROBE = bool(os.environ.get("MEMSTREAM_RETRIEVAL_AKS_PROBE"))
_RETRIEVAL_AKS_PROBE_BUF = []
# MEMSTREAM_TAIL_PROBE=1: measure the overlap between the keys kk_adjacent would
# drop (redundant) and the keys carrying the attention mass (what top-k keeps).
_TAIL_PROBE = bool(os.environ.get("MEMSTREAM_TAIL_PROBE"))
# MEMSTREAM_COMPRESS_LAYERS=<i>,<j>,... -- same, but for an explicit (possibly
# non-contiguous) layer set. Indices are bounds-checked against the model's layer count
# in patch_hf, since a typo'd index would otherwise just run that layer dense.
def _parse_compress_layer_set(spec):
    """'0,5,6' -> sorted tuple of distinct non-negative layer ids; raises on anything else."""
    toks = spec.split(",")
    if not all(t.strip().isdigit() for t in toks):
        raise ValueError(f"MEMSTREAM_COMPRESS_LAYERS={spec!r}: expected <start>:<end> or a "
                         f"comma-separated list of non-negative layer indices, e.g. 0,5,6,10")
    layers = [int(t) for t in toks]
    if len(set(layers)) != len(layers):
        raise ValueError(f"MEMSTREAM_COMPRESS_LAYERS={spec!r}: duplicate layer index")
    return tuple(sorted(layers))

_cl = os.environ.get("MEMSTREAM_COMPRESS_LAYERS")
_COMPRESS_LAYER_SET = _parse_compress_layer_set(_cl) if _cl and ":" not in _cl else None
_COMPRESS_LAYERS = tuple(int(x) for x in _cl.split(":")) if _cl and ":" in _cl else None
if _COMPRESS_LAYERS:
    print(f"[MEMSTREAM-KNOB] compression restricted to layers "
          f"[{_COMPRESS_LAYERS[0]}, {_COMPRESS_LAYERS[1]}) -- all others run DENSE", flush=True)
if _COMPRESS_LAYER_SET:
    print(f"[MEMSTREAM-KNOB] compressing layers {list(_COMPRESS_LAYER_SET)} "
          f"-- all others run DENSE", flush=True)

def _parse_qframe_config(prefill_method):
    """
    Parse `swa_qframe_<n>_<lam>[_z]` -> (n: int, lam: float in [0,1], use_z: bool).

    STRICT on purpose. `lam` is required and never defaulted, because the arm name IS the
    results directory (`{save_dir}/{prefill_method}-{retrieval_method}`): an omitted lam
    would let two runs with different lam land in indistinguishable paths. Unknown tokens
    raise rather than being ignored, so a typo cannot silently run a different arm -- the
    failure mode that made swa_vv_adj_2 execute as swa_kk_adjacent_2.

    Called from BOTH __init__ (validate at construction, fail before any GPU work) and
    _apply_compression (parse at use). One function, so the two cannot diverge.
    """
    body = prefill_method.split("_after")[0]
    body = re.sub(r"_nd=\d+", "", body)
    toks = body.split("_")
    if len(toks) < 4 or toks[0] != "swa" or toks[1] != "qframe":
        raise ValueError(
            f"malformed qframe arm {prefill_method!r}: expected swa_qframe_<n>_<lam>, "
            f"e.g. swa_qframe_18_75")
    n_str, lam_str, rest = toks[2], toks[3], toks[4:]
    if not n_str.isdigit():
        raise ValueError(f"qframe keep-count must be a non-negative integer, got {n_str!r} "
                         f"in {prefill_method!r}")
    if not lam_str.isdigit():
        raise ValueError(
            f"qframe lam must be an integer percent 0-100, got {lam_str!r} in "
            f"{prefill_method!r}. lam is REQUIRED -- write e.g. swa_qframe_{n_str}_75")
    lam = int(lam_str)
    if not (0 <= lam <= 100):
        raise ValueError(f"qframe lam must be 0-100, got {lam} in {prefill_method!r}")
    use_z = False
    for t in rest:
        if t == "z":
            use_z = True
        elif t:
            raise ValueError(f"unrecognised qframe token {t!r} in {prefill_method!r}")
    return int(n_str), lam / 100.0, use_z


_TAIL_PROBE_BUF = []
_TAIL_PROBE_SEEN = []   # layer ids, one entry per recorded (layer, window)
_TAIL_PROBE_FLUSH = int(os.environ.get('MEMSTREAM_TAIL_PROBE_FLUSH', '100'))  # progress-log cadence, in windows
# the O(frames^2) best-match-over-window stat is expensive to run on every
# call; off by default for full-video runs, enable with =1 for a spot check.
_TAIL_PROBE_HEAVY = bool(os.environ.get('MEMSTREAM_TAIL_PROBE_HEAVY'))
_TAIL_PROBE_CHUNK = int(os.environ.get('MEMSTREAM_TAIL_PROBE_CHUNK', '8'))
_TAIL_PROBE_NLAYERS = int(os.environ.get("MEMSTREAM_TAIL_PROBE_NLAYERS", "36"))
_TAIL_PROBE_MIN_KEYS = int(os.environ.get("MEMSTREAM_TAIL_PROBE_MIN_KEYS", "30000"))

# MEMSTREAM_TSM_PROBE=1: Foote checkerboard event segmentation over the per-frame
# self-similarity matrix, per layer. Width/height are in FRAMES (Qwen3 temporal
# units), matching the reference in morph_graphs/.../events/tsm.py.
_TSM_PROBE = bool(os.environ.get("MEMSTREAM_TSM_PROBE"))
_TSM_WIDTH = int(os.environ.get("MEMSTREAM_TSM_WIDTH", "6"))       # half-window
_TSM_HEIGHT = float(os.environ.get("MEMSTREAM_TSM_HEIGHT", "1.0"))  # z-score threshold
_TSM_SIGMA = float(os.environ.get("MEMSTREAM_TSM_SIGMA", "0.5"))
# Dump the full frame x frame similarity matrix every Nth window (0 = never).
# n^2 floats per layer-window, so keep N large; for heatmaps only.
_TSM_DUMP_EVERY = int(os.environ.get("MEMSTREAM_TSM_DUMP_EVERY", "0") or 0)

# MEMSTREAM_CRETR=<r>: COMPRESSED RETRIEVAL. The retrieval budget is a fixed number of
# TOKENS, spent today as `topk` blocks loaded densely. This spends it as MORE blocks
# loaded sparsely: keep every r-th patch along each grid axis (1/r^2 of each block's
# visual tokens) and retrieve proportionally more blocks, so total loaded tokens are
# unchanged and only temporal COVERAGE grows.
#
# Timestamp and vision-boundary tokens are NEVER compressed: on Qwen3 the visual tokens
# carry t_index=0 and ALL temporal position lives in the interleaved timestamp text, so
# striding those away would destroy the model's sense of when a frame happened. They are
# also why the slot count is derived from measured widths rather than r^2 -- there are
# r^2 times more prefixes to pay for, and the grid rarely divides evenly.
# MEMSTREAM_VRETR=1: retrieve in VALUE space. Block signatures are mean-pooled VALUES
# instead of keys, and they are scored against the question's mean-pooled VALUES rather
# than its queries. Note this is a genuinely different similarity notion, not a fix:
# no layer of the model ever computes v_question . v_frame, whereas q.k is exactly what
# attention uses to decide where to look.
_VRETR = bool(os.environ.get("MEMSTREAM_VRETR"))
# MEMSTREAM_FUSE_RETR=1: rank blocks by RRF over BOTH signature spaces -- the usual
# query-vs-key score and the value-vs-value score -- then take the same top-k. Budget
# is unchanged; only the ranking differs. Offline on one chunk this was the largest
# matched-budget recall gain measured (+0.085 vs QK alone, where either space alone
# gives +0.00/+0.040), because the two spaces retrieve substantially different blocks
# (Jaccard 0.30).
_FUSE_RETR = bool(os.environ.get("MEMSTREAM_FUSE_RETR"))
_FUSE_RRF_K = int(os.environ.get("MEMSTREAM_FUSE_RRF_K", "60"))
if _FUSE_RETR:
    print(f"[MEMSTREAM-KNOB] fused retrieval active: RRF(k={_FUSE_RRF_K}) over "
          f"query-key AND value-value rankings", flush=True)
if _VRETR:
    print("[MEMSTREAM-KNOB] value-space retrieval active: block sigs and question "
          "embedding both from VALUES", flush=True)
_CRETR_R = int(os.environ.get("MEMSTREAM_CRETR", "0") or 0)
# Override the derived slot count, for deliberately UNMATCHED budget studies.
_CRETR_SLOTS = int(os.environ.get("MEMSTREAM_CRETR_SLOTS", "0") or 0)
if _CRETR_R > 1:
    print(f"[MEMSTREAM-KNOB] compressed retrieval active: keep 1/{_CRETR_R}^2 of each "
          f"retrieved block's visual tokens, retrieve proportionally more blocks "
          f"(slots override={_CRETR_SLOTS or 'derived'})", flush=True)
if _RETRIEVAL_AKS > 1:
    print(f"[MEMSTREAM-KNOB] retrieval-time AKS active: keep 1/{_RETRIEVAL_AKS} of each "
          f"block's tokens for the retrieval signature (encoding unchanged)", flush=True)
import torch.nn.functional as F
from einops import rearrange
from .compression_methods import (
    apply_pooled_compression,
    apply_kk_guided_compression,
    apply_budgeted_kk_compression,
    apply_gt_alloc_compression,
    apply_patch_sample_compression,
    apply_frame_sample_compression,
    apply_clustered_compression,
    apply_tome_with_frame_sampling,
    apply_mixed_tome_compression,
    rebuild_mixed_window,
    frame_keep_to_sparse_indices,
    frame_query_affinity,
    RecencyProfile,
    no_compression
)

from model.qwen3_utils import get_qwen3_vl_retrieved_position_ids
from . import diagnostics

# Check if logger output should be enabled
if not os.environ.get('ENABLE_LOGGING'):
    logger.disabled = True

# CudaCache / MemoryUnit / VectorTensor live in memory_units.py; re-exported here
# so existing `from .kv_cache_manager import ...` call sites keep working.
from .memory_units import CudaCache, MemoryUnit, VectorTensor


GLOBAL_STREAM = None



def _tail_probe_selftest():
    """Synthetic Qwen3-like window with VARIABLE per-frame text widths -- the exact
    layout that made v1 drift. Runs once at probe startup so a geometry error
    fails on the compute node instead of silently producing a plausible number."""
    n_init, bs, nfr, d = 14, 12, 8, 16
    lens = [3, 5, 4, 6, 3, 5, 4, 3]
    keys, vm = [torch.randn(n_init, d)], [torch.zeros(n_init, dtype=torch.bool)]
    static = torch.randn(bs // 2, d)
    for f in range(nfr):
        keys.append(torch.randn(lens[f], d)); vm.append(torch.zeros(lens[f], dtype=torch.bool))
        keys.append(torch.cat([static, torch.randn(bs // 2, d)]))
        vm.append(torch.ones(bs, dtype=torch.bool))
    K = torch.cat(keys); VM = torch.cat(vm); n = K.size(0)
    k = K[None, None].expand(1, 2, n, d).contiguous()
    q = torch.randn(1, 8, 4, d)
    r = _tail_redundancy_probe(q, k, bs, 0, visual_mask=VM[None], len_q=4, n_init=n_init)
    assert r is not None, "probe returned None on the self-test layout"
    assert r["n_frames"] == nfr, f"FRAME GEOMETRY WRONG: got {r['n_frames']}, expected {nfr}"
    assert abs(r["chance_kk_2"] - 0.75) < 0.02, f"chance baseline off: {r['chance_kk_2']}"
    print(f"[TAIL-PROBE] selftest PASS: {nfr} frames recovered with variable text "
          f"widths {lens}; chance_kk_2={r['chance_kk_2']:.3f}", flush=True)


def _checkerboard_kernel(width, sigma, device, dtype):
    """Foote's Gaussian-tapered checkerboard. Port of
    morph_graphs/anubhav/morph_graph/events/tsm.py::checkerboard_kernel.
    np.meshgrid's default 'xy' indexing is irrelevant here -- the taper is
    symmetric in x and y -- so torch's 'ij' gives an identical kernel.
    """
    cb = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=device, dtype=dtype)
    cb = torch.kron(cb, torch.ones((width, width), device=device, dtype=dtype))
    axis = torch.linspace(-1.0, 1.0, width * 2, device=device, dtype=dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return cb * torch.exp(-(x ** 2 + y ** 2) / sigma)


def _tsm_novelty(fe, width, sigma):
    """Per-frame checkerboard novelty over the frame self-similarity matrix.

    Same quantity as the reference's per-unit loop, but the (2w x 2w) diagonal
    blocks are gathered from one precomputed S instead of re-multiplying each
    window -- identical values, and it has to run for 36 layers x ~1000 windows.
    Entries outside [w, n-w) stay 0: the window does not fit there.
    """
    n = fe.size(0)
    f = torch.nn.functional.normalize(fe.float(), dim=-1)
    K = _checkerboard_kernel(width, sigma, f.device, f.dtype)
    nov = torch.zeros(n, device=f.device, dtype=f.dtype)
    if n < 2 * width + 1:
        return nov
    S = f @ f.T
    centers = torch.arange(width, n - width, device=f.device)
    off = torch.arange(2 * width, device=f.device)
    rows = centers[:, None] - width + off[None, :]          # (C, 2w)
    sub = S[rows[:, :, None], rows[:, None, :]]             # (C, 2w, 2w)
    nov[centers] = (sub * K).sum((-1, -2))
    return nov


def _tsm_pick_peaks(s, height, distance, lo, hi):
    """Local maxima >= height, thinned to a minimum separation.

    Mirrors scipy.signal.find_peaks(height=, distance=), including its tie-break:
    when two peaks are closer than `distance`, the TALLER one survives.
    Reimplemented rather than imported so the probe carries no scipy dependency
    onto the compute node; the CPU test asserts equality against scipy.
    """
    cand = [i for i in range(max(lo + 1, 1), min(hi, len(s) - 1))
            if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] >= height]
    cand.sort(key=lambda i: -s[i])
    kept = []
    for i in cand:
        if all(abs(i - j) >= distance for j in kept):
            kept.append(i)
    return sorted(kept)


def _tsm_segment(fe, width, sigma, height, seed):
    """Boundaries for one (layer, window), plus a frame-shuffled null.

    The null is the whole point. find_peaks at z>=1 returns peaks from ANY noisy
    curve, so a raw boundary count says nothing about whether events exist.
    Shuffling frame order preserves every frame's content and the novelty curve's
    marginal scale while destroying temporal contiguity: if real events exist the
    real curve must peak HARDER than the shuffle. Real ~ shuffle => no events.
    """
    n = fe.size(0)
    if n < 2 * width + 1:
        return None
    def _run(x):
        nov = _tsm_novelty(x, width, sigma)
        valid = nov[width:n - width]
        # POPULATION std, matching the reference's np.std (ddof=0). torch's default
        # is the sample std (ddof=1), which shifts every z by ~1% at n=48 and would
        # silently move the z>=1.0 boundary threshold away from the reference's.
        mu, sd = float(valid.mean()), float(valid.std(unbiased=False))
        z = ((nov - mu) / (sd + 1e-6)).tolist()
        pk = _tsm_pick_peaks(z, height, width, width, n - width)
        return nov, z, pk, mu, sd
    nov, z, pk, mu, sd = _run(fe)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).to(fe.device)
    nov_s, z_s, pk_s, _, sd_s = _run(fe[perm])
    zi = z[width:n - width]; zs = z_s[width:n - width]
    ev = [b - a for a, b in zip([0] + pk, pk + [n])] if pk else [n]
    out = {
        "tsm_n_frames": int(n), "tsm_valid": int(n - 2 * width), "tsm_width": int(width),
        "tsm_nbound": len(pk), "tsm_nbound_shuf": len(pk_s),
        "tsm_max_z": max(zi) if zi else 0.0,
        "tsm_max_z_shuf": max(zs) if zs else 0.0,
        "tsm_nov_mean": mu, "tsm_nov_sd": sd, "tsm_nov_sd_shuf": sd_s,
        "tsm_event_len_mean": sum(ev) / len(ev),
        "tsm_boundaries": [int(b) for b in pk[:24]],
        # THE CURVES THEMSELVES, valid centers only. Without these the z>=1.0
        # threshold would be frozen at collection time and re-asking "how many
        # events at z>=2?" would mean re-running 36 layers x ~1000 windows on a
        # GPU. They also make the curve plottable. z is recoverable from these
        # alone (z = (nov - mean(nov)) / sd(nov)), so nothing else is needed.
        "tsm_nov": [round(float(v), 4) for v in nov[width:n - width].tolist()],
        "tsm_nov_shuf": [round(float(v), 4) for v in nov_s[width:n - width].tolist()],
    }
    # Full frame x frame similarity matrix, for heatmaps. Off by default: this is
    # ~n^2 floats per layer-window and would dwarf everything else if always on.
    if _TSM_DUMP_EVERY > 0 and (len(_TAIL_PROBE_SEEN) // max(1, _TAIL_PROBE_NLAYERS)) % _TSM_DUMP_EVERY == 0:
        fS = torch.nn.functional.normalize(fe.float(), dim=-1)
        out["tsm_S"] = [round(float(v), 3) for v in (fS @ fS.T).flatten().tolist()]
    return out


def _eff_distinct(p, vecs, uniform=False):
    """Effective number of distinct directions the attention output draws from.

        eff = 1 / sum_ij p_i p_j cos(v_i, v_j)  ==  1 / || sum_i p_i vhat_i ||^2

    The identity makes this O(n*d) instead of O(n^2), so no subsampling.
    Reads as: 1.0 if every weighted token points the same way; m if the mass is
    spread over m mutually orthogonal directions. This is the quantity a
    "compression removes redundancy" claim predicts should RISE after dropping
    duplicates -- unlike attention mass, which such a drop necessarily lowers.
    """
    s = float(p.sum())
    if s <= 0 or p.numel() == 0:
        return float("nan")
    ph = (p / s).unsqueeze(-1)
    if uniform:
        # PURE DIRECTION DIVERSITY. The mass-weighted form silently degenerates into a
        # measure of mass CONCENTRATION once p is skewed, which real attention always
        # is: with lognormal mass, a maximally diverse 57-token set and a random one
        # score 4.94 vs 4.83 -- indistinguishable -- while under uniform weights the
        # same two sets score 49.1 vs 17.5. Both are reported so the two effects can
        # be told apart instead of being confounded.
        ph = torch.full_like(ph, 1.0 / p.numel())
    vh = torch.nn.functional.normalize(vecs.float(), dim=-1)
    r = (ph * vh).sum(0)
    d = float((r * r).sum())
    return 1.0 / d if d > 1e-12 else float("nan")


def _tail_redundancy_probe(local_h_q, local_h_k, block_size, layer_idx,
                           visual_mask=None, len_q=0, n_init=0, pre_rope_k=None,
                           global_k=None, local_h_v=None):
    """Do the keys kk_adjacent would DISCARD carry the attention mass top-k needs?

    v2, after an audit found three defects in v1:
      * v1 rolled the FLAT key axis by `block_size`. On Qwen3 `block_size` is
        visual-only (510) while the real per-frame stride is
        block_size + n_block_prefix_tokens (~521) and frame widths VARY, so each
        patch was compared to one ~10 positions away, drifting through the
        window. v2 indexes VISUAL tokens only and reshapes to (frames, patches),
        which is the geometry the real kk criterion uses.
      * v1 let the newest frame be droppable and used no causal mask; real kk
        always concatenates the dense tail back uncompressed. v2 excludes the
        last len_q positions and masks non-causal logits.
      * v1 compared against a 0.750 baseline that ignored the protected first
        block. v2 restricts every quantity to the droppable pool, so the
        chance baseline is exactly the drop fraction, and separately reports
        what share of total mass the pool even holds.
    """
    with torch.no_grad():
        q = local_h_q.float()
        k = local_h_k.float()
        if q.size(1) != k.size(1):                      # GQA: expand KV heads
            assert q.size(1) % k.size(1) == 0
            k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
        n = k.size(-2)

        # causal: query i sits at absolute position n - len_q + i
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        if len_q > 0 and len_q <= n:
            qpos = torch.arange(n - len_q, n, device=k.device)[:, None]
            kpos = torch.arange(n, device=k.device)[None, :]
            logits = logits.masked_fill((kpos > qpos)[None, None], float("-inf"))
        # JOINT softmax over local window AND global/sink KV -- the model softmaxes
        # the concatenation in finalize(). Normalising over the local window alone
        # renormalises local mass to 1 and hides the sinks entirely.
        n_glob = 0
        if global_k is not None:
            gk = global_k.float()
            if q.size(1) != gk.size(1):
                gk = gk.repeat_interleave(q.size(1) // gk.size(1), dim=1)
            gl = torch.matmul(q, gk.transpose(-1, -2)) / math.sqrt(q.size(-1))
            n_glob = gl.size(-1)
            logits = torch.cat([logits, gl], dim=-1)
        p_full = torch.softmax(logits, dim=-1).mean(dim=(0, 1, 2))
        p_all = p_full[:n]                                            # local part, TRUE scale

        # visual tokens only -- text/timestamp positions are never kk candidates
        if visual_mask is not None:
            vm = visual_mask[0] if visual_mask.dim() > 1 else visual_mask
            vidx = torch.nonzero(vm.to(k.device)[:n], as_tuple=False).flatten()
        else:
            vidx = torch.arange(n_init, n, device=k.device)
        nfr = vidx.numel() // block_size
        if nfr < 2:
            return None
        vidx = vidx[: nfr * block_size].view(nfr, block_size)          # (frames, patches)

        kf = k.mean(dim=1)[0]                                          # heads pooled
        grid = kf[vidx]                                                # (frames, patches, d)
        nov = 1.0 - torch.nn.functional.cosine_similarity(
            grid[1:], grid[:-1], dim=-1)                               # (frames-1, patches)

        cand = vidx[1:].reshape(-1)                                    # frame 0 has no predecessor
        novf = nov.reshape(-1)
        keep_tail = cand < (n - len_q)                                 # real kk never drops the dense tail
        cand, novf = cand[keep_tail], novf[keep_tail]
        if cand.numel() < 8:
            return None

        pool_mass = float(p_all[cand].sum())
        tot = float(p_full.sum())            # TRUE total incl. global/sink keys
        # Decompose the PROTECTED mass into its three disjoint sources, so we can
        # see which of them is actually load-bearing. Sinks and the dense tail are
        # protected for principled reasons; frame 0 is protected only because the
        # adjacent-novelty criterion has no predecessor for it -- an artifact.
        dev = k.device
        idx_all = torch.arange(n, device=dev)
        m_sink = torch.zeros(n, dtype=torch.bool, device=dev); m_sink[:n_init] = True
        m_tail = torch.zeros(n, dtype=torch.bool, device=dev)
        if len_q > 0:
            m_tail[max(0, n - len_q):] = True
        m_tail &= ~m_sink
        m_f0 = torch.zeros(n, dtype=torch.bool, device=dev)
        m_f0[vidx[0]] = True
        m_f0 &= ~m_sink & ~m_tail
        m_pool = torch.zeros(n, dtype=torch.bool, device=dev); m_pool[cand] = True
        m_other = ~(m_sink | m_tail | m_f0 | m_pool)     # text/timestamp + unpaired visual
        out = {"layer": int(layer_idx), "n_keys": int(n), "n_frames": int(nfr),
               "pool_size": int(cand.numel()), "n_init": int(n_init), "len_q": int(len_q),
               "pool_mass_share": pool_mass / tot,
               "protected_mass_share": 1.0 - pool_mass / tot,
               "mass_sinks": float(p_all[m_sink].sum()) / tot,
               "mass_dense_tail": float(p_all[m_tail].sum()) / tot,
               "mass_frame0": float(p_all[m_f0].sum()) / tot,
               "mass_other": float(p_all[m_other].sum()) / tot,
               "n_sinks": int(m_sink.sum()), "n_tail": int(m_tail.sum()),
               "n_frame0": int(m_f0.sum()), "n_other": int(m_other.sum())}

        # PER-FRAME ATTENTION MASS: one number per frame, the share of TOTAL mass
        # (denominator includes sinks and global KV, so these do NOT sum to 1)
        # landing on that frame's visual tokens. This is the series to lay against
        # the TSM boundaries -- does attention pool inside events, or spike at the
        # edges between them? attn_visual_share is the sum, i.e. how much of the
        # model's attention is on visual tokens at all, which is the context every
        # per-frame number has to be read against.
        fm = p_all[vidx].sum(dim=1) / tot                              # (frames,)
        out["attn_per_frame"] = [round(float(v), 7) for v in fm.tolist()]
        out["attn_visual_share"] = float(fm.sum())

        # RAW QK, PRE-SOFTMAX. attention = softmax(QK/sqrt(d)), and the exponential is
        # strongly winner-take-all, so a frame can be middling in raw similarity and
        # still dominate the mass (or vice versa). Storing both says how much of the
        # attention pattern is already present in the similarity versus manufactured by
        # the softmax. Masked (non-causal) entries are -inf, so they are excluded from
        # the mean rather than averaged in -- otherwise every key a query cannot see
        # would drag its score to -inf.
        lg = logits[..., :n]
        fin = torch.isfinite(lg)
        lsum = torch.where(fin, lg, torch.zeros_like(lg)).sum(dim=(0, 1, 2))
        lcnt = fin.sum(dim=(0, 1, 2)).clamp(min=1)
        lmean = lsum / lcnt                                            # (n,) per key
        # MEAN over the frame's patches (a typical similarity), where attn_per_frame is
        # a SUM (total mass). Frames are equal-width so this is only a rescale, but the
        # two are different quantities and should not be read off the same axis.
        out["qk_per_frame"] = [round(float(v), 5) for v in lmean[vidx].mean(dim=1).tolist()]
        # Where in the window does the text/timestamp mass sit? Qwen3's visual
        # tokens carry zero temporal position, so these tokens are the model's
        # only handle on "when". Concentrated near the query => recency. Spread
        # across the window => the model is reading the whole timeline.
        oth = torch.nonzero(m_other, as_tuple=False).flatten()
        if oth.numel() >= 4:
            q4 = torch.tensor_split(oth, 4)
            om = float(p_all[oth].sum()) or 1.0
            out["other_by_quartile"] = [float(p_all[t].sum()) / om for t in q4]
            out["other_tokens_by_quartile"] = [int(t.numel()) for t in q4]
        sk = torch.nonzero(m_sink, as_tuple=False).flatten()
        out["sink_per_token"] = [float(p_all[i]) for i in sk.tolist()]

        # LOOKBACK CURVE: how much redundancy exists at each temporal distance?
        # kk_adjacent/ToMe only ever compare against lookback=1, so any patch that
        # is redundant against an OLDER frame is invisible to them. That pool grows
        # with n_local, which is exactly the "bigger window => more compressible"
        # question. novelty = 1 - cos(patch_j frame f, patch_j frame f-d).
        gn = torch.nn.functional.normalize(grid.float(), dim=-1)      # (frames, patches, d)

        # TSM EVENT SEGMENTATION. Foote checkerboard novelty over the per-frame
        # embedding (patches mean-pooled), run per LAYER because redundancy
        # structure differs ~10x across layers -- a single stack-wide segmentation
        # would average exactly the thing we are trying to see.
        if _TSM_PROBE:
            seg = _tsm_segment(grid.float().mean(dim=1), _TSM_WIDTH, _TSM_SIGMA,
                               _TSM_HEIGHT, seed=1000003 * int(layer_idx) + nfr)
            if seg is not None:
                out.update(seg)

        lb = {}
        for dd in (1, 2, 4, 8, 16, 32):
            if nfr > dd:
                nv = 1.0 - (gn[dd:] * gn[:-dd]).sum(-1)               # (frames-dd, patches)
                lb[dd] = float(nv.mean())
        out["novelty_by_lookback"] = lb
        # PER-FRAME novelty spread. Every method we run compresses at a FIXED ratio
        # on every frame. That is only sensible if novelty is roughly uniform across
        # frames; if it is concentrated (few high-novelty frames, many near-
        # duplicates) then a fixed quota destroys informative frames to hit a target
        # it should be meeting from the redundant ones. Report the distribution so
        # we can tell which regime we are in.
        nv1 = 1.0 - (gn[1:] * gn[:-1]).sum(-1)                 # (frames-1, patches)
        # PRE-RoPE novelty on the same patches. The real kk_adjacent ranks pre-RoPE
        # (compression runs before position_embedding); this probe ranks post-RoPE.
        # Qwen3 visual tokens are supposed to carry t_index=0, so patch j should get
        # the SAME rotation every frame and the two should agree -- measure it.
        if pre_rope_k is not None:
            pk = pre_rope_k.float()
            if pk.size(1) != k.size(1):
                pk = pk.repeat_interleave(k.size(1) // pk.size(1), dim=1)
            pkf = pk.mean(dim=1)[0]
            pgrid = torch.nn.functional.normalize(pkf[vidx].float(), dim=-1)
            nv_pre = 1.0 - (pgrid[1:] * pgrid[:-1]).sum(-1)
            out["novelty_pre_rope"] = float(nv_pre.mean())
            out["novelty_post_rope"] = float(nv1.mean())
            # per-patch agreement of the RANKING, which is what selection depends on
            a = nv1.reshape(-1).float(); b = nv_pre.reshape(-1).float()
            am, bm = a.mean(), b.mean()
            out["novelty_rope_corr"] = float(((a-am)*(b-bm)).sum() /
                                             (((a-am).pow(2).sum()*((b-bm).pow(2).sum())).sqrt() + 1e-9))
        per_frame = nv1.mean(dim=-1)                            # mean novelty of each frame
        pf = per_frame.detach().float().cpu()
        q = torch.quantile(pf, torch.tensor([0.05,0.25,0.50,0.75,0.95]))
        out["frame_novelty"] = {
            "mean": float(pf.mean()), "sd": float(pf.std()),
            "cv": float(pf.std()/ (pf.mean()+1e-9)),            # >0.5 => strongly concentrated
            "q05": float(q[0]), "q25": float(q[1]), "q50": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]),
            "ratio_q95_q05": float(q[4]/(q[0]+1e-9)),
        }
        # within-frame spread too: are the redundant PATCHES concentrated in some
        # frames, or spread evenly? (the axis a per-frame threshold would exploit)
        within = nv1.std(dim=-1).mean()
        out["frame_novelty"]["within_frame_sd"] = float(within)
        out["frame_novelty"]["between_frame_sd"] = float(pf.std())
        # best match over ALL earlier frames, for the most recent frames (cost-capped)
        probe_fr = min(8, nfr - 1) if _TAIL_PROBE_HEAVY else 0
        if probe_fr > 0:
            cur = gn[-probe_fr:]                                       # (probe_fr, patches, d)
            best = []
            for i in range(probe_fr):
                f_abs = nfr - probe_fr + i
                prev = gn[:f_abs]                                      # all earlier frames
                # same spatial position only, across every earlier frame
                sim = (prev * cur[i][None]).sum(-1)                    # (f_abs, patches)
                best.append((1.0 - sim.max(dim=0).values).mean())
            out["novelty_best_over_window"] = float(torch.stack(best).mean())
            out["novelty_lookback1_same_frames"] = float(
                (1.0 - (gn[-probe_fr:] * gn[-probe_fr-1:-1]).sum(-1)).mean())
        # (2) DOES ATTENTION IDENTIFY REDUNDANT PATCHES? Correlate each candidate
        # patch's TRUE attention mass against its novelty. ~0 => attention carries no
        # information about redundancy, so attention-based selection (top-k/top-p)
        # and content-based selection (kk/ToMe) are orthogonal criteria.
        am = p_all[cand].float(); nvv = novf.float()
        amm, nvm = am.mean(), nvv.mean()
        out["corr_attn_novelty"] = float(((am-amm)*(nvv-nvm)).sum() /
            (((am-amm).pow(2).sum()*(nvv-nvm).pow(2).sum()).sqrt() + 1e-9))
        # mass carried by the most-redundant quartile vs the least-redundant quartile
        o_ = nvv.argsort(); qn = max(1, cand.numel()//4)
        out["mass_most_redundant_q"] = float(am[o_[:qn]].sum() / (am.sum()+1e-9))
        out["mass_least_redundant_q"] = float(am[o_[-qn:]].sum() / (am.sum()+1e-9))
        out["global_mass_true"] = float(p_full[n:].sum() / (float(p_full.sum())+1e-9)) if n_glob else 0.0

        # (1) GLOBAL novelty: best match against ALL earlier frames, not just f-1.
        # Bounds what any dedup method could remove, and is the quantity that should
        # grow with window size if larger windows really do accumulate redundancy.
        if _TAIL_PROBE_HEAVY and nfr > 2:
            # EVERY frame against EVERY earlier frame -- the full O(frames^2) measure,
            # chunked over target frames so the (F, P, P) similarity tensor never
            # materialises for the whole window at once. Chunk size is the only knob;
            # the computation itself is exact.
            CH = _TAIL_PROBE_CHUNK
            sp_num = ap_num = cnt = 0.0
            bm = [0.0]*6; bm_n = 0
            mg = {"m12": 0.0, "m1med": 0.0, "z1": 0.0, "sd": 0.0, "s1": 0.0, "smed": 0.0}
            mg_tie = [0.0]*4; mg_tiefrac = [0.0]*4; mg_n = 0
            indeg = torch.zeros(nfr, dtype=torch.float32, device=gn.device)
            for f0 in range(1, nfr, CH):
                f1 = min(f0 + CH, nfr)
                cur = gn[f0:f1]                                        # (c, P, d)
                for i in range(cur.size(0)):
                    fa = f0 + i
                    prev = gn[:fa]                                     # (fa, P, d)
                    c = cur[i]                                         # (P, d)
                    # same spatial position, best over all earlier frames
                    sims = (prev * c[None]).sum(-1)                   # (fa, P)
                    sp = sims.amax(dim=0)                             # (P,)
                    sp_num += float((1.0 - sp).mean()); cnt += 1
                    # WHERE does the best match live? distance = fa - argmax_frame.
                    # d==1 means the previous frame really was the best match; larger
                    # distances mean adjacent-frame comparison saw only a transient
                    # (occlusion, mid-pan, wrong phase of a cycle, alternating shot).
                    amax = sims.argmax(dim=0)                        # (P,) source frame per patch
                    dist = (fa - amax).float()
                    for bi, (blo, bhi) in enumerate(((1,1),(2,3),(4,7),(8,15),(16,31),(32,10**6))):
                        bm[bi] += float(((dist >= blo) & (dist <= bhi)).float().mean())
                    bm_n += 1
                    # MARGIN. in-degree counts argmax votes, and an argmax is only
                    # meaningful if the winner actually BEATS the runners-up. If a
                    # patch sits at ~equal similarity to every earlier frame, its
                    # vote is a coin flip: the concentration measured above would be
                    # noise, and -- the part that matters for an I-frame scheme --
                    # discarding the "winner" would cost nothing, since a near-equal
                    # substitute is still in the cache. So record how peaked each
                    # patch's across-frame similarity profile is, not just where it
                    # peaks. n_tie(eps) = how many frames are within eps of the best,
                    # i.e. how many interchangeable references that patch has.
                    if fa >= 3:
                        t2 = sims.topk(2, dim=0).values                 # (2, P)
                        s1, s2 = t2[0], t2[1]
                        med = sims.median(dim=0).values
                        mu = sims.mean(dim=0); sd = sims.std(dim=0)
                        mg["m12"] += float((s1 - s2).mean())
                        mg["m1med"] += float((s1 - med).mean())
                        mg["z1"] += float(((s1 - mu) / (sd + 1e-6)).mean())
                        mg["sd"] += float(sd.mean())
                        mg["s1"] += float(s1.mean()); mg["smed"] += float(med.mean())
                        for ei, eps in enumerate((0.005, 0.01, 0.02, 0.05)):
                            tied = (sims >= (s1 - eps)[None]).float().sum(0)
                            mg_tie[ei] += float(tied.mean())
                            mg_tiefrac[ei] += float((tied / fa).mean())
                        mg_n += 1
                    # PER-FRAME IN-DEGREE: how many later patches pick frame j as their
                    # best match. Concentrated counts => natural reference ("I") frames
                    # exist and a keep-dense/compress-dependents scheme has something to
                    # build on. Flat counts => matches are diffuse and there are none.
                    indeg.scatter_add_(0, amax, torch.ones_like(amax, dtype=indeg.dtype))
                    # content match: best counterpart ANYWHERE in ANY earlier frame
                    best = None
                    for g0 in range(0, fa, CH):
                        blk = prev[g0:min(g0 + CH, fa)]                # (b, P, d)
                        m = torch.einsum("pd,bqd->bpq", c, blk).amax(dim=(0, 2))
                        best = m if best is None else torch.maximum(best, m)
                    ap_num += float((1.0 - best).mean())
            out["novelty_global_samepos"] = sp_num / cnt
            if bm_n:
                out["bestmatch_dist_hist"] = [x / bm_n for x in bm]   # d=1,2-3,4-7,8-15,16-31,32+
                tot = float(indeg.sum()) or 1.0
                sh = (indeg / tot)
                srt = torch.sort(sh, descending=True).values
                out["indeg_top1_share"] = float(srt[0])
                out["indeg_top5_share"] = float(srt[:5].sum())
                out["indeg_top10_share"] = float(srt[:10].sum())
                out["indeg_uniform_top5"] = 5.0 / nfr               # what flat would give
                # Gini: 0 = perfectly flat, 1 = one frame takes everything
                srt_a = torch.sort(sh).values
                idx = torch.arange(1, nfr + 1, device=sh.device, dtype=torch.float32)
                out["indeg_gini"] = float((2 * (idx * srt_a).sum()) / (nfr * srt_a.sum() + 1e-9) - (nfr + 1) / nfr)
                out["indeg_nframes"] = int(nfr)
            if mg_n:
                for _k, _v in mg.items():
                    out["margin_" + _k] = _v / mg_n
                out["margin_tie_count"] = [x / mg_n for x in mg_tie]      # eps=.005,.01,.02,.05
                out["margin_tie_frac"] = [x / mg_n for x in mg_tiefrac]
                out["margin_n"] = int(mg_n)
            out["novelty_global_anypos"] = ap_num / cnt
            out["novelty_global_nframes"] = int(cnt)
            out["novelty_lookback1_ref"] = float(nv1.mean())

        order = novf.argsort()                                         # most redundant first
        for r, tag in ((2, "keep_2"), (4, "kk_2")):
            nd = int(cand.numel() * (r - 1) / r)
            dropped = cand[order[:nd]]
            # chance baseline for this ratio is exactly nd/pool_size
            out[f"massdrop_{tag}"] = float(p_all[dropped].sum()) / pool_mass if pool_mass > 0 else 0.0
            out[f"chance_{tag}"] = nd / cand.numel()
            # WHAT COULD A BETTER CRITERION ACHIEVE AT THE SAME TOKEN COUNT?
            # kk_2 discards ~67% of pool mass while discarding 75% of pool tokens --
            # only ~8 points better than random. That indicts the CRITERION, not
            # compression itself. Dropping the same NUMBER of tokens chosen by lowest
            # mass is the floor any selector could reach; by highest mass is the
            # ceiling. floor vs massdrop_{tag} is the headroom a better criterion has.
            # Cheap (one sort of the pool) and deliberately outside the HEAVY block,
            # so it is measured on every probe run.
            if 0 < nd < cand.numel() and pool_mass > 0:
                srt = torch.sort(p_all[cand]).values
                out[f"massdrop_{tag}_floor"] = float(srt[:nd].sum()) / pool_mass
                out[f"massdrop_{tag}_ceil"] = float(srt[-nd:].sum()) / pool_mass

            # DOES DROPPING DUPLICATES ACTUALLY DIVERSIFY THE OUTPUT?
            # Mass is the wrong objective: any drop lowers it. The claim worth testing
            # is that removing redundant tokens stops repeated content being weighted
            # in proportion to how often it repeats. Pruning + renormalising preserves
            # RELATIVE weights among survivors exactly, so the only thing that can
            # change is which directions contribute -- which is what eff_distinct sees.
            # The random arm is the control: if a random drop of the same size raises
            # eff_distinct just as much, the novelty criterion earns nothing.
            keep = torch.ones(cand.numel(), dtype=torch.bool, device=cand.device)
            keep[order[:nd]] = False

            # WHAT swa_vv_adj_<r> WOULD SELECT: the identical adjacent-frame criterion
            # scored on VALUES instead of keys. Recording it here answers two things the
            # accuracy arm cannot: whether the value criterion picks DIFFERENT tokens at
            # all (vv_overlap_*), and whether its selection carries more independent
            # directions (effdist_v_vvadj_*). A high overlap would explain an accuracy
            # null as "same tokens", not "different tokens, same outcome".
            vkeep = None
            if local_h_v is not None:
                _vvv = local_h_v.float().mean(dim=1)[0]
                if _vvv.size(0) >= n:
                    _vg = _vvv[vidx]                                       # (frames,patches,d)
                    _vnov = 1.0 - torch.nn.functional.cosine_similarity(
                        _vg[1:], _vg[:-1], dim=-1)                          # (frames-1,patches)
                    _vnf = _vnov.reshape(-1)[keep_tail]
                    if _vnf.numel() == cand.numel():
                        vorder = _vnf.argsort()                             # most redundant first
                        vkeep = torch.ones(cand.numel(), dtype=torch.bool, device=cand.device)
                        vkeep[vorder[:nd]] = False
                        out[f"vv_overlap_{tag}"] = float((keep & vkeep).sum()) / max(1, int(keep.sum()))
            gen = torch.Generator(device="cpu").manual_seed(12345 + int(layer_idx))
            rnd = torch.ones(cand.numel(), dtype=torch.bool)
            rnd[torch.randperm(cand.numel(), generator=gen)[:nd]] = False
            rnd = rnd.to(cand.device)
            # DOES THE VDIV CRITERION ACTUALLY RAISE POOL-LEVEL DIVERSITY?
            # swa_vdiv_<s> scores centrality WITHIN each frame, but the redundancy the
            # eff_distinct diagnosis found is pool-level -- mostly content repeating
            # ACROSS frames. Both variants are evaluated here at the SAME token count
            # as kk_2, so an accuracy null can be attributed to the criterion or to its
            # scope, instead of being ambiguous between the two.
            if local_h_v is not None and 0 < nd < cand.numel():
                _vv = local_h_v.float().mean(dim=1)[0]
                if _vv.size(0) >= n and vidx.size(0) > 1:
                    _grid = vidx[1:]                                          # (f,bs)
                    # Same eligibility as `cand`: the dense tail is never a candidate.
                    # Including it would be catastrophic here rather than merely wrong --
                    # the tail carries ~31% of visual mass, so it would dominate the
                    # mass-weighted eff_distinct and make every selection look terrible.
                    _elig = _grid < (n - len_q) if len_q > 0 else torch.ones_like(_grid, dtype=torch.bool)
                    _vn = torch.nn.functional.normalize(_vv[_grid], dim=-1)      # (f,bs,d)
                    _vn = _vn * _elig.unsqueeze(-1).to(_vn.dtype)
                    _bs = _vn.size(1)
                    _keep_per = max(1, min(_bs, int(round(_bs * (1.0 - nd / cand.numel())))))
                    for _tag, _res in (("vdivframe", _vn.sum(dim=1, keepdim=True)),
                                       ("vdivpool", _vn.sum(dim=(0, 1), keepdim=True))):
                        _cen = (_vn * _res).sum(-1) - _elig.to(_vn.dtype)        # (f,bs)
                        _cen = torch.where(_elig, _cen, torch.full_like(_cen, float('inf')))
                        _sel = _cen.topk(_keep_per, dim=-1, largest=False).indices
                        _idx = torch.gather(_grid, 1, _sel)
                        _ok = torch.gather(_elig, 1, _sel)
                        _idx = _idx[_ok]
                        out[f"effdist_v_{_tag}_{tag}"] = _eff_distinct(p_all[_idx], _vv[_idx])
                        out[f"effdistU_v_{_tag}_{tag}"] = _eff_distinct(p_all[_idx], _vv[_idx], uniform=True)
                        out[f"n_{_tag}_{tag}"] = int(_idx.numel())

            for src, vv in (("v", local_h_v), ("k", None)):
                vec = None
                if src == "k":
                    vec = kf                                   # heads-pooled keys
                elif vv is not None:
                    vec = vv.float().mean(dim=1)[0]            # heads-pooled values
                if vec is None or vec.size(0) < n:
                    continue
                out[f"effdist_{src}_dense_{tag}"] = _eff_distinct(p_all[cand], vec[cand])
                out[f"effdist_{src}_{tag}"] = _eff_distinct(p_all[cand[keep]], vec[cand[keep]])
                out[f"effdist_{src}_rand_{tag}"] = _eff_distinct(p_all[cand[rnd]], vec[cand[rnd]])
                out[f"effdistU_{src}_dense_{tag}"] = _eff_distinct(p_all[cand], vec[cand], uniform=True)
                out[f"effdistU_{src}_{tag}"] = _eff_distinct(p_all[cand[keep]], vec[cand[keep]], uniform=True)
                out[f"effdistU_{src}_rand_{tag}"] = _eff_distinct(p_all[cand[rnd]], vec[cand[rnd]], uniform=True)
                if vkeep is not None:
                    out[f"effdist_{src}_vvadj_{tag}"] = _eff_distinct(p_all[cand[vkeep]], vec[cand[vkeep]])
                    out[f"effdistU_{src}_vvadj_{tag}"] = _eff_distinct(p_all[cand[vkeep]], vec[cand[vkeep]], uniform=True)
        return out


class ContextManager:
    def __init__(self, 
                 layer_idx,
                 position_embedding,
                 n_init, n_local, 
                 block_size, max_cached_block, topk, 
                 encode_chunk_size, chunk_size, exc_block_size, 
                 fattn: bool = False,
                 async_global_stream: bool = False,
                 pin_memory: bool = False,
                 prefill_method: str = "swa",     # swa: sliding window attention + sink tokens 
                 retrieval_method: str = "rekv",  # rekv: avg pool keys and question features and compute cosine similarity 
                 head_idx: int = -1,              # -1 for all heads, 0 for first head, 1 for second head, etc.
                 store_full: bool = False,
                 store_adjacent_similarities: bool = False,
                 store_attn_scores: str = None, # store attention scores - options: "prefill", "retrieval", "decode", "all" 
    ):

        # Analysis env toggles (STORAGE_ON, STORE_ATTN_SCORES, MEMSTREAM_*_DIR)
        # may force store_full / store_attn_scores; inert in normal runs.
        store_full, store_adjacent_similarities, store_attn_scores = \
            diagnostics.apply_env_overrides(
                store_full, store_adjacent_similarities, store_attn_scores)

        self.layer_idx = layer_idx
        self.length = 0  # number of tokens in the KV-Cache
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.block_size = block_size
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size
        assert exc_block_size <= n_local # no global token in input
        self.topk = topk
        self.chunk_size = chunk_size

        # use torch impl if storing attention scores or running top-k / top-p attention
        if store_attn_scores == "all" or store_attn_scores == "prefill":
            fattn = False
        if (os.environ.get("MEMSTREAM_TOPK_ATTN") or os.environ.get("MEMSTREAM_TOPP_ATTN")
                or os.environ.get("MEMSTREAM_ATTN_SPLIT")):
            fattn = False

        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        self.fattn = fattn

        self.initialized = False
        self.load_count = 0
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory
        global GLOBAL_STREAM
        
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

        self.grid_hw = (int(math.sqrt(block_size)), int(math.sqrt(block_size)))

        """ prefill methods """
        self.prefill_method = prefill_method
        self.encode_chunk_size = encode_chunk_size
        
        # Parse nd parameter (number of dense frames) from prefill_method if present
        # e.g., "swa_kk_adjacent_4_nd=2" or "swa_dilated_4_nd=2_after_10f"
        self.nd = 1  # default: keep 1 frame as dense
        if "_nd=" in self.prefill_method:
            # Extract nd value, e.g., "swa_kk_adjacent_4_nd=2" -> 2
            nd_str = self.prefill_method.split("_nd=")[1].split("_")[0]
            self.nd = int(nd_str)

        # Validate the qframe arm name at CONSTRUCTION, not at the first compressed window
        # ten minutes into a job. Same parser the dispatch uses, so the two cannot diverge.
        self._qframe_cfg = None
        if self.prefill_method.startswith("swa_qframe_"):
            self._qframe_cfg = _parse_qframe_config(self.prefill_method)
            if self.layer_idx == 0:
                _n, _lam, _z = self._qframe_cfg
                # ABSENCE OF THIS LINE IN A JOB LOG MEANS THE ARM DID NOT PARSE, i.e. the
                # code snapshot the job ran is stale. Cheapest possible staleness detector.
                print(f"[MEMSTREAM-KNOB] swa_qframe: n={_n} lam={_lam:.2f} z={_z} "
                      f"nd={self.nd} -- keeps {_n} sparse + {self.nd} dense frames/window, "
                      f"selected by pre-RoPE query-frame affinity minus lam*recency profile",
                      flush=True)

        """ retrieval methods """
        # rekv - avg pooling - apply dot product with avg query - sum to get score 
        supported_retrieval_methods = ["rekv"] 
        # topk selection methods 
        supported_retrieval_methods += ["topk_watershed_rekv"]
        # qt to vb methods (deprecated for now)
        supported_retrieval_methods += ["qt_to_vb_max", "qt_to_vb_min", "qt_to_vb_mean"]
        
        # Check if it's a uniform sampling method with fallback (e.g., "uniform_0_14_rekv")
        # Pattern: uniform_X_Y_METHOD means layers X to Y-1 use uniform, others use METHOD
        is_uniform_with_fallback = re.match(r"uniform_(\d+)_(\d+)_(\w+)", retrieval_method)
        self.use_uniform_sampling = False
        if is_uniform_with_fallback:
            uniform_layer_start = int(is_uniform_with_fallback.group(1))
            uniform_layer_end = int(is_uniform_with_fallback.group(2))
            fallback_method = is_uniform_with_fallback.group(3)
            
            # Validate fallback method
            assert fallback_method in supported_retrieval_methods, f"Fallback method '{fallback_method}' is not supported!"
            
            # Check if this layer should use uniform sampling (0_4 --> 0,1,2,3)
            self.use_uniform_sampling = (uniform_layer_start <= layer_idx < uniform_layer_end)
            
            if self.use_uniform_sampling:
                logger.info(f"Layer {layer_idx}: Using uniform sampling (within range [{uniform_layer_start}, {uniform_layer_end}))")
                self.retrieval_method = retrieval_method  # Keep the full string for logging
            else:
                logger.info(f"Layer {layer_idx}: Using {fallback_method} retrieval (outside uniform range [{uniform_layer_start}, {uniform_layer_end}))")
                self.retrieval_method = fallback_method  # Use fallback method for non-uniform layers
        else:
            self.retrieval_method = retrieval_method 

        self.reset_retrieval()

        logger.info(f"context manager is using prefill: {self.prefill_method}, retrieval: {self.retrieval_method} using head {head_idx} (-1 for all heads)")

        self.head_idx = head_idx
        self.store_full = store_full
        self.store_adjacent_similarities = store_adjacent_similarities
        
        assert store_attn_scores is None or store_attn_scores in ["prefill", "retrieval", "decode", "all"]
        self.store_attn_scores = store_attn_scores

        if self.store_full or self.store_adjacent_similarities:
            logger.warning("storing full!")

    def _remove_lru_blocks(self, u, num_remove: Optional[int] = None, ignore_blocks=None):

        if num_remove is None:
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block

        if num_remove <= 0:
            return

        lst = list(self.cached_blocks[u].items())
        lst.sort(key=lambda x: x[1])


        removed = 0
        for i in range(len(lst)):
            idx = lst[i][0]
            if ignore_blocks is None or (idx not in ignore_blocks):
                self.global_blocks[u][idx].offload()
                self.cached_blocks[u].pop(idx)
                removed += 1

            if removed >= num_remove:
                return

    # handle GQA, k: (batch_size, n_head_kv, length, dim_head) -> (batch_size, n_head, length, dim_head)
    def _from_group_kv(self, tensor):
        # tensor: (batch_size, n_head_kv, length, dim_head)
        assert tensor.dim() == 4 
        assert tensor.size(1) == self.num_heads_kv
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view((self.num_units, self.unit_size_kv, 1, length, dim_head))  # (batch_size, n_head_kv, 1, length, dim_head)
        tensor = tensor.expand((self.num_units, self.unit_size_kv, num_group, length, dim_head)).reshape((self.num_units, self.num_heads, length, dim_head))  # (batch_size, n_head, length, dim_head)
        return tensor
    
    def init(
        self, 
        local_q, local_k, local_v,
        global_q, global_k, global_v
    ):
        """
        Only use the metadata of these parameters, such as shape, dtype, and device.
        """
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert (_t.size(1) == num_heads or _t.size(1) == num_heads_kv)
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.num_units = batch_size
        self.unit_size = num_heads
        self.unit_size_kv = num_heads_kv

        self.global_blocks = [[] for _ in range(self.num_units)] # context memory's KV-Cache: [ batch_size x [memory_unit] ]
        self.cached_blocks = [{} for _ in range(self.num_units)] # relavency scores of blocks: batch_size x {block_id: block_score}
        self.num_global_block = 0  # Number of blocks offloaded to CPU
        self.num_blocks_seen = 0  # Total number of blocks processed (including those in local window)

        # context memory's representative keys: batch_size x (n_blocks, hidden_dim)
        self.block_k = [VectorTensor(
            dim_head * self.unit_size, global_k.dtype, global_k.device
        ) for _ in range(self.num_units)]
        # Value-space twin of block_k, populated only when MEMSTREAM_VRETR is set so the
        # default path allocates nothing extra.
        self.block_v = [VectorTensor(
            dim_head * self.unit_size, global_k.dtype, global_k.device
        ) for _ in range(self.num_units)] if (_VRETR or _FUSE_RETR) else None

        # local KV - (batch_size, n_head_kv, 0, dim_head)
        self.local_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device) 
        self.local_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device)


        """ Qwen3-VL parameters """ 
        # store the timestamp and boundary embeddings for all frames 
        self.full_timestamp_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device) 
        self.full_timestamp_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device) 
        self.full_boundary_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device)
        self.full_boundary_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device)
        
        # adaptive block-size or not (Qwen3-VL vs. Qwen2.5/2-VL)
        self.has_adaptive_block_size = False 
        # keep track of block-size <t1><vision_start><f1><vision_end> 
        self.adaptive_block_sizes = []
        # compressed retrieval; populated per video by _cretr_setup()
        self._cretr_sel = None
        self._cretr_kept = 0
        self._cretr_stage = None
        # number of prefix tokens (e.g. timestamp) to skip in representative vec; set per-video via _initialize_cache_for_video
        self.n_block_prefix_tokens = 0 
        # Sliding window of position ids for the current local window (kept in sync with local_k/local_v)
        self.sliding_window_pos_ids = torch.empty((3, 1, 0), dtype=torch.long)

        
        # blockwise cosine similarity
        self.block_sims = None # (batch_size, n_blocks, n_blocks) 

        # For event-chunking: store adjacent key similarities
        self.adjacent_key_similarities = [] if self.store_adjacent_similarities else None
        self.last_block_key = None  # Store last key from previous block for similarity computation
        self._aks_prev_block_k = None  # previous block's tokens, for retrieval-time AKS novelty

        # swa_qframe: running mean of per-frame query affinity, indexed by frames_ago.
        # Per-LAYER (one ContextManager per decoder layer) and per-VIDEO (clear_cache()
        # destroys the managers, so anything allocated here is fresh for the next video).
        self._qframe_profile = RecencyProfile()

        # Running boolean mask (mirrors self.local_k length) — True where tokens are visual.
        # Populated on the first append() call that carries a visual_pos_masks argument.
        # Stays None for models that don't pass visual_pos_masks (Qwen2-VL, etc.).
        self.visual_mask_buffer = None

        # global KV that are not yet processed into blocks.
        # 2 x (batch_size, n_head_kv, length, dim_head)
        self.global_remainder = (
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_v.dtype, device=global_v.device),
        )
        self.global_remainder_visual_mask = None

        # init KV
        self.init_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_exc = False
        self.dtype = local_q.dtype

        # full KV (for debugging or full attention prefill)
        if self.store_full:
            diagnostics.init_full_storage(self, global_k)

        # set the position embedding to the max length for initializing cache 
        # for local attention - only first n_init positions are used, so still valid 
        # for global attention - although use the n_local position, k, v are 0 - so effectively an identity operation 
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )
        self.pos_emb_device = local_k.device 

        # buffering global KV during attention computations
        # (2, batch_size, n_head_kv, L, dim_head)
        # L = n_init + n_retrieve
        buffer_len = self.topk * self.block_size + self.n_init
        self.global_buffer = torch.zeros(
                (2, self.num_units, self.unit_size_kv, buffer_len , dim_head),
                dtype = global_k.dtype, device=global_k.device
            )
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.num_units,
            self.unit_size_kv * self.block_size * dim_head * 2,
            local_k.dtype
        )  # (max_cached_block * batch_size, block_size * D * 2)

        self.initialized = True

    def set_retrieval(self):
        self.to_retrieve = True

    def reset_retrieval(self):
        self.similarity = None
        self.full_similarity = None
        self.retrieved_block_indices = None
        self.to_retrieve = False      

    def reset_position_embedding(self, n_local):
        self.n_local = n_local
        if hasattr(self.position_embedding, "initialized_for_qa"):
            self.position_embedding.initialized_for_qa = False
        # Qwen3-VL: the RoPE module is shared by every layer and outlives clear_cache(), so its
        # video-specific T-RoPE tables must be invalidated here or the next video may reuse the
        # previous video's tables. The window-skip in _recompute_cos_sin_tables_for_window keys
        # on the exact position ids the cached table was built from (_built_window_pos_ids);
        # clear that key too so the next video starts from a rebuild.
        if hasattr(self.position_embedding, "_has_trope_tables"):
            self.position_embedding._has_trope_tables = False
            self.position_embedding.curr_block_sizes = []
            self.position_embedding._built_window_pos_ids = None
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, self.pos_emb_device, self.dim_head 
        )

    def _cretr_block_tokens(self, new_block_size):
        """Tokens a single retrieved block occupies in the buffer."""
        visual = self._cretr_kept if self._cretr_sel is not None else new_block_size
        return visual + self.n_block_prefix_tokens

    def _cretr_setup(self, new_block_size):
        """Pick the kept patches for compressed retrieval and re-derive the slot count.

        Selection is a 2-D stride over the patch grid, matching the encode-side convention
        where dilation_rate is a per-axis stride -- kept patches stay spread across the frame
        instead of collapsing into one corner.

        The slot count is derived from MEASURED widths, not from r^2. Two things make the
        nominal ratio wrong: the grid rarely divides evenly (30x17 at r=2 keeps 15x9=135, a
        3.78x reduction, not 4x), and every extra block drags an uncompressed timestamp
        prefix along with it. Deriving from widths keeps the token budget actually matched,
        which is the whole premise of the comparison.
        """
        self._cretr_sel = None
        self._cretr_kept = new_block_size
        self._cretr_stage = None
        if _CRETR_R <= 1:
            return

        gh, gw = self.grid_hw
        assert gh * gw == new_block_size, (
            f"compressed retrieval needs the patch grid to match the block: "
            f"{gh}x{gw}={gh * gw} vs block_size={new_block_size}"
        )
        dev = self.global_buffer.device
        r = _CRETR_R
        rows = torch.arange(0, gh, r, device=dev)
        cols = torch.arange(0, gw, r, device=dev)
        self._cretr_sel = (rows.view(-1, 1) * gw + cols.view(1, -1)).flatten()
        self._cretr_kept = int(self._cretr_sel.numel())

        full = new_block_size + self.n_block_prefix_tokens
        comp = self._cretr_kept + self.n_block_prefix_tokens
        self.topk = _CRETR_SLOTS or max(1, (self.topk * full) // comp)

        # The CPU path's BlockUnit.load() writes a full block, so compressed loads stage
        # through this scratch tensor and gather out of it.
        self._cretr_stage = torch.zeros(
            (2, self.unit_size_kv, new_block_size, self.dim_head),
            dtype=self.global_buffer.dtype, device=dev,
        )
        # Fail here, with the number to set, rather than inside CudaCache.alloc()'s bare assert.
        if self.max_cached_block < self.topk:
            raise ValueError(
                f"compressed retrieval needs max_cached_block >= topk ({self.topk}), got "
                f"{self.max_cached_block}. One question retrieves {self.topk} blocks and the "
                f"LRU will not evict a block it is about to read, so CudaCache.alloc() would "
                f"assert on an exhausted idle set partway through a question."
            )
        logger.info(
            f"[compressed-retrieval] grid {gh}x{gw}, r={r}: keep {self._cretr_kept}/"
            f"{new_block_size} visual tokens per block ({new_block_size / self._cretr_kept:.2f}x), "
            f"{self._topk_base} -> {self.topk} slots, "
            f"{self._topk_base * full} -> {self.topk * comp} tokens"
        )

    def initialize_cache_for_video(self, new_block_size, cache_position_ids, video_grid_thw, max_frames, init_block_sizes=None):
        """ Update the cache size to match the new block size.
            Args:
                new_block_size: (int) number of tokens per frame
                cache_position ids: (3, 1, n_video_tokens)
                max_frames: min number of frames that we must encode 
                init_block_sizes: used for Qwen3-VL - handles variable block-sizes for each frame + timestamps 
        """
        # store height and width of the video (needed by the compressed-retrieval grid below,
        # so it is set here rather than at the end of this function)
        self.grid_hw = (video_grid_thw[0, 1].item(), video_grid_thw[0, 2].item())

        # This runs once per VIDEO. Compressed retrieval rewrites self.topk, so the base value
        # has to be restored first or each video would scale the previous video's slot count.
        if not hasattr(self, "_topk_base"):
            self._topk_base = self.topk
        self.topk = self._topk_base
        # Belt and braces: the qframe recency profile is already per-video by virtue of
        # clear_cache() destroying the ContextManagers, but this hook is the documented
        # once-per-video entry point, so reset here too against any future manager reuse.
        self._qframe_profile = RecencyProfile()
        self._cretr_setup(new_block_size)
        per_block_tokens = self._cretr_block_tokens(new_block_size)

        new_unit_size = self.unit_size_kv * new_block_size * self.dim_head * 2
        self.global_buffer = torch.zeros(
            (2, self.num_units, self.unit_size_kv, self.topk * per_block_tokens + self.n_init, self.dim_head),
            dtype=self.global_buffer[0].dtype, device=self.global_buffer[0].device
        )
        self.cuda_cache.update_cache_size(new_unit_size)

        # set adaptive block sizes 
        self.has_adaptive_block_size = init_block_sizes is not None

        if self._cretr_sel is not None and not self.has_adaptive_block_size:
            raise NotImplementedError(
                "compressed retrieval requires the adaptive-block (Qwen3-VL) path. On the "
                "fixed-width path _build_retrieved_position_ids returns None and the QA RoPE "
                "tables are rebuilt from the buffer length alone, which would hand contiguous "
                "positions to strided patches and silently corrupt the M-RoPE grid."
            )

        if self.has_adaptive_block_size:
            self.position_embedding._update_cos_sin_tables_pos(self.pos_emb_device, cache_position_ids, self.n_init, 
                init_block_sizes, new_block_size)
        else: 
            self.position_embedding._update_cos_sin_tables_pos(self.pos_emb_device, cache_position_ids, self.n_init) 


        full_block_size = new_block_size + self.n_block_prefix_tokens  # accounts for timestamp prefix (0 for Qwen2/2.5-VL)
        n_local_frames = self.n_local // full_block_size  # update n_local to be the closest multiple of full block size  
        self.n_local_frames = min(n_local_frames, max_frames-1)  # choose n_local_frames to be 1 LESS than max_frames 
        self.n_local = self.n_local_frames * full_block_size

        # Retrieving more frames than the sliding window was built for makes the QA context exceed
        # n_local, which does not raise anywhere downstream -- it silently produces degenerate
        # generations (repetition loops, numeric garbage). Refuse to run instead.
        # The real constraint is TOKENS, not frames: compressed retrieval deliberately holds
        # more blocks than the window has frame-slots, which is safe precisely because each
        # block is narrower. Checking frames here would reject a run that fits comfortably.
        retrieved_tokens = self.topk * per_block_tokens
        if retrieved_tokens > self.n_local:
            raise ValueError(
                f"retrieval would pack {retrieved_tokens} tokens ({self.topk} blocks x "
                f"{per_block_tokens}) into a window built for {self.n_local}. This silently "
                f"degrades generation quality. Either lower retrieve_size, raise n_local to "
                f">= {retrieved_tokens}, or lower tokens-per-frame (block_size="
                f"{new_block_size}, +{self.n_block_prefix_tokens} prefix"
                + (f", compressed to {self._cretr_kept} visual tokens" if self._cretr_sel is not None else "")
                + ")."
            )

        logger.info(f"updated self.n_local: {self.n_local} - {self.n_local_frames} - {self.n_local_frames} - {new_block_size} - {self.global_buffer[0].shape} - {cache_position_ids.shape}")

    def set_retrieved_block_indices(self, retrieved_block_indices):
        # retrieved_block_indices (list): batch_size x n_frames
        if isinstance(retrieved_block_indices, torch.Tensor):
            retrieved_block_indices = retrieved_block_indices.cpu().tolist()
        self.retrieved_block_indices = retrieved_block_indices
        
        # Log external similarities if available and method is patched
        if hasattr(self, 'similarity') and self.similarity is not None and hasattr(self._calc_block_topk, '__closure__'):
            try:
                for cell in self._calc_block_topk.__closure__ or []:
                    if hasattr(cell.cell_contents, '_log_similarity'):
                        sim = self.similarity if isinstance(self.similarity, torch.Tensor) else torch.tensor(self.similarity)
                        cell.cell_contents._log_similarity(self.layer_idx, sim.clone().detach().cpu())
                        break
            except (ValueError, AttributeError):
                pass

    def reciprocal_rank_stream_fusion(self, retrieved_block_indices_lst, sims_lst, k=60, weights=None, break_ties="sims", n_pairs_to_keep=5): 
        """RRF fusion for dual-stream: top n_pairs get both features, rest use tie-breaker.
        
        Args:
            retrieved_block_indices_lst: [even_frames, odd_frames] with global indices
            sims_lst: [even_sims, odd_sims] sorted similarity scores
            k: RRF constant
            weights: [even_weight, odd_weight]
            break_ties: "sims" for relative importance, "odd_only" for only odd frames
            n_pairs_to_keep: Number of top pairs to keep both features from
        """
        weights = weights or [0.5, 0.5]
        
        # Build rank dictionaries: global_frame_idx -> rank
        even_ranks = {frame_idx: rank for rank, frame_idx in enumerate(retrieved_block_indices_lst[0], 1)}
        odd_ranks = {frame_idx: rank for rank, frame_idx in enumerate(retrieved_block_indices_lst[1], 1)}
        
        # Score pairs by combined RRF
        n_pairs = (len(retrieved_block_indices_lst[0]) + len(retrieved_block_indices_lst[1])) // 2
        pair_scores = {
            i: [weights[0] / (k + even_ranks[i*2]), weights[1] / (k + odd_ranks[i*2+1])]
            for i in range(n_pairs)
        }
        
        # Sort pairs by total score (tie-break by best individual rank)
        fused_pairs = sorted(pair_scores.keys(), key=lambda i: (
            -sum(pair_scores[i]),
            min(even_ranks.get(i*2, float('inf')), odd_ranks.get(i*2+1, float('inf')))
        ))
        
        selected_frames = []
        
        # Top n pairs: keep both even and odd 
        for pair_idx in fused_pairs[:n_pairs_to_keep]:
            selected_frames.extend([pair_idx * 2, pair_idx * 2 + 1])
        
        # Remaining: tie-breaker based on strategy
        if break_ties == "odd_only":
            # Always use odd frames (uniform_sample_8) for remaining
            for pair_idx in fused_pairs[n_pairs_to_keep:]:
                if len(selected_frames) >= self.topk:
                    break
                selected_frames.append(pair_idx * 2 + 1)  # odd frame
        
        elif break_ties == "sims" and sims_lst:
            even_sims_total = sum(sims_lst[0])
            odd_sims_total = sum(sims_lst[1])
            
            # Pre-compute lookups for O(1) access
            even_idx_to_sim = {retrieved_block_indices_lst[0][i]: sims_lst[0][i] for i in range(len(sims_lst[0]))}
            odd_idx_to_sim = {retrieved_block_indices_lst[1][i]: sims_lst[1][i] for i in range(len(sims_lst[1]))}
            
            for pair_idx in fused_pairs[n_pairs_to_keep:]:
                if len(selected_frames) >= self.topk:
                    break
                
                even_frame, odd_frame = pair_idx * 2, pair_idx * 2 + 1
                even_relative = even_idx_to_sim[even_frame] / even_sims_total
                odd_relative = odd_idx_to_sim[odd_frame] / odd_sims_total
                
                selected_frames.append(even_frame if even_relative > odd_relative else odd_frame)
        
        return torch.tensor([selected_frames[:self.topk]]), pair_scores

    def reciprocal_rank_fusion(self, retrieved_block_indices_lst, k=60, weights=None):
        """ compute rrf algorithm for fusing retrievals across internal and external 
            retrieved_block_indices_lst: list[list of len self.topk] - retrieved block indices for each retrieval
            k: int - constant for rrf 
            weights: int - if we wish to weight different retrievals 
        """ 
   
        if weights is None:
            weights = [1.0] * len(retrieved_block_indices_lst)

        # get ranks for each retrieval 
        rL_lst = [] 
        for L in retrieved_block_indices_lst:
            rL = {d: i+1 for i, d in enumerate(L)}
            rL_lst.append(rL)

        # get all ids across all retrievals        
        all_ids = set()
        for L in retrieved_block_indices_lst: 
            all_ids = all_ids.union(L)

        scores = defaultdict(float)
        for d in all_ids:
            for i, rL in enumerate(rL_lst):
                if d in rL:
                    scores[d] += weights[i] * (1.0 / (k + rL[d]))

        # tie-break by best individual retrieval, then prefer earlier retrievals
        def best_rank(d):
            ranks = [rL.get(d, float('inf')) for rL in rL_lst]
            return min(ranks), ranks

        fused = sorted(all_ids, key=lambda d: (-scores[d], best_rank(d)))
        fused = torch.tensor(fused)
        fused = fused[None, :self.topk]
        return fused, scores 

    def _get_adaptive_timestamp_bounds(self, block_idx):
        """Return packed timestamp-KV bounds for an adaptive Qwen3 frame."""
        timestamp_lengths = [block_size - self.block_size - 2 for block_size in self.adaptive_block_sizes]
        assert all(length >= 0 for length in timestamp_lengths)
        st = sum(timestamp_lengths[:block_idx])
        return st, st + timestamp_lengths[block_idx]

    def _get_adaptive_full_block_start(self, block_idx, init_ed):
        """Return the start offset of a full adaptive frame block in global_remainder."""
        return init_ed + sum(self.adaptive_block_sizes[:block_idx])

    def _load_retrieved_blocks_from_cpu(self, global_h_k, global_h_v, init_ed):
        """Load selected blocks from offloaded CPU memory into the retrieval buffer."""
        for u in range(self.num_units):
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block
            for b_idx in self.retrieved_block_indices[u]:
                if b_idx not in self.cached_blocks[u]:
                    num_remove += 1
            self._remove_lru_blocks(u, num_remove, self.retrieved_block_indices[u])

        self.load_count += 1
        for u in range(self.num_units):
            for b_idx in self.retrieved_block_indices[u]:
                self.cached_blocks[u][b_idx] = self.load_count

        ed = init_ed
        loaded_block_indices = [[] for _ in range(self.num_units)]

        for u in range(self.num_units):
            assert self.retrieved_block_indices[u][-1] < self.num_global_block, \
                f'{self.retrieved_block_indices[u][-1]}, {self.num_global_block}'

            for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                if self.has_adaptive_block_size:
                    ts_st, ts_ed = self._get_adaptive_timestamp_bounds(b_idx)
                    ts_len = ts_ed - ts_st
                    boundary_st = b_idx * 2
                    boundary_ed = boundary_st + 2
                    assert boundary_ed <= self.full_boundary_k.size(-2)
                    assert ed + ts_len + self._cretr_block_tokens(self.block_size) - self.n_block_prefix_tokens + 2 <= global_h_k.size(-2)
                    # load the timestamp embeddings
                    if ts_len > 0:
                        global_h_k[u, :, ed:ed + ts_len, :] = self.full_timestamp_k[u, :, ts_st:ts_ed, :]
                        global_h_v[u, :, ed:ed + ts_len, :] = self.full_timestamp_v[u, :, ts_st:ts_ed, :]
                        ed += ts_len

                    # load the boundary start token
                    global_h_k[u, :, ed:ed + 1, :] = self.full_boundary_k[u, :, boundary_st:boundary_st + 1, :]
                    global_h_v[u, :, ed:ed + 1, :] = self.full_boundary_v[u, :, boundary_st:boundary_st + 1, :]
                    ed += 1

                    # load the visual tokens
                    visual_st = ed
                    if self._cretr_sel is not None:
                        # BlockUnit.load() always writes a full block, so stage it and gather
                        # the kept patches out. Timestamp/boundary tokens above stay dense.
                        self.global_blocks[u][b_idx].load(
                            (self._cretr_stage[0], self._cretr_stage[1])
                        )
                        ed = visual_st + self._cretr_kept
                        global_h_k[u, :, visual_st:ed, :] = self._cretr_stage[0][:, self._cretr_sel, :]
                        global_h_v[u, :, visual_st:ed, :] = self._cretr_stage[1][:, self._cretr_sel, :]
                    else:
                        ed = visual_st + self.block_size
                        self.global_blocks[u][b_idx].load(
                            (global_h_k[u, :, visual_st:ed, :], global_h_v[u, :, visual_st:ed, :])
                        )

                    # load the boundary end token 
                    global_h_k[u, :, ed:ed + 1, :] = self.full_boundary_k[u, :, boundary_st + 1:boundary_ed, :]
                    global_h_v[u, :, ed:ed + 1, :] = self.full_boundary_v[u, :, boundary_st + 1:boundary_ed, :]
                    ed += 1
                else:
                    st = init_ed + cnt * self.block_size
                    ed = st + self.block_size
                    self.global_blocks[u][b_idx].load(
                        (global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :])
                    )
                
                loaded_block_indices[u].append(b_idx)
                

        return ed, loaded_block_indices 

    def _load_retrieved_blocks_from_remainder(self, global_h_k, global_h_v, init_ed):
        """Load selected blocks from the GPU remainder into the retrieval buffer."""
        ed = init_ed
        loaded_block_indices = [[] for _ in range(self.num_units)]

        for u in range(self.num_units):
            for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                if self.has_adaptive_block_size:
                    remainder_st = self._get_adaptive_full_block_start(b_idx, init_ed)
                    remainder_ed = remainder_st + self.adaptive_block_sizes[b_idx]
                    if remainder_st >= self.global_remainder[0].size(2):
                        break

                    ts_st, ts_ed = self._get_adaptive_timestamp_bounds(b_idx)
                    ts_len = ts_ed - ts_st
                    boundary_st = b_idx * 2
                    boundary_ed = boundary_st + 2
                    assert boundary_ed <= self.full_boundary_k.size(-2)
                    assert ed + ts_len + self.block_size + 2 <= global_h_k.size(-2)
                    # load the timestamp embeddings
                    if ts_len > 0:
                        global_h_k[u, :, ed:ed + ts_len, :] = self.full_timestamp_k[u, :, ts_st:ts_ed, :]
                        global_h_v[u, :, ed:ed + ts_len, :] = self.full_timestamp_v[u, :, ts_st:ts_ed, :]
                        ed += ts_len

                    # load the boundary start token 
                    global_h_k[u, :, ed:ed + 1, :] = self.full_boundary_k[u, :, boundary_st:boundary_st + 1, :]
                    global_h_v[u, :, ed:ed + 1, :] = self.full_boundary_v[u, :, boundary_st:boundary_st + 1, :]
                    ed += 1

                    # load the visual tokens
                    assert self.global_remainder_visual_mask is not None
                    visual_mask = self.global_remainder_visual_mask[u, remainder_st:remainder_ed]
                    block_k = self.global_remainder[0][u, :, remainder_st:remainder_ed, :][:, visual_mask.to(self.global_remainder[0].device), :]
                    block_v = self.global_remainder[1][u, :, remainder_st:remainder_ed, :][:, visual_mask.to(self.global_remainder[1].device), :]
                    assert block_k.size(-2) == self.block_size, f"{block_k.size(-2)} vs. {self.block_size}"
                    if self._cretr_sel is not None:
                        block_k = block_k[:, self._cretr_sel, :]
                        block_v = block_v[:, self._cretr_sel, :]
                    n_visual = block_k.size(-2)
                    global_h_k[u, :, ed:ed + n_visual, :] = block_k
                    global_h_v[u, :, ed:ed + n_visual, :] = block_v
                    ed += n_visual

                    # load the boundary end token 
                    global_h_k[u, :, ed:ed + 1, :] = self.full_boundary_k[u, :, boundary_st + 1:boundary_ed, :]
                    global_h_v[u, :, ed:ed + 1, :] = self.full_boundary_v[u, :, boundary_st + 1:boundary_ed, :]
                    ed += 1
                else:
                    remainder_st = init_ed + b_idx * self.block_size
                    remainder_ed = remainder_st + self.block_size
                    if remainder_st >= self.global_remainder[0].size(2):
                        break

                    st = init_ed + cnt * self.block_size
                    ed = st + self.block_size
                    global_h_k[u, :, st:ed] = self.global_remainder[0][u, :, remainder_st:remainder_ed]
                    global_h_v[u, :, st:ed] = self.global_remainder[1][u, :, remainder_st:remainder_ed]
                
                loaded_block_indices[u].append(b_idx)

        return ed, loaded_block_indices 

    def _remainder_visual_mask(self, global_k):
        """Visual-token mask for `global_remainder[:, :, n_init:]`, as a bool vector.

        The two visual masks are NOT interchangeable: `visual_mask_buffer` mirrors `local_k`
        (trimmed to n_local) while `global_remainder_visual_mask` mirrors `global_remainder`;
        they diverge as soon as either is trimmed, so the length is asserted here.
        """
        assert self.global_remainder_visual_mask is not None, (
            "global_remainder_visual_mask missing; adaptive block sizes require it"
        )
        mask = self.global_remainder_visual_mask[:, self.n_init:].squeeze(0)
        assert mask.size(-1) == global_k.size(-2), (
            f"remainder visual mask ({mask.size(-1)}) does not match remainder length "
            f"({global_k.size(-2)}); n_init={self.n_init}, "
            f"remainder={self.global_remainder[0].size(-2)}, "
            f"local_k={self.local_k.size(-2)}, num_global_block={self.num_global_block}"
        )
        return mask.to(global_k.device)

    def _build_retrieved_position_ids(self, loaded_block_indices, init_ed):
        """Build position ids in the same packed order as global_h_k/global_h_v."""
        if self.has_adaptive_block_size:
            return get_qwen3_vl_retrieved_position_ids(
                init_ed,
                loaded_block_indices[0],
                self.adaptive_block_sizes,
                self.block_size,
                self.grid_hw,
                self.global_buffer.device,
                keep_index=self._cretr_sel,
            )

        # Non-adaptive (Qwen2/2.5-VL, LLaVA-OV, ...): blocks are fixed-width and contiguous, so the
        # QA RoPE tables are built from `global_buffer_size` alone by RotaryEmbeddingQwenVL. Return
        # None so the attention forward keeps taking the reference `global_buffer_size` path —
        # returning a synthetic arange here silently suppressed the interleaved-table rebuild.
        return None

    def get_retrieved_kv(self, query=None, ext_kwargs=None, query_v=None):

        """retrieve context blocks with retrieved_block_indices
        query: (batch_size, num_heads, length, dim_head)
        ext_kwargs: dict of external model results 
            {
                'ext_question_features': question_features,
                'ext_video_features': video_features,
                'ext_retrieved_block_indices': ext_model.retrieved_block_indices,
                'ext_similarity': ext_model.similarity,
                'ext_merge_method': ext_merge_method,
                'ext_weights': ext_weights,
            }
        return [init_k, retrieved_k] and the respective v
        """

        if query is not None:  # retrieve based on the attention score between query and context's representative keys
            block_topk = self._calc_block_topk(query, ext_kwargs, query_v=query_v)
            self.set_retrieved_block_indices(block_topk)

        assert len(self.retrieved_block_indices) == self.num_units, f"retrieved_block_indices: {len(self.retrieved_block_indices)} vs. {self.num_units}"

        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        with torch.cuda.stream(GLOBAL_STREAM):
            # Determine if we need to load from offloaded blocks (CPU) or from GPU remainder
            load_from_blocks = self.init_exc or ("chunked" in self.prefill_method and self.num_blocks_seen > self.n_local_frames)
            if load_from_blocks:  # init KV were loaded in global_h_k, context KV were offloaded in global_blocks
                # no need to load init KV
                init_st = 0
                init_ed = init_st + self.init_k.size(-2)
                assert self.global_buffer_init_st == init_st or self.global_buffer_init_ed == init_ed

                # load retrieved context KV
                ed, loaded_block_indices = self._load_retrieved_blocks_from_cpu(global_h_k, global_h_v, init_ed)

            else:  # init KV and context are in self.global_remainder
                # load init KV
                init_st = 0
                init_ed = init_st + self.n_init
                global_h_k[:, :, init_st:init_ed] = self.global_remainder[0][:, :, init_st:init_ed]
                global_h_v[:, :, init_st:init_ed] = self.global_remainder[1][:, :, init_st:init_ed]

                # load retrieved context KV
                ed, loaded_block_indices = self._load_retrieved_blocks_from_remainder(global_h_k, global_h_v, init_ed)

            global_h_k = global_h_k[:, :, :ed, :]
            global_h_v = global_h_v[:, :, :ed, :]

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)



        retrieved_position_ids = self._build_retrieved_position_ids(loaded_block_indices, init_ed)
        if retrieved_position_ids is not None:
            assert retrieved_position_ids.size(-1) + self.n_init == global_h_k.size(-2), f"{retrieved_position_ids.size(-1)} + {self.n_init} vs. {global_h_k.size(-2)}"
        return global_h_k, global_h_v, retrieved_position_ids
    

    def _get_representative_keys(self, global_k): 
        """ obtain representative key features based on retrieval method 
            Args: 
                global_k: torch.Tensor of shape 
                - (batch_size, num_heads, length, dim_head)
        """

        batch_size, num_heads, length, dim_head = global_k.shape 
        assert num_heads == self.unit_size and dim_head == self.dim_head and length % self.block_size == 0 

        global_k = rearrange(global_k, "b h n d -> b n h d") # (batch_size, length - n_init, num_heads, dim_head)
        global_k = rearrange(global_k, "b (m n) h d -> b m n h d", m=length // self.block_size, n=self.block_size)

        if "rekv" in self.retrieval_method: 
            global_k = rearrange(global_k, "b n m h d -> b n m (h d)") 
            # MEMSTREAM_RETRIEVAL_AKS=<r>: build each block's retrieval signature from
            # only the 1/r tokens LEAST similar to the same position in the previous
            # block (the adjacent-novelty criterion AKS applies at encode time),
            # instead of meaning over every token. Encoding and storage are untouched:
            # this changes only what a block looks like to the retriever.
            # One block == one frame and token position is consistent across frames,
            # so position-wise comparison against block m-1 is the same criterion.
            if _RETRIEVAL_AKS > 1:
                prev = torch.roll(global_k, shifts=1, dims=1)
                prev[:, 0] = global_k[:, 0]          # first block has no predecessor
                sim = torch.nn.functional.cosine_similarity(
                    global_k.float(), prev.float(), dim=-1)          # (b, block, token)
                keep = max(1, global_k.size(2) // _RETRIEVAL_AKS)
                idx = sim.argsort(dim=-1)[..., :keep]                # least similar == most novel
                sel = torch.gather(global_k, 2,
                                   idx.unsqueeze(-1).expand(-1, -1, -1, global_k.size(-1)))
                # retrieval runs once per QUESTION, so cap the probe buffer at the
                # first question's full layer sweep rather than letting it grow unbounded.
                if _RETRIEVAL_AKS_PROBE and len(_RETRIEVAL_AKS_PROBE_BUF) < 36:
                    _RETRIEVAL_AKS_PROBE_BUF.append({
                        "layer": self.layer_idx,
                        "dense": global_k.mean(dim=-2).detach().float().cpu(),
                        "aks": sel.mean(dim=-2).detach().float().cpu(),
                        "keep": keep, "n_tokens": global_k.size(2),
                    })
                global_k = sel
            global_k = global_k.mean(dim=-2, keepdim=False)  # (batch_size, block_num, dim)
        elif self.retrieval_method == "qt_to_vb_min": 
            global_k = global_k.amin(dim=2, keepdim=False)  # (batch_size, block_num, num_heads, dim_head)
            global_k = rearrange(global_k, "b n h d -> b h n d")
        elif self.retrieval_method == "qt_to_vb_max": 
            global_k = global_k.amax(dim=2, keepdim=False)  # (batch_size, block_num, num_heads, dim_head)
            global_k = rearrange(global_k, "b n h d -> b h n d")
        else: 
            raise NotImplementedError(f"{self.retrieval_method} not supported!")
        return global_k 

    # Get the indices of the top-k vectors in self.block_k[u] that have the highest similarity with global_h_q[u].
    # ret: batch_size x topk
    def _calc_block_topk(
        self, global_h_q, ext_kwargs=None, query_v=None
    ):  
        """
        ext_kwargs: dict of external model results 
            {
                'ext_question_features': question_features,
                'ext_video_features': video_features,
                'ext_retrieved_block_indices': ext_model.retrieved_block_indices,
                'ext_similarity': ext_model.similarity,
                'ext_merge_method': ext_merge_method,
                'ext_weights': ext_weights,
            }
        """

        # (batch_size, num_heads, length, dim_head) — only the store_full diagnostic reads it
        global_h_q_full = global_h_q.clone() if self.store_full else None

        # Handle uniform sampling retrieval method (if this layer is in the uniform range)
        if self.use_uniform_sampling:
            # Determine the number of blocks available
            if self.num_global_block > 0:
                # Use offloaded blocks (CPU path)
                total_blocks = self.num_global_block
            elif not self.init_exc and self.global_remainder[0].size(-2) > self.n_init:
                # Use global_remainder (GPU path - prefill phase)
                global_k = self.global_remainder[0][:, :, self.n_init:, :]
                if self.has_adaptive_block_size:
                    # must be the remainder's own mask: visual_mask_buffer mirrors local_k, which
                    # is trimmed to n_local independently of how the remainder is consumed
                    visual_pos_masks = self._remainder_visual_mask(global_k)
                    global_k = global_k[:, :, visual_pos_masks]
                total_blocks = global_k.size(-2) // self.block_size
            else:
                total_blocks = 0
            
            if total_blocks > 0:
                # Uniformly sample topk frames from ALL available blocks (0 to total_blocks)
                num_samples = min(self.topk, total_blocks)
                sampled_indices = torch.linspace(0, total_blocks - 1, num_samples, dtype=torch.long)
                ret = [sampled_indices.tolist() for _ in range(self.num_units)]
                
                # Create dummy similarity for logging (all equal since it's uniform)
                self.similarity = torch.ones((self.num_units, total_blocks)) / total_blocks
                
                return ret
            else:
                ret = [[] for _ in range(self.num_units)]
                return ret

        if "rekv" in self.retrieval_method:
            # VALUE-VALUE RETRIEVAL. Both sides of the comparison are VALUES:
            #   score = cos( mean(question VALUES), mean(frame VALUES) )
            # The question's query tensor is discarded entirely here -- this is not
            # query-against-value. `global_h_q` keeps its name only because it is the
            # variable the pooling and scoring below already read; its CONTENTS are the
            # question's values once this branch fires.
            # GQA: values carry unit_size_kv heads vs the query's unit_size, so expand
            # first or the assert below fires on a mismatched signature width.
            # Pool the question's VALUES separately. Fusion needs BOTH embeddings, so
            # this no longer overwrites global_h_q unless we are in pure VRETR mode.
            pooled_v = None
            if (_VRETR or _FUSE_RETR) and query_v is not None:
                _qv = query_v
                if _qv.size(1) != self.unit_size:
                    assert self.unit_size % _qv.size(1) == 0, f"{self.unit_size} vs {_qv.size(1)}"
                    _qv = _qv.repeat_interleave(self.unit_size // _qv.size(1), dim=1)
                pooled_v = _qv.mean(dim=2, keepdim=False).reshape(
                    self.num_units, self.dim_head * self.unit_size)
                if _VRETR and not _FUSE_RETR:
                    global_h_q = _qv      # pure value-value: query tensor unused
            global_h_q = global_h_q.mean(dim=2, keepdim=False)  # (batch_size, num_heads, dim_head)
            assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
            global_h_q = global_h_q.reshape(self.num_units, self.dim_head * self.unit_size)  # (batch_size, dim_head * num_heads)

        _rrf_k = (ext_kwargs or {}).get("ext_rrf_k")
        rrf_k = 60 if _rrf_k is None else _rrf_k

        logits = None

        # Load from GPU 
        if self.num_global_block <= self.topk:        
            if not self.init_exc:  # The local window has not yet been filled, i.e., KV-Cache offloading has not been activated. Retrieval needs to be performed within the local window.
                assert self.global_remainder[0].size(-2) > self.n_init, f'{self.global_remainder[0].shape}'
                # VALUE-SPACE RETRIEVAL also applies on this path. It is a SEPARATE
                # retrieval route used while the local window is still filling, and it
                # reads global_remainder directly rather than the block_k store. Leaving
                # it on keys would score a value-space question embedding against
                # key-space frames -- a silent mixed-space comparison that still returns
                # plausible indices, so it would not surface as an error.
                _rem = self.global_remainder[1 if _VRETR else 0]
                global_k = _rem[:, :, self.n_init:, :]  # (batch_size, n_head_kv, length - n_init, dim_head)

                if self.has_adaptive_block_size:
                    visual_pos_masks = self._remainder_visual_mask(global_k)
                    global_k = global_k[:, :, visual_pos_masks]

                global_k = self._from_group_kv(global_k)  # (batch_size, num_heads, length - n_init, dim_head)
                assert global_k.size(-2) % self.block_size == 0, f'{global_k.shape}'
                block_num = global_k.size(-2) // self.block_size  # number of frames in local window
                
                # all key video tokens — only the store_full diagnostic reads it
                global_k_full = global_k.clone() if self.store_full else None

                # get representative keys 
                global_k = self._get_representative_keys(global_k)

                if "rekv" in self.retrieval_method: 
                    similarity = torch.matmul(global_k.float(), global_h_q[:, :, None].float()).squeeze(dim=-1)  # (batch_size, block_num)     
                    key_norms = global_k.float().norm(dim=-1)
                    query_norm = global_h_q.float().norm(dim=-1)
                    self.similarity = similarity / (key_norms * query_norm)

                elif self.retrieval_method == "qt_to_vb_max": 
                    similarity = torch.einsum("bhnd,bhqd->bhqn", global_k.float(), global_h_q.float()) # (batch_size, num_heads, num_qtokens, block_num) 
                    # cosine normalization to match CPU path behavior
                    key_norms = global_k.float().norm(dim=-1, keepdim=False)  # (batch_size, num_heads, block_num)
                    query_norms = global_h_q.float().norm(dim=-1, keepdim=False)  # (batch_size, num_heads, num_qtokens)
                    denom = key_norms[:, :, None, :] * query_norms[:, :, :, None]
                    similarity = similarity / (denom + 1e-6)
                    similarity = similarity.mean(dim=(1,2), keepdim=False) # (batch_size, num_heads, block_num)
                    self.similarity = similarity  

                # full token-wise similarity between global_h_q_full and global_k_full
                if self.store_full:
                    diagnostics.record_full_similarity_prefill(self, global_k_full, global_h_q_full)

                # return all blocks if fewer than topk
                if block_num <= self.topk:
                    ret = [list(range(block_num)) for _ in range(self.num_units)]
                else: 
                    logits = similarity # (batch_size, block_num)

            else:  # The local window is already filled, but the number of input frames is less than 'topk'.
                ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
                    
        # Load from CPU 
        else:   
            if "rekv" in self.retrieval_method: 
                if ext_kwargs is not None and ext_kwargs['ext_merge_method'] == 'l2_concat':
                    logits = torch.stack([self.block_k[u].get_cosine_similarity_with_external(global_h_q[u], **ext_kwargs) for u in range(self.num_units)])  # (batch_size, block_num)
                else: 
                    if _FUSE_RETR and self.block_v is not None and pooled_v is not None:
                        # Rank in each space independently, then RRF. Scores are NOT
                        # added directly: cosine in key space and in value space have
                        # different scales and offsets, so a raw sum would silently let
                        # whichever space has the wider spread dominate the ranking.
                        lk = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
                        lv = torch.stack([self.block_v[u].get_cosine_similarity(pooled_v[u]) for u in range(self.num_units)])
                        logits = torch.zeros_like(lk)
                        for _l in (lk, lv):
                            _rank = _l.argsort(dim=-1, descending=True).argsort(dim=-1).float()
                            logits = logits + 1.0 / (_FUSE_RRF_K + _rank + 1.0)
                    else:
                        _store = self.block_v if (_VRETR and self.block_v is not None) else self.block_k
                        _emb = pooled_v if (_VRETR and self.block_v is not None) else global_h_q
                        logits = torch.stack([_store[u].get_cosine_similarity(_emb[u]) for u in range(self.num_units)])  # (batch_size, block_num)
            elif "qt" in self.retrieval_method:
                with_softmax = "softmax" in self.retrieval_method
                logits = torch.stack([self.block_k[u].get_cosine_similarity_tokenwise(global_h_q[u], with_softmax=with_softmax, head_idx=self.head_idx) for u in range(self.num_units)])  # (batch_size, block_num)

            if self.store_full:
                diagnostics.record_full_similarity_offloaded(self, global_h_q_full)
            
        if logits is not None:
            assert self.topk % self.chunk_size == 0
            
            # Chunk logits by averaging within chunk_size groups
            remainder_size = logits.shape[-1] % self.chunk_size
            pattern = "b h (m n) -> b h m n" if logits.dim() == 3 else "b (m n) -> b m n"
            cat_dim = 2 if logits.dim() == 3 else 1
            
            # Average complete chunks
            chunked_logits = rearrange(logits[..., :logits.shape[-1] - remainder_size], pattern, n=self.chunk_size).mean(dim=-1)
            
            # Handle remainder if exists
            if remainder_size > 0:
                remainder_logits = logits[..., -remainder_size:].mean(dim=-1, keepdim=True)
                chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=cat_dim)
            
            # Filter by even/odd if dual_stream is enabled 
            if "dual_stream" in self.prefill_method:
                # Determine strategy from retrieval_method suffix
                if self.retrieval_method.endswith("even_rekv"):
                    stream_to_use = "even"
                elif self.retrieval_method.endswith("odd_rekv"):
                    stream_to_use = "odd"
                elif self.retrieval_method.endswith("interleave_rekv"):
                    stream_to_use = "interleave"
                elif "rrf_rekv" in self.retrieval_method:
                    # Parse: sims_np=5_rrf_rekv or odd_only_np=5_rrf_rekv
                    match = re.search(r'(sims|odd_only)_np=(\d+)_rrf', self.retrieval_method)
                    if match:
                        stream_to_use = f"{match.group(1)}_rrf"
                        n_pairs_from_name = int(match.group(2))
                    else:
                        # Fallback for legacy names
                        if "sims_rrf" in self.retrieval_method:
                            stream_to_use = "sims_rrf"
                        else:
                            stream_to_use = "odd_only_rrf"
                        n_pairs_from_name = 5  # default
                else:
                    raise ValueError(f"Invalid retrieval method: {self.retrieval_method}")
                
                if stream_to_use == "interleave":
                    # Split into even/odd streams
                    even_indices = torch.arange(0, chunked_logits.shape[-1], 2)
                    odd_indices = torch.arange(1, chunked_logits.shape[-1], 2)
                    
                    # Min-max normalize each stream separately
                    chunked_logits[:, even_indices] = (chunked_logits[:, even_indices] - chunked_logits[:, even_indices].min()) / (chunked_logits[:, even_indices].max() - chunked_logits[:, even_indices].min() + 1e-8)
                    chunked_logits[:, odd_indices] = (chunked_logits[:, odd_indices] - chunked_logits[:, odd_indices].min()) / (chunked_logits[:, odd_indices].max() - chunked_logits[:, odd_indices].min() + 1e-8)
                    
                    logger.info(f"Merged streams with normalization: even={chunked_logits[:, even_indices].shape}, odd={chunked_logits[:, odd_indices].shape}, combined={chunked_logits.shape}")
                
                elif stream_to_use in ["sims_rrf", "odd_only_rrf"]:


                    # Min-max normalize even and odd streams separately
                    even_indices = torch.arange(0, chunked_logits.shape[-1], 2) # (batch_size, num_qtokens, chunk_size/2)
                    odd_indices = torch.arange(1, chunked_logits.shape[-1], 2) # (batch_size, num_qtokens, chunk_size/2)

                    # Min-max normalize even and odd streams separately
                    chunked_logits[:, even_indices] = (chunked_logits[:, even_indices] - chunked_logits[:, even_indices].min()) / (chunked_logits[:, even_indices].max() - chunked_logits[:, even_indices].min() + 1e-8)
                    chunked_logits[:, odd_indices] = (chunked_logits[:, odd_indices] - chunked_logits[:, odd_indices].min()) / (chunked_logits[:, odd_indices].max() - chunked_logits[:, odd_indices].min() + 1e-8)
                    
                    # get rankings for even and odd streams 
                    even_rankings = chunked_logits[:, even_indices].argsort(dim=1, descending=True)[0].tolist()
                    odd_rankings = chunked_logits[:, odd_indices].argsort(dim=1, descending=True)[0].tolist()

                    # update rankings to contain global indices 
                    even_rankings = [x * 2 for x in even_rankings] 
                    odd_rankings = [x * 2 + 1 for x in odd_rankings]
                    
                    # Get similarity scores for even and odd streams (sorted)
                    even_sims = chunked_logits[:, even_indices].sort(dim=1, descending=True)[0][0].tolist()
                    odd_sims = chunked_logits[:, odd_indices].sort(dim=1, descending=True)[0][0].tolist()

                    # Apply RRF between streams with pair scoring and tie-breaking
                    break_ties = "sims" if stream_to_use == "sims_rrf" else "odd_only" 
                    fused_indices, _ = self.reciprocal_rank_stream_fusion(
                        [even_rankings, odd_rankings], 
                        [even_sims, odd_sims], 
                        weights=[0.5, 0.5], 
                        k=rrf_k, 
                        break_ties=break_ties,
                        n_pairs_to_keep=n_pairs_from_name
                    )
                    
                    # fused_indices already contains global frame indices
                    global_indices = fused_indices[0].tolist()

                    # Reassign scores based on RRF ranking
                    chunked_logits_rrf = torch.zeros_like(chunked_logits)
                    for rank, global_idx in enumerate(global_indices):
                        if global_idx < chunked_logits.shape[-1]:
                            chunked_logits_rrf[0, global_idx] = len(global_indices) - rank
                    chunked_logits = chunked_logits_rrf
                    
                    logger.info(f"RRF stream fusion: selected {len(global_indices)} frames (top 5 pairs = both, rest = tie-break)")


                else:
                    # Single stream filtering (even or odd)
                    stream_indices = torch.arange(0, chunked_logits.shape[-1], 2) if stream_to_use == "even" else torch.arange(1, chunked_logits.shape[-1], 2)
                    chunked_logits = chunked_logits[:, stream_indices]
                    logger.info(f"Filtered to {stream_to_use} stream: {chunked_logits.shape}")

            self.similarity = chunked_logits
            if chunked_logits.dim() == 3: 
               chunked_logits = chunked_logits.mean(dim=1) # (batch_size, 1)

            if self.retrieval_method == "topk_watershed_rekv":
                raise RuntimeError("not vendored: topk_watershed_rekv (model/attention/utils.py)")
                topk_result = select_topk_watershed(chunked_logits.cpu().squeeze(dim=0).numpy(), self.topk)
                self.sorted_block_indices = torch.tensor(topk_result['frame_indices'], dtype=torch.long)[None, :]
            else:
                # Sort similarity scores and apply RRF fusion if needed
                sorted_block_indices = chunked_logits.argsort(dim=1, descending=True) 
                self.sorted_block_indices = sorted_block_indices.cpu().tolist()

            current_rankings = self.sorted_block_indices[0]  # batch_size=1
            
            # Apply layerwise RRF (fuse with previous layer rankings)
            if ext_kwargs and ext_kwargs.get('ext_prev_layer_full_rankings'):
                prev_rankings = ext_kwargs['ext_prev_layer_full_rankings']
                weights = ext_kwargs['ext_weights']
                fused_indices, _ = self.reciprocal_rank_fusion([current_rankings] + prev_rankings, k=rrf_k, weights=weights)
                current_rankings = fused_indices[0].tolist()
            
            # Apply external RRF (fuse with external retrieval)
            if ext_kwargs and ext_kwargs.get('ext_merge_method') == 'rrf':
                ext_rankings = ext_kwargs['ext_retrieved_block_indices'][0]
                # off-by-1 => temporal-group rounding differs from the QA model's truncation
                assert len(current_rankings) == len(ext_rankings), (
                    f"block count mismatch: internal {len(current_rankings)} vs "
                    f"external {len(ext_rankings)} (see pecore_rekv.calc_topk)"
                )
                fused_indices, _ = self.reciprocal_rank_fusion([current_rankings, ext_rankings], k=rrf_k, weights=ext_kwargs['ext_weights'])
                current_rankings = fused_indices[0].tolist()
            
            # Step 3: Take top-k from final rankings
            ret = torch.tensor(current_rankings[:self.topk//self.chunk_size], dtype=torch.long).unsqueeze(0) 

            ret = ret.sort(dim=1)[0][:, :, None]  # (batch_size, topk//chunk_size, 1)
            ret = ret * self.chunk_size + torch.arange(self.chunk_size, device=ret.device)[None, None, :]  # (batch_size, topk//chunk_size, chunk_size)
            ret = ret.reshape(self.num_units, -1)  # (batch_size, topk)
            
            ret = ret.cpu().tolist()
            
            # Map filtered even/odd indices back to global indices
            if "dual_stream" in self.prefill_method and stream_to_use in ["even", "odd"]:
                for u in range(self.num_units):
                    ret[u] = [idx * 2 if stream_to_use == "even" else idx * 2 + 1 for idx in ret[u]]  # 0→0, 1→2, 2→4, ... 0→1, 1→3, 2→5, ...
            
            # NOTE: The last chunk might cause an index overflow
            for u in range(self.num_units):
                ret[u] = list(filter(lambda idx: idx < logits.shape[-1], ret[u]))

        return ret

    # load init KV
    def get_global_hidden_and_mask(self, exc_length):
        
        # self.global_buffer - keeps track of cache (prior to updates)
        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        global_remainder_ed = self._global_remainder_ed + exc_length
        global_remainder_st = self._global_remainder_st
        global_remainder_len = global_remainder_ed - global_remainder_st

        # prepare init KV-Cache until it's full
        if not self.init_exc and global_remainder_len > self.n_local:

            global_k = self.global_remainder[0]
            global_v = self.global_remainder[1]

            append_init_len = min(
                self.n_init - self.init_k.size(-2), # How many init tokens still needed
                global_remainder_len - self.n_local # How many "excess" tokens available
            )

            # Move first set of tokens from local cache to init cache
            self.init_k = torch.cat(
                (self.init_k, global_k[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            self.init_v = torch.cat(
                (self.init_v, global_v[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            global_remainder_st += append_init_len
            global_remainder_len -= append_init_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True  # init KV-Cache is full

        # update remainder start and end for transfering 
        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

        # load init KV
        init_st = 0
        init_ed = init_st + self.init_k.size(-2)
        if self.global_buffer_init_st != init_st or self.global_buffer_init_ed != init_ed:  # init KV haven't been loaded into global_h_kv
            global_h_k[:, :, init_st: init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st: init_ed, :].copy_(self.init_v, non_blocking=True)

        self.global_buffer_init_st = init_st
        self.global_buffer_init_ed = init_ed

        global_h_k = global_h_k[:, :, :init_ed, :]
        global_h_v = global_h_v[:, :, :init_ed, :]


        return global_h_k, global_h_v

    def _parse_adaptive_config(self, prefill_method):
        """
        Parse adaptive configuration from prefill_method string.
        
        Format: swa_kk_adjacent_adaptive_4[_m10][_10_25_65][_temporal|_taligned][_mean]
        
        Args:
            prefill_method: Method string containing adaptive configuration
            
        Returns:
            tuple: (is_adaptive, adaptive_config, cleaned_prefill_method)
                - is_adaptive: bool indicating if adaptive mode is enabled
                - adaptive_config: dict with 'min_m', 'tier_percentages', and 'allocation_mode' keys
                - cleaned_prefill_method: method string with adaptive config removed
        """
        adaptive = "_adaptive" in prefill_method
        adaptive_config = {}
        
        if adaptive:
            # Check for allocation mode first (temporal_aligned or content_based)
            allocation_mode = 'content_based'  # default
            if '_temporal' in prefill_method or '_taligned' in prefill_method:
                allocation_mode = 'temporal_aligned'
                # Remove the mode suffix for parsing
                prefill_method = prefill_method.replace('_temporal', '').replace('_taligned', '')
            
            # Remove _mean suffix if present (just a label, doesn't affect logic)
            prefill_method = prefill_method.replace('_mean', '')
            
            # Pattern: _adaptive_4_m10_10_25_65 or _adaptive_4_50_0_50 or _adaptive_4
            adaptive_match = re.search(r'_adaptive(_\d+(?:_m\d+)?(?:_\d+_\d+_\d+)?)', prefill_method)
            if adaptive_match:
                adaptive_suffix = adaptive_match.group(1)  # e.g., "_4_m16" or "_4_50_0_50" or "_4"
                parts = adaptive_suffix.strip('_').split('_')
                dilation_rate = parts[0]  # Save for later
                
                # Check for min_m (e.g., parts=["4", "m16"] or ["4", "m16", "10", "25", "65"])
                if len(parts) >= 2 and parts[1].startswith('m'):
                    adaptive_config['min_m'] = int(parts[1][1:])
                    percentage_start = 2
                else:
                    adaptive_config['min_m'] = 5  # default
                    percentage_start = 1
                
                # Check for tier percentages (e.g., parts=["4", "10", "25", "65"] or ["4", "m16", "10", "25", "65"])
                if len(parts) >= percentage_start + 3:
                    adaptive_config['tier_percentages'] = [
                        int(parts[percentage_start])/100,
                        int(parts[percentage_start+1])/100,
                        int(parts[percentage_start+2])/100
                    ]
                else:
                    adaptive_config['tier_percentages'] = [0.10, 0.25, 0.65]  # defaults
                
                # Set the allocation mode
                adaptive_config['allocation_mode'] = allocation_mode
            
            # Remove _adaptive and all config, replace with just _<dilation_rate>
            # E.g., swa_kk_adjacent_adaptive_4_m16 -> swa_kk_adjacent_4
            prefill_method = re.sub(r'_adaptive_\d+(?:_m\d+)?(?:_\d+_\d+_\d+)?', f'_{dilation_rate}', prefill_method)
        
        return adaptive, adaptive_config, prefill_method

    def _frame_geometry(self, local_k, visual_pos_masks):
        """Locate per-frame spans in a mixed text/visual sliding window (Qwen3-VL).

        Each frame is emitted as `<timestamp text><|vision_start|>[patches]<|vision_end|>`, so
        frame i occupies boundary_indices[2i]..boundary_indices[2i+1] inclusive, preceded by its
        variable-length timestamp text.

        Returns:
            (prefix_len, visual_starts, visual_ends, boundary_indices)
        """
        prefix_len = self.n_init if (
            not self.init_exc
            and local_k.size(-2) >= self.n_init
            and not visual_pos_masks[0, :self.n_init].any()
        ) else 0

        frame_visual_mask = visual_pos_masks[:, prefix_len:]
        prev_visual_mask = F.pad(frame_visual_mask[:, :-1], (1, 0), value=False)
        next_visual_mask = F.pad(frame_visual_mask[:, 1:], (0, 1), value=False)
        boundary_mask = (~frame_visual_mask) & (prev_visual_mask | next_visual_mask)

        visual_starts = torch.nonzero(
            frame_visual_mask[0] & ~prev_visual_mask[0], as_tuple=False).squeeze(-1) + prefix_len
        visual_ends = torch.nonzero(
            frame_visual_mask[0] & ~next_visual_mask[0], as_tuple=False).squeeze(-1) + prefix_len + 1
        boundary_indices = torch.nonzero(
            boundary_mask[0], as_tuple=False).squeeze(-1) + prefix_len

        if boundary_indices.numel() != 2 * visual_starts.numel():
            raise AssertionError(
                "mixed text/visual window is not frame-aligned for frame sampling: "
                f"n_visual_starts={visual_starts.numel()}, n_visual_ends={visual_ends.numel()}, "
                f"n_boundary={boundary_indices.numel()} (expected {2 * visual_starts.numel()}), "
                f"local_k_len={local_k.size(-2)}, init_exc={self.init_exc}, prefix_len={prefix_len}"
            )
        return prefix_len, visual_starts, visual_ends, boundary_indices

    def _apply_compression(
        self, local_k, local_v, local_q, len_k,
        start_compression, should_compress, 
        custom_prefill_method=None,
        visual_pos_masks=None,
    ):
        """
        Apply compression based on prefill_method.

        local_k: (batch, n_head_kv, total_len, dim_head)
        local_v: (batch, n_head_kv, total_len, dim_head)
        local_q: (batch, n_head, len_q, dim_head)
        start_compression: bool - whether to start compression (len_k - len_q > 0 - so after first frame is prefilled)
        should_compress: bool - whether to compress 
        visual_pos_masks: (batch_size, length) - mask of length input_ids, where visual tokens are located (True)
        
        Returns:
            tuple: (compressed_k, compressed_v, subsample_mask, sliding_window)
        """
        len_q = local_q.size(-2)

        # If we're not starting compression yet, use standard path
        if not start_compression:
            return no_compression(local_k, local_v, self.n_local)

        # MEMSTREAM_COMPRESS_LAYERS=<start>:<end> -- apply the prefill compression
        # ONLY to layers in [start, end); every other layer runs dense. Lets us ask
        # WHERE in the stack compression helps or hurts, which the aggregate
        # accuracy of a whole-stack arm cannot separate.
        if _COMPRESS_LAYERS is not None:
            _lo, _hi = _COMPRESS_LAYERS
            if not (_lo <= self.layer_idx < _hi):
                return no_compression(local_k, local_v, self.n_local)
        if _COMPRESS_LAYER_SET is not None and self.layer_idx not in _COMPRESS_LAYER_SET:
            return no_compression(local_k, local_v, self.n_local)

        # remove _nd=X from prefill_method
        prefill_method = custom_prefill_method if custom_prefill_method else self.prefill_method
        prefill_method = prefill_method.replace(f"_nd={self.nd}", "")


        # "swa_dual_stream_{method1}_AND_{method2}": an even-iter query attends only the even
        # previous frames with method1, an odd-iter query only the odd frames with method2
        # (strict stream separation; the current frame is always dense).
        if prefill_method.startswith("swa_dual_stream_"):

            if not should_compress:
                return no_compression(local_k, local_v, self.n_local)

            method_1, method_2 = prefill_method.split("swa_dual_stream_")[1].split("_AND_")

            # the first iteration is even 
            query_is_even = self.num_blocks_seen % 2 == 0
            method = method_1 if query_is_even else method_2
            method = "swa_" + method

            # In dual stream section, ToMe is not compatible with other methods (TODO: add support later)
            if "tome" in method_1 or "tome" in method_2:
                raise ValueError("ToMe compression is not compatible with dual_stream.")
            
            h, w = self.grid_hw
            n_init = self.n_init if self.init_exc else 0
            b, nh, total_len, d = local_k.shape

            # Separate init and frame tokens
            st = 0 if self.init_exc else self.n_init
            init_k, frame_k = local_k[:, :, :st, :], local_k[:, :, st:, :]
            init_v, frame_v = local_v[:, :, :st, :], local_v[:, :, st:, :]


            # Reshape to frames: (b, nh, m, h*w, d)
            m = frame_k.size(2) // (h * w) 
            m_prev = m - 1 # (ignore the query frame)
            frame_k = frame_k.view(b, nh, m, h*w, d)
            frame_v = frame_v.view(b, nh, m, h*w, d)

            prev_frame_k = frame_k[:, :, :-1, :, :]
            prev_frame_v = frame_v[:, :, :-1, :, :]

            # add current frame (dense) - (b, nh, h*w, d)
            curr_frame_k = frame_k[:, :, -1, :, :]
            curr_frame_v = frame_v[:, :, -1, :, :]
            
            # for first iteration, apply dense attention on the current frame only 
            if self.num_blocks_seen == 1:
                # enforce strict separation between both streams 
                local_k = torch.cat([init_k, curr_frame_k], dim=2) 
                local_v = torch.cat([init_v, curr_frame_v], dim=2)
                return no_compression(local_k, local_v, self.n_local)
            
            # Compute global frame indices: [num_blocks_seen - m, ..., num_blocks_seen - 1]
            global_start = self.num_blocks_seen - m_prev
            global_indices = torch.arange(global_start, self.num_blocks_seen, device=local_k.device)
            
            # Determine query's stream and select matching frames
            stream_mask = (global_indices % 2 == 0) if query_is_even else (global_indices % 2 == 1)

            # Extract and flatten stream frames
            frame_k_stream = prev_frame_k[:, :, stream_mask, :, :].reshape(b, nh, -1, d)
            frame_v_stream = prev_frame_v[:, :, stream_mask, :, :].reshape(b, nh, -1, d)

            # concatenate previous and current frames 
            frame_k_stream = torch.cat([frame_k_stream, curr_frame_k], dim=2)
            frame_v_stream = torch.cat([frame_v_stream, curr_frame_v], dim=2)
            
            # Reassemble with init tokens
            local_k_stream = torch.cat([init_k, frame_k_stream], dim=2)
            local_v_stream = torch.cat([init_v, frame_v_stream], dim=2)

            # get the length of the stream passed into compression function 
            len_k_stream = local_k_stream.size(-2)

            # create a global subsample mask - (b, nh, m_prev, h*w) - sparse_frames only 
            full_subsample_mask = torch.zeros(b, nh, m_prev, h, w, dtype=torch.bool, device=local_k.device)

            out = self._apply_compression(
                local_k_stream, local_v_stream, local_q,
                len_k_stream, start_compression, should_compress,
                custom_prefill_method=method
            )
            compressed_k, compressed_v, subsample_mask, sliding_window = out 
             
            # check the shape of the subsample_mask
            if subsample_mask.dim() == 3:
                # (m_prev, h, w) - set subsample_mask on frames in stream_mask 
                subsample_mask = subsample_mask.unsqueeze(0).unsqueeze(0) # (1, 1, m_prev, h, w)
                full_subsample_mask[:, :, stream_mask] = subsample_mask.expand(b, nh, -1, -1, -1) 
            elif subsample_mask.dim() == 5:
                # (b, nh, m, h, w) - set subsample_mask on stream_mask_frames 
                # check that num frames in stream_mask equals num frames in subsample_mask 
                assert stream_mask.long().sum() == subsample_mask.size(2) 
                full_subsample_mask[:, :, stream_mask] = subsample_mask 
            else:
                raise ValueError(f"Unsupported subsample_mask shape: {subsample_mask.shape}")


            return compressed_k, compressed_v, full_subsample_mask, sliding_window

        # prefill with average/max pooling on all previous frames and dense attention on the current frame 
        # example: swa_pooled_4 (2x2 pooling)
        if prefill_method.startswith(("swa_pooled_", "swa_max_pooled_")):
            if not should_compress:
                return no_compression(local_k, local_v, self.n_local)
            
            use_max_pool = prefill_method.startswith("swa_max_pooled_")
            method_suffix = prefill_method.split("_", 3)[-1]  # Get everything after swa_[max_]pooled_
            pool_ratio_str = method_suffix.split("_after")[0]
            pool_ratio = int(pool_ratio_str)
            
            return apply_pooled_compression(
                local_k, local_v, local_q,
                self.n_init, self.init_exc, self.grid_hw,
                pool_ratio, use_max_pool, nd=self.nd
            )
        
        # prefill with dilated attention on all previous frames and dense attention on the current frame
        # example: swa_dilated_4 (dilation-rate = 4)
        # Mixed text/visual windows (Qwen3-VL) are handled by the shared kk/dilated path below,
        # which strips the per-frame timestamp tokens before compressing and restores them after.
        elif prefill_method.startswith("swa_dilated_") and visual_pos_masks is None:
            if not should_compress:
                return no_compression(local_k, local_v, self.n_local)
            
            method_suffix = prefill_method.split("_", 2)[-1]  # Get everything after swa_dilated_
            dilation_rate_str = method_suffix.split("_after")[0]
            dilation_rate = int(dilation_rate_str)
            
            return apply_patch_sample_compression(
                local_k, local_v, local_q,
                self.n_init, self.init_exc, self.grid_hw,
                method='dilated',
                dilation_rate=dilation_rate,
                nd=self.nd
            )

        elif prefill_method.startswith("swa_random_patch_"): 
            if not should_compress:
                return no_compression(local_k, local_v, self.n_local)
            
            method_suffix = prefill_method.split("swa_random_patch_")[-1]  # Get everything after swa_random_patch_
            scale_factor_str = method_suffix.split("_after")[0]
            scale_factor = int(scale_factor_str)
            
            return apply_patch_sample_compression(
                local_k, local_v, local_q,
                self.n_init, self.init_exc, self.grid_hw,
                method='random',
                scale_factor=scale_factor,
                nd=self.nd
            )

        elif prefill_method.startswith("swa_tome_patch_"):
            if not should_compress:
                return no_compression(local_k, local_v, self.n_local)
            
            method_suffix = prefill_method.split("swa_tome_patch_")[-1]  # Get everything after swa_tome_patch_
            
            # Check if this combines ToMe with frame sampling
            # Format: swa_tome_patch_0.5_with_uniform_sample_8
            # or: swa_tome_patch_0.5_with_uniform_sample_8_niter_5
            if "_with_uniform_sample_" in method_suffix:
                merge_ratio_str = method_suffix.split("_with_uniform_sample_")[0]
                merge_ratio = float(merge_ratio_str)
                
                # Extract n_frames_to_sample and optionally niter
                remaining_suffix = method_suffix.split("_with_uniform_sample_")[1]
                
                # Check if niter is specified
                num_iters = 2  # default value
                if "_niter_" in remaining_suffix:
                    n_frames_to_sample_str = remaining_suffix.split("_niter_")[0]
                    niter_str = remaining_suffix.split("_niter_")[1].split("_after")[0]
                    num_iters = int(niter_str)
                else:
                    n_frames_to_sample_str = remaining_suffix.split("_after")[0]
                
                # Values <= 1 are a FRACTION of the window's frames (1.0 = every frame),
                # resolved per encode call in apply_tome_with_frame_sampling so it is
                # window-size-invariant across tpf/fps. Values > 1 are an absolute count.
                n_frames_to_sample = float(n_frames_to_sample_str)
                if n_frames_to_sample > 1:
                    n_frames_to_sample = int(n_frames_to_sample)

                if visual_pos_masks is not None:
                    return apply_mixed_tome_compression(
                        local_k, local_v, local_q, visual_pos_masks,
                        self._frame_geometry(local_k, visual_pos_masks),
                        self.grid_hw, self.nd,
                        merge_ratio=merge_ratio,
                        n_frames_to_sample=n_frames_to_sample,
                        num_iters=num_iters,
                    )
                
                # Apply hybrid compression: frame sampling + ToMe
                return apply_tome_with_frame_sampling(
                    local_k, local_v, local_q,
                    self.n_init, self.init_exc, self.grid_hw,
                    n_frames_to_sample=n_frames_to_sample,
                    merge_ratio=merge_ratio,
                    nd=self.nd,
                    num_iters=num_iters
                )
            else:
                # Standard ToMe without frame sampling
                merge_ratio_str = method_suffix.split("_after")[0]
                merge_ratio = float(merge_ratio_str)

                if visual_pos_masks is not None:
                    return apply_mixed_tome_compression(
                        local_k, local_v, local_q, visual_pos_masks,
                        self._frame_geometry(local_k, visual_pos_masks),
                        self.grid_hw, self.nd,
                        merge_ratio=merge_ratio,
                    )
                
                return apply_patch_sample_compression(
                    local_k, local_v, local_q,
                    self.n_init, self.init_exc, self.grid_hw,
                    method='tome',
                    merge_ratio=merge_ratio,
                    nd=self.nd
                )
        
        # prefill with clustered attention on all previous frames and dense attention on the current frame 
        # example: swa_clustered_30_10 # 30 clusters, 10 iterations 
        elif prefill_method.startswith("swa_clustered_"): 
            if not should_compress:
                return no_compression(local_k, local_v, self.n_local)

            method_suffix = prefill_method.split("_", 2)[-1]  # Get everything after swa_clustered_
            cluster_nc_niter_str = method_suffix.split("_after")[0]
            n_clusters, n_iters = tuple([int(x) for x in cluster_nc_niter_str.split('_')])

            return apply_clustered_compression(
                local_k, local_v, local_q,
                self.n_init, self.init_exc, self.grid_hw,
                n_clusters=n_clusters, n_iters=n_iters,
                nd=self.nd
            )
        
        # prefill with uniform sampling across previous frames and dense attention on the current frame 
        elif prefill_method.startswith("swa_uniform_sample_") \
            or prefill_method.startswith("swa_random_sample_") \
            or prefill_method.startswith("swa_recent_sample_") \
            or prefill_method.startswith("swa_event_sample_"):

            if not should_compress:
                return no_compression(local_k, local_v, self.n_local)

            method_suffix = prefill_method.split("_", 3)[-1]  # Get everything after swa_uniform_sample
            n_frames_to_sample_str = method_suffix.split("_after")[0]
            n_frames_to_sample = int(n_frames_to_sample_str)
            
            method = prefill_method.split('_', 3)[1] # "uniform", "random", "event"

            if visual_pos_masks is not None:
                # Mixed text/visual window (Qwen3-VL): frame sampling must drop WHOLE frames
                # (timestamp text + boundary tokens + patches together) so the window's
                # text:visual ratio is preserved. That is the point of this arm — patch-level
                # methods delete only visual tokens and inflate the timestamp share ~10x.
                if method == "event":
                    raise NotImplementedError(
                        "swa_event_sample_<n> is not supported on mixed text/visual windows: its "
                        "frame scoring reshapes the sparse region as (m h w), which assumes a "
                        "purely visual window. Use swa_uniform_sample_<n> or swa_random_sample_<n>."
                    )

                prefix_len, visual_starts, visual_ends, boundary_indices = self._frame_geometry(
                    local_k, visual_pos_masks)

                n_frames = visual_starts.numel()
                dense_frames = min(self.nd, n_frames)
                sparse_frames = max(n_frames - dense_frames, 0)
                nh = local_k.size(1)

                # The newest nd frames always stay dense, matching every other prefill method;
                # only the older ("sparse") frames are subject to sampling.
                n_keep = max(0, min(sparse_frames, n_frames_to_sample))
                if sparse_frames > 0 and n_keep > 0:
                    if method == "uniform":
                        sel = torch.linspace(
                            0, sparse_frames - 1, n_keep, device=local_k.device).round().long()
                    elif method == "random":
                        sel = torch.randperm(sparse_frames, device=local_k.device)[:n_keep]
                    elif method == "recent":
                        # The newest n_keep sparse frames: the explicit "is this just a
                        # shorter sliding window?" control for any query-aware frame
                        # selector. Selecting by raw encode attention lands on this set
                        # ~77-79% of the time, so without it a qframe result cannot be
                        # distinguished from window truncation.
                        sel = torch.arange(
                            sparse_frames - n_keep, sparse_frames, device=local_k.device)
                    else:
                        # NEVER fall through. `method` is token 1 of the arm name, so an
                        # unrecognised swa_<x>_sample_<n> used to execute silently as
                        # "random" -- the same silent-fallback class of bug that made
                        # swa_vv_adj_2 run as swa_kk_adjacent_2.
                        raise ValueError(
                            f"unhandled frame-sample method {method!r} from prefill_method "
                            f"{prefill_method!r}; expected uniform/random/recent")
                    sel = torch.unique(sel)
                else:
                    sel = torch.empty(0, dtype=torch.long, device=local_k.device)

                keep_sparse = torch.zeros(
                    sparse_frames, dtype=torch.bool, device=local_k.device)
                if sel.numel() > 0:
                    keep_sparse[sel] = True

                sparse_indices = frame_keep_to_sparse_indices(
                    keep_sparse, visual_starts, visual_ends, nh, local_k.device)
                return rebuild_mixed_window(
                    local_k, local_v,
                    (prefix_len, visual_starts, visual_ends, boundary_indices),
                    sparse_indices,
                )

            return apply_frame_sample_compression(
                local_k, local_v, local_q, 
                self.n_init, self.init_exc, self.grid_hw,
                n_frames_to_sample=n_frames_to_sample,
                method=method,
                nd=self.nd
            )

        # QUERY-FRAME top-k FRAME selection: swa_qframe_<n>_<lam>[_z]
        # Keep the n sparse frames the INCOMING frame attends to most, dropping the rest
        # whole. Raw affinity is recency-dominated (top-18-of-72 reproduces the 18 newest
        # frames ~77-79% of the time vs 25% chance), which would make the arm a disguised
        # shorter window; subtracting lam * (running recency profile) trades that off.
        # lam=0 is raw (~= swa_recent_sample_<n>), lam=1 is exact de-meaning (chance recency).
        elif prefill_method.startswith("swa_qframe_"):

            if not should_compress:
                return no_compression(local_k, local_v, self.n_local)

            n_cfg, lam, use_z = _parse_qframe_config(prefill_method)

            if visual_pos_masks is None:
                # G1: ONE call site only. Refusing here (rather than falling through to a
                # purely-visual implementation) structurally removes the two-call-site
                # divergence that made swa_vv_adj_2 silently run as swa_kk_adjacent_2:
                # there is no second path to forget to update.
                raise NotImplementedError(
                    "swa_qframe_<n>_<lam> requires mixed text/visual windows: it needs the "
                    "per-frame geometry from visual_pos_masks and per-layer recency state. "
                    "Use swa_uniform_sample_<n> / swa_recent_sample_<n> on purely-visual windows."
                )

            prefix_len, visual_starts, visual_ends, boundary_indices = self._frame_geometry(
                local_k, visual_pos_masks)

            n_frames = visual_starts.numel()
            dense_frames = min(self.nd, n_frames)
            sparse_frames = max(n_frames - dense_frames, 0)
            nh = local_k.size(1)
            n_keep = max(0, min(sparse_frames, n_cfg))

            keep_sparse = torch.zeros(
                sparse_frames, dtype=torch.bool, device=local_k.device)

            # Lazily allocated so a ContextManager that never ran init() (dispatch tests)
            # still works, and so the attribute cannot be missing on any future path.
            prof_obj = getattr(self, "_qframe_profile", None)
            if prof_obj is None:
                prof_obj = self._qframe_profile = RecencyProfile()

            if sparse_frames > 0 and n_keep > 0:
                per_head, qk = frame_query_affinity(
                    local_k, local_q, visual_starts, visual_ends, sparse_frames)
                # G8: one NaN would poison the running mean for the rest of the video.
                if not torch.isfinite(qk).all():
                    raise AssertionError("swa_qframe: non-finite query-frame affinity")
                if use_z:
                    # Guard against a single high-norm head dominating the plain mean.
                    _mu = per_head.mean(dim=1, keepdim=True)
                    _sd = per_head.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
                    qk = ((per_head - _mu) / _sd).mean(dim=0)

                # frames_ago: 0 == the incoming frame. Sparse frame i is
                # (n_frames - 1 - i) frames old, so this DESCENDS with i (oldest first).
                frames_ago = (n_frames - 1) - torch.arange(
                    sparse_frames, device=qk.device)
                # G4: a flip or off-by-one here turns the arm into an ANTI-recency
                # selector with no other visible symptom.
                assert int(frames_ago.max()) == n_frames - 1, \
                    f"qframe frames_ago max {int(frames_ago.max())} != {n_frames - 1}"
                assert int(frames_ago.min()) == dense_frames, \
                    f"qframe frames_ago min {int(frames_ago.min())} != {dense_frames}"

                prof, _n_unseen = prof_obj.lookup(frames_ago)
                # +1e-6*arange breaks ties toward RECENT frames (arange ascends with i,
                # but topk takes largest, and i ascends oldest->newest). Without it the
                # all-zero-score case (empty profile at lam=1) silently selects the
                # OLDEST frames via topk's index ordering. Precedent: akv2's stripe
                # tie-break.
                score = (qk - lam * prof.to(qk.device)
                         + 1e-6 * torch.arange(sparse_frames, device=qk.device, dtype=qk.dtype))
                keep_sparse[score.topk(n_keep).indices] = True

                # DECISION-GATE RECORDER. MEMSTREAM_QFRAME_DIR=<dir> dumps the raw
                # head-pooled PRE-RoPE affinity together with the profile slice, so the
                # ENTIRE lam->overlap curve is recomputable offline from ONE encode-only
                # run -- no re-run per lam. This exists because lam may be INERT pre-RoPE:
                # top-k is shift-invariant, so only the profile's SHAPE matters, and much
                # of the post-RoPE recency gradient is RoPE's own long-range decay, which
                # the pre-RoPE keys scored here do not have. If the profile is flat, every
                # lam is the same arm and the designed mechanism is dead.
                _qf_dir = os.environ.get("MEMSTREAM_QFRAME_DIR")
                if _qf_dir:
                    import json as _json
                    os.makedirs(_qf_dir, exist_ok=True)
                    _sel = torch.nonzero(keep_sparse, as_tuple=False).squeeze(-1)
                    _recent = set(range(sparse_frames - n_keep, sparse_frames))
                    _rec = {
                        "video_id": os.environ.get("CURRENT_VIDEO_ID", ""),
                        "layer": int(self.layer_idx),
                        "window_idx": int(self.num_blocks_seen),
                        "n_frames": int(n_frames),
                        "sparse_frames": int(sparse_frames),
                        "n_keep": int(n_keep),
                        "lam": float(lam),
                        "n_unseen": int(_n_unseen),
                        # both oldest-first, aligned with the sparse frame index
                        "qk_prerope": [round(float(x), 4) for x in qk.tolist()],
                        "profile": [round(float(x), 4) for x in prof.tolist()],
                        "per_head_std": round(float(per_head.std(dim=0).mean()), 5),
                        # Full (nh_kv, sparse_frames) matrix, oldest-first per head. Lets us
                        # measure offline how far each head's own top-k departs from the
                        # head-pooled top-k -- i.e. whether forcing one shared frame set
                        # on every head (which the mixed-window rebuild requires) throws
                        # away per-head preferences. 3dp: ~8x72 floats per record.
                        "qk_prerope_per_head": [
                            [round(float(x), 3) for x in row] for row in per_head.tolist()],
                        "recency_overlap": round(
                            len(_recent & set(_sel.tolist())) / max(1, n_keep), 4),
                        "chance": round(n_keep / max(1, sparse_frames), 4),
                    }
                    with open(os.path.join(
                            _qf_dir, f"qframe_L{int(self.layer_idx)}.jsonl"), "a") as _f:
                        _f.write(_json.dumps(_rec) + "\n")

                # AFTER scoring, never before: a window must not de-mean itself.
                prof_obj.update(frames_ago, qk)

            # G5
            assert int(keep_sparse.sum()) == n_keep, \
                f"qframe kept {int(keep_sparse.sum())} frames, expected {n_keep}"

            sparse_indices = frame_keep_to_sparse_indices(
                keep_sparse, visual_starts, visual_ends, nh, local_k.device)
            assert len(sparse_indices) == sparse_frames

            out = rebuild_mixed_window(
                local_k, local_v,
                (prefix_len, visual_starts, visual_ends, boundary_indices),
                sparse_indices,
            )
            # G7: the rebuild's `.view(nh, -1)` silently depends on every head keeping the
            # same tokens. Frame-level selection satisfies that by construction; assert it
            # so a future per-head variant cannot break it quietly.
            _kept = out[2]
            assert torch.equal(_kept[0], _kept[0, :1].expand_as(_kept[0])), \
                "qframe kept indices differ across heads"
            return out

        # prefill with kk-guided
        elif prefill_method.startswith("swa_kk_full_") \
            or prefill_method.startswith("swa_kk_adjacent_") \
            or prefill_method.startswith("swa_kk_keep_") \
            or prefill_method.startswith("swa_vdiv_") \
            or prefill_method.startswith("swa_vv_adj_") \
            or prefill_method.startswith("swa_akv2_") \
            or prefill_method.startswith("swa_gtalloc_") \
            or prefill_method.startswith("swa_allocvec_") \
            or prefill_method.startswith("swa_dilated_"):

            if not should_compress:
                return no_compression(local_k, local_v, self.n_local)

            # Parse adaptive configuration if present
            adaptive, adaptive_config, prefill_method = self._parse_adaptive_config(prefill_method)

            # swa_kk_keep_<s> keeps round(h*w/s) patches (1/s of the grid), which the per-axis
            # stride of swa_kk_adjacent_<r> cannot express. It is deliberately a DIFFERENT
            # prefix: kk_adjacent's "_2" already means stride-2 = 1/4 kept, so reusing that
            # token would silently reinterpret every existing arm name and result dir.
            keep_k = None
            akv2_params = None
            gtalloc_floor = None
            gtalloc_vector = False
            postrope = False
            # Selection-criterion knobs. These MUST be initialised here and passed
            # explicitly at BOTH call sites below. They used to be read via
            # locals().get(...) at the purely-visual call site only, which meant the
            # mixed-window path (every Qwen3-VL run, since timestamps interleave text
            # with visual tokens) silently dropped them and fell back to the default
            # criterion -- no error, no log line, just a different arm than the name
            # says. That defect made swa_vv_adj_2 run as swa_kk_adjacent_2.
            score_on = "k"
            vdiv_rounds = 1
            if prefill_method.startswith("swa_allocvec_"):
                # Arbitrary per-frame keep counts (allocation-search harness): the tag
                # after swa_allocvec_ only names the candidate; counts come from
                # MEMSTREAM_ALLOCVEC_JSON = {video_id: {"keeps": [count per global
                # frame]}}. Selection inside each frame stays the v1 criterion
                # (per-head adjacent dissimilarity, stripe for frame 0).
                method = "gtalloc"
                gtalloc_vector = True
                gtalloc_dense = None
                if not hasattr(self, "_allocvec_sets"):
                    import json as _json
                    _p = os.environ.get("MEMSTREAM_ALLOCVEC_JSON")
                    if not _p:
                        raise ValueError("swa_allocvec requires MEMSTREAM_ALLOCVEC_JSON")
                    self._allocvec_sets = _json.load(open(_p))
                _vid = os.environ.get("CURRENT_VIDEO_ID", "")
                if _vid not in self._allocvec_sets:
                    raise ValueError(f"allocvec: video {_vid!r} not in MEMSTREAM_ALLOCVEC_JSON")
                _entry = self._allocvec_sets[_vid]
                if not isinstance(_entry, dict) or "keeps" not in _entry:
                    raise ValueError(f"allocvec: entry for {_vid!r} must be {{'keeps': [...]}}")
                self._allocvec_keeps = [int(x) for x in _entry["keeps"]]
                dilation_rate = None
                top_p = False
            elif prefill_method.startswith("swa_gtalloc_"):
                # GT-allocation ORACLE: swa_gtalloc_f<floor>. Frames in the per-video
                # union of clue windows (MEMSTREAM_GTALLOC_JSON) keep ALL patches;
                # others keep <floor> by the v1 criterion. Ceiling probe, not a method.
                method = "gtalloc"
                _body = prefill_method.split("swa_gtalloc_f")[-1].split("_after")[0]
                gtalloc_floor = int(_body.split("_d")[0])
                gtalloc_dense = int(_body.split("_d")[1]) if "_d" in _body else None
                if not hasattr(self, "_gtalloc_sets"):
                    import json as _json
                    _p = os.environ.get("MEMSTREAM_GTALLOC_JSON")
                    if not _p:
                        raise ValueError("swa_gtalloc requires MEMSTREAM_GTALLOC_JSON")
                    self._gtalloc_sets = {k: set(v) for k, v in _json.load(open(_p)).items()}
                _vid = os.environ.get("CURRENT_VIDEO_ID", "")
                if _vid not in self._gtalloc_sets:
                    raise ValueError(f"gtalloc: video {_vid!r} not in MEMSTREAM_GTALLOC_JSON")
                self._gtalloc_dense = self._gtalloc_sets[_vid]
                dilation_rate = None
                top_p = False
            elif prefill_method.startswith("swa_akv2_"):
                # AKS-2: count-targeted, budget-reallocated, query-guided selection.
                # Name grammar: swa_akv2_<avg_keep>[_a<alpha*100>][_f<floor_frac*100>]
                #   swa_akv2_120        -> avg 120 kept tokens/frame, alpha=0.50, floor=0.25
                #   swa_akv2_120_a100   -> pure novelty scoring under the budget
                #   swa_akv2_120_a0_f50 -> pure query guidance, floor = 60 tokens/frame
                # avg_keep is a COUNT, not a ratio: the retained budget is fixed as the
                # input tokens-per-frame scales (the point of the method).
                method = "akv2"
                body = prefill_method.split("swa_akv2_")[-1].split("_after")[0]
                parts = body.split("_")
                akv2_params = {"avg_keep": int(parts[0]), "alpha": 0.5, "floor_frac": 0.25}
                for p in parts[1:]:
                    if p.startswith("a") and p[1:].isdigit():
                        akv2_params["alpha"] = int(p[1:]) / 100.0
                    elif p.startswith("f") and p[1:].isdigit():
                        akv2_params["floor_frac"] = int(p[1:]) / 100.0
                    else:
                        raise ValueError(f"unrecognised akv2 token '{p}' in {prefill_method}")
                dilation_rate = None
                top_p = False
            elif prefill_method.startswith("swa_vv_adj_"):
                # VALUE-space twin of swa_kk_adjacent_<r>: identical adjacent-frame
                # criterion and identical kept-token count, scoring VALUES instead of
                # keys. The only difference from kk_adjacent is which tensor is read,
                # which is exactly the contrast the eff_distinct result points at.
                method = "adjacent"
                score_on = "v"
                # Set dilation_rate_STR, not dilation_rate: the shared line below does
                # `dilation_rate = int(dilation_rate_str)` whenever keep_k is None, so
                # assigning the int here leaves the str unbound and raises UnboundLocalError
                # at the first compressed window.
                dilation_rate_str = prefill_method.split("swa_vv_adj_")[-1].split("_after")[0]
                top_p = False
            elif prefill_method.startswith("swa_vdiv_"):
                # VALUE-DIVERSITY selector. swa_vdiv_<s>[_rN] keeps round(h*w/s)
                # patches per frame -- the SAME count as swa_kk_keep_<s> -- so the two
                # differ only in criterion (value redundancy vs key novelty), which is
                # the comparison worth running. _rN sets the re-scoring rounds.
                method = "vdiv"
                _tok = prefill_method.split("swa_vdiv_")[-1].split("_after")[0].split("_")
                keep_denom = float(_tok[0])
                if keep_denom < 1:
                    raise ValueError(
                        f"swa_vdiv_<s> needs s >= 1 (got {keep_denom}); s is a divisor of "
                        f"the grid, so s=2 keeps half and s=4 keeps a quarter."
                    )
                gh, gw = self.grid_hw
                keep_k = max(1, min(int(round(gh * gw / keep_denom)), gh * gw))
                vdiv_rounds = 1
                for _t in _tok[1:]:
                    if _t.startswith("r") and _t[1:].isdigit():
                        vdiv_rounds = int(_t[1:])
                    elif _t:
                        raise ValueError(f"unrecognised vdiv token '{_t}' in {prefill_method}")
                dilation_rate = None
                top_p = False
            elif prefill_method.startswith("swa_kk_keep_"):
                method = "adjacent"
                keep_denom = float(prefill_method.split("swa_kk_keep_")[-1].split("_after")[0])
                if keep_denom < 1:
                    raise ValueError(
                        f"swa_kk_keep_<s> needs s >= 1 (got {keep_denom}); s is a divisor of the "
                        f"grid, so s=2 keeps half and s=4 keeps a quarter."
                    )
                gh, gw = self.grid_hw
                keep_k = max(1, min(int(round(gh * gw / keep_denom)), gh * gw))
                dilation_rate = 1  # unused on this path; frame 0 uses the diagonal mask
                top_p = False
            elif prefill_method.startswith("swa_dilated_"):
                # Same-budget control for AKS: keeps the identical dilated index set that
                # swa_kk_adjacent_<r> uses for its first frame, but applies it to every frame
                # instead of ranking patches by adjacent-frame key similarity.
                method = "dilated"
                dilation_rate_str = prefill_method.split("_", 2)[-1].split("_after")[0]
                top_p = False
            else:
                method_suffix = prefill_method.split("_", 3)[-1]
                dilation_rate_str = method_suffix.split("_after")[0]
                method = prefill_method.split('_', 3)[2] # "full" or "adjacent"

                top_p = prefill_method.endswith("_top_p") # swa_kk_adjacent_4_top_p -> True
                if "top_p" in dilation_rate_str:
                    dilation_rate_str = dilation_rate_str.split("_top_p")[0]
                # swa_kk_adjacent_<r>_postrope: novelty measured in rope space —
                # the later frame's keys get the constant inter-block rotation R(delta)
                # before the adjacent cosine. Selection budget/machinery unchanged.
                postrope = "_postrope" in prefill_method
                if "postrope" in dilation_rate_str:
                    dilation_rate_str = dilation_rate_str.split("_postrope")[0]
            if keep_k is None and akv2_params is None and gtalloc_floor is None and not gtalloc_vector:
                dilation_rate = int(dilation_rate_str)

            # Compress visual tokens only, then rebuild the mixed local window
            # around each frame's timestamp and boundary tokens.
            if visual_pos_masks is not None:

                visual_token_indices = torch.nonzero(visual_pos_masks[0], as_tuple=False).squeeze(-1)
                visual_local_k = local_k[:, :, visual_token_indices.to(local_k.device), :]
                visual_local_v = local_v[:, :, visual_token_indices.to(local_v.device), :]

                # Frame spans come from the shared derivation; it also enforces that every
                # frame has both of its boundary tokens.
                prefix_len, visual_starts, visual_ends, boundary_indices = self._frame_geometry(
                    local_k, visual_pos_masks)

                # Patch-level methods additionally require a *uniform* per-frame visual width,
                # because they reshape the sparse region to (m, h, w) to select patches. Frame
                # sampling has no such requirement, so this check stays here rather than moving
                # into _frame_geometry.
                if not (
                    visual_token_indices.numel() % self.exc_block_size == 0
                    and bool(torch.all((visual_ends - visual_starts) == self.exc_block_size))
                    and visual_local_k.size(-2) % self.exc_block_size == 0
                ):
                    raise AssertionError(
                        "mixed text/visual window has non-uniform frame widths for AKS compression: "
                        f"n_visual_starts={visual_starts.numel()}, n_visual_ends={visual_ends.numel()}, "
                        f"n_boundary={boundary_indices.numel()}, "
                        f"n_visual_tokens={visual_token_indices.numel()}, "
                        f"visual_local_k_len={visual_local_k.size(-2)}, "
                        f"exc_block_size={self.exc_block_size}, prefix_len={prefix_len}, "
                        f"local_k_len={local_k.size(-2)}, init_exc={self.init_exc}, "
                        f"frame_lens={(visual_ends - visual_starts).tolist()}"
                    )

                if method == "dilated":
                    # Fixed-stride selection applied to every sparse frame. This is exactly the
                    # index set apply_kk_guided_compression uses for its frame 0, so the kept-token
                    # budget matches swa_kk_adjacent_<r> and the only difference is *which* patches
                    # survive -- the control that separates budget cost from selection quality.
                    gh, gw = self.grid_hw
                    n_sparse = max(
                        (visual_local_k.size(-2) - self.exc_block_size * self.nd) // self.exc_block_size,
                        0,
                    )
                    h_idx = torch.arange(0, gh, dilation_rate, device=local_k.device)
                    w_idx = torch.arange(0, gw, dilation_rate, device=local_k.device)
                    spatial_mask = torch.zeros(gh, gw, dtype=torch.bool, device=local_k.device)
                    spatial_mask[h_idx[:, None], w_idx[None, :]] = True
                    subsample_mask = spatial_mask.view(1, 1, 1, gh, gw).expand(
                        1, local_k.size(1), n_sparse, gh, gw
                    )
                elif method == "akv2":
                    # Mask only; the mixed-window rebuild below gathers the kept tokens.
                    # Per-frame counts vary but are shared across heads, which is what
                    # the rebuild's (nh, -1) reshape requires.
                    _, _, subsample_mask, _ = apply_budgeted_kk_compression(
                        visual_local_k, visual_local_v, local_q,
                        0, True, self.grid_hw,
                        nd=self.nd,
                        dense_len=self.exc_block_size * self.nd,
                        **akv2_params,
                    )
                elif method == "gtalloc":
                    _n_sp = max((visual_local_k.size(-2) - self.exc_block_size * self.nd) // self.exc_block_size, 0)
                    _gs = self.num_blocks_seen - _n_sp
                    if gtalloc_vector:
                        if _gs + _n_sp > len(self._allocvec_keeps):
                            raise ValueError(
                                f"allocvec: window needs frames up to {_gs + _n_sp} but keeps "
                                f"has only {len(self._allocvec_keeps)} entries")
                        _kv = [self._allocvec_keeps[_gs + _j] for _j in range(_n_sp)]
                        _, _, subsample_mask, _ = apply_gt_alloc_compression(
                            visual_local_k, visual_local_v, local_q,
                            0, True, self.grid_hw,
                            dense_frame_mask=None, floor_keep=None,
                            nd=self.nd, dense_len=self.exc_block_size * self.nd,
                            keep_vector=_kv,
                        )
                    else:
                        _dm = [(_gs + _j) in self._gtalloc_dense for _j in range(_n_sp)]
                        _, _, subsample_mask, _ = apply_gt_alloc_compression(
                            visual_local_k, visual_local_v, local_q,
                            0, True, self.grid_hw,
                            dense_frame_mask=_dm, floor_keep=gtalloc_floor,
                            nd=self.nd, dense_len=self.exc_block_size * self.nd,
                            dense_keep=gtalloc_dense,
                        )
                else:
                    _pr_kwargs = {}
                    if postrope and self.sliding_window_pos_ids.size(-1) > 0 and len(self.adaptive_block_sizes) >= 2:
                        _bs0 = self.adaptive_block_sizes[0]
                        _delta = float(self.sliding_window_pos_ids[0, 0, _bs0] - self.sliding_window_pos_ids[0, 0, 0])
                        _pos = torch.full((3, 1, 1), _delta, dtype=torch.float32)
                        _cosd, _sind = self.position_embedding._generate_cos_sin_tables_pos(_pos)
                        _pr_kwargs = {"postrope_cos": _cosd.reshape(-1), "postrope_sin": _sind.reshape(-1)}
                    _, _, subsample_mask, _ = apply_kk_guided_compression(
                        visual_local_k, visual_local_v, local_q,
                        0, True, self.grid_hw,
                        method=method,
                        dilation_rate=dilation_rate,
                        nd=self.nd,
                        dense_len=self.exc_block_size * self.nd,
                        top_p=top_p,
                        keep_k=keep_k,
                        score_on=score_on,
                        vdiv_rounds=vdiv_rounds,
                        **_pr_kwargs,
                    )

                sparse_frames = visual_starts.numel() - min(
                    self.nd, visual_starts.numel())
                nh = local_k.size(1)
                flat_subsample_mask = subsample_mask.flatten(start_dim=3) if sparse_frames > 0 else None
                sparse_indices = []
                for frame_idx in range(sparse_frames):
                    selected = torch.nonzero(
                        flat_subsample_mask[0, :, frame_idx], as_tuple=False
                    )[:, 1].view(nh, -1).to(local_k.device)
                    sparse_indices.append(
                        visual_starts[frame_idx].to(local_k.device) + selected)
                return rebuild_mixed_window(
                    local_k, local_v,
                    (prefix_len, visual_starts, visual_ends, boundary_indices),
                    sparse_indices,
                )

            else:
                if method == "dilated":
                    raise ValueError(
                        "swa_dilated_<r> reached the kk-guided path without visual_pos_masks; "
                        "purely-visual windows are handled by the dedicated swa_dilated_ branch."
                    )
                if method == "gtalloc":
                    _st = 0 if self.init_exc else self.n_init
                    _n_sp = max((local_k.size(-2) - _st - self.exc_block_size * self.nd) // self.exc_block_size, 0)
                    _gs = self.num_blocks_seen - _n_sp
                    if gtalloc_vector:
                        if _gs + _n_sp > len(self._allocvec_keeps):
                            raise ValueError(
                                f"allocvec: window needs frames up to {_gs + _n_sp} but keeps "
                                f"has only {len(self._allocvec_keeps)} entries")
                        _kv = [self._allocvec_keeps[_gs + _j] for _j in range(_n_sp)]
                        return apply_gt_alloc_compression(
                            local_k, local_v, local_q,
                            self.n_init, self.init_exc, self.grid_hw,
                            dense_frame_mask=None, floor_keep=None,
                            nd=self.nd, dense_len=self.exc_block_size * self.nd,
                            keep_vector=_kv,
                        )
                    _dm = [(_gs + _j) in self._gtalloc_dense for _j in range(_n_sp)]
                    return apply_gt_alloc_compression(
                        local_k, local_v, local_q,
                        self.n_init, self.init_exc, self.grid_hw,
                        dense_frame_mask=_dm, floor_keep=gtalloc_floor,
                        nd=self.nd, dense_len=self.exc_block_size * self.nd,
                        dense_keep=gtalloc_dense,
                    )
                if method == "akv2":
                    return apply_budgeted_kk_compression(
                        local_k, local_v, local_q,
                        self.n_init, self.init_exc, self.grid_hw,
                        nd=self.nd,
                        dense_len=self.exc_block_size * self.nd,
                        **akv2_params,
                    )
                compressed_local_k, compressed_local_v, subsample_mask, sliding_window = apply_kk_guided_compression(
                    local_k, local_v, local_q,
                    self.n_init, self.init_exc, self.grid_hw,
                    method=method,
                    dilation_rate=dilation_rate,
                    nd=self.nd,
                    dense_len=self.exc_block_size * self.nd,
                    top_p=top_p,
                    keep_k=keep_k,
                    vdiv_rounds=vdiv_rounds,
                    score_on=score_on,
                )
                return compressed_local_k, compressed_local_v, subsample_mask, sliding_window

        elif self.prefill_method == "swa": 
            return no_compression(local_k, local_v, self.n_local)

        else: 
            raise ValueError(f"Unsupported prefill method: {self.prefill_method}")
        
    def _append(
        self,
        local_q, local_k, local_v, global_q,
        sliding_window_pos_ids = None, block_sizes=None,
        visual_pos_masks = None,
    ):
        """calculate attention results 

        Args:
            local_q (_type_): (batch_size, num_heads, length, dim_head)
            local_k (_type_): (batch_size, num_heads, length, dim_head)
            local_v (_type_): (batch_size, num_heads, length, dim_head)
            global_q (_type_): (batch_size, num_heads, length, dim_head)
            sliding_window_pos_ids (_type_): (3, 1, length)
            block_sizes (__type__)L list of length self.n_local_frames 
            visual_pos_masks (_type_): (batch_size, length) - mask of length input_ids, where visual tokens are located (True)
        Returns:
            chunk_o: (batch_size, num_heads, length, dim_head)
        """
        
        len_q = local_q.size(-2)         
        len_k = local_k.size(-2)

        # Adjacent frame-level cosine similarities for event chunking (analysis only)
        if self.store_adjacent_similarities and self.adjacent_key_similarities is not None and len_k - len_q >= self.n_init:
            diagnostics.record_adjacent_similarity(self, local_k, len_q)

        # Determine compression threshold
        should_compress = True
        if "_after_exc" in self.prefill_method:
            should_compress = self.init_exc
        elif "_after_" in self.prefill_method and "f" in self.prefill_method.split("_after_")[-1]:
            frame_threshold = int(''.join(filter(str.isdigit, self.prefill_method.split("_after_")[-1].split('f')[0])))
            should_compress = self.num_blocks_seen >= frame_threshold

        # start compression after prefilling sink tokens and the first frame
        start_compression = len_k - len_q > self.n_init

        # Apply compression method based on prefill_method.
        # visual_pos_masks is forwarded so compression methods can handle
        # extract/compress/scatter internally (e.g. kk_adjacent for InternVL3.5).
        compressed_k, compressed_v, subsample_mask, sliding_window = self._apply_compression(
            local_k, local_v, local_q, len_k,
            start_compression, should_compress,
            visual_pos_masks=visual_pos_masks,
        )

        # Apply position embedding to compressed keys
        if subsample_mask is not None:
            # Compression with subsampling - need special positional embedding handling
            # Use nd to determine how many dense frames to keep
            if "tome" in self.prefill_method:
                local_h_q, local_h_k = self.position_embedding(
                    local_q, compressed_k, right=len_k, init_exc=self.init_exc,
                    subsample_window=(self.init_exc, len_k - self.nd*len_q), subsample_mask=subsample_mask, 
                    height_width = self.grid_hw,
                )
            else:
                local_h_q, local_h_k = self.position_embedding(
                    local_q, compressed_k, right=len_k, init_exc=self.init_exc,
                    subsample_window=(self.init_exc, len_k - self.nd*len_q), subsample_mask=subsample_mask
                )
        else:
            # No compression or early in prefill - standard positional embedding
            local_h_q, local_h_k = self.position_embedding(local_q, compressed_k, init_exc=self.init_exc)
        
        local_h_v = compressed_v

        # one record per layer, and only once the sliding window is FULL -- sampling
        # the first encode calls measures novelty against a single predecessor frame
        # in a ~1k-key window, which is not the regime any arm actually runs in.
        if _TAIL_PROBE and not _TAIL_PROBE_SEEN and not globals().get("_TAIL_SELFTEST_DONE"):
            globals()["_TAIL_SELFTEST_DONE"] = True
            _tail_probe_selftest()
        # Record once per layer PER WINDOW (capped), not once per layer total. The
        # single-window version could not distinguish "this layer is redundant" from
        # "this layer was redundant in the first 2 minutes of the video".
        # FULL-VIDEO probing: record on EVERY encode call whose window is full, for
        # every layer, for the whole video. Statistics are then aggregated over all
        # windows offline. Sampling a single window (or a few CONSECUTIVE calls,
        # which overlap by all but encode_chunk_size frames) is not representative
        # of a video and previously produced misleading "stable" readings.
        sliding_window_mask = None

        # store attention scores or not 
        get_score = self.store_attn_scores in ["prefill", "all"]

        # compute attention with local KV
        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        # lets the top-p stats attribute kept-key counts to a layer
        attn.layer_idx = self.layer_idx
        attn.append(
            local_h_q, local_h_k, local_h_v, 
            end=False, 
            get_score=get_score,
            sliding_window=sliding_window,
            custom_mask=sliding_window_mask
        )

        # load init KV and update cache state
        # Note: get_global_hidden_and_mask() updates cache management state even if we don't use the returned KV
        with torch.cuda.stream(GLOBAL_STREAM):
            global_h_q = global_q
            global_h_k, global_h_v = self.get_global_hidden_and_mask(exc_length=global_q.size(-2))

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        if _TAIL_PROBE and local_h_k.size(-2) >= _TAIL_PROBE_MIN_KEYS:
            _r = _tail_redundancy_probe(local_h_q, local_h_k, self.block_size, self.layer_idx,
                                        visual_mask=visual_pos_masks, len_q=len_q, n_init=self.n_init,
                                        pre_rope_k=compressed_k, global_k=global_h_k,
                                        local_h_v=local_h_v)
            if _r is not None:
                _r["window_idx"] = _TAIL_PROBE_SEEN.count(self.layer_idx)
                _TAIL_PROBE_SEEN.append(self.layer_idx)
                _TAIL_PROBE_BUF.append(_r)
                # APPEND-ONLY JSONL. A full video is ~750 windows x 36 layers ~= 27k
                # records; rewriting a growing JSON array on every flush would
                # re-serialise tens of MB dozens of times. Appending also means a
                # preempted run keeps everything written so far.
                _out = os.environ.get("TAIL_PROBE_OUT")
                if _out:
                    import json as _json
                    with open(_out, "a") as _f:
                        _f.write(_json.dumps(_r) + "\n")
                    _TAIL_PROBE_BUF.clear()          # streamed out; do not hold in RAM
                    if _TAIL_PROBE_SEEN.count(0) % _TAIL_PROBE_FLUSH == 0 and self.layer_idx == 0:
                        print(f"[TAIL-PROBE] {_TAIL_PROBE_SEEN.count(0)} windows recorded "
                              f"(n_keys={_r['n_keys']}) -> {_out}", flush=True)

        # compute attention with sinks (global KV)       
        attn.append(
            global_h_q, global_h_k, global_h_v, 
            end=True,  # the final append operation
            get_score=get_score,
            sliding_window=None,
            complement_sliding_window=True,
        )

        o, score_list = attn.get_result()

        # Env-gated attention-score diagnostics (attention stats CSV, textmass,
        # masknull, proxyfid) — all moved to diagnostics.py; inert in normal runs.
        if get_score and start_compression:
            diagnostics.record_prefill_attention_diagnostics(
                self, score_list, local_q, local_k, visual_pos_masks, subsample_mask,
            )

        assert not torch.isnan(o).any(), "attention output is nan!"

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        return o.view((self.batch_size, self.num_heads, -1, self.dim_head))

    def _append_global(
        self
    ):
        """offload context memory
        """

        global_remainder_ed = self._global_remainder_ed
        global_remainder_st = self._global_remainder_st
        global_remainder_len = global_remainder_ed - global_remainder_st

        # Offload to CPU if init cache is full OR if chunked prefilling
        offload = self.init_exc or ("chunked" in self.prefill_method and global_remainder_len > self.n_init)

        if offload:
            # First time offloading (chunked prefill path): save init tokens
            if not self.init_exc and global_remainder_st == 0:
                self.init_k = self.global_remainder[0][:, :, :self.n_init, :]
                self.init_v = self.global_remainder[1][:, :, :self.n_init, :]
                self.global_buffer[0][:, :, :self.n_init, :].copy_(self.init_k, non_blocking=True)
                self.global_buffer[1][:, :, :self.n_init, :].copy_(self.init_v, non_blocking=True)
                global_remainder_st += self.n_init
                global_remainder_len -= self.n_init

            # Offload blocks — adaptive blocks consume full timestamp+visual spans,
            # but store only fixed-width visual KV in CPU memory.
            while global_remainder_len > 0:
                curr_full_block_size = self.adaptive_block_sizes[self.num_global_block] if self.has_adaptive_block_size else self.block_size
                if global_remainder_len < curr_full_block_size:
                    break  # not enough tokens for the next block yet
                global_remainder_len -= curr_full_block_size

                block_k = self.global_remainder[0][:, :, global_remainder_st:global_remainder_st + curr_full_block_size, :]
                block_v = self.global_remainder[1][:, :, global_remainder_st:global_remainder_st + curr_full_block_size, :]
                if self.has_adaptive_block_size:
                    assert self.global_remainder_visual_mask is not None
                    visual_mask = self.global_remainder_visual_mask[:, global_remainder_st:global_remainder_st + curr_full_block_size]
                    block_k = block_k[:, :, visual_mask[0].to(block_k.device), :]
                    block_v = block_v[:, :, visual_mask[0].to(block_v.device), :]
                    assert block_k.size(-2) == self.block_size, f"{block_k.size(-2)} vs. {self.block_size}"

                # Context KV-Cache
                for u in range(self.num_units):
                    self.global_blocks[u].append((
                        MemoryUnit(
                            (
                                block_k[u],
                                block_v[u]
                            ),
                            self.cuda_cache,
                            False,
                            self.pin_memory
                        )
                    ))

                # Representative vector for retrieval — reduce over token dim, skip prefix if present
                global_block_k = {}
                reduce_ops = {"rekv": "mean", "min": "amin", "max": "amax"}
                op = next((v for k, v in reduce_ops.items() if k in self.retrieval_method), None)
                if op is not None:
                    global_block_k = self._from_group_kv(block_k)
                    # Retrieval-time AKS: the block's retrieval signature is built HERE,
                    # once, at offload -- not at query time -- so this is where the
                    # adjacent-novelty selection has to live. The stored KV (block_k /
                    # block_v above) stays DENSE; only what the retriever sees changes.
                    if _RETRIEVAL_AKS > 1:
                        gk = rearrange(global_block_k, "u h n d -> u n (h d)")
                        prev = self._aks_prev_block_k
                        if prev is not None and prev.shape == gk.shape:
                            sim = torch.nn.functional.cosine_similarity(
                                gk.float(), prev.float(), dim=-1)          # (units, tokens)
                            keep = max(1, gk.size(1) // _RETRIEVAL_AKS)
                            idx = sim.argsort(dim=-1)[:, :keep]            # least similar == most novel
                            sel = torch.gather(
                                global_block_k, 2,
                                idx[:, None, :, None].expand(-1, global_block_k.size(1), -1,
                                                             global_block_k.size(-1)))
                        else:
                            sel = global_block_k                            # first block: no predecessor
                        self._aks_prev_block_k = gk.detach()
                        global_block_k = sel
                    global_block_k = getattr(global_block_k, op)(dim=-2, keepdim=False)
                    global_block_k = global_block_k.reshape(self.num_units, -1)[:, None, :]

                global_block_v = None
                if _VRETR or _FUSE_RETR:
                    # Same pooling as the key signature, on the block's VALUES.
                    gbv = self._from_group_kv(block_v)
                    gbv = rearrange(gbv, "b h n d -> b n (h d)")
                    gbv = getattr(gbv, op)(dim=-2, keepdim=False)
                    global_block_v = gbv.reshape(self.num_units, -1)[:, None, :]

                for u in range(self.num_units):
                    self.block_k[u].append(global_block_k[u])
                    if global_block_v is not None:
                        self.block_v[u].append(global_block_v[u])
                
                self.num_global_block += 1
                global_remainder_st += curr_full_block_size
            

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st 

    def append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
        curr_position_ids = None, block_sizes = None, 
        visual_pos_masks = None,
    ):
        # Pre-allocate GPU Memory.
        if not self.initialized:
            self.init(local_q, local_k, local_v, global_q, global_k, global_v)

        if self.has_adaptive_block_size: 
            assert curr_position_ids is not None and block_sizes is not None 

        input_length = local_q.size(-2)
        
        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # append local KV
        existing_kv_len = self.local_k.size(-2)  # length BEFORE this chunk (used for buffer bootstrap)
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        kv_length = self.local_k.size(-2)

        # Maintain visual mask buffer — mirrors self.local_k length token-for-token.
        # On the first call that carries a mask, bootstrap with False for all previously
        # accumulated tokens (init / system-prompt tokens, which are never visual).
        if visual_pos_masks is not None:
            if self.visual_mask_buffer is None:
                self.visual_mask_buffer = torch.zeros(
                    visual_pos_masks.shape[0], existing_kv_len,
                    dtype=torch.bool, device=visual_pos_masks.device,
                )
            self.visual_mask_buffer = torch.cat(
                [self.visual_mask_buffer, visual_pos_masks], dim=-1
            )

        if self.has_adaptive_block_size and visual_pos_masks is not None:
            prev_visual_mask = F.pad(visual_pos_masks[:, :-1], (1, 0), value=False)
            next_visual_mask = F.pad(visual_pos_masks[:, 1:], (0, 1), value=False)
            boundary_mask = (~visual_pos_masks) & (prev_visual_mask | next_visual_mask)
            timestamp_mask = (~visual_pos_masks) & (~boundary_mask)

            assert boundary_mask.sum().item() == 2 * len(block_sizes)
            self.full_timestamp_k = torch.cat(
                [self.full_timestamp_k, global_k[:, :, timestamp_mask[0].to(global_k.device), :]], dim=-2
            )
            self.full_timestamp_v = torch.cat(
                [self.full_timestamp_v, global_v[:, :, timestamp_mask[0].to(global_v.device), :]], dim=-2
            )
            self.full_boundary_k = torch.cat(
                [self.full_boundary_k, global_k[:, :, boundary_mask[0].to(global_k.device), :]], dim=-2
            )
            self.full_boundary_v = torch.cat(
                [self.full_boundary_v, global_v[:, :, boundary_mask[0].to(global_v.device), :]], dim=-2
            )

        # append global remainder
        with torch.cuda.stream(GLOBAL_STREAM):
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)
            # tracking the whole cache as well (same values as self.local_k and self.local_v)
            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
            )
            if visual_pos_masks is not None:
                if self.global_remainder_visual_mask is None:
                    self.global_remainder_visual_mask = torch.zeros(
                        visual_pos_masks.shape[0], self._global_remainder_ed,
                        dtype=torch.bool, device=visual_pos_masks.device,
                    )
                self.global_remainder_visual_mask = torch.cat(
                    [self.global_remainder_visual_mask, visual_pos_masks], dim=-1
                )
            if self.store_full:
                diagnostics.append_full_storage(self, global_k, global_v, visual_pos_masks)

        # apply RoPE to global_q
        with torch.cuda.stream(GLOBAL_STREAM):
            max_pos = int(self.n_local * (1 + 1 / self.n_local_frames)) if self.has_adaptive_block_size else self.n_local
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(global_q, max_pos,
                block_sizes=block_sizes, curr_position_ids=curr_position_ids
            )

        o_list = [] 

        # Build iteration boundaries from per-block sizes or fall back to fixed stride
        if block_sizes is not None:
            ends = list(accumulate(block_sizes))
            boundaries = list(zip([0] + ends[:-1], ends))
        else:
            boundaries = [
                (st, min(st + self.exc_block_size, input_length)) for st in range(0, input_length, self.exc_block_size)
            ]

        for block_idx, (st, ed) in enumerate(boundaries):
            # append this block's size so [-n_local_frames:] only sees frames processed so far
            if block_sizes is not None:
                self.adaptive_block_sizes.append(block_sizes[block_idx])

            # append to sliding window position ids as we shift window
            if self.has_adaptive_block_size:
                # curr_position_ids are chunk-cumulative (block b starts at ~b*block_len), so rebase
                # each block to local zero before stitching with max()+1 — otherwise the chunk offset
                # is double-counted and positions inflate with encode_chunk_size (the ecs bug).
                blk_pos = curr_position_ids[..., st:ed]
                blk_pos = blk_pos - blk_pos.min()
                shift = self.sliding_window_pos_ids.max() + 1 if self.sliding_window_pos_ids.numel() else 0
                self.sliding_window_pos_ids = torch.cat(
                    (
                        self.sliding_window_pos_ids.to(curr_position_ids.device),
                        blk_pos + shift,
                    ),
                    dim=-1
                )
                # Invariant: sliding_window_pos_ids keeps the *full* position history (it is never
                # trimmed, unlike local_k / visual_mask_buffer) and is indexed by the cumulative
                # offset sum(adaptive_block_sizes[:first_frame_in_window]).
                assert self.sliding_window_pos_ids.size(-1) == sum(self.adaptive_block_sizes), (
                    f"sliding_window_pos_ids ({self.sliding_window_pos_ids.size(-1)}) drifted from "
                    f"sum(adaptive_block_sizes) ({sum(self.adaptive_block_sizes)}) at block {block_idx}"
                )

            # calculate attention results
            if self.has_adaptive_block_size:
                # use the actual block sizes to compute the local window size when shifting window 
                if self.init_exc: 
                    n_local = sum(self.adaptive_block_sizes[self.num_blocks_seen-self.n_local_frames:self.num_blocks_seen])
                    kv_st = max(kv_length + st - input_length - n_local, 0)
                else:
                    kv_st = max(kv_length + st - input_length - self.n_local, 0) 
                kv_ed = kv_length + ed - input_length 
            else:
                kv_st = max(kv_length + st - input_length - self.n_local, 0)
                kv_ed = kv_length + ed - input_length

            # init_exc flips on a TOKEN threshold while this gate counts FRAMES, so at the
            # smallest budgets a layer can cross this gate while still in the pre-slide
            # kv_st=0 regime (not init_exc). In that state the shared cos/sin table must NOT
            # be read as-is: it is a singleton shared by all layers, and a layer that already
            # slid has rebuilt it down to its own trailing window. So rebuild before reading —
            # over ALL blocks in the cache (matching kv_st=0), right-justified to the anchor.
            if self.has_adaptive_block_size and self.num_blocks_seen > self.n_local_frames:
                if not self.init_exc:
                    full_block_sizes = self.adaptive_block_sizes[:self.num_blocks_seen + 1]
                    assert self.sliding_window_pos_ids.size(-1) == sum(full_block_sizes), (
                        self.sliding_window_pos_ids.shape, sum(full_block_sizes)
                    )
                    self.position_embedding._recompute_cos_sin_tables_for_window(
                        self.pos_emb_device,
                        self.sliding_window_pos_ids,
                        self.n_init,
                        full_block_sizes,
                        max_pos,
                    )
                    _tbl_len = self.position_embedding._cos_cached.size(-2)
                    if _tbl_len != kv_ed:
                        raise AssertionError(
                            "deferred-slide rebuild does not tile the local cache: rebuilt table "
                            f"has {_tbl_len} rows but kv_ed={kv_ed}. In the pre-slide regime the "
                            f"cache must hold exactly n_init + all blocks: num_blocks_seen="
                            f"{self.num_blocks_seen}, n_local_frames={self.n_local_frames}, "
                            f"n_local={self.n_local}, "
                            f"adaptive_block_sizes[-5:]={self.adaptive_block_sizes[-5:]}"
                        )
                else:
                    window_block_sizes = self.adaptive_block_sizes[self.num_blocks_seen-self.n_local_frames:self.num_blocks_seen + 1]
                    sw_st = sum(self.adaptive_block_sizes[:self.num_blocks_seen - self.n_local_frames])
                    window_pos_ids = self.sliding_window_pos_ids[..., sw_st:]
                    assert window_pos_ids.size(-1) == sum(window_block_sizes), (
                        window_pos_ids.shape, sum(window_block_sizes), window_block_sizes
                    )
                    self.position_embedding._recompute_cos_sin_tables_for_window(
                        self.pos_emb_device,
                        window_pos_ids,
                        self.n_init,
                        window_block_sizes, # +1 because we want to include the current block size
                        max_pos,
                    )

            if self.has_adaptive_block_size and (kv_ed - kv_st) < (ed - st):
                raise AssertionError(
                    "adaptive sliding window underflow: the KV window is shorter than the query chunk "
                    f"(kv_st={kv_st}, kv_ed={kv_ed}, span={kv_ed - kv_st}) < (st={st}, ed={ed}, "
                    f"span={ed - st}); num_blocks_seen={self.num_blocks_seen}, "
                    f"n_local_frames={self.n_local_frames}, n_local={self.n_local}, "
                    f"adaptive_block_sizes[-5:]={self.adaptive_block_sizes[-5:]}"
                )


            chunk_o = self._append(
                local_q[:, :, st:ed, :],
                self.local_k[:, :, kv_st: kv_ed, :],
                self.local_v[:, :, kv_st: kv_ed, :],
                global_q[:, :, st:ed, :],
                # Pass the full-window slice of the buffer so _append sees masks for
                # all kv_st:kv_ed tokens, not just the current chunk.
                visual_pos_masks=self.visual_mask_buffer[:, kv_st:kv_ed] \
                    if self.visual_mask_buffer is not None else None,
            )
            o_list.append(chunk_o)

            # offload context memory
            with torch.cuda.stream(GLOBAL_STREAM):
                self._append_global()

            if self.async_global_stream:
                torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)
            
            # Track total blocks seen (for compression logic)
            num_tokens_added = ed - st
            if self.has_adaptive_block_size and num_tokens_added > self.n_init:
                self.num_blocks_seen += 1
            elif num_tokens_added > self.n_init and num_tokens_added % self.block_size == 0:
                self.num_blocks_seen += num_tokens_added // self.block_size
            
        self.length += input_length

        # restrict the length of local KV-cache to n_local_frames (frame-aware) or self.n_local (fixed)
        if self.has_adaptive_block_size:
            st_idx = max(0, self.num_blocks_seen - self.n_local_frames)
            n_local = sum(self.adaptive_block_sizes[st_idx:self.num_blocks_seen])

            # This trim runs once per append() while the window slide runs per block; that gap
            # is fine because the per-block kv_st computation is what enforces the window.
            # Keep the original TOKEN gate here — it is what produced all completed results.
            if self.local_k.size(-2) > self.n_local:
                self.local_k = self.local_k[:, :, -n_local:, :]
                self.local_v = self.local_v[:, :, -n_local:, :]
                if self.visual_mask_buffer is not None:
                    self.visual_mask_buffer = self.visual_mask_buffer[:, -n_local:]
                # NOTE: sliding_window_pos_ids is deliberately *not* trimmed here — it holds the
                # full position history and is indexed by cumulative block offsets (see `sw_st`).
                # Trimming it would silently break that indexing.

        elif self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local:, :]
            self.local_v = self.local_v[:, :, -self.n_local:, :] 
            if self.visual_mask_buffer is not None:
                raise NotImplementedError("Visual mask buffer not implemented for fixed window size")

        # update global remainder
        assert self._global_remainder_ed == self.global_remainder[0].size(-2)
        assert not self.init_exc or self._global_remainder_st == self._global_remainder_ed, f'self.init_exc: {self.init_exc}, global_remainder_st: {self._global_remainder_st}, global_remainder_ed: {self._global_remainder_ed}'
        with torch.cuda.stream(GLOBAL_STREAM):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st:, :],
                self.global_remainder[1][:, :, self._global_remainder_st:, :]
            )
            if self.global_remainder_visual_mask is not None:
                self.global_remainder_visual_mask = self.global_remainder_visual_mask[:, self._global_remainder_st:]

        ret = torch.cat(o_list, dim=-2)
        return ret
    
    def get_all_token_similarity(self, query: torch.Tensor, mode=None):
        """Token-level (mode="qk") or blockwise (mode="kk") similarity over the
        store_full debug cache; see diagnostics.get_all_token_similarity."""
        return diagnostics.get_all_token_similarity(self, query, mode=mode)

    def size(self, *args, **kwargs):
        return self.length

    def calculate_cpu_memory(self):
        memory = 0
        for u in range(self.num_units):
            for block in self.global_blocks[u]:
                memory += block.calculate_cpu_memory()
        return memory
    
    def get_adjacent_key_similarities(self):
        return diagnostics.get_adjacent_key_similarities(self)

    def get_similarity_statistics(self):
        return diagnostics.get_similarity_statistics(self)

    def find_event_boundaries(self, threshold=None, method='percentile', percentile=25):
        return diagnostics.find_event_boundaries(
            self, threshold=threshold, method=method, percentile=percentile)
