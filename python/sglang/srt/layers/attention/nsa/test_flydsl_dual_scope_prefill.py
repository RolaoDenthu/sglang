"""Standalone equivalence harness for the dual-scope prefill scaffold.

Runs on CPU with tiny synthetic inputs (no GPU / FlyDSL required).  Validates
the PyTorch dual-scope reference (the live TEMP path) against an INDEPENDENT
hand-rolled dense masked-softmax oracle built from the pre-quantization float
KV, exercising:
  * the paged-pool byte layout / gather / dequant (fp8 ue8m0 nope + bf16 rope),
  * dual-scope (SWA + extra) accumulation into one softmax,
  * invalid (idx==-1) and padded (beyond topk_length) slot masking,
  * attn_sink folded into the denominator,
  * lonely-query (no valid key) zero rows,
  * output shape / dtype / finiteness.

Run:
    python -m sglang.srt.layers.attention.nsa.test_flydsl_dual_scope_prefill
    python python/sglang/srt/layers/attention/nsa/test_flydsl_dual_scope_prefill.py

``build_synthetic_dual_scope`` is reusable: the future kernel-vs-reference test
will build the same inputs and assert cosine-diff < 3e-2 (see TODO slot below).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.attention.nsa.flydsl_dual_scope_prefill import (
    BYTES_PER_TOKEN_DATA,
    BYTES_PER_TOKEN_SCALE,
    D_NOPE,
    D_QK,
    D_ROPE,
    D_V,
    N_NOPE_TILES,
    QUANT_BLOCK,
    _USE_FLYDSL_KERNEL,
    _flydsl_available,
    _flydsl_dual_scope_kernel_impl,
    _torch_reference_dual_scope,
    flydsl_dual_scope_prefill,
)


@dataclass
class SyntheticDualScope:
    q: torch.Tensor                      # [T, 1, H, 512] bf16
    swa_k_cache: torch.Tensor            # [Nb_m, bs_m, 1, 584] uint8
    swa_indices: torch.Tensor            # [T, 1, topk_m] int32
    swa_topk_length: Optional[torch.Tensor]
    extra_k_cache: Optional[torch.Tensor]
    extra_indices: Optional[torch.Tensor]
    extra_topk_length: Optional[torch.Tensor]
    compress_ratio: int
    softmax_scale: float
    attn_sink: Optional[torch.Tensor]
    head_dim_v: int
    # Independent oracle material: per-global-token dequantized float KV.
    dense_kv_main: torch.Tensor          # [Nb_m * bs_m, 512] f32
    dense_kv_extra: Optional[torch.Tensor]


def _make_scope_pool(num_blocks: int, block_size: int, seed: int, device, dtype_scale_lo=124, dtype_scale_hi=131, value_scale: float = 0.30):
    """Build a raw paged KV pool + the dense (pre-pack) dequantized float KV.

    Layout matches DeepSeekV4SingleKVPool / the Triton reference exactly:
        per block: [block_size * 576 data bytes][block_size * 8 scale bytes]
        per token: nope(448 fp8) ++ rope(128 raw = 64 bf16); scale 7 ue8m0 + pad.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_tok = num_blocks * block_size

    # nope: fp8 e4m3fn is the source of truth (no separate quantization step).
    nope_fp8 = (torch.randn(n_tok, D_NOPE, generator=g) * value_scale).to(torch.float8_e4m3fn)
    nope_u8 = nope_fp8.view(torch.uint8)                       # [n_tok, 448]

    # ue8m0 scale bytes near 127 (exp2(byte-127) ~ O(1)).
    scale_u8 = torch.randint(
        dtype_scale_lo, dtype_scale_hi, (n_tok, N_NOPE_TILES), generator=g, dtype=torch.int32
    ).to(torch.uint8)                                          # [n_tok, 7]

    # rope: bf16 values stored as raw bytes.
    rope_bf16 = (torch.randn(n_tok, D_ROPE, generator=g) * value_scale).to(torch.bfloat16)
    rope_u8 = rope_bf16.view(torch.uint8)                      # [n_tok, 128]

    # ----- dense (pre-pack) dequantized float KV, the independent oracle -----
    scale_f = torch.exp2(scale_u8.to(torch.float32) - 127.0)   # [n_tok, 7]
    nope_dq = (
        nope_fp8.to(torch.float32).reshape(n_tok, N_NOPE_TILES, QUANT_BLOCK)
        * scale_f.unsqueeze(-1)
    ).reshape(n_tok, D_NOPE)
    dense_kv = torch.cat([nope_dq, rope_bf16.to(torch.float32)], dim=-1)  # [n_tok, 512]

    # ----- pack into the data-major raw byte pool -----
    data = torch.cat(
        [nope_u8.reshape(num_blocks, block_size, D_NOPE),
         rope_u8.reshape(num_blocks, block_size, D_ROPE * 2)],
        dim=-1,
    )                                                          # [Nb, bs, 576]
    scale_sec = torch.zeros(num_blocks, block_size, BYTES_PER_TOKEN_SCALE, dtype=torch.uint8)
    scale_sec[..., :N_NOPE_TILES] = scale_u8.reshape(num_blocks, block_size, N_NOPE_TILES)
    pool = torch.cat(
        [data.reshape(num_blocks, block_size * BYTES_PER_TOKEN_DATA),
         scale_sec.reshape(num_blocks, block_size * BYTES_PER_TOKEN_SCALE)],
        dim=1,
    )                                                          # [Nb, bs*584]
    kv_cache = pool.view(
        num_blocks, block_size, 1, BYTES_PER_TOKEN_DATA + BYTES_PER_TOKEN_SCALE
    ).to(device)
    return kv_cache, dense_kv.to(device)


def _make_indices(T: int, topk: int, n_tok: int, seed: int, device):
    """Random valid indices with -1 padding + a per-token topk_length cutoff.

    Guarantees coverage of every masking branch:
      * token 0 is lonely (all -1),
      * token 1 has a valid index sitting beyond its topk_length (must mask),
      * scattered -1 sentinels elsewhere.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, n_tok, (T, topk), generator=g, dtype=torch.int32)
    topk_len = torch.randint(1, topk + 1, (T,), generator=g, dtype=torch.int32)

    # scattered -1 padding
    pad_mask = torch.rand(T, topk, generator=g) < 0.15
    idx[pad_mask] = -1

    # token 0: lonely
    idx[0, :] = -1
    topk_len[0] = topk

    # token 1: valid index beyond the cutoff -> must be masked by topk_length
    cut = max(1, topk // 2)
    topk_len[1] = cut
    idx[1, :cut] = torch.randint(0, n_tok, (cut,), generator=g, dtype=torch.int32)
    idx[1, cut:] = torch.randint(0, n_tok, (topk - cut,), generator=g, dtype=torch.int32)

    return idx.to(device), topk_len.to(device)


def build_synthetic_dual_scope(
    T: int = 6,
    H: int = 4,
    topk_main: int = 64,
    topk_extra: int = 64,
    block_size_main: int = 16,
    block_size_extra: int = 16,
    num_blocks_main: int = 8,
    num_blocks_extra: int = 8,
    compress_ratio: int = 4,
    with_extra: bool = True,
    with_attn_sink: bool = True,
    with_topk_length: bool = True,
    seed: int = 0,
    device: str = "cpu",
    value_scale: float = 0.30,
    scale_lo: int = 124,
    scale_hi: int = 131,
) -> SyntheticDualScope:
    """Build tiny synthetic dual-scope inputs in the real pool layout/dtypes.

    ``value_scale`` scales q / nope / rope magnitudes (controls softmax sharpness:
    larger -> peakier attention), and ``scale_lo/scale_hi`` set the ue8m0
    per-tile exponent spread (wider -> larger per-tile dynamic range, which
    stresses the fp8 dequant + online-softmax accumulation).  Sweeping these
    makes the fidelity check representative of "all types of input" rather than a
    single calm distribution.
    """
    dev = torch.device(device)
    g = torch.Generator(device="cpu").manual_seed(seed + 100)

    q = (torch.randn(T, 1, H, D_QK, generator=g) * value_scale).to(torch.bfloat16).to(dev)
    softmax_scale = 1.0 / math.sqrt(D_QK)

    swa_k_cache, dense_main = _make_scope_pool(
        num_blocks_main, block_size_main, seed + 1, dev,
        dtype_scale_lo=scale_lo, dtype_scale_hi=scale_hi, value_scale=value_scale,
    )
    swa_idx, swa_len = _make_indices(
        T, topk_main, num_blocks_main * block_size_main, seed + 2, dev
    )
    swa_indices = swa_idx.unsqueeze(1)  # [T, 1, topk_main]
    swa_topk_length = swa_len if with_topk_length else None

    if with_extra:
        extra_k_cache, dense_extra = _make_scope_pool(
            num_blocks_extra, block_size_extra, seed + 3, dev,
            dtype_scale_lo=scale_lo, dtype_scale_hi=scale_hi, value_scale=value_scale,
        )
        extra_idx, extra_len = _make_indices(
            T, topk_extra, num_blocks_extra * block_size_extra, seed + 4, dev
        )
        extra_indices = extra_idx.unsqueeze(1)
        extra_topk_length = extra_len if with_topk_length else None
    else:
        extra_k_cache = extra_indices = extra_topk_length = dense_extra = None

    attn_sink = (torch.randn(H, generator=g) * 0.5).to(torch.float32).to(dev) if with_attn_sink else None

    return SyntheticDualScope(
        q=q,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices,
        swa_topk_length=swa_topk_length,
        extra_k_cache=extra_k_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        compress_ratio=compress_ratio,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        head_dim_v=D_V,
        dense_kv_main=dense_main,
        dense_kv_extra=dense_extra,
    )


def _dense_oracle(s: SyntheticDualScope) -> torch.Tensor:
    """Independent hand-rolled dense masked-softmax over the pre-pack float KV.

    Uses ``dense_kv_*`` (gathered by index) instead of the byte-unpacking path,
    so agreement with the reference validates the pool-layout / gather / dequant
    / masking logic end-to-end.
    """
    q3 = s.q.squeeze(1).to(torch.bfloat16).to(torch.float32)  # [T, H, 512]
    T, H, _ = q3.shape

    def scope(dense_kv, indices, topk_len):
        idx = indices.squeeze(1).to(torch.int64)              # [T, topk]
        topk = idx.shape[1]
        valid = idx != -1
        if topk_len is not None:
            col = torch.arange(topk, device=idx.device).unsqueeze(0)
            valid = valid & (col < topk_len.to(torch.int64).unsqueeze(1))
        gathered = dense_kv[idx.clamp(min=0).reshape(-1)].reshape(T, topk, D_QK)
        gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        return gathered, valid

    kv_list, valid_list = [], []
    kv, valid = scope(s.dense_kv_main, s.swa_indices, s.swa_topk_length)
    kv_list.append(kv)
    valid_list.append(valid)
    if s.dense_kv_extra is not None:
        kv, valid = scope(s.dense_kv_extra, s.extra_indices, s.extra_topk_length)
        kv_list.append(kv)
        valid_list.append(valid)

    kv = torch.cat(kv_list, dim=1).to(torch.bfloat16).to(torch.float32)
    valid = torch.cat(valid_list, dim=1)

    scores = torch.einsum("thd,tkd->thk", q3, kv) * float(s.softmax_scale)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    m = scores.amax(dim=-1)
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    p = torch.where(
        torch.isinf(scores), torch.zeros_like(scores),
        torch.exp(scores - m_safe.unsqueeze(-1)),
    )
    l = p.sum(dim=-1)
    acc = torch.einsum("thk,tkd->thd", p, kv)

    if s.attn_sink is not None:
        denom = l + torch.exp(s.attn_sink.to(torch.float32).reshape(1, H) - m_safe)
    else:
        denom = l
    lonely = l == 0.0
    denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)
    out = acc / denom.unsqueeze(-1)
    out = torch.where(lonely.unsqueeze(-1), torch.zeros_like(out), out)
    return out[..., : s.head_dim_v].to(torch.bfloat16)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0
    return float((a @ b) / denom)


def _check(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def run_case(label: str, **kwargs) -> None:
    print(f"\n=== {label} ===")
    s = build_synthetic_dual_scope(**kwargs)

    ref = _torch_reference_dual_scope(
        q=s.q,
        swa_k_cache=s.swa_k_cache,
        swa_indices=s.swa_indices,
        swa_topk_length=s.swa_topk_length,
        extra_k_cache=s.extra_k_cache,
        extra_indices=s.extra_indices,
        extra_topk_length=s.extra_topk_length,
        compress_ratio=s.compress_ratio,
        softmax_scale=s.softmax_scale,
        attn_sink=s.attn_sink,
        head_dim_v=s.head_dim_v,
    )

    T, _, H, _ = s.q.shape
    _check(f"shape == [T={T}, H={H}, {s.head_dim_v}]", tuple(ref.shape) == (T, H, s.head_dim_v))
    _check("dtype == bf16", ref.dtype == torch.bfloat16)
    _check("all finite", bool(torch.isfinite(ref.to(torch.float32)).all()))

    # token 0 is lonely (all-invalid SWA + extra) -> must be an exact zero row
    if s.swa_topk_length is not None:
        _check("lonely token 0 -> zero row", bool((ref[0].to(torch.float32) == 0).all()))

    oracle = _dense_oracle(s)
    cos = _cosine(ref, oracle)
    max_abs = float((ref.to(torch.float32) - oracle.to(torch.float32)).abs().max())
    print(f"  cosine(ref, oracle) = {cos:.6f}   max_abs = {max_abs:.3e}")
    _check("cosine(ref, oracle) > 0.999", cos > 0.999)
    _check("max_abs < 5e-2", max_abs < 5e-2)

    # When the kernel is the live path (_USE_FLYDSL_KERNEL=True) the public entry
    # routes through the GPU FlyDSL kernel, which requires CUDA and is not
    # bit-identical to this CPU reference; that path is validated numerically by
    # run_kernel_launch_case below.  Only assert exact public==reference equality
    # when the reference is the live path (flag False).
    if not _USE_FLYDSL_KERNEL:
        pub = flydsl_dual_scope_prefill(
            q=s.q,
            swa_k_cache=s.swa_k_cache,
            swa_indices=s.swa_indices,
            swa_topk_length=s.swa_topk_length,
            extra_k_cache=s.extra_k_cache,
            extra_indices=s.extra_indices,
            extra_topk_length=s.extra_topk_length,
            compress_ratio=s.compress_ratio,
            softmax_scale=s.softmax_scale,
            attn_sink=s.attn_sink,
            head_dim_v=s.head_dim_v,
        )
        _check("public entry == reference", bool(torch.equal(pub, ref)))


def run_kernel_launch_case(label: str, ref_subset: Optional[int] = None, strict: bool = True, **kwargs) -> None:
    """Phase-1 inner-math test: build synthetic inputs, launch the FlyDSL
    dual-scope kernel, and assert it numerically MATCHES the PyTorch reference
    (cosine > 0.97).  Reports cosine / max-abs / max-rel per case.  Skips
    cleanly when CUDA / FlyDSL (gfx950) is unavailable so the CPU reference
    harness still runs.

    ``ref_subset``: when set, the kernel still runs on the FULL token set
    (cheap: one CTA/token), but the dense PyTorch reference is only built for
    the first ``ref_subset`` query tokens and the kernel output is sliced to the
    same subset for comparison.  This keeps the reference's [T, topk, 512]
    gather tractable for the large-T cases (T=512/1024) without diluting the
    correctness signal.
    """
    print(f"\n=== [kernel-launch] {label} ===")
    if not torch.cuda.is_available():
        print("  [SKIP] no CUDA device (FlyDSL kernel requires gfx950 GPU)")
        return
    if not _flydsl_available():
        print("  [SKIP] FlyDSL / gfx950 not available")
        return

    s = build_synthetic_dual_scope(device="cuda", **kwargs)
    out = _flydsl_dual_scope_kernel_impl(
        q=s.q,
        swa_k_cache=s.swa_k_cache,
        swa_indices=s.swa_indices,
        swa_topk_length=s.swa_topk_length,
        extra_k_cache=s.extra_k_cache,
        extra_indices=s.extra_indices,
        extra_topk_length=s.extra_topk_length,
        compress_ratio=s.compress_ratio,
        softmax_scale=s.softmax_scale,
        attn_sink=s.attn_sink,
        head_dim_v=s.head_dim_v,
    )

    T, _, H, _ = s.q.shape

    # Optionally restrict the dense reference (and the kernel output it is
    # compared against) to the first ``n`` query tokens.  Per-token inputs
    # (q, indices, topk_length) are sliced consistently; the paged pools and
    # attn_sink are shared and stay full.
    if ref_subset is not None and ref_subset < T:
        n = ref_subset
        print(f"  [ref-subset] dense reference on first {n} of T={T} tokens "
              f"(kernel ran full T)")
        ref_q = s.q[:n].contiguous()
        ref_swa_idx = s.swa_indices[:n].contiguous()
        ref_swa_len = s.swa_topk_length[:n].contiguous() if s.swa_topk_length is not None else None
        ref_extra_idx = s.extra_indices[:n].contiguous() if s.extra_indices is not None else None
        ref_extra_len = s.extra_topk_length[:n].contiguous() if s.extra_topk_length is not None else None
        out_cmp = out[:n].contiguous()
    else:
        n = T
        ref_q = s.q
        ref_swa_idx = s.swa_indices
        ref_swa_len = s.swa_topk_length
        ref_extra_idx = s.extra_indices
        ref_extra_len = s.extra_topk_length
        out_cmp = out

    ref = _torch_reference_dual_scope(
        q=ref_q,
        swa_k_cache=s.swa_k_cache,
        swa_indices=ref_swa_idx,
        swa_topk_length=ref_swa_len,
        extra_k_cache=s.extra_k_cache,
        extra_indices=ref_extra_idx,
        extra_topk_length=ref_extra_len,
        compress_ratio=s.compress_ratio,
        softmax_scale=s.softmax_scale,
        attn_sink=s.attn_sink,
        head_dim_v=s.head_dim_v,
    ).to(out.device)

    _check(
        f"kernel out shape == [T={T}, H={H}, {s.head_dim_v}]",
        tuple(out.shape) == (T, H, s.head_dim_v),
    )
    _check("kernel out dtype == bf16", out.dtype == torch.bfloat16)
    _check("kernel out all finite", bool(torch.isfinite(out.to(torch.float32)).all()))

    of = out_cmp.to(torch.float32)
    rf = ref.to(torch.float32)
    cos = _cosine(of, rf)
    max_abs = float((of - rf).abs().max())
    denom = rf.abs().clamp_min(1e-4)
    max_rel = float(((of - rf).abs() / denom).max())
    print(f"  cosine(kernel, ref) = {cos:.6f}   max_abs = {max_abs:.3e}   "
          f"max_rel = {max_rel:.3e}   (compared on {n} tokens)")
    if strict:
        _check("cosine(kernel, ref) > 0.97", cos > 0.97)
    else:
        # Diagnostic mode: report the gate result but do NOT abort, so every
        # regime (C4 / C128-base) prints its cosine even if one is lossy.
        print(f"  [{'PASS' if cos > 0.97 else 'FAIL'}] cosine(kernel, ref) > 0.97 "
              f"(diagnostic, non-fatal)")


# Larger shape exercising grid_h>1 head-tiling and multi-tile runtime scf.for
# loops: H=20 -> h_q_pad=32 -> grid_h=2; pool block_size=64; topk 128 (4 SWA
# tiles) + 256 (8 EXTRA tiles).  Keyword set reused by both the CPU reference
# checks and the on-box kernel-launch checks below.
_LARGE_CASE = dict(
    H=20,
    topk_main=128,
    topk_extra=256,
    block_size_main=64,
    block_size_extra=64,
    num_blocks_main=8,
    num_blocks_extra=8,
    compress_ratio=4,
)

# ---------------------------------------------------------------------------
# C4 production-shape case (compress_ratio=4).  Exercises the C4 extra scope:
# block_size_extra = page_size // 4 = 64, a LARGE extra topk (512 -> 16 runtime
# EXTRA tiles) on top of the SWA main scope (topk 256 -> 8 main tiles), at the
# real model dims H=128 (h_q_pad=128 -> grid_h=8), head_dim_v=512, and the real
# main block_size 256.  Mixed valid / -1 / over-topk_length indices + attn_sink
# are all populated by build_synthetic_dual_scope / _make_indices.
_C4_CASE = dict(
    H=128,
    topk_main=256,
    topk_extra=512,
    block_size_main=256,
    block_size_extra=64,
    num_blocks_main=8,
    num_blocks_extra=64,
    compress_ratio=4,
)

# C128 production-shape case (compress_ratio=128 -> routes to the specialized C128
# kernel via the tuned _C128_PLAN=(128,64,8)).  The C128 regime is the tiny extra
# scope: SWA topk_main=128 + topk_extra=64, block_size_extra = page_size // 128 = 2,
# at the real model dims H=128, head_dim_v=512, real main block_size 256.  This
# ratio-128 path carries the bulk of GSM8K's layers yet had NO kernel-vs-reference
# numerics coverage before (only the C4 path was launch-tested).
_C128_CASE = dict(
    H=128,
    topk_main=128,
    topk_extra=64,
    block_size_main=256,
    block_size_extra=2,
    num_blocks_main=8,
    num_blocks_extra=64,
    compress_ratio=128,
)

# Large-T cases: full sequence length 512 / 1024 with the real H=128 head count
# and dual scope populated, modest topk so the dense reference's [T, topk, 512]
# gather stays tractable.  The kernel runs the full T cheaply (one CTA/token);
# the dense reference is restricted to a token subset (ref_subset) for the
# largest case.
_LARGE_T_512 = dict(
    T=512,
    H=128,
    topk_main=64,
    topk_extra=64,
    block_size_main=256,
    block_size_extra=64,
    num_blocks_main=8,
    num_blocks_extra=64,
    compress_ratio=4,
)
_LARGE_T_1024 = dict(
    T=1024,
    H=128,
    topk_main=64,
    topk_extra=64,
    block_size_main=256,
    block_size_extra=64,
    num_blocks_main=8,
    num_blocks_extra=64,
    compress_ratio=4,
)


def run_fidelity_sweep() -> None:
    """Representative kernel-vs-reference fidelity sweep over all input types.

    For each production regime (C4 + C128/base) sweep value magnitude (softmax
    sharpness) x ue8m0 per-tile scale spread (dynamic range) x seed, run the
    FlyDSL kernel against the PyTorch reference, and print every case's
    cosine / max_abs / max_rel sorted worst-first.  This pinpoints which input
    regime (if any) is lossy enough to explain a GSM8K accuracy drop, rather
    than relying on one calm synthetic distribution.
    """
    print("\n" + "=" * 78)
    print("REPRESENTATIVE FIDELITY SWEEP (kernel vs reference, all input types)")
    print("=" * 78)
    if not torch.cuda.is_available() or not _flydsl_available():
        print("  [SKIP] FlyDSL / gfx950 GPU not available")
        return

    regimes = {"C4": _C4_CASE, "C128": _C128_CASE}
    # value_scale: 0.30 ~ near-uniform softmax, 1.0 moderate, 3.0 peaky attention.
    value_scales = [0.30, 1.0, 3.0]
    # ue8m0 exponent byte spread: narrow ~O(1) vs wide ~2^-9..2^10 dynamic range.
    scale_spreads = [(124, 131), (118, 137)]
    seeds = [0, 7]

    rows = []
    for rname, base in regimes.items():
        for vs in value_scales:
            for (lo, hi) in scale_spreads:
                for sd in seeds:
                    kw = dict(base)
                    kw.update(value_scale=vs, scale_lo=lo, scale_hi=hi, seed=sd)
                    s = build_synthetic_dual_scope(device="cuda", **kw)
                    out = _flydsl_dual_scope_kernel_impl(
                        q=s.q,
                        swa_k_cache=s.swa_k_cache,
                        swa_indices=s.swa_indices,
                        swa_topk_length=s.swa_topk_length,
                        extra_k_cache=s.extra_k_cache,
                        extra_indices=s.extra_indices,
                        extra_topk_length=s.extra_topk_length,
                        compress_ratio=s.compress_ratio,
                        softmax_scale=s.softmax_scale,
                        attn_sink=s.attn_sink,
                        head_dim_v=s.head_dim_v,
                    )
                    ref = _torch_reference_dual_scope(
                        q=s.q,
                        swa_k_cache=s.swa_k_cache,
                        swa_indices=s.swa_indices,
                        swa_topk_length=s.swa_topk_length,
                        extra_k_cache=s.extra_k_cache,
                        extra_indices=s.extra_indices,
                        extra_topk_length=s.extra_topk_length,
                        compress_ratio=s.compress_ratio,
                        softmax_scale=s.softmax_scale,
                        attn_sink=s.attn_sink,
                        head_dim_v=s.head_dim_v,
                    ).to(out.device)
                    of = out.to(torch.float32)
                    rf = ref.to(torch.float32)
                    cos = _cosine(of, rf)
                    max_abs = float((of - rf).abs().max())
                    denom = rf.abs().clamp_min(1e-4)
                    max_rel = float(((of - rf).abs() / denom).max())
                    rows.append((cos, max_abs, max_rel, rname, vs, (lo, hi), sd))

    rows.sort(key=lambda r: r[0])  # worst cosine first
    print(f"\n{'regime':<6} {'vscale':>7} {'spread':>11} {'seed':>5} "
          f"{'cosine':>10} {'max_abs':>11} {'max_rel':>11}")
    print("-" * 72)
    for cos, max_abs, max_rel, rname, vs, spread, sd in rows:
        flag = "  <-- LOSSY" if cos < 0.999 else ""
        print(f"{rname:<6} {vs:>7.2f} {str(spread):>11} {sd:>5} "
              f"{cos:>10.6f} {max_abs:>11.3e} {max_rel:>11.3e}{flag}")
    worst = rows[0]
    print(f"\nWORST cosine = {worst[0]:.6f}  (regime={worst[3]}, value_scale={worst[4]}, "
          f"scale_spread={worst[5]}, seed={worst[6]})")
    print("If cosine ~1.0 across ALL rows the kernel is faithful on synthetic inputs, "
          "and the GSM8K drop points to real-data distribution rather than kernel math.")


def main() -> None:
    torch.manual_seed(0)
    run_case("dual scope (SWA + C4 extra), sink + topk_length", compress_ratio=4)
    run_case("dual scope, no attn_sink", with_attn_sink=False)
    run_case("dual scope, no topk_length", with_topk_length=False)
    run_case("SWA only (no extra scope)", with_extra=False)
    run_case("C128 extra (compress_ratio=128)", compress_ratio=128)
    run_case("LARGE: H=20 (grid_h>1), bs=64, multi-tile (4 SWA + 8 EXTRA)", **_LARGE_CASE)
    # C4 production-shape reference-vs-oracle check (small T=6 keeps the oracle
    # cheap; the C4-distinguishing knobs are block_size_extra=64 + large extra
    # topk + compress_ratio=4, all exercised here).
    run_case("C4 prod-shape: H=128, bs_m=256, bs_e=64, topk 256+512", **_C4_CASE)
    print("\nAll dual-scope reference equivalence checks passed.")

    # Compile-first FlyDSL kernel launch tests (skip cleanly off-box).
    run_kernel_launch_case("dual scope (SWA + extra)", compress_ratio=4)
    run_kernel_launch_case("SWA only (no extra scope)", with_extra=False)
    run_kernel_launch_case("no attn_sink / no topk_length", with_attn_sink=False, with_topk_length=False)
    run_kernel_launch_case(
        "LARGE: H=20 (grid_h>1), bs=64, multi-tile (4 SWA + 8 EXTRA)", **_LARGE_CASE
    )
    # NEW axes (not previously covered): C4 production shape + large T.
    run_kernel_launch_case(
        "C4 prod-shape: H=128 (grid_h=8), bs_m=256, bs_e=64, topk 256+512 "
        "(8 SWA + 16 EXTRA tiles)",
        strict=False,
        **_C4_CASE,
    )
    run_kernel_launch_case(
        "C128 prod-shape: H=128, ratio=128 -> BASE kernel, topk 128+64, bs_e=2 "
        "(GSM8K bulk-layer path -- previously untested)",
        strict=False,
        **_C128_CASE,
    )
    run_kernel_launch_case(
        "LARGE-T=512: H=128, dual scope (kernel full T, ref full T)",
        **_LARGE_T_512,
    )
    run_kernel_launch_case(
        "LARGE-T=1024: H=128, dual scope (kernel full T, ref subset=256)",
        ref_subset=256,
        **_LARGE_T_1024,
    )

    # Representative fidelity sweep across input distributions for both regimes.
    run_fidelity_sweep()


if __name__ == "__main__":
    main()
