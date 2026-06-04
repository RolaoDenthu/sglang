"""Fused dual-scope sparse-attention prefill for DeepSeek V4 / NSA (gfx950).

This module replaces the broken two-kernel + Python-LSE-merge prefill path with
a SINGLE fused dual-scope kernel that attends over both the sliding-window (SWA,
"main") scope and the compressed (C4 / C128, "extra") scope in one shared
online-softmax pass.  It is the FlyDSL analogue of the authoritative Triton
reference ``_fused_gather_attn_dsv4_dual_scope_kernel`` in
``nsa/triton_decode/triton_mla_kernels_decode_fused.py``.

Components:
  * ``flydsl_dual_scope_prefill``      -- public entry matching the backend hook;
                                          the live prefill path on gfx950.
  * ``_build_dual_scope_kernel``       -- the FlyDSL kernel: gather + e4m3fn fp8
                                          dequant + QK + exp2 online softmax + PV
                                          over both scopes in one fused pass.
  * ``_flydsl_dual_scope_kernel_impl`` -- thin launcher that normalizes inputs,
                                          builds/caches the kernel, and launches it.
  * ``_torch_reference_dual_scope``    -- faithful pure-PyTorch correctness ORACLE
                                          used by the tests (NOT a live fallback).

The FlyDSL kernel is the live path (``_USE_FLYDSL_KERNEL = True``) and matches
both the PyTorch oracle and the production Triton kernel on synthetic, scaled,
and real captured tensors.  The FlyDSL import is lazy/guarded so this module
still imports cleanly on a CPU host where FlyDSL (a remote-container dependency)
is absent; the PyTorch oracle and the test harness run without FlyDSL present.

----------------------------------------------------------------------------
Paged KV pool byte layout (per block of ``block_size`` token slots), mirrored
from ``DeepSeekV4SingleKVPool.get_bytes_per_token`` and the Triton kernel:

    bytes [0 .. block_size*576)                : token DATA section (data-major)
        per slot s (0 .. block_size-1) -> 576 bytes at [s*576 .. (s+1)*576):
            nope : bytes [s*576       .. s*576+448)  -> 448 fp8 (e4m3) values
            rope : bytes [s*576+448   .. s*576+576)  -> 64 bf16 values (128 bytes, LE)
    bytes [block_size*576 .. block_size*584)   : token SCALE section
        per slot s -> 8 bytes at [block_size*576 + s*8 .. +8):
            7 x ue8m0 scale bytes (one per 64-dim nope tile) + 1 byte pad

Dequant (per gathered key token):
    nope_f32[d] = fp8_to_f32(nope_byte[d]) * exp2(scale_byte[d // 64] - 127)
    rope_f32[d] = bf16_to_f32(rope value d)
    kv[0:512]   = concat(nope_f32[0:448], rope_f32[0:64])          # K == V (MLA)
----------------------------------------------------------------------------
"""


import functools
import math
import os
import time
from typing import Optional, Tuple

import torch

# ----------------------------------------------------------------------------
# Diagnostic timing.  Gated on SGLANG_FLYDSL_TIMING=1 -- when enabled, every
# prefill call appends a per-call attribution line to
# SGLANG_FLYDSL_TIMING_FILE (default /tmp/flydsl_timing.log):
#   host-prep GPU time (allocs/pads/copies) | JIT build/recompile (host, MLIR)
#   | pure kernel exec | post (out slice/contiguous).  This is the definitive
# way to split a TTFT regression into "host overhead" vs "kernel recompiles"
# vs "kernel is genuinely slow", given rocprofv3 corrupts CUDA-graph dispatch
# and torch-profiler traces get truncated on server kill.
# ----------------------------------------------------------------------------
_FLYDSL_TIMING: bool = os.environ.get("SGLANG_FLYDSL_TIMING", "0") == "1"
_FLYDSL_TIMING_FILE: str = os.environ.get(
    "SGLANG_FLYDSL_TIMING_FILE", "/tmp/flydsl_timing.log"
)

# ----------------------------------------------------------------------------
# FlyDSL is a remote-container (gfx950) dependency.  Import its API surface at
# MODULE level -- guarded so the module still imports on a CPU host where FlyDSL
# is absent (the PyTorch reference path + CPU harness need no FlyDSL).
#
# IMPORTANT: these MUST be module globals (not imported inside the kernel
# builder).  When the FlyDSL names live in the builder's local scope they become
# closure free-vars of the @flyc.kernel function (and its nested _as_ptr helper),
# which trips FlyDSL's ASTRewriter freevar-count check
# ("requires N free vars, not N-1").  The known-good artefact kernels
# (flydsl_nsa_prefill.py / flydsl_swa_prefill.py) import at module level for
# exactly this reason; we mirror that import surface here.
# ----------------------------------------------------------------------------
try:  # pragma: no cover - exercised only on the gfx950 box
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import (
        arith,
        buffer_ops,
        const_expr,
        gpu,
        range_constexpr,
        rocdl,
    )
    from flydsl.expr.typing import T, Vector as Vec
    from flydsl.expr.utils.arith import _to_raw as _raw
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import (
        arith as _mlir_arith,
        fly as _fly,
        llvm as _llvm,
        memref as _memref,
        scf as _scf,
    )

    _HAS_FLYDSL = True
except Exception:  # FlyDSL absent (CPU host): reference path + harness still run
    _HAS_FLYDSL = False

# ----------------------------------------------------------------------------
# Layout / dim constants (must stay in sync with the pool + Triton reference).
# ----------------------------------------------------------------------------
D_QK: int = 512                 # query / key head dim (448 nope + 64 rope)
D_V: int = 512                  # value (output) head dim
D_NOPE: int = 448               # fp8 nope dims
D_ROPE: int = 64                # bf16 rope dims
QUANT_BLOCK: int = 64           # nope dims per ue8m0 scale tile
N_NOPE_TILES: int = D_NOPE // QUANT_BLOCK            # 7
BYTES_PER_TOKEN_DATA: int = D_NOPE + D_ROPE * 2      # 448 + 128 = 576
BYTES_PER_TOKEN_SCALE: int = 8                       # 7 ue8m0 + 1 pad

_LOG2E: float = 1.4426950408889634


# Internal flag: when True (the production default) the public entry routes
# prefill through the live FlyDSL dual-scope kernel
# (_flydsl_dual_scope_kernel_impl). When False the public entry uses the
# pure-PyTorch oracle (_torch_reference_dual_scope) -- this branch exists for
# the tests/oracle path only, not as a runtime fallback.
_USE_FLYDSL_KERNEL: bool = True

# ----------------------------------------------------------------------------
# Host-prep buffer cache.  The launcher feeds the kernel several CONSTANT dummy
# tensors when an optional input is absent (all-(-1) extra indices, full-length
# topk_length, zero attn_sink).  These are read-only kernel inputs whose content
# depends only on (shape, fill-value, dtype, device), so we cache and reuse them
# across calls instead of rebuilding with ``torch.full`` / ``torch.zeros`` every
# prefill call.  Reuse is launch-order-safe: all launches share one CUDA stream,
# so a reused buffer's reads are serialized after the prior launch that read it.
# ----------------------------------------------------------------------------
_DUMMY_CACHE: dict = {}


def _cached_full(shape, fill_value, dtype, device) -> torch.Tensor:
    key = (shape, int(fill_value), dtype, str(device))
    t = _DUMMY_CACHE.get(key)
    if t is None:
        t = torch.full(shape, fill_value, dtype=dtype, device=device)
        _DUMMY_CACHE[key] = t
    return t


def _cached_zeros(shape, dtype, device) -> torch.Tensor:
    key = ("zeros", shape, dtype, str(device))
    t = _DUMMY_CACHE.get(key)
    if t is None:
        t = torch.zeros(shape, dtype=dtype, device=device)
        _DUMMY_CACHE[key] = t
    return t


# C128 / low-topk routing: the C128 regime is a TINY 192-key problem
# (topk_main=128 + topk_extra=64, bs_main=256, bs_extra=2, H=128, head_dim=512).
# An earlier gsm8k-only autotune (8-shot/200q) had picked the BASE kernel for
# ratio==128, but that comparison predated the PV transpose-read (ds_read_b64_tr_b16)
# and conflated the tiny C128 slice with the decode-bound gsm8k total.  A fresh,
# direct C128-regime microbenchmark + gsm8k re-measurement (8-shot/400q,
# DeepSeek-V4-Pro FP8, tp8) overturned it: the specialized (128,64,8) C128 kernel
# with the transpose-read is faster than base across the ENTIRE T range with NO
# accuracy cost:
#   per-call kernel (HIP events): spec+DS_TR vs base ->
#     T=6  1.23x | T=190 2.01x | T=680 2.47x | T=2048 2.47x | T=8192 2.57x
#   gsm8k end-to-end: base acc 0.940 / 24.24 s ; spec+DS_TR acc 0.943 / 23.45 s
#     (cosine vs torch ref = 0.99999750 at every T)
# The 1-CTA/token BLOCK_H=128 tile dequants the fp8 KV ONCE per token (vs the base
# kernel's BLOCK_H=16 = 8 CTAs/token, each redundantly dequanting the same KV), and
# the transpose-read collapses the PV V-operand LDS traffic ~2.5x.
# => default plan is the specialized (128,64,8); route ratio==128 there.  The base
# kernel remains the fallback for all other ratios and is still reachable for
# ratio==128 via the "base" env override below (autotuning / regression bisects).
_C128_BLOCK_H: int = 128
_C128_BLOCK_N: int = 64
_C128_N_WAVES: int = 8


def _c128_plan():
    """C128 tiling plan: ``(block_h, block_n, n_waves)`` for the specialized
    kernel, or ``None`` to route compress_ratio==128 to the BASE kernel.

    The tuned DEFAULT is the specialized ``(128, 64, 8)`` (it beats base across the
    whole T range with the transpose-read on -- see the comment above).  Overridable
    via env ``SGLANG_C128_TILING``:
        * "base"            -> route ratio==128 to the base kernel
        * "BH,BN,NW"        -> specialized C128 with that tiling
        * unset/invalid     -> the tuned default (specialized 128,64,8)
    Parsed once at import (a server restart re-reads it).
    """
    import os as _os

    raw = _os.environ.get("SGLANG_C128_TILING", "").strip().lower()
    if raw == "base":
        return None
    if raw:
        parts = raw.split(",")
        if len(parts) == 3:
            try:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                pass
    return (_C128_BLOCK_H, _C128_BLOCK_N, _C128_N_WAVES)


_C128_PLAN = _c128_plan()

# C4 / high-topk routing: the C4 regime (compress_ratio==4, the dominant
# high-topk prefill: topk_main=128 + topk_extra=512, bs_main=256, bs_extra=64,
# H=128, head_dim=512) used to launch the FIXED _build_dual_scope_kernel_c4 at
# BLOCK_N=64 / BLOCK_H=128 / 4 waves -- 1 CTA/token staging a 96KB LDS tile
# (KV 64KB bf16 + P 32KB f32).  Two 96KB CTAs (192KB) exceed the 160KB gfx950
# (MI350X) group-segment budget, so only 1 CTA co-resides per CU: the
# rocdl.waves_per_eu desired==2 hint resolves to final==1 (occupancy 1).
#
# Halving BLOCK_N (64 -> 32) halves the LDS to 48KB and lifts per-CU occupancy
# from 1 to 2-3 -- but it ran SLOWER end-to-end: rocprof showed the kernel is
# stall/VGPR-spill bound (PV f32 O-accumulator -> VGPR256 + spill, ~74% cycles
# stalled, MfmaUtil ~5%), NOT LDS-occupancy bound, so more co-resident CTAs did
# not help and the extra KV tiles/barriers cost more.  The bn32 route is
# therefore REVERTED (default _c4_plan() -> None -> legacy fixed BLOCK_N=64).
# Profiling the BLOCK_N=64 baseline (rocprofv3 counters) shows it is NOT
# MFMA-bound (MfmaUtil ~6%) and NOT VMEM-bound (MemUnitStalled ~0%): a native
# fp8 QK/PV MFMA path was prototyped and then removed because halving matmul
# cost can recover at most a few % of wall time.  The dominant costs are LDS
# bank conflicts (~39%) and barrier/latency stalls at occupancy 1, so the real
# levers are LDS layout (bank-conflict removal) and occupancy.  The
# parameterized builder (_build_dual_scope_kernel_c128) and the _C4_* knobs
# below remain available for A/B via SGLANG_C4_TILING but are no longer default.
_C4_BLOCK_H: int = 128
_C4_BLOCK_N: int = 32
_C4_N_WAVES: int = 4
_C4_WAVES_PER_EU: int = 2


def _c4_plan():
    """C4 tiling plan for compress_ratio==4: ``(block_h, block_n, n_waves,
    waves_per_eu)`` to route through the parameterized C128-style builder, or
    ``None`` to use the LEGACY fixed ``_build_dual_scope_kernel_c4`` (BLOCK_N=64).

    The DEFAULT is ``None`` -- the LEGACY fixed BLOCK_N=64 C4 kernel.  The
    BLOCK_N=32 occupancy-2 experiment was reverted: halving BLOCK_N doubled
    per-CU occupancy but ran SLOWER end-to-end (the kernel is stall/VGPR-spill
    bound, not LDS-occupancy bound), so BLOCK_N=64 is the baseline the fp8-MFMA
    work builds on.  Still overridable for A/B and autotuning via env
    ``SGLANG_C4_TILING``:
        * "legacy"          -> the fixed BLOCK_N=64 C4 kernel (default)
        * "BH,BN,NW"        -> parameterized C4 with that tiling (waves_per_eu=2)
        * "BH,BN,NW,WPE"    -> parameterized C4 with explicit waves_per_eu
        * unset/invalid     -> the legacy fixed BLOCK_N=64 C4 kernel
    Parsed once at import (a server restart re-reads it).
    """
    import os as _os

    raw = _os.environ.get("SGLANG_C4_TILING", "").strip().lower()
    if raw == "legacy":
        return None
    if raw:
        parts = raw.split(",")
        if len(parts) in (3, 4):
            try:
                bh, bn, nw = int(parts[0]), int(parts[1]), int(parts[2])
                wpe = int(parts[3]) if len(parts) == 4 else _C4_WAVES_PER_EU
                return (bh, bn, nw, wpe)
            except ValueError:
                pass
    return None


_C4_PLAN = _c4_plan()

# High-concurrency routing for C4 kernel: when T_tok >= this threshold, use
# BLOCK_H=64 (2 CTAs/token, occupancy=2, LDS=80KB) instead of BLOCK_H=128
# (1 CTA/token, occupancy=1, LDS=96KB).  At high T, SM is already full from
# the token dimension, so trading 1 CTA/token for occupancy=2 hides barrier
# stalls behind a second resident CTA.  Tune via SGLANG_C4_HIGH_CONC_THRESH.
_C4_HIGH_CONC_THRESH: int = int(
    _os_ldspad.environ.get("SGLANG_C4_HIGH_CONC_THRESH", "").strip() or "256"
)
# BLOCK_H=64 tiling for high-concurrency C4: (block_h, block_n, n_waves, waves_per_eu)
_C4_HIGH_CONC_PLAN: tuple = (64, 64, 4, 2)

# LDS bank-conflict padding (per-row stride pad, in elements) for the C4/C128
# dual-scope kernels.  The lds_kv ([key][dim] bf16) and lds_p ([head_row][key]
# f32) strides must avoid being multiples of the 32 LDS banks; these pads are
# the tuned defaults (see the LDS layout comment in _build_dual_scope_kernel_c4)
# and are overridable via env for autotuning sweeps.
#
# NOTE (tuning result): driving SQ_LDS_BANK_CONFLICT to ~0 is possible (e.g.
# KV pad 12 -> 0.2%) but is a ~10% perf REGRESSION: the residual ~18% conflict
# is the PV V-read's two-k-group 2-way collision, and removing it needs a bf16
# stride not divisible by 8, which breaks the 16-byte row alignment the QK
# ds_read_b128 vector loads depend on.  The conflict is overlapped behind the
# occupancy-1 latency/barrier stalls (the true bottleneck), so we keep the pad
# that is fastest end-to-end (16-byte aligned), not the one with lowest conflict.
import os as _os_ldspad
_C4_KV_PAD: int = int(_os_ldspad.environ.get("SGLANG_C4_KV_PAD", "").strip() or "16")
_C4_P_PAD: int = int(_os_ldspad.environ.get("SGLANG_C4_P_PAD", "").strip() or "4")

# gfx950 only: route the high-frequency PV V-operand read through the CDNA4 LDS
# transpose-read intrinsic (ds_read_b64_tr_b16) instead of 8 strided scalar bf16
# loads per (ks_pv, d) tile.  Each 16x16 transpose read returns the 4 contraction
# keys at the lane's head already laid out as the MFMA-B operand, so 2 reads
# replace 8 scalar loads (~2.5x less PV LDS traffic).  Verified bit-exact vs the
# scalar path for all (ks_pv, d, stride/pad) in standalone spikes, and a measured
# end-to-end win on BOTH the C4 and C128 regimes (gsm8k acc unchanged, faster).
# => ON by default; disable for A/B or regression bisects with SGLANG_C4_DS_TR=0.
_C4_DS_TR: bool = (
    _os_ldspad.environ.get("SGLANG_C4_DS_TR", "").strip().lower()
    not in ("0", "false", "no", "off")
)

# Intra-wave software pipelining of the PV-GEMM V-operand LDS reads.  At
# occupancy-1 the CTA's 4 waves land 1-per-SIMD, so a stalled LDS read idles the
# whole SIMD (no second wave to switch to).  The only latency-hiding lever left
# is per-wave ILP: emit a BATCH of _C4_PV_PIPE independent V-reads (into distinct
# registers) before their consuming MFMAs, so several reads are in flight during
# the ~20-40cy LDS round-trips instead of one-at-a-time read->dependent-MFMA.
# This is a pure emission-order reorder (per-accumulator MFMA order is unchanged)
# so it is bit-exact vs the unpipelined path (verified: max_abs_diff==0).
#
# NOTE (tuning result -- kept OFF): this does NOT help and is a documented
# dead-end.  The C4 kernel is REGISTER-bound, not schedule-bound: VGPR is pinned
# at the 256 arch cap (the f32 O-accumulator alone is D_BLKS*4 = 128 VGPR and is
# live across the whole tile), so there is no headroom to hold loads in flight --
# every prefetched fragment spills to scratch (488B -> 512-624B) and the spill
# costs more than the LDS latency it hides.  Measured (T=2048, gfx950, DS_TR on):
# PV_PIPE off 1.905ms ; W=2 2.111ms (+11%) ; W=4 1.899ms (~neutral) ; W=8 2.078ms
# (+9%).  Hiding LDS latency here is gated on first shrinking the O-accumulator's
# register footprint (split-D multi-pass / LDS-resident acc), not on pipelining.
# OFF by default; A/B via env.  SGLANG_C4_PV_PIPE = PV d-loop batch width (def 4).
_C4_PIPE: bool = (
    _os_ldspad.environ.get("SGLANG_C4_PIPE", "").strip().lower()
    in ("1", "true", "yes", "on")
)
_C4_PV_PIPE: int = int(_os_ldspad.environ.get("SGLANG_C4_PV_PIPE", "").strip() or "4")

# D4 (deferred / lazy softmax O-rescale).  The online-softmax loop rescales the
# 512-wide fp32 O-accumulator by corr = exp2(m_old - m_new) on every KV tile
# (D_BLKS x vector<4xf32> = 128 VGPR worth of fmuls, on the PV-MFMA dependency
# chain).  On tiles where no output row's running max advances this tile,
# m_new == m_old so corr == 1.0 and the rescale is an identity multiply.  D4
# skips it: a wave-uniform predicate (any owned row's max strictly increased) is
# reduced across the 64-lane wave and the rescale is emitted under scf.if, so the
# fmuls are genuinely branched-over (not exec-masked) when no max advances.
#
# Bit-exact vs the unconditional path: the skip branch is taken only when the
# wave-wide max delta is <= 0, i.e. corr == 1.0 for every row, so o*corr == o.
# When the predicate is true the original (corr possibly < 1) rescale runs
# verbatim.  m_new / l_new / p are computed identically in both cases.
# ON by default (bit-exact, latency-neutral-to-faster); disable via env
# SGLANG_C4_LAZY_RESCALE in {0,false,no,off}.
_C4_LAZY_RESCALE: bool = (
    _os_ldspad.environ.get("SGLANG_C4_LAZY_RESCALE", "1").strip().lower()
    not in ("0", "false", "no", "off")
)


# ============================================================================
# Shape / dtype normalization helpers
# ============================================================================
def _normalize_q(q: torch.Tensor) -> torch.Tensor:
    """Backend passes q as [T, 1, H, D_QK] (after unsqueeze(1)) or [T, H, D_QK].

    Returns a contiguous [T, H, D_QK] tensor.
    """
    if q.ndim == 4:
        # [T, s_q=1, H, D] -> squeeze the singleton seq dim.
        assert q.shape[1] == 1, f"unexpected q seq dim: {q.shape}"
        q = q.squeeze(1)
    assert q.ndim == 3, f"q must be 3D after normalize, got {q.shape}"
    assert q.shape[-1] == D_QK, f"q head dim {q.shape[-1]} != {D_QK}"
    return q.contiguous()


def _normalize_indices(indices: torch.Tensor) -> torch.Tensor:
    """Backend passes indices as [T, 1, topk] (after unsqueeze(1)) or [T, topk]."""
    if indices.ndim == 3:
        assert indices.shape[1] == 1, f"unexpected indices seq dim: {indices.shape}"
        indices = indices.squeeze(1)
    assert indices.ndim == 2, f"indices must be 2D after normalize, got {indices.shape}"
    return indices.contiguous()


def _block_layout(kv_cache: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Reinterpret a paged KV cache as a flat data-major uint8 tensor.

    ``kv_cache`` is the 4D view the backend builds:
        [num_blocks, block_size, 1, kv_cache_total_dim]  (uint8, total_dim == 584)
    The backend already sliced off page padding before viewing, so the per-block
    byte run is exactly ``block_size * 584`` and data-major.

    Returns ``(u8_flat[num_blocks, block_size*584], num_blocks, block_size)``.
    """
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    u8 = kv_cache.contiguous().view(torch.uint8).reshape(num_blocks, -1)
    per_block_bytes = u8.shape[1]
    expected = block_size * (BYTES_PER_TOKEN_DATA + BYTES_PER_TOKEN_SCALE)
    assert per_block_bytes == expected, (
        f"per-block bytes {per_block_bytes} != expected {expected} "
        f"(block_size={block_size}); pool layout assumption violated"
    )
    return u8, num_blocks, block_size


# ============================================================================
# Pure-PyTorch dequant of a gathered scope (byte-unpacking path == oracle)
# ============================================================================
def _dequant_scope(
    kv_cache: torch.Tensor,
    indices: torch.Tensor,          # [T, topk] int
    topk_length: Optional[torch.Tensor],  # [batch] int, or None
    s_q: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather + dequant one scope into dense bf16 KV and a validity mask.

    Returns:
        kv:    [T, topk, D_QK] float32  (dequantized key == value; invalid -> 0)
        valid: [T, topk]       bool     (False for idx==-1 or beyond topk_length)
    """
    device = kv_cache.device
    T, topk = indices.shape
    u8, num_blocks, block_size = _block_layout(kv_cache)
    u8_flat = u8.reshape(-1)  # [num_blocks * per_block_bytes]
    per_block_bytes = u8.shape[1]

    idx = indices.to(torch.int64)

    # Validity: idx == -1 sentinel, plus optional per-batch topk_length cutoff.
    valid = idx != -1
    if topk_length is not None:
        n_batch = topk_length.shape[0]
        sq = max(1, s_q)
        batch_of_t = torch.div(
            torch.arange(T, device=device), sq, rounding_mode="floor"
        ).clamp_(max=n_batch - 1)
        # per-token cutoff length, broadcast over the topk axis
        cutoff = topk_length.to(torch.int64)[batch_of_t]  # [T]
        col = torch.arange(topk, device=device).unsqueeze(0)  # [1, topk]
        valid = valid & (col < cutoff.unsqueeze(1))

    idx_c = idx.clamp_(min=0)
    block = torch.div(idx_c, block_size, rounding_mode="floor")  # [T, topk]
    off = idx_c - block * block_size

    # Absolute byte base of each gathered token's DATA and SCALE sections.
    block_base = block * per_block_bytes
    data_base = block_base + off * BYTES_PER_TOKEN_DATA              # [T, topk]
    scale_base = block_base + block_size * BYTES_PER_TOKEN_DATA + off * BYTES_PER_TOKEN_SCALE

    # --- nope: 448 fp8 bytes -> float32, scaled per 64-dim tile -----------------
    nope_off = data_base.unsqueeze(-1) + torch.arange(D_NOPE, device=device)  # [T,topk,448]
    nope_u8 = u8_flat[nope_off.reshape(-1)].reshape(T, topk, D_NOPE)
    nope_f = nope_u8.view(torch.float8_e4m3fn).to(torch.float32)

    scale_idx = scale_base.unsqueeze(-1) + torch.arange(N_NOPE_TILES, device=device)
    scale_u8 = u8_flat[scale_idx.reshape(-1)].reshape(T, topk, N_NOPE_TILES)
    scale_f = torch.exp2(scale_u8.to(torch.float32) - 127.0)  # ue8m0 -> f32
    # broadcast each tile scale over its 64 dims
    nope_dq = (
        nope_f.reshape(T, topk, N_NOPE_TILES, QUANT_BLOCK)
        * scale_f.unsqueeze(-1)
    ).reshape(T, topk, D_NOPE)

    # --- rope: 128 raw bytes -> 64 bf16 -> float32 -----------------------------
    rope_off = (
        data_base.unsqueeze(-1) + D_NOPE + torch.arange(D_ROPE * 2, device=device)
    )
    rope_u8 = u8_flat[rope_off.reshape(-1)].reshape(T, topk, D_ROPE * 2)
    rope_f = rope_u8.view(torch.bfloat16).to(torch.float32).reshape(T, topk, D_ROPE)

    kv = torch.cat([nope_dq, rope_f], dim=-1)  # [T, topk, 512]
    kv = torch.where(valid.unsqueeze(-1), kv, torch.zeros_like(kv))
    return kv, valid


# ============================================================================
# Faithful pure-PyTorch dual-scope reference (the correctness oracle)
# ============================================================================
def _torch_reference_dual_scope(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_length: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    compress_ratio: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
) -> torch.Tensor:
    """Dense-equivalent of the Triton dual-scope online-softmax kernel.

    Mirrors ``_fused_gather_attn_dsv4_dual_scope_kernel`` exactly (numerically
    identical up to bf16-matmul rounding):

      * gather + dequant SWA (main) and C4/C128 (extra) keys from the paged pool,
      * QK with K == V (MLA latent), score *= softmax_scale,
      * mask invalid / padded slots (idx==-1 or beyond topk_length) to -inf
        BEFORE softmax so zero-padding never pollutes the denominator,
      * one shared softmax over [SWA ++ extra] with the attn_sink folded into the
        denominator (denom = sum exp(score - m) + exp(attn_sink - m)),
      * lonely queries (no valid key) emit a zero row,
      * output is the probability-weighted sum of V, cast to bf16.

    The Triton kernel does this in log2 space with exp2 and a running (m, l, acc)
    state processed SWA-first then extra; an online softmax over a fixed key set
    is mathematically identical to the dense masked softmax computed here, which
    is easier to read and audit.  Returns [T, H, head_dim_v] bf16.
    """
    q3 = _normalize_q(q)                       # [T, H, 512]
    device = q3.device
    T, H, _ = q3.shape
    qf = q3.to(torch.bfloat16).to(torch.float32)

    swa_idx = _normalize_indices(swa_indices)  # [T, topk_m]
    s_q_main = max(1, T // swa_topk_length.shape[0]) if swa_topk_length is not None else 1
    kv_main, valid_main = _dequant_scope(
        swa_k_cache, swa_idx, swa_topk_length, s_q_main
    )

    kv_list = [kv_main]
    valid_list = [valid_main]

    if extra_k_cache is not None and extra_indices is not None:
        extra_idx = _normalize_indices(extra_indices)
        s_q_extra = (
            max(1, T // extra_topk_length.shape[0])
            if extra_topk_length is not None
            else 1
        )
        kv_extra, valid_extra = _dequant_scope(
            extra_k_cache, extra_idx, extra_topk_length, s_q_extra
        )
        kv_list.append(kv_extra)
        valid_list.append(valid_extra)

    kv = torch.cat(kv_list, dim=1)        # [T, K, 512]
    valid = torch.cat(valid_list, dim=1)  # [T, K]

    # QK^T in bf16 (mirrors the bf16 MFMA path), accumulate in float32.
    kv_bf = kv.to(torch.bfloat16).to(torch.float32)
    scores = torch.einsum("thd,tkd->thk", qf, kv_bf) * float(softmax_scale)  # [T,H,K]

    neg_inf = float("-inf")
    valid_h = valid.unsqueeze(1)  # [T, 1, K]
    scores = torch.where(valid_h, scores, torch.full_like(scores, neg_inf))

    m = scores.amax(dim=-1)                          # [T, H]
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    p = torch.where(
        torch.isinf(scores), torch.zeros_like(scores),
        torch.exp(scores - m_safe.unsqueeze(-1)),
    )                                                # [T, H, K]
    l = p.sum(dim=-1)                                # [T, H]
    acc = torch.einsum("thk,tkd->thd", p, kv_bf)     # [T, H, 512]

    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).reshape(1, H)            # [1, H]
        denom = l + torch.exp(sink - m_safe)
    else:
        denom = l

    lonely = l == 0.0
    denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)
    out = acc / denom.unsqueeze(-1)
    out = torch.where(lonely.unsqueeze(-1), torch.zeros_like(out), out)

    out = out[..., :head_dim_v].to(torch.bfloat16).contiguous()  # [T, H, head_dim_v]
    return out


# ============================================================================
# FlyDSL fused dual-scope kernel (live prefill path): gather + fp8 dequant +
# QK + exp2 online softmax + PV over SWA (main) and C4/C128 (extra) in one pass.
# ============================================================================
def _flydsl_available() -> bool:
    """True only on a gfx950 host with FlyDSL importable."""
    if not _HAS_FLYDSL:
        return False
    try:
        return str(get_rocm_arch()).startswith("gfx950")
    except Exception:
        return False


@functools.lru_cache(maxsize=64)
def _build_dual_scope_kernel(
    h_q: int,
    head_dim: int = D_QK,
    head_dim_v: int = D_V,
    topk_main: int = 2048,
    topk_extra: int = 2048,
    block_size_main: int = 64,
    block_size_extra: int = 64,
    tile_m: int = 16,
    block_n: int = 32,
    block_h: int = 16,
    sm_scale: Optional[float] = None,
    has_attn_sink: bool = True,
    has_topk_length_main: bool = True,
    has_topk_length_extra: bool = True,
    waves_per_eu: int = 2,
):
    """Build (and cache) the fused dual-scope FlyDSL kernel.

    The lru_cache key is the full set of compile-time-shaping args (dtype is
    fixed bf16/fp8, dims/topk/block sizes/BLOCK_H/sink/length flags select a
    distinct kernel specialization).  Returns a python launcher.

    The kernel implements the full dual-scope inner math: per-tile gather +
    e4m3fn fp8 dequant into LDS, bf16-MFMA QK, exp2 online softmax with running
    (m, l, acc) state shared across the SWA then EXTRA scopes, and bf16-MFMA PV,
    finalized with attn_sink folding and a lonely-query zero row.

    FlyDSL is imported at MODULE level (guarded); this builder is only reachable
    on the gfx950 box where the import succeeds.
    """
    assert _HAS_FLYDSL, "FlyDSL is not importable (build is gfx950-container only)"
    assert head_dim == D_QK, f"only head_dim={D_QK} is supported"
    assert tile_m == 16 and block_n == 32, "MFMA tiling is fixed at 16x32"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    gpu_arch = get_rocm_arch()

    # ---- tiling -----------------------------------------------------------
    # One CTA per query token (grid.y); BLOCK_H heads tiled in grid.x.  The
    # 16x16x32 bf16 MFMA M-dimension is the BLOCK_H=16 heads of this token
    # (NOT 16 tokens like the artefacts), the N-dimension is BLOCK_N keys, the
    # K-dimension is head_dim.  topk_length is read 1:1 per query token.
    WARP_SIZE = 64
    MFMA_N = 16
    MFMA_K = 32
    BLOCK_SIZE = WARP_SIZE
    K_STEPS_QK = head_dim // MFMA_K     # 16
    N_BLKS_S = block_n // MFMA_N        # 2
    D_BLKS = head_dim // MFMA_N         # 32
    KV_TILES_MAIN = topk_main // block_n
    KV_TILES_EXTRA = topk_extra // block_n

    BLOCK_H = block_h
    assert BLOCK_H == tile_m == MFMA_N, "BLOCK_H must equal MFMA M=16"
    TOK_OUT_STRIDE = h_q * head_dim_v               # Out token stride (elems)
    HEAD_OUT_STRIDE = head_dim_v                    # Out head stride (elems)
    Q_STOK = h_q * head_dim                         # Q token stride (elems)

    # ---- pool byte geometry (data-major: 576 data + 8 scale per slot) -----
    SLOT_DATA_BYTES = D_NOPE + D_ROPE * 2           # 576
    SLOT_SCALE_BYTES = BYTES_PER_TOKEN_SCALE        # 8
    PBB_MAIN = block_size_main * (SLOT_DATA_BYTES + SLOT_SCALE_BYTES)
    PBB_EXTRA = block_size_extra * (SLOT_DATA_BYTES + SLOT_SCALE_BYTES)
    NOPE_HALF = D_NOPE // 2                          # 224
    NOPE_CHUNKS = NOPE_HALF // 8                     # 28
    ROPE_HALF = D_ROPE // 2                          # 32
    ROPE_CHUNKS = ROPE_HALF // 8                     # 4

    SM_LOG2E = float(sm_scale) * _LOG2E
    BIG = 1.0e30                                     # finite mask sentinel

    # ---- LDS layout -------------------------------------------------------
    # KV tile staged as dequantized bf16 [block_n, head_dim]; P (attn weights)
    # staged as f32 [BLOCK_H, block_n].
    LDS_KV_BF16 = block_n * head_dim                # 16384 bf16
    LDS_KV_BYTES = LDS_KV_BF16 * 2                  # 32768
    LDS_P_F32 = BLOCK_H * block_n                    # 512
    LDS_P_BYTES = LDS_P_F32 * 4                      # 2048
    LDS_TOTAL = LDS_KV_BYTES + LDS_P_BYTES

    alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name="dual_scope_smem")
    base_off = alloc._align(alloc.ptr, 16)
    alloc.ptr = base_off + LDS_TOTAL
    kv_lds_off = base_off
    p_lds_off = base_off + LDS_KV_BYTES

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def dual_scope_prefill_kernel(
        Q: fx.Tensor,             # [T, h_q, head_dim]            bf16
        KV_Main: fx.Tensor,       # [Nb_main, per_block_bytes]    uint8 raw pool
        Indices_Main: fx.Tensor,  # [T, topk_main]                int32
        TopkLen_Main: fx.Tensor,  # [T]                           int32 (1:1 per tok)
        KV_Extra: fx.Tensor,      # [Nb_extra, per_block_bytes]   uint8 raw pool
        Indices_Extra: fx.Tensor, # [T, topk_extra]               int32
        TopkLen_Extra: fx.Tensor, # [T]                           int32 (1:1 per tok)
        AttnSink: fx.Tensor,      # [h_q]                         f32
        Out: fx.Tensor,           # [T, h_q, head_dim_v]          bf16
        total_tokens: fx.Int32,
    ):
        # ----------------------------------------------------------------
        # LLVM pointer helpers (mirror the artefact kernels exactly).
        # ----------------------------------------------------------------
        def _ptr_ty():
            return ir.Type.parse("!llvm.ptr")

        def _as_ptr(t):
            v = t
            if hasattr(v, "ir_value") and not isinstance(v, ir.Value):
                v = v.ir_value()
            return _fly.extract_aligned_pointer_as_index(_ptr_ty(), v)

        q_ptr           = _as_ptr(Q)
        out_ptr         = _as_ptr(Out)
        sink_ptr        = _as_ptr(AttnSink)
        main_pool_ptr   = _as_ptr(KV_Main)
        main_idx_ptr    = _as_ptr(Indices_Main)
        main_tkl_ptr    = _as_ptr(TopkLen_Main)
        extra_pool_ptr  = _as_ptr(KV_Extra)
        extra_idx_ptr   = _as_ptr(Indices_Extra)
        extra_tkl_ptr   = _as_ptr(TopkLen_Extra)

        # ---- typed constants / fast-math wrappers ----------------------
        v4f32_type  = Vec.make_type(4, fx.Float32)
        vec8bf16_ty = Vec.make_type(8, fx.BFloat16)
        fm = arith.FastMathFlags.fast

        def _fadd(a, b): return arith.addf(_raw(a), _raw(b), fastmath=fm)
        def _fmul(a, b): return arith.mulf(_raw(a), _raw(b), fastmath=fm)
        def _fsub(a, b): return arith.subf(_raw(a), _raw(b), fastmath=fm)
        def _fmax(a, b): return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result
        def _fmin(a, b): return arith.MinNumFOp(_raw(a), _raw(b), fastmath=fm).result
        def _exp2(x):    return rocdl.exp2(T.f32, _raw(x))

        c_one  = fx.Float32(1.0)
        c_big  = fx.Float32(BIG)
        c_log2e = fx.Float32(_LOG2E)
        c_sm_log2e = fx.Float32(SM_LOG2E)

        # ---- load helpers ----------------------------------------------
        def load_i32(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.i32)
            return _llvm.LoadOp(T.i32, gep).result

        def load_bf16_elem(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.bf16)
            return _llvm.LoadOp(T.bf16, gep).result

        def load_f32_elem(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.f32)
            return _llvm.LoadOp(T.f32, gep).result

        def load_u8_i32(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            u8 = _llvm.LoadOp(T.i8, gep).result
            return _mlir_arith.ExtUIOp(T.i32, u8).result

        def load_i64(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            return _llvm.LoadOp(T.i64, gep).result

        def load_bf16_byte(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            return _llvm.LoadOp(T.bf16, gep).result

        def extract_byte_i32(v_i64, j):
            sh  = _mlir_arith.ShRUIOp(v_i64, _raw(fx.Int64(j * 8))).result
            msk = _mlir_arith.AndIOp(sh, _raw(fx.Int64(0xFF))).result
            return _mlir_arith.TruncIOp(T.i32, msk).result

        def fp8_to_f32(byte_i32):
            return rocdl.cvt_f32_fp8(T.f32, byte_i32, fx.Int32(0))

        def ue8m0_to_f32(u8_i32):
            f = _mlir_arith.UIToFPOp(T.f32, u8_i32).result
            return _exp2(_fsub(f, fx.Float32(127.0)))

        def to_bf16(f32val):
            return fx.Float32(f32val).to(fx.BFloat16)

        def mfma_bf16(acc, a, b):
            return rocdl.mfma_f32_16x16x32_bf16(v4f32_type, [a, b, acc, 0, 0, 0])

        # ---- thread / block ids ----------------------------------------
        #   grid = (cdiv(h_q, BLOCK_H), total_tokens); block = (BLOCK_SIZE,1,1).
        tid        = fx.Index(gpu.thread_idx.x)
        bid_h_tile = fx.Index(gpu.block_idx.x)
        pid_t      = fx.Index(gpu.block_idx.y)

        lane     = tid % fx.Index(MFMA_N)        # 0..15 : MFMA row(A)/col(B/C)
        k_group  = tid // fx.Index(MFMA_N)       # 0..3  : MFMA K-subgroup
        head_base = bid_h_tile * fx.Index(BLOCK_H)
        kv_row   = tid % fx.Index(block_n)       # 0..31 : gather key row
        g_half   = tid // fx.Index(block_n)      # 0/1   : gather head-dim half

        # ---- LDS ------------------------------------------------------
        lds_base = alloc.get_base()
        lds_kv = SmemPtr(lds_base, kv_lds_off, T.bf16, shape=(LDS_KV_BF16,)).get()
        lds_p  = SmemPtr(lds_base, p_lds_off,  T.f32,  shape=(LDS_P_F32,)).get()

        topklen_main_val  = load_i32(main_tkl_ptr,  pid_t)
        topklen_extra_val = load_i32(extra_tkl_ptr, pid_t)

        # ---- preload Q head tile into K_STEPS_QK x vector<8xbf16> -------
        # A-operand row m = lane -> head (head_base + lane); k_group selects
        # the 8 contraction dims [ks*32 + k_group*8 : +8].
        q_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_elems = []
            for j in range_constexpr(8):
                q_eoff = (
                    pid_t * fx.Index(Q_STOK)
                    + (head_base + lane) * fx.Index(head_dim)
                    + fx.Index(ks * MFMA_K + j) + k_group * fx.Index(8)
                )
                q_elems.append(load_bf16_elem(q_ptr, q_eoff))
            q_packs.append(Vec.from_elements(q_elems, fx.BFloat16))

        # ---- online-softmax carry (finite -BIG sentinel, never -inf) ---
        # Flat loop-carried state for the runtime scf.for tile loops:
        #   [0:4]            m_i   -- per-row running max (4 MFMA rows / lane)
        #   [4:8]            l_i   -- per-row running softmax denominator
        #   [8:8+D_BLKS]     o_acc -- D_BLKS x vector<4xf32> output accumulator
        # All entries are raw IR values (vector<4xf32> for o_acc) so they can be
        # carried directly through scf.for / yield.
        _init = (
            [_raw(fx.Float32(-BIG)) for _ in range_constexpr(4)]
            + [_raw(fx.Float32(0.0)) for _ in range_constexpr(4)]
            + [_raw(Vec.filled(4, 0.0, fx.Float32)) for _ in range_constexpr(D_BLKS)]
        )

        def emit_tile(kv_tile, m_run, l_run, o_acc,
                      pool_ptr, idx_ptr, tkl_val, topk, bs, pbb):
            # kv_tile is a RUNTIME scf.for induction value (fx.Index); the carry
            # (m_run, l_run, o_acc) is passed in explicitly per iteration.
            kv_pos_base = kv_tile * fx.Index(block_n)

            # === gather + dequant one BLOCK_N tile -> LDS (bf16) =========
            idx_i32 = load_i32(idx_ptr, pid_t * fx.Index(topk)
                               + kv_pos_base + kv_row)
            idx_c = _mlir_arith.MaxSIOp(idx_i32, _raw(fx.Int32(0))).result
            blk = _mlir_arith.DivUIOp(idx_c, _raw(fx.Int32(bs))).result
            off = _mlir_arith.RemUIOp(idx_c, _raw(fx.Int32(bs))).result
            blk_i = fx.Index(blk)
            off_i = fx.Index(off)
            data_base = blk_i * fx.Index(pbb) + off_i * fx.Index(SLOT_DATA_BYTES)
            scale_base = (blk_i * fx.Index(pbb)
                          + fx.Index(bs * SLOT_DATA_BYTES)
                          + off_i * fx.Index(SLOT_SCALE_BYTES))

            # phase A: nope (448 fp8) -> bf16, per-64 ue8m0 scale folded in
            for dc in range_constexpr(NOPE_CHUNKS):
                abs_dim = g_half * fx.Index(NOPE_HALF) + fx.Index(dc * 8)
                tile = abs_dim // fx.Index(QUANT_BLOCK)
                scale_f = ue8m0_to_f32(load_u8_i32(pool_ptr, scale_base + tile))
                raw_i64 = load_i64(pool_ptr, data_base + abs_dim)
                bvals = []
                for j in range_constexpr(8):
                    fv = _fmul(fp8_to_f32(extract_byte_i32(raw_i64, j)), scale_f)
                    bvals.append(to_bf16(fv))
                Vec.from_elements(bvals, fx.BFloat16).store(
                    lds_kv, [kv_row * fx.Index(head_dim) + abs_dim]
                )

            # phase B: rope (128 raw bytes -> 64 bf16 LE) -> bf16
            for dc in range_constexpr(ROPE_CHUNKS):
                rope_idx = g_half * fx.Index(ROPE_HALF) + fx.Index(dc * 8)
                bvals = []
                for j in range_constexpr(8):
                    boff = (data_base + fx.Index(D_NOPE)
                            + (rope_idx + fx.Index(j)) * fx.Index(2))
                    bvals.append(load_bf16_byte(pool_ptr, boff))
                Vec.from_elements(bvals, fx.BFloat16).store(
                    lds_kv, [kv_row * fx.Index(head_dim) + fx.Index(D_NOPE) + rope_idx]
                )

            gpu.barrier()

            # === QK GEMM : S[BLOCK_H, BLOCK_N] = Q @ K^T (bf16 MFMA) =====
            s_acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(N_BLKS_S)]
            for ks in range_constexpr(K_STEPS_QK):
                q_a = q_packs[ks]
                for nb in range_constexpr(N_BLKS_S):
                    key = fx.Index(nb * MFMA_N) + lane
                    koff = (key * fx.Index(head_dim)
                            + fx.Index(ks * MFMA_K) + k_group * fx.Index(8))
                    k_b = Vec.load(vec8bf16_ty, lds_kv, [koff])
                    s_acc[nb] = mfma_bf16(s_acc[nb], q_a, k_b)

            # === validity mask (recomputed per key column) ==============
            mask_f = []
            for nb in range_constexpr(N_BLKS_S):
                key_pos = kv_pos_base + fx.Index(nb * MFMA_N) + lane
                ki = load_i32(idx_ptr, pid_t * fx.Index(topk) + key_pos)
                ne = arith.cmpi(arith.CmpIPredicate.ne, ki, fx.Int32(-1))
                kp_i32 = _mlir_arith.IndexCastOp(T.i32, _raw(key_pos)).result
                lt = arith.cmpi(arith.CmpIPredicate.slt, kp_i32, tkl_val)
                good = _mlir_arith.AndIOp(_raw(ne), _raw(lt)).result
                mask_f.append(_mlir_arith.UIToFPOp(T.f32, good).result)

            # === online softmax (log2 space) ===========================
            s_scaled = [
                [_fmul(Vec(s_acc[nb])[r], c_sm_log2e) for r in range_constexpr(4)]
                for nb in range_constexpr(N_BLKS_S)
            ]
            # masked columns: drive far negative for the max (finite, no NaN)
            s_for_max = [
                [_fadd(s_scaled[nb][r], _fmul(_fsub(mask_f[nb], c_one), c_big))
                 for r in range_constexpr(4)]
                for nb in range_constexpr(N_BLKS_S)
            ]
            local_max = [_fmax(s_for_max[0][r], s_for_max[1][r]) for r in range_constexpr(4)]
            row_max = list(local_max)
            for xor_off in [8, 4, 2, 1]:
                so = fx.Int32(xor_off)
                for r in range_constexpr(4):
                    row_max[r] = _fmax(
                        row_max[r],
                        fx.Float32(row_max[r]).shuffle_xor(so, fx.Int32(WARP_SIZE)),
                    )

            m_new = [_fmax(m_run[r], row_max[r]) for r in range_constexpr(4)]
            corr  = [_exp2(_fsub(m_run[r], m_new[r])) for r in range_constexpr(4)]

            p_vals   = [[None] * 4 for _ in range_constexpr(N_BLKS_S)]
            tile_sum = [_raw(fx.Float32(0.0)) for _ in range_constexpr(4)]
            for nb in range_constexpr(N_BLKS_S):
                for r in range_constexpr(4):
                    p = _fmul(_exp2(_fsub(s_scaled[nb][r], m_new[r])), mask_f[nb])
                    p_vals[nb][r] = p
                    tile_sum[r] = _fadd(tile_sum[r], p)
            for xor_off in [8, 4, 2, 1]:
                so = fx.Int32(xor_off)
                for r in range_constexpr(4):
                    tile_sum[r] = _fadd(
                        tile_sum[r],
                        fx.Float32(tile_sum[r]).shuffle_xor(so, fx.Int32(WARP_SIZE)),
                    )

            l_new = [_fadd(_fmul(corr[r], l_run[r]), tile_sum[r]) for r in range_constexpr(4)]
            new_o = []
            for d in range_constexpr(D_BLKS):
                ov = Vec(o_acc[d])
                new_o.append(
                    Vec.from_elements(
                        [_fmul(ov[r], corr[r]) for r in range_constexpr(4)], fx.Float32
                    )
                )

            # === stage P (f32) -> LDS, transposed [head_row][key] =======
            for nb in range_constexpr(N_BLKS_S):
                for r in range_constexpr(4):
                    p_row = k_group * fx.Index(4) + fx.Index(r)
                    p_col = fx.Index(nb * MFMA_N) + lane
                    _memref.store(_raw(p_vals[nb][r]), lds_p,
                                  [_raw(p_row * fx.Index(block_n) + p_col)])
            gpu.barrier()

            # === PV GEMM : O += P @ V (bf16 MFMA) =======================
            p_base = lane * fx.Index(block_n) + k_group * fx.Index(8)
            p_a = Vec.from_elements(
                [to_bf16(_memref.load(lds_p, [_raw(p_base + fx.Index(j))]))
                 for j in range_constexpr(8)],
                fx.BFloat16,
            )
            for d in range_constexpr(D_BLKS):
                vvals = []
                for j in range_constexpr(8):
                    key = k_group * fx.Index(8) + fx.Index(j)
                    voff = key * fx.Index(head_dim) + fx.Index(d * MFMA_N) + lane
                    vvals.append(_memref.load(lds_kv, [_raw(voff)]))
                v_b = Vec.from_elements(vvals, fx.BFloat16)
                new_o[d] = mfma_bf16(new_o[d], p_a, v_b)
            gpu.barrier()
            return m_new, l_new, new_o

        # ==== dual-scope online softmax: SWA (main) then EXTRA ==========
        # RUNTIME scf.for carry loops: the per-tile body is emitted ONCE (not
        # compile-time unrolled), so code size / JIT time stay bounded even at
        # production topk (e.g. extra topk 2048 / block_n 32 -> 64 tiles).  The
        # trip count KV_TILES_* = ceil(topk / BLOCK_N) is derived from the tensor
        # shapes; the EXTRA loop continues into the SAME shared carried state
        # (m_i, l_i, o_acc) produced by the SWA loop.  Per-key masking beyond
        # topk_length / idx==-1 still zeroes any over-run column exactly as
        # before, so a fixed (padded) trip count is numerically safe.
        loop_results = _init
        for kv_tile, _carry in range(0, KV_TILES_MAIN, 1, init=_init):
            m_run = [_carry[r] for r in range_constexpr(4)]
            l_run = [_carry[4 + r] for r in range_constexpr(4)]
            o_acc = [_carry[8 + d] for d in range_constexpr(D_BLKS)]
            m_new, l_new, new_o = emit_tile(
                kv_tile, m_run, l_run, o_acc,
                main_pool_ptr, main_idx_ptr, topklen_main_val,
                topk_main, block_size_main, PBB_MAIN,
            )
            loop_results = yield list(m_new) + list(l_new) + list(new_o)

        for kv_tile, _carry in range(0, KV_TILES_EXTRA, 1, init=loop_results):
            m_run = [_carry[r] for r in range_constexpr(4)]
            l_run = [_carry[4 + r] for r in range_constexpr(4)]
            o_acc = [_carry[8 + d] for d in range_constexpr(D_BLKS)]
            m_new, l_new, new_o = emit_tile(
                kv_tile, m_run, l_run, o_acc,
                extra_pool_ptr, extra_idx_ptr, topklen_extra_val,
                topk_extra, block_size_extra, PBB_EXTRA,
            )
            loop_results = yield list(m_new) + list(l_new) + list(new_o)

        # ==== finalize : normalize, fold attn_sink, lonely -> zero ======
        m_run = [loop_results[r] for r in range_constexpr(4)]
        l_run = [loop_results[4 + r] for r in range_constexpr(4)]
        o_acc = [loop_results[8 + d] for d in range_constexpr(D_BLKS)]
        for r in range_constexpr(4):
            head = head_base + k_group * fx.Index(4) + fx.Index(r)
            l_r = l_run[r]
            m_r = m_run[r]
            # lonely (l==0): l>=1 whenever any key was valid (max key p=1)
            lonely_f = _fsub(c_one, _fmin(_fmul(l_r, c_big), c_one))
            one_m = _fsub(c_one, lonely_f)
            m_safe = _fmul(m_r, one_m)          # 0 when lonely (mirrors ref)
            if const_expr(has_attn_sink):
                sink = load_f32_elem(sink_ptr, head)
                sink_term = _exp2(_fsub(_fmul(sink, c_log2e), m_safe))
                denom = _fadd(l_r, sink_term)
            else:
                denom = l_r
            safe_denom = _fadd(_fmul(denom, one_m), lonely_f)   # ==1 when lonely
            inv = arith.divf(_raw(c_one), _raw(safe_denom), fastmath=fm)
            for d in range_constexpr(D_BLKS):
                o_norm = _fmul(Vec(o_acc[d])[r], inv)
                out_off = (pid_t * fx.Index(TOK_OUT_STRIDE)
                           + head * fx.Index(HEAD_OUT_STRIDE)
                           + fx.Index(d * MFMA_N) + lane)
                out_gep = buffer_ops.get_element_ptr(
                    out_ptr, fx.Int64(out_off), elem_type=T.bf16
                )
                _llvm.StoreOp(to_bf16(o_norm).ir_value(), out_gep)
        return

    @flyc.jit
    def launch_dual_scope_prefill(
        Q, KV_Main, Indices_Main, TopkLen_Main,
        KV_Extra, Indices_Extra, TopkLen_Extra,
        AttnSink, Out, total_tokens: fx.Int32,
        stream: "fx.Stream" = fx.Stream(None),
    ):
        alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalize()

        tokens_idx = fx.Index(total_tokens)
        grid_h = fx.Index((h_q + BLOCK_H - 1) // BLOCK_H)
        launcher = dual_scope_prefill_kernel(
            Q, KV_Main, Indices_Main, TopkLen_Main,
            KV_Extra, Indices_Extra, TopkLen_Extra,
            AttnSink, Out, total_tokens,
        )

        passthrough = []
        for pair in [
            ("denormal-fp-math-f32", "preserve-sign,preserve-sign"),
            ("no-nans-fp-math", "true"),
            ("unsafe-fp-math", "true"),
        ]:
            passthrough.append(
                ir.ArrayAttr.get(
                    [ir.StringAttr.get(pair[0]), ir.StringAttr.get(pair[1])]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough)
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                    T.i32, int(waves_per_eu)
                )

        launcher.launch(
            grid=(grid_h, tokens_idx, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_dual_scope_prefill(*args, **kwargs)

    return _launch


# ============================================================================
# C4 / high-topk dual-scope kernel (separate specialized variant).
#
# This mirrors Triton's SECOND, high-topk config of the very same dual-scope
# kernel (BLOCK_H=128, BLOCK_N=64, num_warps=4, num_stages=1) -- a distinct
# function, NOT a parameterized generalization of _build_dual_scope_kernel.
# The inner MATH is byte-for-byte the same pipeline as the validated kernel
# above (gather -> e4m3fn fp8 dequant -> bf16 QK MFMA -> exp2 online softmax ->
# bf16 PV MFMA -> attn_sink fold + lonely-zero writeback).  The ONLY changes
# are the TILING / WAVE structure:
#
#   * 1 CTA per query token handles ALL BLOCK_H=128 heads, so the (expensive)
#     fp8 gather+dequant of each KV tile is done ONCE for all 128 heads instead
#     of being redundantly repeated by the 8 separate BLOCK_H=16 head-tile CTAs
#     the current kernel launches (grid.x = h_q/16 = 8).  Eliminating that 8x
#     redundant dequant is the whole point of this variant.
#   * BLOCK_SIZE=256 (4 wavefronts of 64 == num_warps=4).  Each wave OWNS
#     HEADS_PER_WAVE=32 heads (= M_SUBS=2 MFMA m-subtiles), so the online-softmax
#     cross-key reduction stays entirely within a wave (shuffle_xor over its
#     16-lane group), exactly like the single-wave kernel above.
#   * BLOCK_N=64 keys/tile (vs 32): QK has N_BLKS_S=4 n-blocks, PV K_STEPS_PV=2.
#   * The 64-key x 512-dim dequantized bf16 KV tile is staged into LDS
#     cooperatively by all 256 threads: thread (wave w, lin l) dequants key row
#     l, head-dim quarter w (112 nope + 16 rope dims).  KV LDS 64KB + P LDS 32KB
#     = 96KB, within the 160KB gfx950 (MI350X) group-segment budget.
# ============================================================================
@functools.lru_cache(maxsize=64)
def _build_dual_scope_kernel_c4(
    h_q: int,
    head_dim: int = D_QK,
    head_dim_v: int = D_V,
    topk_main: int = 128,
    topk_extra: int = 512,
    block_size_main: int = 256,
    block_size_extra: int = 64,
    block_n: int = 64,
    block_h: int = 128,
    sm_scale: Optional[float] = None,
    has_attn_sink: bool = True,
    has_topk_length_main: bool = True,
    has_topk_length_extra: bool = True,
    waves_per_eu: int = 1,
):
    """Build (and cache) the C4 / high-topk fused dual-scope FlyDSL kernel.

    Numerically identical inner pipeline to ``_build_dual_scope_kernel``; only
    the tiling/wave structure differs (see the module banner above).  Returns a
    python launcher with the same call signature as the base kernel's launcher.
    """
    assert _HAS_FLYDSL, "FlyDSL is not importable (build is gfx950-container only)"
    assert head_dim == D_QK, f"only head_dim={D_QK} is supported"
    assert block_n == 64, "C4 kernel tiling is fixed at BLOCK_N=64"
    assert block_h == 128, "C4 kernel tiling is fixed at BLOCK_H=128"
    assert h_q % block_h == 0, f"h_q={h_q} must be a multiple of BLOCK_H={block_h}"
    assert topk_main % block_n == 0, "topk_main must be a multiple of BLOCK_N=64"
    assert topk_extra % block_n == 0, "topk_extra must be a multiple of BLOCK_N=64"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    gpu_arch = get_rocm_arch()

    # ---- tiling -----------------------------------------------------------
    WARP_SIZE = 64
    N_WAVES = 4                                     # == num_warps
    BLOCK_SIZE = WARP_SIZE * N_WAVES                # 256
    MFMA_N = 16
    MFMA_K = 32
    BLOCK_H = block_h                               # 128
    HEADS_PER_WAVE = BLOCK_H // N_WAVES             # 32
    M_SUBS = HEADS_PER_WAVE // MFMA_N               # 2 MFMA m-subtiles / wave
    K_STEPS_QK = head_dim // MFMA_K                 # 16
    N_BLKS_S = block_n // MFMA_N                     # 4
    K_STEPS_PV = block_n // MFMA_K                   # 2
    D_BLKS = head_dim // MFMA_N                      # 32
    KV_TILES_MAIN = topk_main // block_n
    KV_TILES_EXTRA = topk_extra // block_n

    TOK_OUT_STRIDE = h_q * head_dim_v
    HEAD_OUT_STRIDE = head_dim_v
    Q_STOK = h_q * head_dim

    # ---- pool byte geometry (data-major: 576 data + 8 scale per slot) -----
    SLOT_DATA_BYTES = D_NOPE + D_ROPE * 2           # 576
    SLOT_SCALE_BYTES = BYTES_PER_TOKEN_SCALE        # 8
    PBB_MAIN = block_size_main * (SLOT_DATA_BYTES + SLOT_SCALE_BYTES)
    PBB_EXTRA = block_size_extra * (SLOT_DATA_BYTES + SLOT_SCALE_BYTES)

    # gather: 256 threads = 64 key rows x 4 head-dim quarters (one quarter/wave)
    NOPE_QUARTER = D_NOPE // N_WAVES                 # 112
    NOPE_CHUNKS = NOPE_QUARTER // 8                  # 14
    ROPE_QUARTER = D_ROPE // N_WAVES                 # 16
    ROPE_CHUNKS = ROPE_QUARTER // 8                  # 2

    SM_LOG2E = float(sm_scale) * _LOG2E
    BIG = 1.0e30

    # ---- LDS layout -------------------------------------------------------
    # Per-row padding breaks the multiple-of-32-banks stride that otherwise
    # collapses every key (resp. head-row) onto the same LDS banks.  lds_kv is
    # accessed as [key][dim]: the QK read takes 16-byte (8 bf16) vectors while
    # the high-frequency PV read takes scalar bf16 with the 8 contraction keys
    # one k-group apart (8 keys) -- to avoid those 8-apart keys aliasing the
    # same banks the bf16 stride must NOT be a multiple of 8 (4*stride mod 32).
    # lds_p [head_row][key] f32: the P-store has the two k-groups (head-rows 4
    # apart) aliasing unless the f32 stride is not a multiple of 8.  _C4_KV_PAD
    # / _C4_P_PAD are tuned to minimize SQ_LDS_BANK_CONFLICT (sweep) and are
    # env-overridable.
    KV_LDS_STRIDE = head_dim + _C4_KV_PAD             # padded bf16 per key
    P_LDS_STRIDE = block_n + _C4_P_PAD                 # padded f32 per head-row
    LDS_KV_BF16 = block_n * KV_LDS_STRIDE
    LDS_KV_BYTES = LDS_KV_BF16 * 2
    LDS_P_F32 = BLOCK_H * P_LDS_STRIDE
    LDS_P_BYTES = LDS_P_F32 * 4
    LDS_TOTAL = LDS_KV_BYTES + LDS_P_BYTES

    alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name="dual_scope_c4_smem")
    base_off = alloc._align(alloc.ptr, 16)
    alloc.ptr = base_off + LDS_TOTAL
    kv_lds_off = base_off
    p_lds_off = base_off + LDS_KV_BYTES

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def dual_scope_prefill_kernel_c4(
        Q: fx.Tensor,             # [T, h_q, head_dim]            bf16
        KV_Main: fx.Tensor,       # [Nb_main, per_block_bytes]    uint8 raw pool
        Indices_Main: fx.Tensor,  # [T, topk_main]                int32
        TopkLen_Main: fx.Tensor,  # [T]                           int32 (1:1 per tok)
        KV_Extra: fx.Tensor,      # [Nb_extra, per_block_bytes]   uint8 raw pool
        Indices_Extra: fx.Tensor, # [T, topk_extra]               int32
        TopkLen_Extra: fx.Tensor, # [T]                           int32 (1:1 per tok)
        AttnSink: fx.Tensor,      # [h_q]                         f32
        Out: fx.Tensor,           # [T, h_q, head_dim_v]          bf16
        total_tokens: fx.Int32,
    ):
        # ---- LLVM pointer helpers (mirror the base kernel exactly) ------
        def _ptr_ty():
            return ir.Type.parse("!llvm.ptr")

        def _as_ptr(t):
            v = t
            if hasattr(v, "ir_value") and not isinstance(v, ir.Value):
                v = v.ir_value()
            return _fly.extract_aligned_pointer_as_index(_ptr_ty(), v)

        q_ptr           = _as_ptr(Q)
        out_ptr         = _as_ptr(Out)
        sink_ptr        = _as_ptr(AttnSink)
        main_pool_ptr   = _as_ptr(KV_Main)
        main_idx_ptr    = _as_ptr(Indices_Main)
        main_tkl_ptr    = _as_ptr(TopkLen_Main)
        extra_pool_ptr  = _as_ptr(KV_Extra)
        extra_idx_ptr   = _as_ptr(Indices_Extra)
        extra_tkl_ptr   = _as_ptr(TopkLen_Extra)

        # ---- typed constants / fast-math wrappers ----------------------
        v4f32_type  = Vec.make_type(4, fx.Float32)
        vec8bf16_ty = Vec.make_type(8, fx.BFloat16)
        fm = arith.FastMathFlags.fast

        def _fadd(a, b): return arith.addf(_raw(a), _raw(b), fastmath=fm)
        def _fmul(a, b): return arith.mulf(_raw(a), _raw(b), fastmath=fm)
        def _fsub(a, b): return arith.subf(_raw(a), _raw(b), fastmath=fm)
        def _fmax(a, b): return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result
        def _fmin(a, b): return arith.MinNumFOp(_raw(a), _raw(b), fastmath=fm).result
        def _exp2(x):    return rocdl.exp2(T.f32, _raw(x))

        c_one  = fx.Float32(1.0)
        c_big  = fx.Float32(BIG)
        c_log2e = fx.Float32(_LOG2E)
        c_sm_log2e = fx.Float32(SM_LOG2E)

        # ---- load helpers ----------------------------------------------
        def load_i32(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.i32)
            return _llvm.LoadOp(T.i32, gep).result

        def load_bf16_elem(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.bf16)
            return _llvm.LoadOp(T.bf16, gep).result

        def load_f32_elem(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.f32)
            return _llvm.LoadOp(T.f32, gep).result

        def load_u8_i32(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            u8 = _llvm.LoadOp(T.i8, gep).result
            return _mlir_arith.ExtUIOp(T.i32, u8).result

        def load_i64(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            return _llvm.LoadOp(T.i64, gep).result

        def load_bf16_byte(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            return _llvm.LoadOp(T.bf16, gep).result

        def extract_byte_i32(v_i64, j):
            sh  = _mlir_arith.ShRUIOp(v_i64, _raw(fx.Int64(j * 8))).result
            msk = _mlir_arith.AndIOp(sh, _raw(fx.Int64(0xFF))).result
            return _mlir_arith.TruncIOp(T.i32, msk).result

        def fp8_to_f32(byte_i32):
            return rocdl.cvt_f32_fp8(T.f32, byte_i32, fx.Int32(0))

        def ue8m0_to_f32(u8_i32):
            f = _mlir_arith.UIToFPOp(T.f32, u8_i32).result
            return _exp2(_fsub(f, fx.Float32(127.0)))

        def to_bf16(f32val):
            return fx.Float32(f32val).to(fx.BFloat16)

        def mfma_bf16(acc, a, b):
            return rocdl.mfma_f32_16x16x32_bf16(v4f32_type, [a, b, acc, 0, 0, 0])

        # ---- thread / block ids ----------------------------------------
        #   grid = (h_q/BLOCK_H == 1, total_tokens); block = (256,1,1).
        #   wave = tid//64 (0..3) -> owns 32 heads.  lin = tid%64.
        tid       = fx.Index(gpu.thread_idx.x)
        pid_t     = fx.Index(gpu.block_idx.y)
        wave      = tid // fx.Index(WARP_SIZE)      # 0..3
        lin       = tid % fx.Index(WARP_SIZE)       # 0..63
        lane      = lin % fx.Index(MFMA_N)          # 0..15 MFMA row(A)/col(B/C)
        k_group   = lin // fx.Index(MFMA_N)         # 0..3  MFMA K-subgroup
        wave_head_base = wave * fx.Index(HEADS_PER_WAVE)   # this wave's first head
        kv_row    = lin                              # gather: key row (0..63)
        g_part    = wave                             # gather: head-dim quarter

        # ---- LDS ------------------------------------------------------
        lds_base = alloc.get_base()
        lds_kv = SmemPtr(lds_base, kv_lds_off, T.bf16, shape=(LDS_KV_BF16,)).get()
        lds_p  = SmemPtr(lds_base, p_lds_off,  T.f32,  shape=(LDS_P_F32,)).get()

        topklen_main_val  = load_i32(main_tkl_ptr,  pid_t)
        topklen_extra_val = load_i32(extra_tkl_ptr, pid_t)

        # ---- preload Q head tiles (M_SUBS x K_STEPS_QK x vector<8xbf16>) -
        # A-operand row m = lane -> head (wave_head_base + m_sub*16 + lane);
        # k_group selects the 8 contraction dims [ks*32 + k_group*8 : +8].
        q_packs = [[] for _ in range_constexpr(M_SUBS)]
        for m_sub in range_constexpr(M_SUBS):
            hb = wave_head_base + fx.Index(m_sub * MFMA_N)
            for ks in range_constexpr(K_STEPS_QK):
                q_elems = []
                for j in range_constexpr(8):
                    q_eoff = (
                        pid_t * fx.Index(Q_STOK)
                        + (hb + lane) * fx.Index(head_dim)
                        + fx.Index(ks * MFMA_K + j) + k_group * fx.Index(8)
                    )
                    q_elems.append(load_bf16_elem(q_ptr, q_eoff))
                q_packs[m_sub].append(Vec.from_elements(q_elems, fx.BFloat16))

        # ---- online-softmax carry: per m_sub [m(4), l(4), o_acc(D_BLKS)] -
        PER_SUB = 8 + D_BLKS
        _init = []
        for _ms in range_constexpr(M_SUBS):
            _init += [_raw(fx.Float32(-BIG)) for _ in range_constexpr(4)]
            _init += [_raw(fx.Float32(0.0)) for _ in range_constexpr(4)]
            _init += [_raw(Vec.filled(4, 0.0, fx.Float32)) for _ in range_constexpr(D_BLKS)]

        def emit_tile(kv_tile, carry, pool_ptr, idx_ptr, tkl_val, topk, bs, pbb):
            kv_pos_base = kv_tile * fx.Index(block_n)

            # === gather + dequant one BLOCK_N=64 tile -> LDS (bf16) =======
            # thread (wave w, lin l): key row l, head-dim quarter w.
            idx_i32 = load_i32(idx_ptr, pid_t * fx.Index(topk)
                               + kv_pos_base + kv_row)
            idx_c = _mlir_arith.MaxSIOp(idx_i32, _raw(fx.Int32(0))).result
            blk = _mlir_arith.DivUIOp(idx_c, _raw(fx.Int32(bs))).result
            off = _mlir_arith.RemUIOp(idx_c, _raw(fx.Int32(bs))).result
            blk_i = fx.Index(blk)
            off_i = fx.Index(off)
            data_base = blk_i * fx.Index(pbb) + off_i * fx.Index(SLOT_DATA_BYTES)
            scale_base = (blk_i * fx.Index(pbb)
                          + fx.Index(bs * SLOT_DATA_BYTES)
                          + off_i * fx.Index(SLOT_SCALE_BYTES))

            # phase A: nope quarter (112 fp8) -> bf16, per-64 ue8m0 scale folded
            for dc in range_constexpr(NOPE_CHUNKS):
                abs_dim = g_part * fx.Index(NOPE_QUARTER) + fx.Index(dc * 8)
                tile = abs_dim // fx.Index(QUANT_BLOCK)
                scale_f = ue8m0_to_f32(load_u8_i32(pool_ptr, scale_base + tile))
                raw_i64 = load_i64(pool_ptr, data_base + abs_dim)
                bvals = []
                for j in range_constexpr(8):
                    fv = _fmul(fp8_to_f32(extract_byte_i32(raw_i64, j)), scale_f)
                    bvals.append(to_bf16(fv))
                Vec.from_elements(bvals, fx.BFloat16).store(
                    lds_kv, [kv_row * fx.Index(KV_LDS_STRIDE) + abs_dim]
                )

            # phase B: rope quarter (32 raw bytes -> 16 bf16 LE) -> bf16
            for dc in range_constexpr(ROPE_CHUNKS):
                rope_idx = g_part * fx.Index(ROPE_QUARTER) + fx.Index(dc * 8)
                bvals = []
                for j in range_constexpr(8):
                    boff = (data_base + fx.Index(D_NOPE)
                            + (rope_idx + fx.Index(j)) * fx.Index(2))
                    bvals.append(load_bf16_byte(pool_ptr, boff))
                Vec.from_elements(bvals, fx.BFloat16).store(
                    lds_kv, [kv_row * fx.Index(KV_LDS_STRIDE) + fx.Index(D_NOPE) + rope_idx]
                )

            gpu.barrier()

            # === validity mask (head-independent; computed once / tile) ===
            mask_f = []
            for nb in range_constexpr(N_BLKS_S):
                key_pos = kv_pos_base + fx.Index(nb * MFMA_N) + lane
                ki = load_i32(idx_ptr, pid_t * fx.Index(topk) + key_pos)
                ne = arith.cmpi(arith.CmpIPredicate.ne, ki, fx.Int32(-1))
                kp_i32 = _mlir_arith.IndexCastOp(T.i32, _raw(key_pos)).result
                lt = arith.cmpi(arith.CmpIPredicate.slt, kp_i32, tkl_val)
                good = _mlir_arith.AndIOp(_raw(ne), _raw(lt)).result
                mask_f.append(_mlir_arith.UIToFPOp(T.f32, good).result)

            new_carry = []
            for m_sub in range_constexpr(M_SUBS):
                base = m_sub * PER_SUB
                m_run = [carry[base + r] for r in range_constexpr(4)]
                l_run = [carry[base + 4 + r] for r in range_constexpr(4)]
                o_acc = [carry[base + 8 + d] for d in range_constexpr(D_BLKS)]
                hb = wave_head_base + fx.Index(m_sub * MFMA_N)

                # === QK GEMM : S[16, BLOCK_N] = Q @ K^T (bf16 MFMA) =======
                s_acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(N_BLKS_S)]
                for ks in range_constexpr(K_STEPS_QK):
                    q_a = q_packs[m_sub][ks]
                    for nb in range_constexpr(N_BLKS_S):
                        key = fx.Index(nb * MFMA_N) + lane
                        koff = (key * fx.Index(KV_LDS_STRIDE)
                                + fx.Index(ks * MFMA_K) + k_group * fx.Index(8))
                        k_b = Vec.load(vec8bf16_ty, lds_kv, [koff])
                        s_acc[nb] = mfma_bf16(s_acc[nb], q_a, k_b)

                # === online softmax (log2 space) =========================
                s_scaled = [
                    [_fmul(Vec(s_acc[nb])[r], c_sm_log2e) for r in range_constexpr(4)]
                    for nb in range_constexpr(N_BLKS_S)
                ]
                s_for_max = [
                    [_fadd(s_scaled[nb][r], _fmul(_fsub(mask_f[nb], c_one), c_big))
                     for r in range_constexpr(4)]
                    for nb in range_constexpr(N_BLKS_S)
                ]
                row_max = [s_for_max[0][r] for r in range_constexpr(4)]
                for nb in range_constexpr(N_BLKS_S):
                    if const_expr(nb > 0):
                        for r in range_constexpr(4):
                            row_max[r] = _fmax(row_max[r], s_for_max[nb][r])
                for xor_off in [8, 4, 2, 1]:
                    so = fx.Int32(xor_off)
                    for r in range_constexpr(4):
                        row_max[r] = _fmax(
                            row_max[r],
                            fx.Float32(row_max[r]).shuffle_xor(so, fx.Int32(WARP_SIZE)),
                        )

                m_new = [_fmax(m_run[r], row_max[r]) for r in range_constexpr(4)]
                corr  = [_exp2(_fsub(m_run[r], m_new[r])) for r in range_constexpr(4)]

                p_vals   = [[None] * 4 for _ in range_constexpr(N_BLKS_S)]
                tile_sum = [_raw(fx.Float32(0.0)) for _ in range_constexpr(4)]
                for nb in range_constexpr(N_BLKS_S):
                    for r in range_constexpr(4):
                        p = _fmul(_exp2(_fsub(s_scaled[nb][r], m_new[r])), mask_f[nb])
                        p_vals[nb][r] = p
                        tile_sum[r] = _fadd(tile_sum[r], p)
                for xor_off in [8, 4, 2, 1]:
                    so = fx.Int32(xor_off)
                    for r in range_constexpr(4):
                        tile_sum[r] = _fadd(
                            tile_sum[r],
                            fx.Float32(tile_sum[r]).shuffle_xor(so, fx.Int32(WARP_SIZE)),
                        )

                l_new = [_fadd(_fmul(corr[r], l_run[r]), tile_sum[r]) for r in range_constexpr(4)]
                if const_expr(_C4_LAZY_RESCALE):
                    # D4: skip the O rescale on tiles where no owned row's running
                    # max advances (corr == 1, identity multiply).  Predicate is
                    # reduced across the full 64-lane wave so scf.if is uniform.
                    rm_delta = _fsub(row_max[0], m_run[0])
                    for r in range_constexpr(4):
                        if const_expr(r > 0):
                            rm_delta = _fmax(rm_delta, _fsub(row_max[r], m_run[r]))
                    for xor_off in [32, 16, 8, 4, 2, 1]:
                        rm_delta = _fmax(
                            rm_delta,
                            fx.Float32(rm_delta).shuffle_xor(
                                fx.Int32(xor_off), fx.Int32(WARP_SIZE)
                            ),
                        )
                    should_scale = arith.cmpf(
                        arith.CmpFPredicate.OGT, _raw(rm_delta), _raw(fx.Float32(0.0))
                    )
                    o_raw = [_raw(o_acc[d]) for d in range_constexpr(D_BLKS)]
                    if_op = _scf.IfOp(
                        _raw(should_scale),
                        [v.type for v in o_raw],
                        has_else=True,
                        loc=ir.Location.unknown(),
                    )
                    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                        scaled_raw = []
                        for d in range_constexpr(D_BLKS):
                            ov = Vec(o_acc[d])
                            scaled_raw.append(
                                _raw(Vec.from_elements(
                                    [_fmul(ov[r], corr[r]) for r in range_constexpr(4)],
                                    fx.Float32,
                                ))
                            )
                        _scf.YieldOp(scaled_raw)
                    if len(if_op.regions[1].blocks) == 0:
                        if_op.regions[1].blocks.append(*[])
                    with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                        _scf.YieldOp(o_raw)
                    new_o = [Vec(r) for r in if_op.results]
                else:
                    new_o = []
                    for d in range_constexpr(D_BLKS):
                        ov = Vec(o_acc[d])
                        new_o.append(
                            Vec.from_elements(
                                [_fmul(ov[r], corr[r]) for r in range_constexpr(4)], fx.Float32
                            )
                        )

                # === stage P (f32) -> LDS, transposed [head_row][key] =====
                for nb in range_constexpr(N_BLKS_S):
                    for r in range_constexpr(4):
                        p_row = hb + k_group * fx.Index(4) + fx.Index(r)
                        p_col = fx.Index(nb * MFMA_N) + lane
                        _memref.store(_raw(p_vals[nb][r]), lds_p,
                                      [_raw(p_row * fx.Index(P_LDS_STRIDE) + p_col)])
                gpu.barrier()

                # === PV GEMM : O += P @ V (bf16 MFMA), K=block_n=2x32 ======
                # LDS transpose-read helper (gfx950 ds_read_b64_tr_b16): returns a
                # vector<4xbf16> = the 4 contraction keys at the calling lane's head,
                # transposed from lds_kv at bf16 element offset `elem_off`.
                def _ds_tr_read(elem_off):
                    base_mr = lds_base
                    if hasattr(base_mr, "ir_value") and not isinstance(base_mr, ir.Value):
                        base_mr = base_mr.ir_value()
                    base_idx = _memref.extract_aligned_pointer_as_index(base_mr)
                    base_i64 = _mlir_arith.IndexCastOp(T.i64, _raw(base_idx)).result
                    eoff_i64 = _mlir_arith.IndexCastOp(T.i64, _raw(elem_off)).result
                    byte_addr = _mlir_arith.AddIOp(
                        _mlir_arith.AddIOp(base_i64, _raw(fx.Int64(kv_lds_off))).result,
                        _mlir_arith.MulIOp(eoff_i64, _raw(fx.Int64(2))).result,
                    ).result
                    p3 = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), byte_addr)
                    vi16 = rocdl.ds_read_tr16_b64(ir.VectorType.get([4], T.i16), p3)
                    return arith.BitcastOp(ir.VectorType.get([4], T.bf16), vi16).result

                # PV V-operand read for contraction step ks_pv, output d-block d.
                # Returns the MFMA-B fragment (vector<8xbf16>) for new_o[d].
                def _read_v(ks_pv, d):
                    if const_expr(_C4_DS_TR):
                        # 2 transpose reads replace 8 strided scalar loads.
                        #   n=lane (head 0-15), kg=k_group (0-3)
                        #   base = (kg*8 + n//4 + ks_pv*MFMA_K)*KV_LDS_STRIDE
                        #          + (n%4)*4 + d*MFMA_N ; read2 = base + 4*stride
                        tr_base = ((k_group * fx.Index(8) + lane // fx.Index(4)
                                    + fx.Index(ks_pv * MFMA_K)) * fx.Index(KV_LDS_STRIDE)
                                   + (lane % fx.Index(4)) * fx.Index(4)
                                   + fx.Index(d * MFMA_N))
                        r1 = _ds_tr_read(tr_base)
                        r2 = _ds_tr_read(tr_base + fx.Index(4 * KV_LDS_STRIDE))
                        return Vec.from_elements(
                            [Vec(r1)[0], Vec(r1)[1], Vec(r1)[2], Vec(r1)[3],
                             Vec(r2)[0], Vec(r2)[1], Vec(r2)[2], Vec(r2)[3]],
                            fx.BFloat16,
                        )
                    vvals = []
                    for j in range_constexpr(8):
                        key = fx.Index(ks_pv * MFMA_K) + k_group * fx.Index(8) + fx.Index(j)
                        voff = key * fx.Index(KV_LDS_STRIDE) + fx.Index(d * MFMA_N) + lane
                        vvals.append(_memref.load(lds_kv, [_raw(voff)]))
                    return Vec.from_elements(vvals, fx.BFloat16)

                for ks_pv in range_constexpr(K_STEPS_PV):
                    p_base = ((hb + lane) * fx.Index(P_LDS_STRIDE)
                              + fx.Index(ks_pv * MFMA_K) + k_group * fx.Index(8))
                    p_a = Vec.from_elements(
                        [to_bf16(_memref.load(lds_p, [_raw(p_base + fx.Index(j))]))
                         for j in range_constexpr(8)],
                        fx.BFloat16,
                    )
                    if const_expr(_C4_PIPE):
                        # Issue _C4_PV_PIPE independent V-reads (distinct regs) then
                        # their MFMAs, so multiple LDS reads are in flight per SIMD.
                        PV_W = _C4_PV_PIPE
                        n_chunks = (D_BLKS + PV_W - 1) // PV_W
                        for c in range_constexpr(n_chunks):
                            d0 = c * PV_W
                            w = min(PV_W, D_BLKS - d0)
                            vb = [_read_v(ks_pv, d0 + i) for i in range_constexpr(w)]
                            for i in range_constexpr(w):
                                new_o[d0 + i] = mfma_bf16(new_o[d0 + i], p_a, vb[i])
                    else:
                        for d in range_constexpr(D_BLKS):
                            new_o[d] = mfma_bf16(new_o[d], p_a, _read_v(ks_pv, d))

                new_carry += list(m_new) + list(l_new) + list(new_o)

            # One end-of-tile barrier instead of one per m_sub: each m_sub writes
            # a DISJOINT lds_p region (p_row = wave_head_base + m_sub*16 + ...) and
            # only READS the shared read-only lds_kv, so there is no inter-m_sub
            # LDS hazard.  The single barrier here gates the next tile's gather
            # (which overwrites lds_kv) against this tile's PV V-reads.  At
            # occupancy-1 every barrier is a fully-exposed stall, so dropping
            # M_SUBS-1 of them per tile directly attacks the ~70% stall bottleneck.
            gpu.barrier()
            return new_carry

        # ==== dual-scope online softmax: SWA (main) then EXTRA ==========
        loop_results = _init
        for kv_tile, _carry in range(0, KV_TILES_MAIN, 1, init=_init):
            new_carry = emit_tile(
                kv_tile, _carry,
                main_pool_ptr, main_idx_ptr, topklen_main_val,
                topk_main, block_size_main, PBB_MAIN,
            )
            loop_results = yield new_carry

        for kv_tile, _carry in range(0, KV_TILES_EXTRA, 1, init=loop_results):
            new_carry = emit_tile(
                kv_tile, _carry,
                extra_pool_ptr, extra_idx_ptr, topklen_extra_val,
                topk_extra, block_size_extra, PBB_EXTRA,
            )
            loop_results = yield new_carry

        # ==== finalize : normalize, fold attn_sink, lonely -> zero ======
        for m_sub in range_constexpr(M_SUBS):
            base = m_sub * PER_SUB
            m_run = [loop_results[base + r] for r in range_constexpr(4)]
            l_run = [loop_results[base + 4 + r] for r in range_constexpr(4)]
            o_acc = [loop_results[base + 8 + d] for d in range_constexpr(D_BLKS)]
            hb = wave_head_base + fx.Index(m_sub * MFMA_N)
            for r in range_constexpr(4):
                head = hb + k_group * fx.Index(4) + fx.Index(r)
                l_r = l_run[r]
                m_r = m_run[r]
                lonely_f = _fsub(c_one, _fmin(_fmul(l_r, c_big), c_one))
                one_m = _fsub(c_one, lonely_f)
                m_safe = _fmul(m_r, one_m)
                if const_expr(has_attn_sink):
                    sink = load_f32_elem(sink_ptr, head)
                    sink_term = _exp2(_fsub(_fmul(sink, c_log2e), m_safe))
                    denom = _fadd(l_r, sink_term)
                else:
                    denom = l_r
                safe_denom = _fadd(_fmul(denom, one_m), lonely_f)
                inv = arith.divf(_raw(c_one), _raw(safe_denom), fastmath=fm)
                for d in range_constexpr(D_BLKS):
                    o_norm = _fmul(Vec(o_acc[d])[r], inv)
                    out_off = (pid_t * fx.Index(TOK_OUT_STRIDE)
                               + head * fx.Index(HEAD_OUT_STRIDE)
                               + fx.Index(d * MFMA_N) + lane)
                    out_gep = buffer_ops.get_element_ptr(
                        out_ptr, fx.Int64(out_off), elem_type=T.bf16
                    )
                    _llvm.StoreOp(to_bf16(o_norm).ir_value(), out_gep)
        return

    @flyc.jit
    def launch_dual_scope_prefill_c4(
        Q, KV_Main, Indices_Main, TopkLen_Main,
        KV_Extra, Indices_Extra, TopkLen_Extra,
        AttnSink, Out, total_tokens: fx.Int32,
        stream: "fx.Stream" = fx.Stream(None),
    ):
        alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalize()

        tokens_idx = fx.Index(total_tokens)
        grid_h = fx.Index((h_q + BLOCK_H - 1) // BLOCK_H)
        launcher = dual_scope_prefill_kernel_c4(
            Q, KV_Main, Indices_Main, TopkLen_Main,
            KV_Extra, Indices_Extra, TopkLen_Extra,
            AttnSink, Out, total_tokens,
        )

        passthrough = []
        for pair in [
            ("denormal-fp-math-f32", "preserve-sign,preserve-sign"),
            ("no-nans-fp-math", "true"),
            ("unsafe-fp-math", "true"),
        ]:
            passthrough.append(
                ir.ArrayAttr.get(
                    [ir.StringAttr.get(pair[0]), ir.StringAttr.get(pair[1])]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough)
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                    T.i32, int(waves_per_eu)
                )

        launcher.launch(
            grid=(grid_h, tokens_idx, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_dual_scope_prefill_c4(*args, **kwargs)

    return _launch


# ============================================================================
# C128 / low-topk dual-scope kernel (separate specialized variant).
#
# This mirrors Triton's FIRST, low-topk config of the dual-scope kernel for the
# C128 regime (compress_ratio == 128): topk_main=128, topk_extra=64
# (total_topk=192), block_size_main=256, block_size_extra=2.  Like the C4
# variant it is a DISTINCT function, NOT a parameterization of the base kernel.
# The inner MATH is byte-for-byte the same pipeline as the base/C4 kernels
# (gather -> e4m3fn fp8 dequant -> bf16 QK MFMA -> exp2 online softmax ->
# bf16 PV MFMA -> attn_sink fold + lonely-zero writeback).
#
# DESIGN (tiling / wave choice + LDS budget):
#   * Triton's low-topk config nominally uses BLOCK_N=256 / num_warps=8 /
#     BLOCK_H=64.  We deliberately DO NOT copy those numbers: a flat
#     BLOCK_N=256-key bf16 KV tile would be 256*512*2 = 256 KB of LDS, which
#     EXCEEDS the 160 KB gfx950 (MI350X) group-segment limit.  The C128 scopes
#     are tiny (128 main + 64 extra = 192 keys total), so we reuse the PROVEN,
#     LDS-sound C4 tiling unchanged: BLOCK_N=64, BLOCK_H=128, 4 waves of 64
#     lanes, 2 MFMA m-subtiles per wave.  This is the simplest provably-correct
#     layout; the exact BLOCK_N will be revisited by the FlyDSL autotuner later.
#   * 1 CTA per query token handles ALL BLOCK_H=128 heads, so the fp8
#     gather+dequant of each 64-key KV tile is done ONCE for all 128 heads.
#     With topk_main=128 / topk_extra=64 this is 2 main tiles + 1 extra tile.
#   * Each wave OWNS 32 heads (2 m-subtiles), so the online-softmax cross-key
#     reduction stays within a wave's 16-lane group (shuffle_xor), exactly like
#     the C4 kernel.  The 64-key x 512-dim KV tile is staged into LDS
#     cooperatively by all 256 threads (key row = lin, head-dim quarter = wave).
#   * LDS: KV 64KB (64*512 bf16) + P 32KB (128*64 f32) = 96KB < 160KB.
#   * bs_extra=2 (vs C4's 64) ONLY changes the block->slot index math
#     (idx//bs, idx%bs) -- a compile-time constant; the gather/tile loop
#     structure is otherwise identical to the C4 kernel.
# ============================================================================
@functools.lru_cache(maxsize=64)
def _build_dual_scope_kernel_c128(
    h_q: int,
    head_dim: int = D_QK,
    head_dim_v: int = D_V,
    topk_main: int = 128,
    topk_extra: int = 64,
    block_size_main: int = 256,
    block_size_extra: int = 2,
    block_n: int = 64,
    block_h: int = 128,
    n_waves: int = 4,
    sm_scale: Optional[float] = None,
    has_attn_sink: bool = True,
    has_topk_length_main: bool = True,
    has_topk_length_extra: bool = True,
    waves_per_eu: int = 1,
):
    """Build (and cache) the C128 / low-topk fused dual-scope FlyDSL kernel.

    Numerically identical inner pipeline to ``_build_dual_scope_kernel`` and
    ``_build_dual_scope_kernel_c4`` (gather -> e4m3fn fp8 dequant -> bf16 QK
    MFMA -> exp2 online softmax -> bf16 PV MFMA -> attn_sink fold + lonely-zero
    writeback).  The ONLY differences are the TILING / WAVE structure, which is
    fully PARAMETERIZED over ``(block_h, block_n, n_waves)`` so the C128 grid
    point can be autotuned for its tiny 192-key problem:

      * ``BLOCK_SIZE = 64 * n_waves`` threads (n_waves wavefronts of 64 lanes).
      * Each wave OWNS ``HEADS_PER_WAVE = block_h // n_waves`` heads
        (= ``M_SUBS = HEADS_PER_WAVE // 16`` MFMA m-subtiles), so the
        online-softmax cross-key reduction stays inside a wave's 16-lane group
        (shuffle_xor), exactly like the C4 kernel.
      * ``grid_h = h_q // block_h`` CTAs per token (grid.x); each CTA handles a
        distinct ``block_h``-head tile (head base folds in block_idx.x).
      * The ``block_n``-key x head_dim dequant is staged into LDS cooperatively
        by ALL ``BLOCK_SIZE`` threads: thread (g_part, kv_row) dequants key row
        ``kv_row = tid % block_n`` and head-dim partition ``g_part = tid //
        block_n`` (``G_PARTS = BLOCK_SIZE // block_n`` partitions of the 448
        nope + 64 rope dims).  This generalizes the C4 fixed
        BLOCK_N=64/quarter-per-wave gather to any valid thread count.

    The default ``(block_h=128, block_n=64, n_waves=4)`` reproduces the original
    C4-style C128 tiling byte-for-byte.  Returns a python launcher with the same
    call signature as the base/C4 kernel launchers.
    """
    assert _HAS_FLYDSL, "FlyDSL is not importable (build is gfx950-container only)"
    assert head_dim == D_QK, f"only head_dim={D_QK} is supported"

    WARP_SIZE = 64
    MFMA_N = 16
    MFMA_K = 32
    N_WAVES = n_waves
    BLOCK_SIZE = WARP_SIZE * N_WAVES
    BLOCK_H = block_h
    G_PARTS = BLOCK_SIZE // block_n                  # head-dim partitions / key row

    assert block_n in (32, 64), "C128 kernel tiling supports BLOCK_N in {32, 64}"
    assert block_n % MFMA_N == 0 and block_n % MFMA_K == 0, "BLOCK_N must tile MFMA"
    assert h_q % block_h == 0, f"h_q={h_q} must be a multiple of BLOCK_H={block_h}"
    assert block_h % N_WAVES == 0, f"BLOCK_H={block_h} must be a multiple of n_waves={N_WAVES}"
    assert (block_h // N_WAVES) % MFMA_N == 0, (
        f"HEADS_PER_WAVE={block_h // N_WAVES} must be a multiple of MFMA_M={MFMA_N}"
    )
    assert BLOCK_SIZE % block_n == 0, (
        f"BLOCK_SIZE={BLOCK_SIZE} must be a multiple of BLOCK_N={block_n}"
    )
    assert D_NOPE % G_PARTS == 0 and (D_NOPE // G_PARTS) % 8 == 0, (
        f"D_NOPE={D_NOPE} not 8-divisible across G_PARTS={G_PARTS}"
    )
    assert D_ROPE % G_PARTS == 0 and (D_ROPE // G_PARTS) % 8 == 0, (
        f"D_ROPE={D_ROPE} not 8-divisible across G_PARTS={G_PARTS}"
    )
    assert topk_main % block_n == 0, f"topk_main must be a multiple of BLOCK_N={block_n}"
    assert topk_extra % block_n == 0, f"topk_extra must be a multiple of BLOCK_N={block_n}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    gpu_arch = get_rocm_arch()

    # ---- tiling (parameterized over block_h / block_n / n_waves) ----------
    HEADS_PER_WAVE = BLOCK_H // N_WAVES              # heads owned by each wave
    M_SUBS = HEADS_PER_WAVE // MFMA_N                # MFMA m-subtiles / wave
    K_STEPS_QK = head_dim // MFMA_K                  # 16
    N_BLKS_S = block_n // MFMA_N                      # 2 (bn=32) or 4 (bn=64)
    K_STEPS_PV = block_n // MFMA_K                    # 1 (bn=32) or 2 (bn=64)
    D_BLKS = head_dim // MFMA_N                       # 32
    KV_TILES_MAIN = topk_main // block_n
    KV_TILES_EXTRA = topk_extra // block_n

    TOK_OUT_STRIDE = h_q * head_dim_v
    HEAD_OUT_STRIDE = head_dim_v
    Q_STOK = h_q * head_dim

    # ---- pool byte geometry (data-major: 576 data + 8 scale per slot) -----
    SLOT_DATA_BYTES = D_NOPE + D_ROPE * 2           # 576
    SLOT_SCALE_BYTES = BYTES_PER_TOKEN_SCALE        # 8
    PBB_MAIN = block_size_main * (SLOT_DATA_BYTES + SLOT_SCALE_BYTES)
    PBB_EXTRA = block_size_extra * (SLOT_DATA_BYTES + SLOT_SCALE_BYTES)

    # gather: BLOCK_SIZE threads = block_n key rows x G_PARTS head-dim parts.
    # thread (g_part, kv_row) dequants key row kv_row, head-dim partition g_part.
    NOPE_PART = D_NOPE // G_PARTS
    NOPE_CHUNKS = NOPE_PART // 8
    ROPE_PART = D_ROPE // G_PARTS
    ROPE_CHUNKS = ROPE_PART // 8

    SM_LOG2E = float(sm_scale) * _LOG2E
    BIG = 1.0e30

    # ---- LDS layout (KV block_n*512 bf16 + P block_h*block_n f32; < 160KB) -
    # Per-row padding breaks the multiple-of-32-banks stride (see the C4 kernel)
    # so the lds_kv / lds_p accesses spread across LDS bank groups.
    KV_LDS_STRIDE = head_dim + _C4_KV_PAD
    P_LDS_STRIDE = block_n + _C4_P_PAD
    LDS_KV_BF16 = block_n * KV_LDS_STRIDE
    LDS_KV_BYTES = LDS_KV_BF16 * 2
    LDS_P_F32 = BLOCK_H * P_LDS_STRIDE
    LDS_P_BYTES = LDS_P_F32 * 4
    LDS_TOTAL = LDS_KV_BYTES + LDS_P_BYTES
    assert LDS_TOTAL <= 160 * 1024, f"LDS {LDS_TOTAL} exceeds 160KB budget"

    alloc = SmemAllocator(
        None, arch=gpu_arch,
        global_sym_name=f"dual_scope_c128_smem_{block_h}_{block_n}_{n_waves}",
    )
    base_off = alloc._align(alloc.ptr, 16)
    alloc.ptr = base_off + LDS_TOTAL
    kv_lds_off = base_off
    p_lds_off = base_off + LDS_KV_BYTES

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def dual_scope_prefill_kernel_c128(  # parameterized (block_h/block_n/n_waves)
        Q: fx.Tensor,             # [T, h_q, head_dim]            bf16
        KV_Main: fx.Tensor,       # [Nb_main, per_block_bytes]    uint8 raw pool
        Indices_Main: fx.Tensor,  # [T, topk_main]                int32
        TopkLen_Main: fx.Tensor,  # [T]                           int32 (1:1 per tok)
        KV_Extra: fx.Tensor,      # [Nb_extra, per_block_bytes]   uint8 raw pool
        Indices_Extra: fx.Tensor, # [T, topk_extra]               int32
        TopkLen_Extra: fx.Tensor, # [T]                           int32 (1:1 per tok)
        AttnSink: fx.Tensor,      # [h_q]                         f32
        Out: fx.Tensor,           # [T, h_q, head_dim_v]          bf16
        total_tokens: fx.Int32,
    ):
        # ---- LLVM pointer helpers (mirror the base kernel exactly) ------
        def _ptr_ty():
            return ir.Type.parse("!llvm.ptr")

        def _as_ptr(t):
            v = t
            if hasattr(v, "ir_value") and not isinstance(v, ir.Value):
                v = v.ir_value()
            return _fly.extract_aligned_pointer_as_index(_ptr_ty(), v)

        q_ptr           = _as_ptr(Q)
        out_ptr         = _as_ptr(Out)
        sink_ptr        = _as_ptr(AttnSink)
        main_pool_ptr   = _as_ptr(KV_Main)
        main_idx_ptr    = _as_ptr(Indices_Main)
        main_tkl_ptr    = _as_ptr(TopkLen_Main)
        extra_pool_ptr  = _as_ptr(KV_Extra)
        extra_idx_ptr   = _as_ptr(Indices_Extra)
        extra_tkl_ptr   = _as_ptr(TopkLen_Extra)

        # ---- typed constants / fast-math wrappers ----------------------
        v4f32_type  = Vec.make_type(4, fx.Float32)
        vec8bf16_ty = Vec.make_type(8, fx.BFloat16)
        fm = arith.FastMathFlags.fast

        def _fadd(a, b): return arith.addf(_raw(a), _raw(b), fastmath=fm)
        def _fmul(a, b): return arith.mulf(_raw(a), _raw(b), fastmath=fm)
        def _fsub(a, b): return arith.subf(_raw(a), _raw(b), fastmath=fm)
        def _fmax(a, b): return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result
        def _fmin(a, b): return arith.MinNumFOp(_raw(a), _raw(b), fastmath=fm).result
        def _exp2(x):    return rocdl.exp2(T.f32, _raw(x))

        c_one  = fx.Float32(1.0)
        c_big  = fx.Float32(BIG)
        c_log2e = fx.Float32(_LOG2E)
        c_sm_log2e = fx.Float32(SM_LOG2E)

        # ---- load helpers ----------------------------------------------
        def load_i32(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.i32)
            return _llvm.LoadOp(T.i32, gep).result

        def load_bf16_elem(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.bf16)
            return _llvm.LoadOp(T.bf16, gep).result

        def load_f32_elem(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.f32)
            return _llvm.LoadOp(T.f32, gep).result

        def load_u8_i32(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            u8 = _llvm.LoadOp(T.i8, gep).result
            return _mlir_arith.ExtUIOp(T.i32, u8).result

        def load_i64(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            return _llvm.LoadOp(T.i64, gep).result

        def load_bf16_byte(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            return _llvm.LoadOp(T.bf16, gep).result

        def extract_byte_i32(v_i64, j):
            sh  = _mlir_arith.ShRUIOp(v_i64, _raw(fx.Int64(j * 8))).result
            msk = _mlir_arith.AndIOp(sh, _raw(fx.Int64(0xFF))).result
            return _mlir_arith.TruncIOp(T.i32, msk).result

        def fp8_to_f32(byte_i32):
            return rocdl.cvt_f32_fp8(T.f32, byte_i32, fx.Int32(0))

        def ue8m0_to_f32(u8_i32):
            f = _mlir_arith.UIToFPOp(T.f32, u8_i32).result
            return _exp2(_fsub(f, fx.Float32(127.0)))

        def to_bf16(f32val):
            return fx.Float32(f32val).to(fx.BFloat16)

        def mfma_bf16(acc, a, b):
            return rocdl.mfma_f32_16x16x32_bf16(v4f32_type, [a, b, acc, 0, 0, 0])

        # ---- thread / block ids ----------------------------------------
        #   grid = (h_q/BLOCK_H, total_tokens); block = (BLOCK_SIZE,1,1).
        #   wave = tid//64 -> owns HEADS_PER_WAVE heads.  lin = tid%64.
        #   gather: kv_row = tid%block_n, g_part = tid//block_n (G_PARTS parts).
        tid       = fx.Index(gpu.thread_idx.x)
        bid_h_tile = fx.Index(gpu.block_idx.x)       # head-tile (grid_h CTAs/token)
        pid_t     = fx.Index(gpu.block_idx.y)
        wave      = tid // fx.Index(WARP_SIZE)      # 0..N_WAVES-1
        lin       = tid % fx.Index(WARP_SIZE)       # 0..63
        lane      = lin % fx.Index(MFMA_N)          # 0..15 MFMA row(A)/col(B/C)
        k_group   = lin // fx.Index(MFMA_N)         # 0..3  MFMA K-subgroup
        # head_tile = this CTA's GLOBAL head offset (grid_h CTAs/token);
        # wave_head_base = this wave's LOCAL head base within the CTA tile.
        # LDS-P / PV index by LOCAL head (lds_p is [BLOCK_H, block_n]); Q load,
        # attn_sink and the output writeback index by head_tile + LOCAL head.
        head_tile = bid_h_tile * fx.Index(BLOCK_H)
        wave_head_base = wave * fx.Index(HEADS_PER_WAVE)
        kv_row    = tid % fx.Index(block_n)          # gather: key row (0..block_n-1)
        g_part    = tid // fx.Index(block_n)         # gather: head-dim partition

        # ---- LDS ------------------------------------------------------
        lds_base = alloc.get_base()
        lds_kv = SmemPtr(lds_base, kv_lds_off, T.bf16, shape=(LDS_KV_BF16,)).get()
        lds_p  = SmemPtr(lds_base, p_lds_off,  T.f32,  shape=(LDS_P_F32,)).get()

        topklen_main_val  = load_i32(main_tkl_ptr,  pid_t)
        topklen_extra_val = load_i32(extra_tkl_ptr, pid_t)

        # ---- preload Q head tiles (M_SUBS x K_STEPS_QK x vector<8xbf16>) -
        q_packs = [[] for _ in range_constexpr(M_SUBS)]
        for m_sub in range_constexpr(M_SUBS):
            hb = wave_head_base + fx.Index(m_sub * MFMA_N)        # LOCAL head base
            hb_glb = head_tile + hb                              # GLOBAL head base
            for ks in range_constexpr(K_STEPS_QK):
                q_elems = []
                for j in range_constexpr(8):
                    q_eoff = (
                        pid_t * fx.Index(Q_STOK)
                        + (hb_glb + lane) * fx.Index(head_dim)
                        + fx.Index(ks * MFMA_K + j) + k_group * fx.Index(8)
                    )
                    q_elems.append(load_bf16_elem(q_ptr, q_eoff))
                q_packs[m_sub].append(Vec.from_elements(q_elems, fx.BFloat16))

        # ---- online-softmax carry: per m_sub [m(4), l(4), o_acc(D_BLKS)] -
        PER_SUB = 8 + D_BLKS
        _init = []
        for _ms in range_constexpr(M_SUBS):
            _init += [_raw(fx.Float32(-BIG)) for _ in range_constexpr(4)]
            _init += [_raw(fx.Float32(0.0)) for _ in range_constexpr(4)]
            _init += [_raw(Vec.filled(4, 0.0, fx.Float32)) for _ in range_constexpr(D_BLKS)]

        def emit_tile(kv_tile, carry, pool_ptr, idx_ptr, tkl_val, topk, bs, pbb):
            kv_pos_base = kv_tile * fx.Index(block_n)

            # === gather + dequant one BLOCK_N tile -> LDS (bf16) =========
            # thread (g_part, kv_row): key row kv_row, head-dim partition g_part
            # (NOPE_PART nope + ROPE_PART rope dims), spread over G_PARTS parts.
            idx_i32 = load_i32(idx_ptr, pid_t * fx.Index(topk)
                               + kv_pos_base + kv_row)
            idx_c = _mlir_arith.MaxSIOp(idx_i32, _raw(fx.Int32(0))).result
            blk = _mlir_arith.DivUIOp(idx_c, _raw(fx.Int32(bs))).result
            off = _mlir_arith.RemUIOp(idx_c, _raw(fx.Int32(bs))).result
            blk_i = fx.Index(blk)
            off_i = fx.Index(off)
            data_base = blk_i * fx.Index(pbb) + off_i * fx.Index(SLOT_DATA_BYTES)
            scale_base = (blk_i * fx.Index(pbb)
                          + fx.Index(bs * SLOT_DATA_BYTES)
                          + off_i * fx.Index(SLOT_SCALE_BYTES))

            # phase A: nope partition (NOPE_PART fp8) -> bf16, ue8m0 scale folded
            for dc in range_constexpr(NOPE_CHUNKS):
                abs_dim = g_part * fx.Index(NOPE_PART) + fx.Index(dc * 8)
                tile = abs_dim // fx.Index(QUANT_BLOCK)
                scale_f = ue8m0_to_f32(load_u8_i32(pool_ptr, scale_base + tile))
                raw_i64 = load_i64(pool_ptr, data_base + abs_dim)
                bvals = []
                for j in range_constexpr(8):
                    fv = _fmul(fp8_to_f32(extract_byte_i32(raw_i64, j)), scale_f)
                    bvals.append(to_bf16(fv))
                Vec.from_elements(bvals, fx.BFloat16).store(
                    lds_kv, [kv_row * fx.Index(KV_LDS_STRIDE) + abs_dim]
                )

            # phase B: rope partition (ROPE_PART*2 raw bytes -> ROPE_PART bf16 LE)
            for dc in range_constexpr(ROPE_CHUNKS):
                rope_idx = g_part * fx.Index(ROPE_PART) + fx.Index(dc * 8)
                bvals = []
                for j in range_constexpr(8):
                    boff = (data_base + fx.Index(D_NOPE)
                            + (rope_idx + fx.Index(j)) * fx.Index(2))
                    bvals.append(load_bf16_byte(pool_ptr, boff))
                Vec.from_elements(bvals, fx.BFloat16).store(
                    lds_kv, [kv_row * fx.Index(KV_LDS_STRIDE) + fx.Index(D_NOPE) + rope_idx]
                )

            gpu.barrier()

            # === validity mask (head-independent; computed once / tile) ===
            mask_f = []
            for nb in range_constexpr(N_BLKS_S):
                key_pos = kv_pos_base + fx.Index(nb * MFMA_N) + lane
                ki = load_i32(idx_ptr, pid_t * fx.Index(topk) + key_pos)
                ne = arith.cmpi(arith.CmpIPredicate.ne, ki, fx.Int32(-1))
                kp_i32 = _mlir_arith.IndexCastOp(T.i32, _raw(key_pos)).result
                lt = arith.cmpi(arith.CmpIPredicate.slt, kp_i32, tkl_val)
                good = _mlir_arith.AndIOp(_raw(ne), _raw(lt)).result
                mask_f.append(_mlir_arith.UIToFPOp(T.f32, good).result)

            new_carry = []
            for m_sub in range_constexpr(M_SUBS):
                base = m_sub * PER_SUB
                m_run = [carry[base + r] for r in range_constexpr(4)]
                l_run = [carry[base + 4 + r] for r in range_constexpr(4)]
                o_acc = [carry[base + 8 + d] for d in range_constexpr(D_BLKS)]
                hb = wave_head_base + fx.Index(m_sub * MFMA_N)

                # === QK GEMM : S[16, BLOCK_N] = Q @ K^T (bf16 MFMA) =======
                s_acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(N_BLKS_S)]
                for ks in range_constexpr(K_STEPS_QK):
                    q_a = q_packs[m_sub][ks]
                    for nb in range_constexpr(N_BLKS_S):
                        key = fx.Index(nb * MFMA_N) + lane
                        koff = (key * fx.Index(KV_LDS_STRIDE)
                                + fx.Index(ks * MFMA_K) + k_group * fx.Index(8))
                        k_b = Vec.load(vec8bf16_ty, lds_kv, [koff])
                        s_acc[nb] = mfma_bf16(s_acc[nb], q_a, k_b)

                # === online softmax (log2 space) =========================
                s_scaled = [
                    [_fmul(Vec(s_acc[nb])[r], c_sm_log2e) for r in range_constexpr(4)]
                    for nb in range_constexpr(N_BLKS_S)
                ]
                s_for_max = [
                    [_fadd(s_scaled[nb][r], _fmul(_fsub(mask_f[nb], c_one), c_big))
                     for r in range_constexpr(4)]
                    for nb in range_constexpr(N_BLKS_S)
                ]
                row_max = [s_for_max[0][r] for r in range_constexpr(4)]
                for nb in range_constexpr(N_BLKS_S):
                    if const_expr(nb > 0):
                        for r in range_constexpr(4):
                            row_max[r] = _fmax(row_max[r], s_for_max[nb][r])
                for xor_off in [8, 4, 2, 1]:
                    so = fx.Int32(xor_off)
                    for r in range_constexpr(4):
                        row_max[r] = _fmax(
                            row_max[r],
                            fx.Float32(row_max[r]).shuffle_xor(so, fx.Int32(WARP_SIZE)),
                        )

                m_new = [_fmax(m_run[r], row_max[r]) for r in range_constexpr(4)]
                corr  = [_exp2(_fsub(m_run[r], m_new[r])) for r in range_constexpr(4)]

                p_vals   = [[None] * 4 for _ in range_constexpr(N_BLKS_S)]
                tile_sum = [_raw(fx.Float32(0.0)) for _ in range_constexpr(4)]
                for nb in range_constexpr(N_BLKS_S):
                    for r in range_constexpr(4):
                        p = _fmul(_exp2(_fsub(s_scaled[nb][r], m_new[r])), mask_f[nb])
                        p_vals[nb][r] = p
                        tile_sum[r] = _fadd(tile_sum[r], p)
                for xor_off in [8, 4, 2, 1]:
                    so = fx.Int32(xor_off)
                    for r in range_constexpr(4):
                        tile_sum[r] = _fadd(
                            tile_sum[r],
                            fx.Float32(tile_sum[r]).shuffle_xor(so, fx.Int32(WARP_SIZE)),
                        )

                l_new = [_fadd(_fmul(corr[r], l_run[r]), tile_sum[r]) for r in range_constexpr(4)]
                if const_expr(_C4_LAZY_RESCALE):
                    # D4: skip the O rescale on tiles where no owned row's running
                    # max advances (corr == 1, identity multiply).  Predicate is
                    # reduced across the full 64-lane wave so scf.if is uniform.
                    rm_delta = _fsub(row_max[0], m_run[0])
                    for r in range_constexpr(4):
                        if const_expr(r > 0):
                            rm_delta = _fmax(rm_delta, _fsub(row_max[r], m_run[r]))
                    for xor_off in [32, 16, 8, 4, 2, 1]:
                        rm_delta = _fmax(
                            rm_delta,
                            fx.Float32(rm_delta).shuffle_xor(
                                fx.Int32(xor_off), fx.Int32(WARP_SIZE)
                            ),
                        )
                    should_scale = arith.cmpf(
                        arith.CmpFPredicate.OGT, _raw(rm_delta), _raw(fx.Float32(0.0))
                    )
                    o_raw = [_raw(o_acc[d]) for d in range_constexpr(D_BLKS)]
                    if_op = _scf.IfOp(
                        _raw(should_scale),
                        [v.type for v in o_raw],
                        has_else=True,
                        loc=ir.Location.unknown(),
                    )
                    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                        scaled_raw = []
                        for d in range_constexpr(D_BLKS):
                            ov = Vec(o_acc[d])
                            scaled_raw.append(
                                _raw(Vec.from_elements(
                                    [_fmul(ov[r], corr[r]) for r in range_constexpr(4)],
                                    fx.Float32,
                                ))
                            )
                        _scf.YieldOp(scaled_raw)
                    if len(if_op.regions[1].blocks) == 0:
                        if_op.regions[1].blocks.append(*[])
                    with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                        _scf.YieldOp(o_raw)
                    new_o = [Vec(r) for r in if_op.results]
                else:
                    new_o = []
                    for d in range_constexpr(D_BLKS):
                        ov = Vec(o_acc[d])
                        new_o.append(
                            Vec.from_elements(
                                [_fmul(ov[r], corr[r]) for r in range_constexpr(4)], fx.Float32
                            )
                        )

                # === stage P (f32) -> LDS, transposed [head_row][key] =====
                for nb in range_constexpr(N_BLKS_S):
                    for r in range_constexpr(4):
                        p_row = hb + k_group * fx.Index(4) + fx.Index(r)
                        p_col = fx.Index(nb * MFMA_N) + lane
                        _memref.store(_raw(p_vals[nb][r]), lds_p,
                                      [_raw(p_row * fx.Index(P_LDS_STRIDE) + p_col)])
                gpu.barrier()

                # === PV GEMM : O += P @ V (bf16 MFMA), K=block_n=2x32 ======
                # LDS transpose-read helper (gfx950 ds_read_b64_tr_b16): returns a
                # vector<4xbf16> = the 4 contraction keys at the calling lane's head,
                # transposed from lds_kv at bf16 element offset `elem_off`.
                def _ds_tr_read(elem_off):
                    base_mr = lds_base
                    if hasattr(base_mr, "ir_value") and not isinstance(base_mr, ir.Value):
                        base_mr = base_mr.ir_value()
                    base_idx = _memref.extract_aligned_pointer_as_index(base_mr)
                    base_i64 = _mlir_arith.IndexCastOp(T.i64, _raw(base_idx)).result
                    eoff_i64 = _mlir_arith.IndexCastOp(T.i64, _raw(elem_off)).result
                    byte_addr = _mlir_arith.AddIOp(
                        _mlir_arith.AddIOp(base_i64, _raw(fx.Int64(kv_lds_off))).result,
                        _mlir_arith.MulIOp(eoff_i64, _raw(fx.Int64(2))).result,
                    ).result
                    p3 = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), byte_addr)
                    vi16 = rocdl.ds_read_tr16_b64(ir.VectorType.get([4], T.i16), p3)
                    return arith.BitcastOp(ir.VectorType.get([4], T.bf16), vi16).result

                # PV V-operand read for contraction step ks_pv, output d-block d.
                # Returns the MFMA-B fragment (vector<8xbf16>) for new_o[d].
                def _read_v(ks_pv, d):
                    if const_expr(_C4_DS_TR):
                        # 2 transpose reads replace 8 strided scalar loads.
                        #   n=lane (head 0-15), kg=k_group (0-3)
                        #   base = (kg*8 + n//4 + ks_pv*MFMA_K)*KV_LDS_STRIDE
                        #          + (n%4)*4 + d*MFMA_N ; read2 = base + 4*stride
                        tr_base = ((k_group * fx.Index(8) + lane // fx.Index(4)
                                    + fx.Index(ks_pv * MFMA_K)) * fx.Index(KV_LDS_STRIDE)
                                   + (lane % fx.Index(4)) * fx.Index(4)
                                   + fx.Index(d * MFMA_N))
                        r1 = _ds_tr_read(tr_base)
                        r2 = _ds_tr_read(tr_base + fx.Index(4 * KV_LDS_STRIDE))
                        return Vec.from_elements(
                            [Vec(r1)[0], Vec(r1)[1], Vec(r1)[2], Vec(r1)[3],
                             Vec(r2)[0], Vec(r2)[1], Vec(r2)[2], Vec(r2)[3]],
                            fx.BFloat16,
                        )
                    vvals = []
                    for j in range_constexpr(8):
                        key = fx.Index(ks_pv * MFMA_K) + k_group * fx.Index(8) + fx.Index(j)
                        voff = key * fx.Index(KV_LDS_STRIDE) + fx.Index(d * MFMA_N) + lane
                        vvals.append(_memref.load(lds_kv, [_raw(voff)]))
                    return Vec.from_elements(vvals, fx.BFloat16)

                for ks_pv in range_constexpr(K_STEPS_PV):
                    p_base = ((hb + lane) * fx.Index(P_LDS_STRIDE)
                              + fx.Index(ks_pv * MFMA_K) + k_group * fx.Index(8))
                    p_a = Vec.from_elements(
                        [to_bf16(_memref.load(lds_p, [_raw(p_base + fx.Index(j))]))
                         for j in range_constexpr(8)],
                        fx.BFloat16,
                    )
                    if const_expr(_C4_PIPE):
                        # Issue _C4_PV_PIPE independent V-reads (distinct regs) then
                        # their MFMAs, so multiple LDS reads are in flight per SIMD.
                        PV_W = _C4_PV_PIPE
                        n_chunks = (D_BLKS + PV_W - 1) // PV_W
                        for c in range_constexpr(n_chunks):
                            d0 = c * PV_W
                            w = min(PV_W, D_BLKS - d0)
                            vb = [_read_v(ks_pv, d0 + i) for i in range_constexpr(w)]
                            for i in range_constexpr(w):
                                new_o[d0 + i] = mfma_bf16(new_o[d0 + i], p_a, vb[i])
                    else:
                        for d in range_constexpr(D_BLKS):
                            new_o[d] = mfma_bf16(new_o[d], p_a, _read_v(ks_pv, d))

                new_carry += list(m_new) + list(l_new) + list(new_o)

            # One end-of-tile barrier instead of one per m_sub: each m_sub writes
            # a DISJOINT lds_p region (p_row = wave_head_base + m_sub*16 + ...) and
            # only READS the shared read-only lds_kv, so there is no inter-m_sub
            # LDS hazard.  The single barrier here gates the next tile's gather
            # (which overwrites lds_kv) against this tile's PV V-reads.  At
            # occupancy-1 every barrier is a fully-exposed stall, so dropping
            # M_SUBS-1 of them per tile directly attacks the ~70% stall bottleneck.
            gpu.barrier()
            return new_carry

        # ==== dual-scope online softmax: SWA (main) then EXTRA ==========
        loop_results = _init
        for kv_tile, _carry in range(0, KV_TILES_MAIN, 1, init=_init):
            new_carry = emit_tile(
                kv_tile, _carry,
                main_pool_ptr, main_idx_ptr, topklen_main_val,
                topk_main, block_size_main, PBB_MAIN,
            )
            loop_results = yield new_carry

        for kv_tile, _carry in range(0, KV_TILES_EXTRA, 1, init=loop_results):
            new_carry = emit_tile(
                kv_tile, _carry,
                extra_pool_ptr, extra_idx_ptr, topklen_extra_val,
                topk_extra, block_size_extra, PBB_EXTRA,
            )
            loop_results = yield new_carry

        # ==== finalize : normalize, fold attn_sink, lonely -> zero ======
        for m_sub in range_constexpr(M_SUBS):
            base = m_sub * PER_SUB
            m_run = [loop_results[base + r] for r in range_constexpr(4)]
            l_run = [loop_results[base + 4 + r] for r in range_constexpr(4)]
            o_acc = [loop_results[base + 8 + d] for d in range_constexpr(D_BLKS)]
            hb = wave_head_base + fx.Index(m_sub * MFMA_N)        # LOCAL head base
            for r in range_constexpr(4):
                # GLOBAL head index (head_tile folds in this CTA's grid_h offset)
                head = head_tile + hb + k_group * fx.Index(4) + fx.Index(r)
                l_r = l_run[r]
                m_r = m_run[r]
                lonely_f = _fsub(c_one, _fmin(_fmul(l_r, c_big), c_one))
                one_m = _fsub(c_one, lonely_f)
                m_safe = _fmul(m_r, one_m)
                if const_expr(has_attn_sink):
                    sink = load_f32_elem(sink_ptr, head)
                    sink_term = _exp2(_fsub(_fmul(sink, c_log2e), m_safe))
                    denom = _fadd(l_r, sink_term)
                else:
                    denom = l_r
                safe_denom = _fadd(_fmul(denom, one_m), lonely_f)
                inv = arith.divf(_raw(c_one), _raw(safe_denom), fastmath=fm)
                for d in range_constexpr(D_BLKS):
                    o_norm = _fmul(Vec(o_acc[d])[r], inv)
                    out_off = (pid_t * fx.Index(TOK_OUT_STRIDE)
                               + head * fx.Index(HEAD_OUT_STRIDE)
                               + fx.Index(d * MFMA_N) + lane)
                    out_gep = buffer_ops.get_element_ptr(
                        out_ptr, fx.Int64(out_off), elem_type=T.bf16
                    )
                    _llvm.StoreOp(to_bf16(o_norm).ir_value(), out_gep)
        return

    @flyc.jit
    def launch_dual_scope_prefill_c128(
        Q, KV_Main, Indices_Main, TopkLen_Main,
        KV_Extra, Indices_Extra, TopkLen_Extra,
        AttnSink, Out, total_tokens: fx.Int32,
        stream: "fx.Stream" = fx.Stream(None),
    ):
        alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalize()

        tokens_idx = fx.Index(total_tokens)
        grid_h = fx.Index((h_q + BLOCK_H - 1) // BLOCK_H)
        launcher = dual_scope_prefill_kernel_c128(
            Q, KV_Main, Indices_Main, TopkLen_Main,
            KV_Extra, Indices_Extra, TopkLen_Extra,
            AttnSink, Out, total_tokens,
        )

        passthrough = []
        for pair in [
            ("denormal-fp-math-f32", "preserve-sign,preserve-sign"),
            ("no-nans-fp-math", "true"),
            ("unsafe-fp-math", "true"),
        ]:
            passthrough.append(
                ir.ArrayAttr.get(
                    [ir.StringAttr.get(pair[0]), ir.StringAttr.get(pair[1])]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough)
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                    T.i32, int(waves_per_eu)
                )

        launcher.launch(
            grid=(grid_h, tokens_idx, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_dual_scope_prefill_c128(*args, **kwargs)

    return _launch


def _flydsl_dual_scope_kernel_impl(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_length: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    compress_ratio: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
) -> torch.Tensor:
    """Thin launcher: normalize inputs, build/cache the kernel, launch, return.

    Allocates the output [T, H, head_dim_v] bf16 and launches the live FlyDSL
    dual-scope kernel, which computes the full attention (gather + fp8 dequant +
    QK + online softmax + PV).  Reached from the public entry whenever
    ``_USE_FLYDSL_KERNEL`` is True and FlyDSL is available on the gfx950 host.
    Q / attn_sink are padded to a multiple of BLOCK_H heads; the extra padding
    heads are sliced off (``[:, :H]``) before returning.
    """
    if _FLYDSL_TIMING:
        torch.cuda.synchronize()
        _ev_start = torch.cuda.Event(enable_timing=True)
        _ev_prep = torch.cuda.Event(enable_timing=True)
        _ev_k0 = torch.cuda.Event(enable_timing=True)
        _ev_k1 = torch.cuda.Event(enable_timing=True)
        _ev_end = torch.cuda.Event(enable_timing=True)
        _ev_start.record()

    q3 = _normalize_q(q)                         # [T, H, D_QK]
    device = q3.device
    T_tok, H, _ = q3.shape

    # ---- specialized-kernel dispatch -----------------------------------
    # compress_ratio is the discriminator (passed directly by the backend):
    #   * compress_ratio == 4   -> the C4 / high-topk BLOCK_H=128 / BLOCK_N=64,
    #                              4-wave kernel (1 CTA/token handles all 128
    #                              heads; gather+dequant done once for all heads).
    #   * compress_ratio == 128 -> the C128 / low-topk kernel at the tuned tiling
    #                              from _C128_PLAN (block_h, block_n, n_waves), OR
    #                              the base kernel when _C128_PLAN is None.
    #   * any other ratio       -> the original BLOCK_H=16 / BLOCK_N=32 base
    #                              kernel as a safe fallback.
    # The selected (block_h, block_n) drive the padding: topk_main / topk_extra
    # are padded up to a multiple of block_n with -1 sentinels (which mask out
    # exactly like real padding); h_q is padded to a multiple of block_h.
    ratio = int(compress_ratio)
    use_c4 = ratio == 4
    # C4 routing: at high T_tok (high concurrency), use BLOCK_H=64 (occupancy=2)
    # instead of BLOCK_H=128 (occupancy=1) to hide barrier stalls behind a second
    # resident CTA.  _C4_PLAN overrides this when set (autotuning / A/B).
    use_c4_high_conc = use_c4 and T_tok >= _C4_HIGH_CONC_THRESH and _C4_PLAN is None
    c4_plan = (
        _C4_HIGH_CONC_PLAN if use_c4_high_conc
        else _C4_PLAN if use_c4
        else None
    )
    use_c4_param = use_c4 and c4_plan is not None  # None -> legacy fixed C4
    c128_plan = _C128_PLAN if ratio == 128 else None
    use_c128 = ratio == 128 and c128_plan is not None  # None -> route to base
    use_specialized = use_c4 or use_c128
    if use_c4_param:
        block_h, block_n = c4_plan[0], c4_plan[1]
    elif use_c4:
        block_h, block_n = 128, 64
    elif use_c128:
        block_h, block_n = c128_plan[0], c128_plan[1]
    else:
        block_h, block_n = 16, 32
    h_q_pad = ((H + block_h - 1) // block_h) * block_h

    def _pad_topk(idx: torch.Tensor) -> torch.Tensor:
        """Pad the topk (column) axis of an indices tensor up to a multiple of
        ``block_n`` with -1 (invalid) sentinels.  No-op when already aligned."""
        tk = idx.shape[1]
        tk_pad = ((tk + block_n - 1) // block_n) * block_n
        if tk_pad == tk:
            return idx
        pad = torch.full((idx.shape[0], tk_pad - tk), -1, dtype=idx.dtype, device=idx.device)
        return torch.cat([idx, pad], dim=1).contiguous()

    # --- main (SWA) scope tensors ---------------------------------------
    swa_idx = _normalize_indices(swa_indices).to(torch.int32).contiguous()  # [T, topk_m]
    if use_specialized:
        swa_idx = _pad_topk(swa_idx)
    topk_main = swa_idx.shape[1]
    swa_u8, _nb_main, bs_main = _block_layout(swa_k_cache)                   # [Nb, pbb] u8

    # --- extra scope tensors (dummy, non-indexed, when absent) ----------
    has_extra = extra_k_cache is not None and extra_indices is not None
    if has_extra:
        extra_idx = _normalize_indices(extra_indices).to(torch.int32).contiguous()
        if use_specialized:
            extra_idx = _pad_topk(extra_idx)
        topk_extra = extra_idx.shape[1]
        extra_u8, _nb_extra, bs_extra = _block_layout(extra_k_cache)
    else:
        # No extra scope: feed an all-invalid (-1) dummy so the kernel's extra
        # tiles mask out to zero contribution (matches the reference, which
        # skips the extra scope entirely).
        topk_extra = topk_main
        # Constant all-(-1) dummy; cached/reused across calls (read-only input).
        extra_idx = _cached_full((T_tok, topk_extra), -1, torch.int32, device)
        extra_u8 = swa_u8
        bs_extra = bs_main

    # --- per-token topk_length (dummy full-length when absent) ----------
    if swa_topk_length is not None:
        topklen_main = (
            swa_topk_length
            if swa_topk_length.dtype == torch.int32 and swa_topk_length.is_contiguous()
            else swa_topk_length.to(torch.int32).contiguous()
        )
    else:
        topklen_main = _cached_full((T_tok,), topk_main, torch.int32, device)
    if has_extra and extra_topk_length is not None:
        topklen_extra = (
            extra_topk_length
            if extra_topk_length.dtype == torch.int32 and extra_topk_length.is_contiguous()
            else extra_topk_length.to(torch.int32).contiguous()
        )
    else:
        topklen_extra = _cached_full((T_tok,), topk_extra, torch.int32, device)

    # --- attn_sink (padded/dummy to h_q_pad) ----------------------------
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).reshape(-1).contiguous()
        if sink.shape[0] < h_q_pad:
            sink = torch.cat(
                [sink, torch.zeros(h_q_pad - sink.shape[0], dtype=torch.float32, device=device)]
            )
    else:
        # Constant zero sink; cached/reused across calls (read-only input).
        sink = _cached_zeros((h_q_pad,), torch.float32, device)

    # Q heads.  FAST PATH (h_q_pad == H, the production C4/C128 case at H=128):
    # no head padding is required, so feed q3 straight through (bf16 +
    # contiguous, already ensured by _normalize_q) and skip the ~1GB
    # zeros-alloc + masked copy entirely.  When padding IS needed, allocate
    # UNINITIALIZED (torch.empty) and copy only the real heads: attention is
    # per-head independent and the padding heads are sliced off the output, so
    # their garbage Q never reaches a returned value.
    if h_q_pad == H:
        q_pad = q3 if q3.dtype == torch.bfloat16 else q3.to(torch.bfloat16)
    else:
        q_pad = torch.empty((T_tok, h_q_pad, D_QK), dtype=torch.bfloat16, device=device)
        q_pad[:, :H] = q3.to(torch.bfloat16)

    # Output tile: the kernel writes the full [T, h_q_pad, head_dim_v] tile
    # (including explicit zero rows for lonely queries), so zeroing the buffer
    # first is wasteful -- allocate uninitialized.
    out = torch.empty((T_tok, h_q_pad, head_dim_v), dtype=torch.bfloat16, device=device)

    # Dispatch on compress_ratio: compress_ratio==4 -> the specialized C4 /
    # high-topk kernel (BLOCK_H=128/BLOCK_N=64, 4 waves); compress_ratio==128 ->
    # the specialized C128 / low-topk kernel at the AUTOTUNED tiling
    # (BLOCK_H=128/BLOCK_N=64, N_WAVES=8 -- see _C128_* above); any other ratio
    # -> the original BLOCK_H=16/BLOCK_N=32 base kernel as a safe fallback. All
    # three builders expose the same launcher call signature.
    _build_kw = dict(
        h_q=h_q_pad,
        head_dim=D_QK,
        head_dim_v=head_dim_v,
        topk_main=topk_main,
        topk_extra=topk_extra,
        block_size_main=bs_main,
        block_size_extra=bs_extra,
        sm_scale=float(softmax_scale),
        has_attn_sink=attn_sink is not None,
        has_topk_length_main=swa_topk_length is not None,
        has_topk_length_extra=(has_extra and extra_topk_length is not None),
    )
    if _FLYDSL_TIMING:
        _ev_prep.record()
        _builder = (
            _build_dual_scope_kernel_c128 if use_c4_param
            else _build_dual_scope_kernel_c4 if use_c4
            else _build_dual_scope_kernel_c128 if use_c128
            else _build_dual_scope_kernel
        )
        _misses_before = _builder.cache_info().misses
        _t_build0 = time.perf_counter()

    if use_c4_param:
        # Parameterized BLOCK_N=32 occupancy-2 C4 path (byte-identical math to
        # the fixed C4 kernel at the default tiling; halved LDS lifts the
        # per-CU occupancy from 1 to 2-3).
        kernel = _build_dual_scope_kernel_c128(
            block_h=c4_plan[0], block_n=c4_plan[1],
            n_waves=c4_plan[2], waves_per_eu=c4_plan[3], **_build_kw,
        )
    elif use_c4:
        kernel = _build_dual_scope_kernel_c4(**_build_kw)
    elif use_c128:
        kernel = _build_dual_scope_kernel_c128(
            block_h=c128_plan[0], block_n=c128_plan[1],
            n_waves=c128_plan[2], **_build_kw,
        )
    else:
        kernel = _build_dual_scope_kernel(**_build_kw)

    if _FLYDSL_TIMING:
        _build_ms = (time.perf_counter() - _t_build0) * 1e3
        _was_miss = _builder.cache_info().misses > _misses_before

    stream = torch.cuda.current_stream()
    if _FLYDSL_TIMING:
        _ev_k0.record()
    kernel(
        q_pad, swa_u8, swa_idx, topklen_main,
        extra_u8, extra_idx, topklen_extra,
        sink, out, total_tokens=int(T_tok),
        stream=fx.Stream(stream.cuda_stream),
    )
    if _FLYDSL_TIMING:
        _ev_k1.record()
    # When no head padding was applied, ``out`` is already [T, H, head_dim_v]
    # and contiguous -- return it directly and skip the ~1GB slice + copy.
    result = out if h_q_pad == H else out[:, :H].contiguous()
    if _FLYDSL_TIMING:
        _ev_end.record()
        torch.cuda.synchronize()
        _prep_ms = _ev_start.elapsed_time(_ev_prep)
        _kern_ms = _ev_k0.elapsed_time(_ev_k1)
        _post_ms = _ev_k1.elapsed_time(_ev_end)
        try:
            with open(_FLYDSL_TIMING_FILE, "a") as _f:
                _f.write(
                    f"[FLYDSL] ratio={ratio} T={T_tok} H={H} h_pad={h_q_pad} "
                    f"bh={block_h} bn={block_n} tk_main={topk_main} "
                    f"tk_extra={topk_extra} build_miss={int(_was_miss)} "
                    f"prep_ms={_prep_ms:.3f} build_ms={_build_ms:.3f} "
                    f"kernel_ms={_kern_ms:.3f} post_ms={_post_ms:.3f}\n"
                )
        except Exception:
            pass
    return result


# ============================================================================
# Public entry -- matches the backend hook signature exactly.
# ============================================================================
def flydsl_dual_scope_prefill(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_length: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    compress_ratio: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
) -> torch.Tensor:
    """Fused dual-scope sparse-attention prefill (SWA + C4/C128).

    Contract (validated against the FlashMLA baseline in the backend hook):
        q            : [T, 1, H, 512] (or [T, H, 512])     bf16
        swa_k_cache  : [Nb_m, block_size_m, 1, 584]        uint8 raw pool
        swa_indices  : [T, 1, topk_m] (or [T, topk_m])     int32
        swa_topk_length   : [batch] int32 or None
        extra_k_cache: [Nb_e, block_size_e, 1, 584] uint8 or None
        extra_indices: [T, 1, topk_e] (or [T, topk_e]) int32 or None
        extra_topk_length : [batch] int32 or None
        compress_ratio    : 4 or 128 (extra-scope pool compression)
        softmax_scale     : float
        attn_sink         : [H] float or None
        head_dim_v        : 512

    Returns [T, H, head_dim_v] bf16; the hook squeezes a 4D result, so a 3D
    return matches ``flash_mla_with_kvcache_entrypoint(...)[0].squeeze(1)``.
    """
    if _USE_FLYDSL_KERNEL:
        # Live kernel path. If FlyDSL is unavailable here we must NOT silently
        # degrade to the PyTorch oracle -- the kernel is the intended live path,
        # so fail hard (strict mode) and let the server crash.
        if not _flydsl_available():
            raise RuntimeError(
                "[FlyDSL] _USE_FLYDSL_KERNEL=True but FlyDSL runtime is "
                "unavailable -- aborting (strict mode)"
            )
        return _flydsl_dual_scope_kernel_impl(
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
            head_dim_v=head_dim_v,
        )

    # Oracle path: only reached when _USE_FLYDSL_KERNEL is False (tests).
    return _torch_reference_dual_scope(
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
        head_dim_v=head_dim_v,
    )
