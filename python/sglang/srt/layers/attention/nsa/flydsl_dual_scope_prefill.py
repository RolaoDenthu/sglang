"""Fused dual-scope sparse-attention prefill for DeepSeek V4 / NSA (gfx950).

This module replaces the broken two-kernel + Python-LSE-merge prefill path with
a SINGLE fused dual-scope kernel that attends over both the sliding-window (SWA,
"main") scope and the compressed (C4 / C128, "extra") scope in one shared
online-softmax pass.  It is the FlyDSL analogue of the authoritative Triton
reference ``_fused_gather_attn_dsv4_dual_scope_kernel`` in
``nsa/triton_decode/triton_mla_kernels_decode_fused.py``.

INCREMENT 1 (this file): scaffold only.
  * ``flydsl_dual_scope_prefill``      -- public entry matching the backend hook.
  * ``_torch_reference_dual_scope``    -- faithful pure-PyTorch correctness oracle
                                          (currently wired as the live TEMP path).
  * ``_build_dual_scope_kernel``       -- FlyDSL kernel SKELETON (structure only,
                                          inner math stubbed with TODO(phase1)).
  * ``_flydsl_dual_scope_kernel_impl`` -- thin launcher, stubbed (NotImplemented),
                                          gated behind ``_USE_FLYDSL_KERNEL``.

The FlyDSL import is lazy/guarded so this module imports cleanly on a CPU host
where FlyDSL (a remote-container dependency) is absent; the reference path and
the test harness run without FlyDSL present.

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

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch

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

# Internal flag: when False the public entry uses the PyTorch reference (TEMP);
# flip to True once the FlyDSL kernel inner-math (phase1) lands and is validated.
_USE_FLYDSL_KERNEL: bool = False


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
# FlyDSL kernel SKELETON (structure only -- inner math stubbed for phase1)
# ============================================================================
def _flydsl_available() -> bool:
    """True only on a gfx950 host with FlyDSL importable."""
    try:
        from flydsl.runtime.device import get_rocm_arch  # noqa: WPS433

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
    fixed bf16/fp8 by phase, dims/topk/block sizes/BLOCK_H/sink/length flags
    select a distinct kernel specialization).  Returns a python launcher.

    NOTE (phase1): this is a STRUCTURE-ONLY skeleton.  The gather / dequant /
    QK / online-softmax / PV math is intentionally left as TODO(phase1) stubs.
    It must not be wired into the live path until those land.
    """
    # FlyDSL is a remote-container dependency; import lazily so this module
    # imports cleanly on CPU hosts.  The whole skeleton lives behind this import.
    import flydsl.compiler as flyc  # noqa: F401
    import flydsl.expr as fx  # noqa: F401
    from flydsl.compiler.kernel_function import CompilationContext  # noqa: F401
    from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl  # noqa: F401,E501
    from flydsl.expr.typing import T, Vector as Vec  # noqa: F401
    from flydsl.expr.utils.arith import _to_raw as _raw  # noqa: F401
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr  # noqa: F401
    from flydsl._mlir import ir  # noqa: F401
    from flydsl._mlir.dialects import arith as _mlir_arith, fly as _fly, llvm as _llvm, memref as _memref  # noqa: F401,E501

    assert head_dim == D_QK, f"phase1 supports head_dim={D_QK} only"
    assert tile_m == 16 and block_n == 32, "phase1 MFMA tiling is fixed at 16x32"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    gpu_arch = get_rocm_arch()

    # ---- tiling -----------------------------------------------------------
    # One CTA per query token (heads tiled by BLOCK_H), matching the Triton
    # baseline grid (cdiv(h_q, BLOCK_H), total_tokens).  DO NOT tile multiple
    # query tokens per block (that reintroduces the tile-anchor indexing bug).
    WARP_SIZE = 64
    MFMA_N = 16
    MFMA_K = 32
    BLOCK_SIZE = WARP_SIZE
    K_STEPS_QK = head_dim // MFMA_K     # 16
    N_BLKS_S = block_n // MFMA_N        # 2
    D_BLKS = head_dim // MFMA_N         # 32
    KV_TILES_MAIN = topk_main // block_n
    KV_TILES_EXTRA = topk_extra // block_n

    # Head tiling: one CTA per query token (grid.y), heads tiled by BLOCK_H
    # (grid.x).  The zero-stub writeback distributes the head_dim_v output
    # columns of each owned head across the BLOCK_SIZE-thread wave.
    BLOCK_H = block_h
    assert head_dim_v % BLOCK_SIZE == 0, (
        f"head_dim_v={head_dim_v} must be divisible by BLOCK_SIZE={BLOCK_SIZE}"
    )
    ELEMS_PER_THREAD = head_dim_v // BLOCK_SIZE     # 512 // 64 = 8
    TOK_OUT_STRIDE = h_q * head_dim_v               # Out token stride (elems)
    HEAD_OUT_STRIDE = head_dim_v                    # Out head stride (elems)

    # ---- LDS layout -------------------------------------------------------
    # KV tile staged as i64 packs (8 bf16/fp8 lanes per i64); P (attn weights)
    # staged as f32.  Same shape as the artefact kernels.
    LDS_KV_I64 = block_n * head_dim // 8
    LDS_KV_BYTES = LDS_KV_I64 * 8
    LDS_P_F32 = tile_m * block_n
    LDS_P_BYTES = LDS_P_F32 * 4
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

        out_ptr = _as_ptr(Out)

        # ----------------------------------------------------------------
        # Thread / block ids.
        #   grid = (cdiv(h_q, BLOCK_H), total_tokens); block = (BLOCK_SIZE,1,1).
        #   One CTA per query token (s_q == 1); heads tiled by BLOCK_H.  This
        #   intentionally differs from the artefacts' (cdiv(tokens, TILE_M), h_q)
        #   grid -- keeping one CTA per token avoids the tile-anchor index bug
        #   and matches topk_length being 1:1 per query token.
        # ----------------------------------------------------------------
        tid        = fx.Index(gpu.thread_idx.x)
        bid_h_tile = fx.Index(gpu.block_idx.x)   # head-tile id
        pid_t      = fx.Index(gpu.block_idx.y)   # query-token id

        # ----------------------------------------------------------------
        # LDS allocation (declared + finalized now; real KV/P staging is phase1).
        # A trivial store + barrier exercises the LDS path so the JIT'd kernel
        # references the finalized shared-memory global.
        # ----------------------------------------------------------------
        lds_base   = alloc.get_base()
        lds_kv_i64 = SmemPtr(lds_base, kv_lds_off, T.i64, shape=(LDS_KV_I64,)).get()
        lds_p_f32  = SmemPtr(lds_base, p_lds_off,  T.f32, shape=(LDS_P_F32,)).get()  # noqa: F841
        _memref.store(_raw(fx.Int64(0)), lds_kv_i64, [_raw(tid)])
        gpu.barrier()

        # ================================================================
        # phase1 inner-math TODOs (intentionally NOT implemented here).
        # ----------------------------------------------------------------
        # TODO(phase1): pre-load this token's Q head tile into K_STEPS_QK i64
        #   register packs (bf16, 8 lanes per i64); init online-softmax carry
        #   m_i=-inf (4), l_i=0 (4), o_acc=0 (D_BLKS x v4f32).
        # TODO(phase1) _process_kv_tile(carry, kv_cache, idx_tensor, kv_pos_base,
        #   block_size, topk_len): per BLOCK_N tile --
        #     gather: load idx at (token, kv_pos_base+row); mask idx==-1 and
        #       beyond topk_len -> invalid via BRANCHLESS bitmask (single LDS
        #       store per thread; MLIR SelectOp caused a prior NameError).
        #     decompose: block_idx = idx // block_size, off = idx % block_size
        #       (power-of-2 shift/mask); data_base = block_idx*per_block_bytes +
        #       off*576; scale_base = block_idx*per_block_bytes +
        #       block_size*576 + off*8.
        #     dequant nope (448 fp8): fp8->f32 via the gfx950 OCP e4m3fn path
        #       (NOT the fnuz rocdl.cvt_f32_fp8 used by the artefacts) * ue8m0
        #       exp2(scale_byte[d//64]-127); keep bf16.  NO fp8 re-quant.
        #     dequant rope (128 raw bytes -> 64 bf16 LE): bitcast; concat after
        #       nope -> 512-dim bf16 K (== V).  Stage K -> LDS i64; barrier.
        #     QK GEMM: S[BLOCK_H,BLOCK_N] += Q_tile @ K_tile^T via
        #       rocdl.mfma_f32_16x16x32_bf16 (bf16 in, f32 accum), K_STEPS_QK steps.
        #     mask invalid -> S=-inf BEFORE the softmax max.
        #     online softmax (log2 space): warp shuffle_xor row-max; m_new=
        #       max(m_run,row_max); corr=exp2(m_run-m_new); p=exp2(S*sm_scale*
        #       LOG2E - m_new); l_new=corr*l_run+sum(p); scale o_acc by corr.
        #     stage P (f32) -> LDS; barrier; PV GEMM: o_acc += P @ V via MFMA.
        # TODO(phase1): SWA (main) tiles first (KV_TILES_MAIN), then EXTRA tiles
        #   (KV_TILES_EXTRA), into ONE shared online-softmax state; early-skip
        #   tiles beyond TopkLen_Main / TopkLen_Extra.
        # TODO(phase1): finalize -- denom = l_i + (exp2((AttnSink-m_i)*LOG2E) if
        #   HAS_ATTN_SINK); output_scale = 1/denom; lonely (l_i==0) -> zero row;
        #   o = (o_acc * output_scale).to(bf16); store [head_dim_v].
        # ================================================================

        # ----------------------------------------------------------------
        # phase1 writeback: store ZEROS for every (token, head) this CTA owns.
        # Each of the BLOCK_SIZE threads strides over the head_dim_v columns
        # (ELEMS_PER_THREAD each) for each of the BLOCK_H heads in this tile.
        # ----------------------------------------------------------------
        zero_bf16 = fx.Float32(0.0).to(fx.BFloat16).ir_value()
        for h_local in range_constexpr(BLOCK_H):
            head = bid_h_tile * fx.Index(BLOCK_H) + fx.Index(h_local)
            for je in range_constexpr(ELEMS_PER_THREAD):
                d = tid + fx.Index(je * BLOCK_SIZE)
                out_elem = (
                    pid_t * fx.Index(TOK_OUT_STRIDE)
                    + head * fx.Index(HEAD_OUT_STRIDE)
                    + d
                )
                out_gep = buffer_ops.get_element_ptr(
                    out_ptr, fx.Int64(out_elem), elem_type=T.bf16
                )
                _llvm.StoreOp(zero_bf16, out_gep)
        return

    @flyc.jit
    def launch_dual_scope_prefill(
        Q, KV_Main, Indices_Main, TopkLen_Main,
        KV_Extra, Indices_Extra, TopkLen_Extra,
        AttnSink, Out, total_tokens,
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

    COMPILE-FIRST milestone: this allocates the output [T, H, head_dim_v] bf16
    and launches the FlyDSL skeleton kernel, which writes ZEROS (inner math is
    still a phase1 TODO).  It is wired behind ``_USE_FLYDSL_KERNEL`` (currently
    False) so the public live path keeps using ``_torch_reference_dual_scope``;
    the kernel is exercised only via the dedicated launch test.
    """
    import flydsl.expr as fx  # remote-container dependency; import lazily.

    q3 = _normalize_q(q)                         # [T, H, D_QK]
    device = q3.device
    T_tok, H, _ = q3.shape

    block_h = 16
    h_q_pad = ((H + block_h - 1) // block_h) * block_h

    # --- main (SWA) scope tensors ---------------------------------------
    swa_idx = _normalize_indices(swa_indices).to(torch.int32).contiguous()  # [T, topk_m]
    topk_main = swa_idx.shape[1]
    swa_u8, _nb_main, bs_main = _block_layout(swa_k_cache)                   # [Nb, pbb] u8

    # --- extra scope tensors (dummy, non-indexed, when absent) ----------
    has_extra = extra_k_cache is not None and extra_indices is not None
    if has_extra:
        extra_idx = _normalize_indices(extra_indices).to(torch.int32).contiguous()
        topk_extra = extra_idx.shape[1]
        extra_u8, _nb_extra, bs_extra = _block_layout(extra_k_cache)
    else:
        topk_extra = topk_main
        extra_idx = torch.zeros((T_tok, topk_extra), dtype=torch.int32, device=device)
        extra_u8 = swa_u8
        bs_extra = bs_main

    # --- per-token topk_length (dummy full-length when absent) ----------
    if swa_topk_length is not None:
        topklen_main = swa_topk_length.to(torch.int32).contiguous()
    else:
        topklen_main = torch.full((T_tok,), topk_main, dtype=torch.int32, device=device)
    if has_extra and extra_topk_length is not None:
        topklen_extra = extra_topk_length.to(torch.int32).contiguous()
    else:
        topklen_extra = torch.full((T_tok,), topk_extra, dtype=torch.int32, device=device)

    # --- attn_sink (padded/dummy to h_q_pad) ----------------------------
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).reshape(-1).contiguous()
        if sink.shape[0] < h_q_pad:
            sink = torch.cat(
                [sink, torch.zeros(h_q_pad - sink.shape[0], dtype=torch.float32, device=device)]
            )
    else:
        sink = torch.zeros(h_q_pad, dtype=torch.float32, device=device)

    # Output is pre-zeroed; the kernel also stores zeros, exercising writeback.
    out = torch.zeros((T_tok, h_q_pad, head_dim_v), dtype=torch.bfloat16, device=device)

    kernel = _build_dual_scope_kernel(
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

    stream = torch.cuda.current_stream()
    kernel(
        q3.contiguous(), swa_u8, swa_idx, topklen_main,
        extra_u8, extra_idx, topklen_extra,
        sink, out, total_tokens=int(T_tok),
        stream=fx.Stream(stream.cuda_stream),
    )
    return out[:, :H].contiguous()


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
    if _USE_FLYDSL_KERNEL and _flydsl_available():
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

    # TEMP: reference path until FlyDSL kernel inner-math lands.
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
