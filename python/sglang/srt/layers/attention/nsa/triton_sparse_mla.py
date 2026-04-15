"""
Triton kernels for sparse MLA attention decode (bf16 and FP8).

bf16 kernel: single-pass, ported from aiter's unified_attention_sparse_mla.
FP8 kernel:  split-k (partial + combine), matching tilelang's two-phase approach.
             4x128 chunked FP8 GEMM with P->FP8 scaling for SV dot.
             topk dimension parallelised across n_groups blocks for CU occupancy.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported, is_hip

_is_hip = is_hip()
_is_gfx95 = is_gfx95_supported()


@triton.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        if val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _triton_sparse_mla_decode_kernel(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    topk_indices_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    topk_count: tl.constexpr,
    query_start_len_ptr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    ROPE_RANK: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    BLOCK_Q: tl.constexpr = 1
    kv_head_idx = 0

    q_block_global_idx = tl.program_id(0)
    q_ind = q_block_global_idx // (num_query_heads // BLOCK_M)
    head_ind = q_block_global_idx % (num_query_heads // BLOCK_M)
    seq_idx = _find_seq_idx(query_start_len_ptr, q_ind, num_seqs)
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx)

    q_block_local_idx = q_ind - q_block_start_idx
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M) + head_ind * BLOCK_M

    offs_lora = tl.arange(0, KV_LORA_RANK)
    offs_rope = tl.arange(KV_LORA_RANK, KV_LORA_RANK + ROPE_RANK)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Load Q_rope: (BLOCK_M, ROPE_RANK)
    q_rope_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_rope[None, :]
    )
    Q_rope = tl.load(
        query_ptr + q_rope_offset,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    # Load Q_lora: (BLOCK_M, KV_LORA_RANK)
    q_lora_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_lora[None, :]
    )
    Q_lora = tl.load(
        query_ptr + q_lora_offset,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], dtype=tl.float32)

    num_tiles = (topk_count + TILE_SIZE - 1) // TILE_SIZE
    for t in range(0, num_tiles):
        tile_start = t * TILE_SIZE
        offs_t = tl.arange(0, TILE_SIZE)
        valid_t = (tile_start + offs_t) < topk_count

        topk_row_ptr = topk_indices_ptr + q_ind * topk_count
        topk_pos = tl.load(topk_row_ptr + tile_start + offs_t, mask=valid_t, other=0)
        valid_t = valid_t & (topk_pos != -1)

        physical_block_idx = topk_pos // BLOCK_SIZE
        slot = topk_pos % BLOCK_SIZE

        S = tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32)

        # K_rope: (ROPE_RANK, TILE_SIZE)
        k_rope_ptrs = (
            key_cache_ptr
            + physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_rope[:, None] * stride_k_cache_3
            + slot[None, :] * stride_k_cache_1
        )
        K_rope = tl.load(k_rope_ptrs, mask=valid_t[None, :], other=0.0)
        S += scale * tl.dot(Q_rope, K_rope)

        # K_lora: (KV_LORA_RANK, TILE_SIZE)
        k_lora_ptrs = (
            key_cache_ptr
            + physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_lora[:, None] * stride_k_cache_3
            + slot[None, :] * stride_k_cache_1
        )
        K_lora = tl.load(k_lora_ptrs, mask=valid_t[None, :], other=0.0)
        S += scale * tl.dot(Q_lora, K_lora)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & valid_t[None, :],
            S,
            float("-inf"),
        )

        # Online softmax update
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        # V_lora: (TILE_SIZE, KV_LORA_RANK)
        v_lora_ptrs = (
            value_cache_ptr
            + physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + slot[:, None] * stride_v_cache_1
            + offs_lora[None, :] * stride_v_cache_3
        )
        V_lora = tl.load(v_lora_ptrs, mask=valid_t[:, None], other=0.0)
        acc += tl.dot(P.to(V_lora.dtype), V_lora)

    # Epilogue: normalize
    acc = acc / L[:, None]

    output_offs_lora = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_lora[None, :]
    )
    tl.store(
        output_ptr + output_offs_lora,
        acc,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
    )


def triton_sparse_mla_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float,
    kv_lora_rank: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sparse MLA decode using pure Triton.

    Args:
        q:             [num_tokens, num_heads, kv_lora_rank + rope_rank], bf16
        kv:            paged KV cache [num_pages, page_size, 1, kv_lora_rank + rope_rank], bf16
        topk_indices:  [num_tokens, topk], int32  (direct physical indices, -1 = invalid)
        sm_scale:      attention scaling factor
        kv_lora_rank:  size of the LoRA / value dimension
        cu_seqlens_q:  [batch + 1], cumulative q sequence lengths; defaults to arange (decode)

    Returns:
        out: [num_tokens, num_heads, kv_lora_rank], bf16
    """
    num_tokens, num_heads, head_dim = q.shape
    rope_rank = head_dim - kv_lora_rank
    block_size = kv.shape[1]
    topk_count = topk_indices.shape[1]

    out = torch.empty(
        (num_tokens, num_heads, kv_lora_rank), dtype=q.dtype, device=q.device
    )

    if cu_seqlens_q is None:
        cu_seqlens_q = torch.arange(
            0, num_tokens + 1, dtype=torch.int32, device=q.device
        )

    num_seqs = cu_seqlens_q.shape[0] - 1

    k = kv
    v = kv[..., :kv_lora_rank]

    BLOCK_M = min(16, num_heads)
    if block_size > 1:
        TILE_SIZE = block_size
    elif _is_hip and not _is_gfx95:
        TILE_SIZE = min(topk_count, 32)
    else:
        TILE_SIZE = min(topk_count, 64)

    if _is_hip and not _is_gfx95:
        nwarps = 2
    else:
        nwarps = 4

    total_num_q_blocks = num_tokens * (num_heads // BLOCK_M)

    _triton_sparse_mla_decode_kernel[(total_num_q_blocks,)](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        topk_indices_ptr=topk_indices,
        scale=sm_scale,
        num_query_heads=num_heads,
        num_queries_per_kv=num_heads,
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        topk_count=topk_count,
        query_start_len_ptr=cu_seqlens_q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        ROPE_RANK=rope_rank,
        KV_LORA_RANK=kv_lora_rank,
        TILE_SIZE=TILE_SIZE,
        num_warps=nwarps,
        num_stages=1,
    )

    return out


def _pick_inner_iter(seq: int, ni: int, cu: int, block_per_cu: int) -> int:
    """Largest power-of-two inner_iter that keeps enough work per CU."""
    max_it = int(seq * ni / (cu * block_per_cu))
    it = ni
    while it >= 2:
        if it <= max_it and ni % it == 0:
            return it
        it //= 2
    return 1


@triton.jit
def _triton_sparse_mla_fp8_partial(
    partial_o_ptr,
    partial_lse_ptr,
    query_ptr,
    kv_cache_ptr,
    topk_indices_ptr,
    scale,
    fp8_max_val,
    fp8_inv_scale,
    num_query_heads: tl.constexpr,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    kv_stride: tl.int64,
    topk_count: tl.constexpr,
    num_tokens: tl.int32,
    BLOCK_M: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_I: tl.constexpr,
    INNER_ITER: tl.constexpr,
    N_GROUPS: tl.constexpr,
    IS_FNUZ: tl.constexpr,
):
    """
    grid: (num_tokens * ceil(heads / BLOCK_M), N_GROUPS)
    Each block processes BLOCK_I * INNER_ITER KV tokens with 4x128 chunked FP8 GEMM.
    Outputs partial_o (bf16) + partial_lse (fp32) per group."""
    head_blocks: tl.constexpr = num_query_heads // BLOCK_M
    po_grp_stride: tl.constexpr = num_query_heads * D_V
    lse_grp_stride: tl.constexpr = num_query_heads

    pid_seq_head = tl.program_id(0)
    group_id = tl.program_id(1)

    token_idx = pid_seq_head // head_blocks
    head_block = pid_seq_head % head_blocks

    if token_idx >= num_tokens:
        return

    offs_m = tl.arange(0, BLOCK_M) + head_block * BLOCK_M
    head_mask = offs_m < num_query_heads

    q_base = token_idx * query_stride_0 + offs_m[:, None] * query_stride_1
    q_mask = head_mask[:, None]

    offs_g = tl.arange(0, GROUP_SIZE)
    Q0 = tl.load(query_ptr + q_base + offs_g[None, :], mask=q_mask, other=0.0)
    Q1 = tl.load(
        query_ptr + q_base + (offs_g + GROUP_SIZE)[None, :], mask=q_mask, other=0.0
    )
    Q2 = tl.load(
        query_ptr + q_base + (offs_g + 2 * GROUP_SIZE)[None, :],
        mask=q_mask,
        other=0.0,
    )
    Q3 = tl.load(
        query_ptr + q_base + (offs_g + 3 * GROUP_SIZE)[None, :],
        mask=q_mask,
        other=0.0,
    )
    offs_tail = tl.arange(0, D_TAIL)
    Q_tail = tl.load(
        query_ptr + q_base + (offs_tail + D_V)[None, :], mask=q_mask, other=0.0
    )

    acc0 = tl.zeros([BLOCK_M, GROUP_SIZE], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, GROUP_SIZE], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M, GROUP_SIZE], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_M, GROUP_SIZE], dtype=tl.float32)
    M_val = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L_val = tl.full([BLOCK_M], 0.0, dtype=tl.float32)

    topk_base = group_id * BLOCK_I * INNER_ITER

    for ki in range(INNER_ITER):
        tile_start = topk_base + ki * BLOCK_I
        offs_t = tl.arange(0, BLOCK_I)
        valid_t = (tile_start + offs_t) < topk_count

        topk_row_ptr = topk_indices_ptr + token_idx * topk_count
        topk_pos = tl.load(topk_row_ptr + tile_start + offs_t, mask=valid_t, other=0)
        valid_t = valid_t & (topk_pos != -1)

        kv_base = topk_pos * kv_stride

        # QK: 4x128 chunked FP8 GEMM + rope tail
        S = tl.zeros([BLOCK_M, BLOCK_I], dtype=tl.float32)
        k_mask = valid_t[None, :]

        K0 = tl.load(
            kv_cache_ptr + kv_base[None, :] + offs_g[:, None],
            mask=k_mask,
            other=0.0,
        )
        S += tl.dot(Q0, K0)
        K1 = tl.load(
            kv_cache_ptr + kv_base[None, :] + (offs_g + GROUP_SIZE)[:, None],
            mask=k_mask,
            other=0.0,
        )
        S += tl.dot(Q1, K1)
        K2 = tl.load(
            kv_cache_ptr + kv_base[None, :] + (offs_g + 2 * GROUP_SIZE)[:, None],
            mask=k_mask,
            other=0.0,
        )
        S += tl.dot(Q2, K2)
        K3 = tl.load(
            kv_cache_ptr + kv_base[None, :] + (offs_g + 3 * GROUP_SIZE)[:, None],
            mask=k_mask,
            other=0.0,
        )
        S += tl.dot(Q3, K3)
        K_tail = tl.load(
            kv_cache_ptr + kv_base[None, :] + (offs_tail + D_V)[:, None],
            mask=k_mask,
            other=0.0,
        )
        S += tl.dot(Q_tail, K_tail)

        S *= scale
        S = tl.where(head_mask[:, None] & valid_t[None, :], S, float("-inf"))

        # Online softmax (scale already includes log2(e), so use exp2)
        m_j = tl.maximum(M_val, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.math.exp2(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.math.exp2(M_val - m_j)

        acc0 *= alpha[:, None]
        acc1 *= alpha[:, None]
        acc2 *= alpha[:, None]
        acc3 *= alpha[:, None]
        L_val = L_val * alpha + l_j
        M_val = m_j

        # SV: scale P to fp8, 4x128 chunked GEMM
        P_scaled = tl.clamp(P * fp8_max_val, -fp8_max_val, fp8_max_val)
        if IS_FNUZ:
            P_fp8 = P_scaled.to(tl.float8e4b8)
        else:
            P_fp8 = P_scaled.to(tl.float8e4nv)

        v_mask = valid_t[:, None]
        V0 = tl.load(
            kv_cache_ptr + kv_base[:, None] + offs_g[None, :],
            mask=v_mask,
            other=0.0,
        )
        acc0 += tl.dot(P_fp8, V0) * fp8_inv_scale
        V1 = tl.load(
            kv_cache_ptr + kv_base[:, None] + (offs_g + GROUP_SIZE)[None, :],
            mask=v_mask,
            other=0.0,
        )
        acc1 += tl.dot(P_fp8, V1) * fp8_inv_scale
        V2 = tl.load(
            kv_cache_ptr + kv_base[:, None] + (offs_g + 2 * GROUP_SIZE)[None, :],
            mask=v_mask,
            other=0.0,
        )
        acc2 += tl.dot(P_fp8, V2) * fp8_inv_scale
        V3 = tl.load(
            kv_cache_ptr + kv_base[:, None] + (offs_g + 3 * GROUP_SIZE)[None, :],
            mask=v_mask,
            other=0.0,
        )
        acc3 += tl.dot(P_fp8, V3) * fp8_inv_scale

    # Normalise partial output
    safe_L = tl.where(L_val > 0, L_val, 1.0)
    inv_L = 1.0 / safe_L
    acc0 *= inv_L[:, None]
    acc1 *= inv_L[:, None]
    acc2 *= inv_L[:, None]
    acc3 *= inv_L[:, None]

    lse = tl.where(L_val > 0, tl.math.log2(L_val) + M_val, float("-inf"))

    # Store partial_o  [num_tokens, N_GROUPS, num_heads, D_V]
    po_base = (
        token_idx * (N_GROUPS * po_grp_stride)
        + group_id * po_grp_stride
        + offs_m[:, None] * D_V
    )
    po_mask = head_mask[:, None]
    tl.store(
        partial_o_ptr + po_base + offs_g[None, :],
        acc0.to(tl.bfloat16),
        mask=po_mask,
    )
    tl.store(
        partial_o_ptr + po_base + (offs_g + GROUP_SIZE)[None, :],
        acc1.to(tl.bfloat16),
        mask=po_mask,
    )
    tl.store(
        partial_o_ptr + po_base + (offs_g + 2 * GROUP_SIZE)[None, :],
        acc2.to(tl.bfloat16),
        mask=po_mask,
    )
    tl.store(
        partial_o_ptr + po_base + (offs_g + 3 * GROUP_SIZE)[None, :],
        acc3.to(tl.bfloat16),
        mask=po_mask,
    )

    # Store partial_lse  [num_tokens, N_GROUPS, num_heads]
    lse_off = (
        token_idx * (N_GROUPS * lse_grp_stride) + group_id * lse_grp_stride + offs_m
    )
    tl.store(partial_lse_ptr + lse_off, lse, mask=head_mask)


@triton.jit
def _triton_sparse_mla_fp8_combine(
    output_ptr,
    partial_o_ptr,
    partial_lse_ptr,
    num_tokens: tl.int32,
    num_query_heads: tl.constexpr,
    D_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    N_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    grid: (num_tokens * ceil(heads / BLOCK_M),)
    Online log-sum-exp weighted merge of partial_o across N_GROUPS.
    Single pass: reads each partial_lse element exactly once (vs 3× in naive approach).
    """
    head_blocks: tl.constexpr = num_query_heads // BLOCK_M
    po_grp_stride: tl.constexpr = num_query_heads * D_V
    lse_grp_stride: tl.constexpr = num_query_heads

    pid = tl.program_id(0)
    token_idx = pid // head_blocks
    head_block = pid % head_blocks

    if token_idx >= num_tokens:
        return

    offs_m = tl.arange(0, BLOCK_M) + head_block * BLOCK_M
    head_mask = offs_m < num_query_heads

    lse_tok_base = token_idx * N_GROUPS * lse_grp_stride
    po_tok_base = token_idx * N_GROUPS * po_grp_stride

    offs_g = tl.arange(0, GROUP_SIZE)
    po_mask = head_mask[:, None]

    lse_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    weight_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_M, GROUP_SIZE], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, GROUP_SIZE], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M, GROUP_SIZE], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_M, GROUP_SIZE], dtype=tl.float32)

    for k in range(N_GROUPS):
        lse_k = tl.load(
            partial_lse_ptr + lse_tok_base + k * lse_grp_stride + offs_m,
            mask=head_mask,
            other=float("-inf"),
        )

        new_max = tl.maximum(lse_max, lse_k)
        correction = tl.math.exp2(lse_max - new_max)

        weight_sum = weight_sum * correction
        acc0 *= correction[:, None]
        acc1 *= correction[:, None]
        acc2 *= correction[:, None]
        acc3 *= correction[:, None]

        w_k = tl.math.exp2(lse_k - new_max)
        weight_sum += w_k

        po_base = po_tok_base + k * po_grp_stride + offs_m[:, None] * D_V

        po0 = tl.load(
            partial_o_ptr + po_base + offs_g[None, :], mask=po_mask, other=0.0
        ).to(tl.float32)
        acc0 += w_k[:, None] * po0

        po1 = tl.load(
            partial_o_ptr + po_base + (offs_g + GROUP_SIZE)[None, :],
            mask=po_mask,
            other=0.0,
        ).to(tl.float32)
        acc1 += w_k[:, None] * po1

        po2 = tl.load(
            partial_o_ptr + po_base + (offs_g + 2 * GROUP_SIZE)[None, :],
            mask=po_mask,
            other=0.0,
        ).to(tl.float32)
        acc2 += w_k[:, None] * po2

        po3 = tl.load(
            partial_o_ptr + po_base + (offs_g + 3 * GROUP_SIZE)[None, :],
            mask=po_mask,
            other=0.0,
        ).to(tl.float32)
        acc3 += w_k[:, None] * po3

        lse_max = new_max

    inv_weight_sum = 1.0 / tl.maximum(weight_sum, 1e-10)
    acc0 *= inv_weight_sum[:, None]
    acc1 *= inv_weight_sum[:, None]
    acc2 *= inv_weight_sum[:, None]
    acc3 *= inv_weight_sum[:, None]

    out_base = token_idx * num_query_heads * D_V + offs_m[:, None] * D_V
    out_mask = head_mask[:, None]

    tl.store(
        output_ptr + out_base + offs_g[None, :],
        acc0.to(tl.bfloat16),
        mask=out_mask,
    )
    tl.store(
        output_ptr + out_base + (offs_g + GROUP_SIZE)[None, :],
        acc1.to(tl.bfloat16),
        mask=out_mask,
    )
    tl.store(
        output_ptr + out_base + (offs_g + 2 * GROUP_SIZE)[None, :],
        acc2.to(tl.bfloat16),
        mask=out_mask,
    )
    tl.store(
        output_ptr + out_base + (offs_g + 3 * GROUP_SIZE)[None, :],
        acc3.to(tl.bfloat16),
        mask=out_mask,
    )


def triton_sparse_mla_decode_fp8(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FP8 sparse MLA decode — split-k partial + combine, matching tilelang.

    Args:
        q:             [num_tokens, num_heads, d_v + d_tail] fp8
        kv:            paged KV cache [num_pages, page_size, 1, d_v + d_tail] fp8
        topk_indices:  [num_tokens, topk] int32 (direct physical indices, -1 = invalid)
        sm_scale:      attention scaling factor
        d_v:           nope / value dimension (default 512)
        cu_seqlens_q:  unused, kept for API compat

    Returns:
        out: [num_tokens, num_heads, d_v] bf16
    """
    from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

    num_tokens, num_heads, head_dim = q.shape
    d_tail = head_dim - d_v
    topk_count = topk_indices.shape[1]

    GROUP_SIZE = 128
    BLOCK_I = 64
    BLOCK_M_PARTIAL = min(16, num_heads)
    BLOCK_M_COMBINE = min(4, num_heads)

    assert d_v % GROUP_SIZE == 0
    assert num_heads % BLOCK_M_PARTIAL == 0
    assert num_heads % BLOCK_M_COMBINE == 0

    ni = topk_count // BLOCK_I
    assert ni * BLOCK_I == topk_count, "topk must be divisible by BLOCK_I"

    if _is_hip:
        if _is_gfx95:
            block_per_cu, cu = 2, 256
        else:
            block_per_cu, cu = 1, 304
        inner_iter = _pick_inner_iter(num_tokens, ni, cu, block_per_cu)
    else:
        inner_iter = 1
    n_groups = ni // inner_iter

    fnuz = is_fp8_fnuz()
    fp8_max_val = 240.0 if fnuz else 448.0
    fp8_inv_scale = 1.0 / fp8_max_val

    partial_o = torch.empty(
        (num_tokens, n_groups, num_heads, d_v),
        dtype=torch.bfloat16,
        device=q.device,
    )
    partial_lse = torch.empty(
        (num_tokens, n_groups, num_heads),
        dtype=torch.float32,
        device=q.device,
    )

    head_blocks_partial = num_heads // BLOCK_M_PARTIAL
    grid_partial = (num_tokens * head_blocks_partial, n_groups)

    kv_flat = kv.view(-1, head_dim)
    _triton_sparse_mla_fp8_partial[grid_partial](
        partial_o_ptr=partial_o,
        partial_lse_ptr=partial_lse,
        query_ptr=q,
        kv_cache_ptr=kv_flat,
        topk_indices_ptr=topk_indices,
        scale=sm_scale * 1.44269504,
        fp8_max_val=fp8_max_val,
        fp8_inv_scale=fp8_inv_scale,
        num_query_heads=num_heads,
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        kv_stride=kv_flat.stride(0),
        topk_count=topk_count,
        num_tokens=num_tokens,
        BLOCK_M=BLOCK_M_PARTIAL,
        D_V=d_v,
        D_TAIL=d_tail,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_I=BLOCK_I,
        INNER_ITER=inner_iter,
        N_GROUPS=n_groups,
        IS_FNUZ=fnuz,
        num_warps=4,
        num_stages=1,
    )

    out = torch.empty(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=q.device
    )

    head_blocks_combine = num_heads // BLOCK_M_COMBINE
    grid_combine = (num_tokens * head_blocks_combine,)

    _triton_sparse_mla_fp8_combine[grid_combine](
        output_ptr=out,
        partial_o_ptr=partial_o,
        partial_lse_ptr=partial_lse,
        num_tokens=num_tokens,
        num_query_heads=num_heads,
        D_V=d_v,
        GROUP_SIZE=GROUP_SIZE,
        N_GROUPS=n_groups,
        BLOCK_M=BLOCK_M_COMBINE,
        num_warps=4,
        num_stages=1,
    )

    return out
