from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def sparse_mla_fwd_decode_partial(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Page_Table,
    Out,
    Out_Lse,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_ob,
    stride_oh,
    stride_os,
    stride_pt_bs,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    NEED_REDUCE: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < Lv

    split_start = split_kv_id * TOKENS_PER_SPLIT

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1073741824.0
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(Q + off_qpe, mask=mask_h[:, None], other=0.0)

    for start_n in range(split_start, split_start + TOKENS_PER_SPLIT, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        kv_loc = tl.load(Page_Table + cur_batch * stride_pt_bs + offs_n)
        valid_mask = kv_loc >= 0
        kv_loc_safe = tl.where(valid_mask, kv_loc, 0)

        offs_buf_k = (
            kv_loc_safe[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        k = tl.load(K_Buffer + offs_buf_k, mask=valid_mask[None, :], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            offs_buf_kpe = (
                kv_loc_safe[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(K_Buffer + offs_buf_kpe, mask=valid_mask[None, :], other=0.0)
            qk += tl.dot(qpe, kpe.to(qpe.dtype))

        qk *= sm_scale
        qk = tl.where(mask_h[:, None] & valid_mask[None, :], qk, float("-inf"))

        offs_buf_v = (
            kv_loc_safe[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Buffer + offs_buf_v,
            mask=valid_mask[:, None] & mask_dv[None, :],
            other=0.0,
        )

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    safe_esum = tl.where(e_sum > 0, e_sum, 1.0)
    result = acc / safe_esum[:, None]

    if NEED_REDUCE:
        offs_o = (
            cur_batch * stride_ob
            + cur_head[:, None] * stride_oh
            + split_kv_id * stride_os
            + offs_dv[None, :]
        )
        tl.store(Out + offs_o, result, mask=mask_h[:, None] & mask_dv[None, :])

        lse = tl.where(e_sum > 0, e_max + tl.log(e_sum), -1073741824.0)
        offs_lse = (
            cur_batch * stride_ob + cur_head * stride_oh + split_kv_id * stride_os
        ) // Lv
        tl.store(Out_Lse + offs_lse, lse, mask=mask_h)
    else:
        offs_o = (
            cur_batch * stride_ob + cur_head[:, None] * stride_oh + offs_dv[None, :]
        )
        tl.store(Out + offs_o, result, mask=mask_h[:, None] & mask_dv[None, :])


@triton.jit
def sparse_mla_fwd_decode_combine(
    Mid_O,
    Mid_Lse,
    O,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    N_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_lse = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv

    for split_id in range(0, N_SPLITS):
        tv = tl.load(Mid_O + offs_v + split_id * stride_mid_os, mask=mask_d, other=0.0)
        tlogic = tl.load(Mid_Lse + offs_lse + split_id * stride_mid_os // Lv)

        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - n_e_max)
        acc += exp_logic * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    safe_esum = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / safe_esum,
        mask=mask_d,
    )


def _compute_splits(bs, topk, num_head_groups, block_n=16):
    """Pick (N_SPLITS, TOKENS_PER_SPLIT) to balance CU utilization."""
    CU = 304
    MIN_TOKENS = block_n * 2
    NI = topk // MIN_TOKENS

    max_inner = max(1, int(bs * num_head_groups * NI / CU))
    inner_iter = NI
    while inner_iter >= 2:
        if inner_iter <= max_inner and NI % inner_iter == 0:
            break
        inner_iter //= 2
    inner_iter = max(1, inner_iter)
    return NI // inner_iter, MIN_TOKENS * inner_iter


def sparse_mla_fwd_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    """Triton sparse MLA decode attention with direct dense page-table index.

    Args:
        q:          [bs, num_heads, head_dim]  bf16
        kv_cache:   [pool_size, (1,) head_dim] bf16
        page_table: [bs, topk] int32  (-1 = invalid)
        sm_scale:   softmax scale
        v_head_dim: value dimension
    """
    bs, num_heads, head_dim = q.shape
    topk = page_table.shape[1]

    BLOCK_N = 16
    BLOCK_H = 16

    if head_dim == 576:
        BLOCK_DMODEL, BLOCK_DPE = 512, 64
    elif head_dim == 288:
        BLOCK_DMODEL, BLOCK_DPE = 256, 32
    else:
        BLOCK_DMODEL, BLOCK_DPE = triton.next_power_of_2(head_dim), 0
    BLOCK_DV = triton.next_power_of_2(v_head_dim)

    k_buffer = kv_cache.view(-1, 1, head_dim)
    v_buffer = k_buffer[:, :, :v_head_dim]
    kv_group_num = num_heads // k_buffer.shape[1]

    num_head_groups = triton.cdiv(num_heads, min(BLOCK_H, kv_group_num))
    N_SPLITS, TOKENS_PER_SPLIT = _compute_splits(bs, topk, num_head_groups, BLOCK_N)
    NEED_REDUCE = N_SPLITS > 1

    o = torch.empty((bs, num_heads, v_head_dim), dtype=q.dtype, device=q.device)

    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    if NEED_REDUCE:
        mid_o = torch.empty(
            (bs, num_heads, N_SPLITS, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        mid_lse = torch.empty(
            (bs, num_heads, N_SPLITS),
            dtype=torch.float32,
            device=q.device,
        )
        out_ptr, lse_ptr = mid_o, mid_lse
        s_ob, s_oh, s_os = mid_o.stride(0), mid_o.stride(1), mid_o.stride(2)
    else:
        out_ptr = o
        lse_ptr = o
        s_ob, s_oh, s_os = o.stride(0), o.stride(1), 0

    sparse_mla_fwd_decode_partial[(bs, num_head_groups, N_SPLITS)](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        page_table,
        out_ptr,
        lse_ptr,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        s_ob,
        s_oh,
        s_os,
        page_table.stride(0),
        kv_group_num=kv_group_num,
        q_head_num=num_heads,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        TOKENS_PER_SPLIT=TOKENS_PER_SPLIT,
        NEED_REDUCE=NEED_REDUCE,
        Lv=v_head_dim,
        num_warps=4,
        num_stages=num_stages,
        **extra_kargs,
    )

    if NEED_REDUCE:
        extra_kargs2 = {}
        if _is_hip:
            extra_kargs2 = {
                "waves_per_eu": 4,
                "matrix_instr_nonkdim": 16,
                "kpack": 2,
            }
        sparse_mla_fwd_decode_combine[(bs, num_heads)](
            mid_o,
            mid_lse,
            o,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            o.stride(0),
            o.stride(1),
            N_SPLITS=N_SPLITS,
            BLOCK_DV=BLOCK_DV,
            Lv=v_head_dim,
            num_warps=4,
            num_stages=2,
            **extra_kargs2,
        )

    return o


# Triton implementation
@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # FP8 constants
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    # Calculate row and column offsets
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    # Create offset arrays
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Mask for valid rows and columns
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input data
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute max along columns (group_size dimension) for each row
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  # Shape: (BLOCK_M,)

    # Clamp amax to avoid division by zero
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if round_scale:
        # Fast round scale using bit manipulation approximation
        # This is a simplified version - the exact bit manipulation is harder in Triton
        # Using log2 + ceil + pow2 as approximation
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    # Store quantized output
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store scales
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # Flatten all dims except last
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s


@triton.jit
def _get_valid_kv_indices_kernel(
    page_table_ptr,  # [bs, topk]
    kv_indptr_ptr,  # [bs + 1]
    kv_indices_ptr,  # [bs * topk] output buffer
    bs: tl.constexpr,
    topk: tl.constexpr,
):
    """
    Extract valid indices (non -1) from page_table into kv_indices.
    Each program handles one batch.
    """
    batch_id = tl.program_id(0)

    # Get the start position for this batch in kv_indices
    dst_start = tl.load(kv_indptr_ptr + batch_id)

    # Load all topk indices for this batch
    src_offset = batch_id * topk
    offsets = tl.arange(0, topk)
    indices = tl.load(page_table_ptr + src_offset + offsets)

    # Count valid indices and compact them
    mask = indices != -1

    # Use prefix sum to compute destination positions for valid elements
    # For each position, count how many valid elements are before it
    prefix_sum = tl.cumsum(mask.to(tl.int32), axis=0) - 1

    # Store valid indices to their compacted positions
    dst_positions = dst_start + prefix_sum
    tl.store(kv_indices_ptr + dst_positions, indices, mask=mask)


def get_valid_kv_indices(
    page_table_1: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
):
    """
    Extract valid indices from page_table_1 into kv_indices buffer.

    Args:
        page_table_1: [bs, topk] page table with -1 as invalid
        kv_indptr: [bs + 1] cumulative count of valid indices per batch
        kv_indices: [bs * topk] pre-allocated output buffer
        bs: batch size
    """
    topk = page_table_1.shape[1]
    grid = (bs,)
    _get_valid_kv_indices_kernel[grid](
        page_table_1,
        kv_indptr,
        kv_indices,
        bs,
        topk,
    )
