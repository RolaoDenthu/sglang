"""DeepSeek-V4 sparse prefill attention on gfx950 through aiter's OPUS kernel.

OPUS attends over two bf16 row sources, each addressed by a CSR index list. The
caller dequantizes rows into bf16 workspaces and builds padded per-query lists
(``SparsePrefillChunkCache``); this module turns those lists into CSR and runs
the kernel.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# (int32 indices, int32 indptr [T + 1])
Csr = Tuple[torch.Tensor, torch.Tensor]


@triton.jit
def _combined_to_csr_kernel(
    indices_ptr,
    indices_stride,
    lens_ptr,
    indptr_ptr,
    out_ptr,
    row_start,
    row_end,
    BLOCK: tl.constexpr,
):
    # OPUS reads every CSR entry, so -1 holes inside the scanned prefix and
    # entries outside [row_start, row_end) are dropped here.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    length = tl.load(lens_ptr + row)
    idx = tl.load(
        indices_ptr + row * indices_stride + cols, mask=cols < length, other=-1
    )
    keep = (idx >= row_start) & (idx < row_end)
    dst = tl.cumsum(keep.to(tl.int32), axis=0) - 1
    start = tl.load(indptr_ptr + row)
    tl.store(out_ptr + start + dst, idx - row_start, mask=keep)


def combined_to_csr(
    indices: torch.Tensor, lens: torch.Tensor, row_start: int, row_end: int
) -> Csr:
    """int32 [T, W] padded index lists with per-row scanned lengths [T] -> the CSR
    (T * W capacity) of their entries in [row_start, row_end), rebased to row_start."""
    num_rows, width = indices.shape
    device = indices.device
    cols = torch.arange(width, device=device, dtype=torch.int32)
    keep = (
        (cols[None, :] < lens[:, None]) & (indices >= row_start) & (indices < row_end)
    )
    indptr = torch.zeros(num_rows + 1, dtype=torch.int32, device=device)
    torch.cumsum(keep.sum(dim=1, dtype=torch.int32), dim=0, out=indptr[1:])
    csr = torch.empty(max(num_rows * width, 1), dtype=torch.int32, device=device)
    if num_rows > 0:
        _combined_to_csr_kernel[(num_rows,)](
            indices,
            indices.stride(0),
            lens,
            indptr,
            csr,
            row_start,
            row_end,
            BLOCK=triton.next_power_of_2(width),
        )
    return csr, indptr


@triton.jit
def _rows_csr_count_kernel(
    indices_ptr,
    indices_stride,
    lens_ptr,
    counts_ptr,
    num_src,
    width,
    num_rows,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    length = tl.load(lens_ptr + row, mask=row < num_src, other=0)
    idx = tl.load(
        indices_ptr + row.to(tl.int64) * indices_stride + cols,
        mask=(cols < length) & (cols < width),
        other=-1,
    )
    keep = (idx >= 0) & (idx < num_rows)
    tl.store(counts_ptr + row, tl.sum(keep.to(tl.int32), axis=0))


@triton.jit
def _rows_csr_fill_kernel(
    indices_ptr,
    indices_stride,
    lens_ptr,
    counts_ptr,
    indptr_ptr,
    empty_indptr_ptr,
    out_ptr,
    num_src,
    width,
    num_rows,
    BLOCK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    START_FROM_INDPTR: tl.constexpr,
    HAS_EMPTY: tl.constexpr,
):
    row = tl.program_id(0)
    if START_FROM_INDPTR:
        start = tl.load(indptr_ptr + row, mask=row > 0, other=0)
    else:
        start = 0
        for t0 in range(0, row, BLOCK_T):
            offs = t0 + tl.arange(0, BLOCK_T)
            start += tl.sum(tl.load(counts_ptr + offs, mask=offs < row, other=0))
    cols = tl.arange(0, BLOCK)
    length = tl.load(lens_ptr + row, mask=row < num_src, other=0)
    idx = tl.load(
        indices_ptr + row.to(tl.int64) * indices_stride + cols,
        mask=(cols < length) & (cols < width),
        other=-1,
    )
    keep = (idx >= 0) & (idx < num_rows)
    dst = tl.cumsum(keep.to(tl.int32), axis=0) - 1
    tl.store(out_ptr + start + dst, idx, mask=keep)
    tl.store(indptr_ptr + row + 1, start + tl.sum(keep.to(tl.int32), axis=0))
    if row == 0:
        tl.store(indptr_ptr, 0)
    if HAS_EMPTY:
        tl.store(empty_indptr_ptr + row + 1, 0)
        if row == 0:
            tl.store(empty_indptr_ptr, 0)


# up to this many rows every fill program sums the counts before its row; past it
# a cumsum launch writes the offsets instead
_ROWS_CSR_INLINE_PREFIX_MAX = 4096


def rows_to_csr(
    indices: torch.Tensor,
    lens: torch.Tensor,
    num_tokens: int,
    num_rows: int,
    *,
    with_empty: bool = False,
) -> Tuple[Csr, Optional[Csr]]:
    """int32 [S, W] pool-row lists with per-row lengths [S] -> the CSR (T * W
    capacity) of the first ``num_tokens`` rows' entries in [0, num_rows); rows past
    S are empty, as are -1 holes. ``with_empty`` also returns a CSR with no entry
    for any row, for a kernel source that must be passed but read nothing."""
    width = indices.shape[1]
    assert indices.stride(1) == 1, indices.stride()
    device = indices.device
    counts = torch.empty(num_tokens, dtype=torch.int32, device=device)
    indptr = torch.empty(num_tokens + 1, dtype=torch.int32, device=device)
    out = torch.empty(max(num_tokens * width, 1), dtype=torch.int32, device=device)
    empty_indptr = (
        torch.empty(num_tokens + 1, dtype=torch.int32, device=device)
        if with_empty
        else None
    )
    if num_tokens == 0:
        indptr.zero_()
        if empty_indptr is not None:
            empty_indptr.zero_()
    else:
        block = triton.next_power_of_2(width)
        num_src = min(indices.shape[0], lens.shape[0])
        _rows_csr_count_kernel[(num_tokens,)](
            indices,
            indices.stride(0),
            lens,
            counts,
            num_src,
            width,
            num_rows,
            BLOCK=block,
        )
        from_indptr = num_tokens > _ROWS_CSR_INLINE_PREFIX_MAX
        if from_indptr:
            torch.cumsum(counts, dim=0, out=indptr[1:])
        _rows_csr_fill_kernel[(num_tokens,)](
            indices,
            indices.stride(0),
            lens,
            counts,
            indptr,
            empty_indptr if with_empty else indptr,
            out,
            num_src,
            width,
            num_rows,
            BLOCK=block,
            BLOCK_T=1024,
            START_FROM_INDPTR=from_indptr,
            HAS_EMPTY=with_empty,
        )
    # no index entry: aiter's split-K auto counts every one as a key
    empty = (out[:0], empty_indptr) if with_empty else None
    return (out, indptr), empty


def opus_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    csr: Csr,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    extend_kv: Optional[torch.Tensor] = None,
    extend_csr: Optional[Csr] = None,
) -> torch.Tensor:
    """Attention of q [T, H, 512] bf16 over the bf16 rows of kv [R, 512] named by
    csr, then those of extend_kv named by extend_csr, with a per-head sink."""
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus

    indices, indptr = csr
    if extend_kv is None:
        # every query reads an empty row of the unused extend source
        extend_kv = kv[:1]
        extend_csr = (indices[:1], torch.zeros_like(indptr))
    extend_indices, extend_indptr = extend_csr
    return pa_sparse_prefill_opus(
        q.contiguous(),
        kv,
        indices,
        indptr,
        extend_kv,
        extend_indices,
        extend_indptr,
        attn_sink,
        softmax_scale,
    )
