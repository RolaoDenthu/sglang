"""The DeepSeek-V4.1 query as V41_NEW rows on gfx950: one pass applies the RoPE to the
last dims of every head (optional) and quantizes the head to MXFP8, writing the 512
e4m3 values and 16 e8m0 scales of a pool row. Bit for bit the RoPE of
``apply_rotary_emb_flat_kernel`` stored as bf16, then ``_mxfp8_quant_kernel``."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

_LAYOUT = KVLayout.V41_NEW


@triton.jit
def _mxfp8_store(
    x,
    out_row,
    rmask,
    col0: tl.constexpr,
    WIDTH: tl.constexpr,
    LIMIT: tl.constexpr,
    D: tl.constexpr,
):
    """x [rows, WIDTH] fp32 as the MXFP8 values of row columns [col0, LIMIT) and
    their scales."""
    rows: tl.constexpr = x.shape[0]
    blocks = tl.reshape(x, (rows, WIDTH // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(blocks), axis=2), 1e-30)
    scale = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    scale = tl.minimum(tl.maximum(scale, 0.0), 254.0)
    values = tl.clamp(blocks / tl.exp2(scale - 127.0)[:, :, None], -448.0, 448.0)
    cols = col0 + tl.arange(0, WIDTH)
    tl.store(
        (out_row + cols[None, :]).to(tl.pointer_type(tl.float8e4nv)),
        tl.reshape(values, (rows, WIDTH)).to(tl.float8e4nv),
        mask=rmask[:, None] & (cols < LIMIT)[None, :],
    )
    block_cols = col0 // 32 + tl.arange(0, WIDTH // 32)
    tl.store(
        out_row + D + block_cols[None, :],
        scale.to(tl.uint8),
        mask=rmask[:, None] & (block_cols < LIMIT // 32)[None, :],
    )


@triton.jit
def _q_to_v41_rows_kernel(
    x_ptr,
    out_ptr,
    freqs_ptr,
    pos_ptr,
    num_rows,
    num_heads,
    sx_tok,
    sx_head,
    so_row,
    sfr_pos,
    D: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    rmask = rows < num_rows
    tok = rows // num_heads
    head = rows % num_heads
    x_row = x_ptr + tok[:, None].to(tl.int64) * sx_tok + head[:, None] * sx_head
    out_row = out_ptr + rows[:, None].to(tl.int64) * so_row
    cols = tl.arange(0, D)
    if HAS_ROPE:
        # the nope blocks over the whole power-of-2 row, the roped ones on their own
        x = tl.load(
            x_row + cols[None, :],
            mask=rmask[:, None] & (cols < NOPE_DIM)[None, :],
            other=0.0,
        )
        _mxfp8_store(x.to(tl.float32), out_row, rmask, 0, D, NOPE_DIM, D)
        rope_cols = tl.arange(0, ROPE_DIM)
        x = tl.load(
            x_row + NOPE_DIM + rope_cols[None, :], mask=rmask[:, None], other=0.0
        ).to(tl.float32)
        pos = tl.load(pos_ptr + tok, mask=rmask, other=0)
        pair = rope_cols // 2 * 2
        fr = freqs_ptr + pos[:, None].to(tl.int64) * sfr_pos + pair[None, :]
        cos = tl.load(fr, mask=rmask[:, None], other=0.0)
        sin = tl.load(fr + 1, mask=rmask[:, None], other=0.0)
        x_sin = x * sin
        x_neg = tl.where((rope_cols % 2 == 0)[None, :], x_sin, -x_sin)
        x_rot = tl.reshape(
            tl.flip(tl.reshape(x_neg, (BLOCK_ROWS, ROPE_DIM // 2, 2)), 2),
            (BLOCK_ROWS, ROPE_DIM),
        )
        roped = (x * cos + x_rot).to(tl.bfloat16).to(tl.float32)
        _mxfp8_store(roped, out_row, rmask, NOPE_DIM, ROPE_DIM, D, D)
    else:
        x = tl.load(x_row + cols[None, :], mask=rmask[:, None], other=0.0)
        _mxfp8_store(x.to(tl.float32), out_row, rmask, 0, D, D, D)


def q_to_v41_rows(
    q: torch.Tensor,
    freqs_cis: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """bf16 q [T, H, 512] -> uint8 V41_NEW rows [T, H, 528]; with ``freqs_cis`` (the
    complex [max_pos, rope_dim / 2] table) and ``positions`` [T], the last rope_dim
    dims of every head are roped first."""
    num_tokens, num_heads, head_dim = q.shape
    assert head_dim == _LAYOUT.data_bytes and q.stride(-1) == 1, q.shape
    assert (freqs_cis is None) == (positions is None)
    out = torch.empty(
        num_tokens,
        num_heads,
        _LAYOUT.bytes_per_token,
        dtype=torch.uint8,
        device=q.device,
    )
    num_rows = num_tokens * num_heads
    if num_rows == 0:
        return out
    has_rope = freqs_cis is not None
    if has_rope:
        freqs = torch.view_as_real(freqs_cis).flatten(-2)
        assert freqs.stride(-1) == 1
        rope_dim = freqs.shape[-1]
        # the roped dims fill whole 32-blocks of a power-of-2 width
        assert rope_dim % 32 == 0 and rope_dim & (rope_dim - 1) == 0, rope_dim
    else:
        # unread without the RoPE
        freqs, rope_dim = out, 32
    block_rows = 16
    _q_to_v41_rows_kernel[(triton.cdiv(num_rows, block_rows),)](
        q,
        out,
        freqs,
        positions,
        num_rows,
        num_heads,
        q.stride(0),
        q.stride(1),
        out.stride(1),
        freqs.stride(0),
        D=head_dim,
        NOPE_DIM=head_dim - rope_dim,
        ROPE_DIM=rope_dim,
        HAS_ROPE=has_rope,
        BLOCK_ROWS=block_rows,
        num_warps=4,
    )
    return out


def is_v41_rows(q: torch.Tensor) -> bool:
    """Whether q is already the V41_NEW rows of a query, not the bf16 query."""
    return q.dtype == torch.uint8 and q.shape[-1] == _LAYOUT.bytes_per_token


def v41_rows_split(rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """uint8 V41_NEW rows [..., 528] -> (e4m3 values [..., 512], e8m0 scales [..., 16])
    views at the row stride."""
    assert is_v41_rows(rows), (rows.dtype, rows.shape)
    return (
        rows[..., : _LAYOUT.data_bytes].view(torch.float8_e4m3fn),
        rows[..., _LAYOUT.data_bytes :],
    )
