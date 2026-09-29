"""Fuse the preparation the TLX KDA kernels expect their caller to do.

The TLX kernels take l2-normalized Q/K, a materialized per-K log decay, and a
sigmoided beta. Doing that with torch ops costs ten-odd elementwise launches,
each streaming the whole ``[T, H, D]`` tensor through HBM -- at T=16384 that
measured 0.33ms against a 1.20ms kernel, eating the kernel's entire advantage.

This does all of it in one pass. Plain Triton, no TLX: the work is a per-head
reduction plus elementwise math, which needs no async copies or MFMA control.

The l2-norm epsilon matches ``fused_recurrent_kda_packed_decode_kernel``
(``sqrt(sum + 1e-6)``, not ``max(sqrt(sum), eps)``) so the two paths stay
numerically comparable.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _kda_prepare_kernel(
    q,
    k,
    a,
    b,
    A_log,
    dt_bias,
    qn,
    kn,
    g,
    beta_out,
    lower_bound,
    stride_q_tok,
    stride_k_tok,
    stride_a_tok,
    stride_b_tok,
    R,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    SIGMOID_BETA: tl.constexpr,
):
    # One program owns BLOCK_R (token, head) rows of D elements: a single
    # (token, head) per program leaves most lanes idle and moves too few bytes
    # to reach HBM bandwidth.
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    o_d = tl.arange(0, BD)
    row_mask = rows < R
    mask = row_mask[:, None] & (o_d < D)[None, :]
    i_t = (rows // H).to(tl.int64)
    i_h = rows % H
    # Inputs keep their token stride (split views of the packed qkv); every
    # head vector is contiguous. Outputs are dense [T, H, D].
    head = (i_h * D)[:, None] + o_d[None, :]
    out = rows.to(tl.int64)[:, None] * D + o_d[None, :]

    b_q = tl.load(q + i_t[:, None] * stride_q_tok + head, mask=mask, other=0.0)
    b_k = tl.load(k + i_t[:, None] * stride_k_tok + head, mask=mask, other=0.0)
    b_q = b_q.to(tl.float32)
    b_k = b_k.to(tl.float32)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q, axis=1) + 1e-6)[:, None]
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k, axis=1) + 1e-6)[:, None]
    tl.store(qn + out, b_q.to(qn.dtype.element_ty), mask=mask)
    tl.store(kn + out, b_k.to(kn.dtype.element_ty), mask=mask)

    b_a = tl.load(a + i_t[:, None] * stride_a_tok + head, mask=mask, other=0.0)
    b_a = b_a.to(tl.float32)
    b_dt = tl.load(dt_bias + head, mask=mask, other=0.0).to(tl.float32)
    A = tl.exp(tl.load(A_log + i_h, mask=row_mask, other=0.0).to(tl.float32))[:, None]
    x = b_a + b_dt
    if USE_LOWER_BOUND:
        b_g = lower_bound * tl.sigmoid(A * x)
    else:
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        b_g = -A * softplus_x
    tl.store(g + out, b_g.to(g.dtype.element_ty), mask=mask)

    # One scalar per (token, head) row.
    b_val = tl.load(b + i_t * stride_b_tok + i_h, mask=row_mask, other=0.0)
    b_val = b_val.to(tl.float32)
    if SIGMOID_BETA:
        b_val = tl.sigmoid(b_val)
    tl.store(beta_out + rows, b_val.to(beta_out.dtype.element_ty), mask=row_mask)


def _prepare_config(rows: int) -> Tuple[int, int]:
    """(BLOCK_R, num_warps). Few rows (decode) are latency-bound and want the
    most programs; many rows (extend) are bandwidth-bound and want bigger tiles.
    Measured on gfx950 with H=12, D=128."""
    return (4, 4) if rows <= 4096 else (16, 4)


def prepare_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_heads: int,
    head_dim: int,
    lower_bound: Optional[float] = None,
    sigmoid_beta: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(q_norm, k_norm, g, beta)`` shaped for the TLX kernels.

    ``q``/``k`` come back bf16 (the TLX contract), ``g`` and ``beta`` fp32.
    ``a`` may be any shape with ``T * num_heads * head_dim`` elements; ``b``
    any shape with ``T * num_heads``.

    Set ``sigmoid_beta=False`` when the caller already activated beta (the
    extend path does, decode does not); beta is then passed through as fp32.
    """
    T = a.numel() // (num_heads * head_dim)

    def rows(x):
        # A token stride is fine (split views of the packed qkv); only the
        # [H, D] block inside a token has to be dense.
        x = x.reshape(T, num_heads, head_dim)
        if x.stride(2) != 1 or x.stride(1) != head_dim:
            x = x.contiguous()
        return x

    q, k, a = rows(q), rows(k), rows(a)
    b = b.reshape(T, num_heads)
    if not b.is_contiguous():
        b = b.contiguous()

    qn = torch.empty(T, num_heads, head_dim, dtype=q.dtype, device=q.device)
    kn = torch.empty(T, num_heads, head_dim, dtype=k.dtype, device=k.device)
    g = torch.empty(T, num_heads, head_dim, dtype=torch.float32, device=q.device)
    beta = torch.empty(T, num_heads, dtype=torch.float32, device=q.device)

    R = T * num_heads
    block_r, num_warps = _prepare_config(R)
    _kda_prepare_kernel[(triton.cdiv(R, block_r),)](
        q,
        k,
        a,
        b,
        A_log.reshape(-1),
        dt_bias.reshape(-1),
        qn,
        kn,
        g,
        beta,
        lower_bound,
        stride_q_tok=q.stride(0),
        stride_k_tok=k.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=num_heads,
        R=R,
        H=num_heads,
        D=head_dim,
        BD=triton.next_power_of_2(head_dim),
        BLOCK_R=block_r,
        SOFTPLUS_THRESHOLD=20.0,
        USE_LOWER_BOUND=lower_bound is not None,
        SIGMOID_BETA=sigmoid_beta,
        num_warps=num_warps,
    )
    return (
        qn.unsqueeze(0),
        kn.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
    )
