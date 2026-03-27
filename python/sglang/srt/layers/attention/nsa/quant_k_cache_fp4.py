"""FP4 (E2M1) quantization for NSA KV cache.

Per-token layout (416 bytes, uint8):
    [nope_fp4_packed(256) | scales(32) | rope_bf16(128)]
"""

import torch
import triton
import triton.language as tl

DIM_NOPE = 512
DIM_ROPE = 64
SCALE_BLOCK_SIZE = 16
NUM_SCALE_BLOCKS = DIM_NOPE // SCALE_BLOCK_SIZE

NOPE_FP4_BYTES = DIM_NOPE // 2
SCALE_BYTES = NUM_SCALE_BLOCKS
ROPE_BF16_BYTES = DIM_ROPE * 2
DIM_QUANT_FP4 = NOPE_FP4_BYTES + SCALE_BYTES + ROPE_BF16_BYTES

_BLOCKS_PER_PROG = 2
_NUM_WARPS = 1
_NUM_STAGES = 0
_GRID_Y = triton.cdiv(NUM_SCALE_BLOCKS, _BLOCKS_PER_PROG) + 1


def quantize_k_cache_fp4_separate(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize nope to FP4, keep rope as bf16 bytes."""
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope

    num_tokens = k_nope_2d.shape[0]
    assert k_nope_2d.shape[1] == DIM_NOPE
    assert k_rope_2d.shape[1] == DIM_ROPE

    k_nope_2d = k_nope_2d.contiguous()
    k_rope_2d = k_rope_2d.contiguous()

    nope_part_bytes = NOPE_FP4_BYTES + SCALE_BYTES
    rope_part_bytes = ROPE_BF16_BYTES

    nope_part = torch.empty(
        (num_tokens, nope_part_bytes), dtype=torch.uint8, device=k_nope_2d.device
    )
    rope_part = torch.empty(
        (num_tokens, rope_part_bytes), dtype=torch.uint8, device=k_rope_2d.device
    )

    out_packed = nope_part[:, :NOPE_FP4_BYTES]
    out_scale = nope_part[:, NOPE_FP4_BYTES:]
    out_rope = rope_part.view(torch.bfloat16)

    grid = (num_tokens, _GRID_Y)

    _quantize_k_cache_fp4_kernel[grid](
        out_packed,
        out_scale,
        out_rope,
        k_nope_2d,
        k_rope_2d,
        out_packed.stride(0),
        out_scale.stride(0),
        out_rope.stride(0),
        k_nope_2d.stride(0),
        k_rope_2d.stride(0),
        NUM_NOPE_BLOCKS=NUM_SCALE_BLOCKS,
        SCALE_BLOCK_SIZE=SCALE_BLOCK_SIZE,
        DIM_ROPE=DIM_ROPE,
        BLOCKS_PER_PROG=_BLOCKS_PER_PROG,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    return (
        nope_part.view(num_tokens, 1, nope_part_bytes),
        rope_part.view(num_tokens, 1, rope_part_bytes),
    )


@triton.jit
def _quant_nope_block(
    out_packed_ptr,
    out_scale_ptr,
    k_nope_ptr,
    token_id,
    block_id,
    out_packed_stride_0,
    out_scale_stride_0,
    k_nope_stride_0,
    SCALE_BLOCK_SIZE: tl.constexpr,
):
    HALF: tl.constexpr = SCALE_BLOCK_SIZE // 2

    block_start = block_id * SCALE_BLOCK_SIZE
    k_base = k_nope_ptr + token_id * k_nope_stride_0

    even_offs = block_start + tl.arange(0, HALF) * 2
    odd_offs = even_offs + 1

    even_vals = tl.load(k_base + even_offs).to(tl.float32)
    odd_vals = tl.load(k_base + odd_offs).to(tl.float32)

    even_abs = tl.abs(even_vals)
    odd_abs = tl.abs(odd_vals)
    block_max = tl.maximum(tl.max(even_abs), tl.max(odd_abs))

    safe_ratio = tl.maximum(block_max / 6.0, 1e-10)
    scale_exp = tl.math.ceil(tl.math.log2(safe_ratio))
    scale_uint8 = (scale_exp + 127.0).to(tl.uint8)
    inv_scale = tl.math.exp2(-scale_exp)

    even_scaled = even_abs * inv_scale
    even_mag = (
        (even_scaled >= 0.25).to(tl.uint8)
        + (even_scaled >= 0.75).to(tl.uint8)
        + (even_scaled >= 1.25).to(tl.uint8)
        + (even_scaled >= 1.75).to(tl.uint8)
        + (even_scaled >= 2.5).to(tl.uint8)
        + (even_scaled >= 3.5).to(tl.uint8)
        + (even_scaled >= 5.0).to(tl.uint8)
    )
    even_sign = (even_vals < 0).to(tl.uint8) << 3
    even_fp4 = even_sign | even_mag

    odd_scaled = odd_abs * inv_scale
    odd_mag = (
        (odd_scaled >= 0.25).to(tl.uint8)
        + (odd_scaled >= 0.75).to(tl.uint8)
        + (odd_scaled >= 1.25).to(tl.uint8)
        + (odd_scaled >= 1.75).to(tl.uint8)
        + (odd_scaled >= 2.5).to(tl.uint8)
        + (odd_scaled >= 3.5).to(tl.uint8)
        + (odd_scaled >= 5.0).to(tl.uint8)
    )
    odd_sign = (odd_vals < 0).to(tl.uint8) << 3
    odd_fp4 = odd_sign | odd_mag

    packed = even_fp4 | (odd_fp4 << 4)

    pack_ptr = out_packed_ptr + token_id * out_packed_stride_0 + block_id * HALF
    tl.store(pack_ptr + tl.arange(0, HALF), packed)

    scale_ptr = out_scale_ptr + token_id * out_scale_stride_0 + block_id
    tl.store(scale_ptr, scale_uint8)


@triton.jit
def _quantize_k_cache_fp4_kernel(
    out_packed_ptr,
    out_scale_ptr,
    out_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    out_packed_stride_0: int,
    out_scale_stride_0: int,
    out_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
):
    token_id = tl.program_id(0)
    group_id = tl.program_id(1)

    num_nope_groups: tl.constexpr = (
        NUM_NOPE_BLOCKS + BLOCKS_PER_PROG - 1
    ) // BLOCKS_PER_PROG

    if group_id < num_nope_groups:
        for sub in tl.static_range(BLOCKS_PER_PROG):
            block_id = group_id * BLOCKS_PER_PROG + sub
            if block_id < NUM_NOPE_BLOCKS:
                _quant_nope_block(
                    out_packed_ptr,
                    out_scale_ptr,
                    k_nope_ptr,
                    token_id,
                    block_id,
                    out_packed_stride_0,
                    out_scale_stride_0,
                    k_nope_stride_0,
                    SCALE_BLOCK_SIZE,
                )
    else:
        offs = tl.arange(0, DIM_ROPE)
        src = k_rope_ptr + token_id * k_rope_stride_0 + offs
        dst = out_rope_ptr + token_id * out_rope_stride_0 + offs
        tl.store(dst, tl.load(src))
