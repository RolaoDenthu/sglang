"""FP4 (E2M1) dequantization for NSA KV cache.

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


def dequantize_k_cache_fp4(quant_k_cache: torch.Tensor) -> torch.Tensor:
    """Dequantize the entire KV pool back to bf16."""
    original_ndim = quant_k_cache.ndim
    if original_ndim == 3:
        quant_k_cache = quant_k_cache.unsqueeze(1)

    num_blocks, block_size, _, dim_quant = quant_k_cache.shape
    assert dim_quant == DIM_QUANT_FP4

    quant_flat = quant_k_cache.view(-1, DIM_QUANT_FP4)
    num_tokens = quant_flat.shape[0]

    output = torch.empty(
        (num_tokens, DIM_NOPE + DIM_ROPE),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    in_packed = quant_flat[:, :NOPE_FP4_BYTES]
    in_scale = quant_flat[:, NOPE_FP4_BYTES : NOPE_FP4_BYTES + SCALE_BYTES]
    in_rope = quant_flat[:, NOPE_FP4_BYTES + SCALE_BYTES :].view(torch.bfloat16)

    grid = (num_tokens, _GRID_Y)

    _dequantize_k_cache_fp4_kernel[grid](
        output,
        in_packed,
        in_scale,
        in_rope,
        output.stride(0),
        in_packed.stride(0),
        in_scale.stride(0),
        in_rope.stride(0),
        NUM_NOPE_BLOCKS=NUM_SCALE_BLOCKS,
        SCALE_BLOCK_SIZE=SCALE_BLOCK_SIZE,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        BLOCKS_PER_PROG=_BLOCKS_PER_PROG,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    if original_ndim == 3:
        return output.view(num_blocks, 1, -1)
    else:
        return output.view(num_blocks, block_size, 1, -1)


def dequantize_k_cache_fp4_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
) -> torch.Tensor:
    """Dequantize selected tokens by page indices."""
    dim_quant = quant_k_cache.shape[-1]
    assert dim_quant == DIM_QUANT_FP4

    quant_flat = quant_k_cache.view(-1, DIM_QUANT_FP4)
    num_tokens = page_table_1_flattened.shape[0]

    output = torch.empty(
        (num_tokens, 1, DIM_NOPE + DIM_ROPE),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    in_packed = quant_flat[:, :NOPE_FP4_BYTES]
    in_scale = quant_flat[:, NOPE_FP4_BYTES : NOPE_FP4_BYTES + SCALE_BYTES]
    in_rope = quant_flat[:, NOPE_FP4_BYTES + SCALE_BYTES :].view(torch.bfloat16)

    grid = (num_tokens, _GRID_Y)

    _dequantize_k_cache_fp4_paged_kernel[grid](
        output,
        in_packed,
        in_scale,
        in_rope,
        page_table_1_flattened,
        output.stride(0),
        in_packed.stride(0),
        in_scale.stride(0),
        in_rope.stride(0),
        NUM_NOPE_BLOCKS=NUM_SCALE_BLOCKS,
        SCALE_BLOCK_SIZE=SCALE_BLOCK_SIZE,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        BLOCKS_PER_PROG=_BLOCKS_PER_PROG,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    return output


@triton.jit
def _e2m1_dequant(fp4_code, scale):
    mag = (fp4_code & 0x07).to(tl.int32)
    sign_bit = ((fp4_code >> 3) & 1).to(tl.int32)

    exp_field = mag >> 1
    mant_bit = (mag & 1).to(tl.float32)

    is_subnormal = exp_field == 0
    sub_val = 0.5 * mant_bit
    norm_val = tl.math.exp2((exp_field - 1).to(tl.float32)) * (1.0 + 0.5 * mant_bit)

    float_val = tl.where(is_subnormal, sub_val, norm_val) * scale
    return tl.where(sign_bit != 0, -float_val, float_val)


@triton.jit
def _dequant_nope_block(
    in_packed_ptr,
    in_scale_ptr,
    output_ptr,
    src_token_id,
    dst_token_id,
    block_id,
    in_packed_stride_0,
    in_scale_stride_0,
    output_stride_0,
    SCALE_BLOCK_SIZE: tl.constexpr,
):
    HALF: tl.constexpr = SCALE_BLOCK_SIZE // 2

    pack_offs = tl.arange(0, HALF)
    packed = tl.load(
        in_packed_ptr + src_token_id * in_packed_stride_0 + block_id * HALF + pack_offs
    )

    scale_uint8 = tl.load(in_scale_ptr + src_token_id * in_scale_stride_0 + block_id)
    scale = tl.math.exp2(scale_uint8.to(tl.float32) - 127.0)

    low_fp4 = packed & 0x0F
    high_fp4 = (packed >> 4) & 0x0F

    low_float = _e2m1_dequant(low_fp4, scale)
    high_float = _e2m1_dequant(high_fp4, scale)

    block_start = block_id * SCALE_BLOCK_SIZE
    out_base = output_ptr + dst_token_id * output_stride_0
    even_offs = block_start + tl.arange(0, HALF) * 2
    odd_offs = even_offs + 1

    tl.store(out_base + even_offs, low_float.to(tl.bfloat16))
    tl.store(out_base + odd_offs, high_float.to(tl.bfloat16))


@triton.jit
def _dequantize_k_cache_fp4_kernel(
    output_ptr,
    in_packed_ptr,
    in_scale_ptr,
    in_rope_ptr,
    output_stride_0: int,
    in_packed_stride_0: int,
    in_scale_stride_0: int,
    in_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
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
                _dequant_nope_block(
                    in_packed_ptr,
                    in_scale_ptr,
                    output_ptr,
                    token_id,
                    token_id,
                    block_id,
                    in_packed_stride_0,
                    in_scale_stride_0,
                    output_stride_0,
                    SCALE_BLOCK_SIZE,
                )
    else:
        offs = tl.arange(0, DIM_ROPE)
        src = in_rope_ptr + token_id * in_rope_stride_0 + offs
        dst = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs
        tl.store(dst, tl.load(src))


@triton.jit
def _dequantize_k_cache_fp4_paged_kernel(
    output_ptr,
    in_packed_ptr,
    in_scale_ptr,
    in_rope_ptr,
    page_table_ptr,
    output_stride_0: int,
    in_packed_stride_0: int,
    in_scale_stride_0: int,
    in_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
):
    out_token_id = tl.program_id(0)
    group_id = tl.program_id(1)

    src_token_id = tl.load(page_table_ptr + out_token_id)

    num_nope_groups: tl.constexpr = (
        NUM_NOPE_BLOCKS + BLOCKS_PER_PROG - 1
    ) // BLOCKS_PER_PROG

    if group_id < num_nope_groups:
        for sub in tl.static_range(BLOCKS_PER_PROG):
            block_id = group_id * BLOCKS_PER_PROG + sub
            if block_id < NUM_NOPE_BLOCKS:
                _dequant_nope_block(
                    in_packed_ptr,
                    in_scale_ptr,
                    output_ptr,
                    src_token_id,
                    out_token_id,
                    block_id,
                    in_packed_stride_0,
                    in_scale_stride_0,
                    output_stride_0,
                    SCALE_BLOCK_SIZE,
                )
    else:
        offs = tl.arange(0, DIM_ROPE)
        src = in_rope_ptr + src_token_id * in_rope_stride_0 + offs
        dst = output_ptr + out_token_id * output_stride_0 + DIM_NOPE + offs
        tl.store(dst, tl.load(src))
