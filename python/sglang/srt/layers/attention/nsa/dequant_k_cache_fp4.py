"""FP4 (E2M1) dequantization for NSA KV cache.

Input layout per token (416 bytes, uint8):
    [nope_fp4_packed(256) | uint8_scales(32) | rope_bf16_bytes(128)]

Output: [num_tokens, 1, 576] bf16
"""

import torch
import triton
import triton.language as tl

DIM_NOPE = 512
DIM_ROPE = 64
SCALE_BLOCK_SIZE = 16
NUM_SCALE_BLOCKS = DIM_NOPE // SCALE_BLOCK_SIZE  # 32

NOPE_FP4_BYTES = DIM_NOPE // 2  # 256
SCALE_BYTES = NUM_SCALE_BLOCKS  # 32
ROPE_BF16_BYTES = DIM_ROPE * 2  # 128
DIM_QUANT_FP4 = NOPE_FP4_BYTES + SCALE_BYTES + ROPE_BF16_BYTES  # 416


def dequantize_k_cache_fp4(quant_k_cache: torch.Tensor) -> torch.Tensor:
    """Dequantize the entire KV pool back to bf16.

    Args:
        quant_k_cache: [M, 1, 416] or [M, 416] uint8

    Returns:
        [M, 1, 576] bf16
    """
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

    grid = (num_tokens, NUM_SCALE_BLOCKS + 1)

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
        BLOCK_SIZE=SCALE_BLOCK_SIZE,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
    )

    if original_ndim == 3:
        return output.view(num_blocks, 1, -1)
    else:
        return output.view(num_blocks, block_size, 1, -1)


def dequantize_k_cache_fp4_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
) -> torch.Tensor:
    """Dequantize selected tokens by page indices (for compressed attention).

    Args:
        quant_k_cache: [total_tokens, 1, 416] uint8
        page_table_1_flattened: [num_tokens] int

    Returns:
        [num_tokens, 1, 576] bf16
    """
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

    grid = (num_tokens, NUM_SCALE_BLOCKS + 1)

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
        BLOCK_SIZE=SCALE_BLOCK_SIZE,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
    )

    return output


@triton.jit
def _e2m1_dequant(fp4_code, scale):
    """Dequantize FP4 E2M1 codes to float32."""
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
    BLOCK_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)

    if block_id < NUM_NOPE_BLOCKS:
        HALF: tl.constexpr = BLOCK_SIZE // 2

        pack_offs = tl.arange(0, HALF)
        packed = tl.load(
            in_packed_ptr + token_id * in_packed_stride_0 + block_id * HALF + pack_offs
        )

        scale_uint8 = tl.load(
            in_scale_ptr + token_id * in_scale_stride_0 + block_id
        )
        scale = tl.math.exp2(scale_uint8.to(tl.float32) - 127.0)

        low_fp4 = packed & 0x0F
        high_fp4 = (packed >> 4) & 0x0F

        low_float = _e2m1_dequant(low_fp4, scale)
        high_float = _e2m1_dequant(high_fp4, scale)

        block_start = block_id * BLOCK_SIZE
        out_base = output_ptr + token_id * output_stride_0
        even_offs = block_start + tl.arange(0, HALF) * 2
        odd_offs = even_offs + 1

        tl.store(out_base + even_offs, low_float.to(tl.bfloat16))
        tl.store(out_base + odd_offs, high_float.to(tl.bfloat16))
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
    BLOCK_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    out_token_id = tl.program_id(0)
    block_id = tl.program_id(1)

    src_token_id = tl.load(page_table_ptr + out_token_id)

    if block_id < NUM_NOPE_BLOCKS:
        HALF: tl.constexpr = BLOCK_SIZE // 2

        pack_offs = tl.arange(0, HALF)
        packed = tl.load(
            in_packed_ptr + src_token_id * in_packed_stride_0 + block_id * HALF + pack_offs
        )

        scale_uint8 = tl.load(
            in_scale_ptr + src_token_id * in_scale_stride_0 + block_id
        )
        scale = tl.math.exp2(scale_uint8.to(tl.float32) - 127.0)

        low_fp4 = packed & 0x0F
        high_fp4 = (packed >> 4) & 0x0F

        low_float = _e2m1_dequant(low_fp4, scale)
        high_float = _e2m1_dequant(high_fp4, scale)

        block_start = block_id * BLOCK_SIZE
        out_base = output_ptr + out_token_id * output_stride_0
        even_offs = block_start + tl.arange(0, HALF) * 2
        odd_offs = even_offs + 1

        tl.store(out_base + even_offs, low_float.to(tl.bfloat16))
        tl.store(out_base + odd_offs, high_float.to(tl.bfloat16))
    else:
        offs = tl.arange(0, DIM_ROPE)
        src = in_rope_ptr + src_token_id * in_rope_stride_0 + offs
        dst = output_ptr + out_token_id * output_stride_0 + DIM_NOPE + offs
        tl.store(dst, tl.load(src))
