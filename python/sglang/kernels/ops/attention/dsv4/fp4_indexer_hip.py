"""AITER adapters for the DeepSeek-V4 FP4 indexer on HIP."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Tuple, Union

import torch

if TYPE_CHECKING:
    from sglang.kernels.ops.attention.dsv4.compress import (
        CompressorDecodePlan,
        CompressorPrefillPlan,
    )


_HEADS = 64
_HEAD_DIM = 128
_ROPE_DIM = 64
_GROUP_SIZE = 32
_KV_BLOCK_SIZE = 64
_Q_SCALE_SHAPE = (1, 4, 16, 4)
# gfx950 has 256 CUs; target four persistent CTAs per CU.
_DECODE_BASE_CTA_TARGET = 1024
# Preserve per-query parallelism when the batch itself exceeds one CTA per CU.
_DECODE_CTAS_PER_QUERY = 4


# ---------------------------------------------------------------------------
# Metadata types
# ---------------------------------------------------------------------------


class FP4DecodeWorkspace(NamedTuple):
    guarded_page_table: torch.Tensor
    cta_info: torch.Tensor
    cta_count: int
    logits: torch.Tensor


class FP4KWriteMetadata(NamedTuple):
    positions: torch.Tensor
    slots: torch.Tensor


# ---------------------------------------------------------------------------
# Metadata preparation
# ---------------------------------------------------------------------------


def _decode_cta_count(num_queries: int, max_seq_len: int) -> int:
    """Choose a bounded persistent grid without exceeding available KV chunks."""
    chunks_per_seq = max(1, (max_seq_len + 255) // 256)
    available_ctas = num_queries * chunks_per_seq
    target_ctas = max(
        _DECODE_BASE_CTA_TARGET,
        num_queries * _DECODE_CTAS_PER_QUERY,
    )
    return min(available_ctas, target_ctas)


def _guard_page_table(page_table: torch.Tensor):
    """Pad page tables for 256-token scheduling and one-chunk lookahead."""
    page_table = page_table.to(dtype=torch.int32).contiguous()
    rows, logical_width = page_table.shape
    padded_width = max(4, (logical_width + 3) // 4 * 4)
    guarded = page_table.new_zeros((rows, padded_width + 4))
    guarded[:, :logical_width].copy_(page_table)
    return guarded, padded_width * _KV_BLOCK_SIZE


def prepare_fp4_decode_workspace(
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
) -> FP4DecodeWorkspace:
    """Build graph-stable page-table, schedule, and logits buffers for decode."""
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
        compute_varctx_schedule,
    )

    page_table, max_seq_len = _guard_page_table(page_table)
    c4_seq_lens = c4_seq_lens.reshape(-1).to(torch.int32).contiguous()
    cta_count = _decode_cta_count(page_table.shape[0], max_seq_len)
    _, cta_info, _ = compute_varctx_schedule(
        c4_seq_lens,
        block_k=256,
        parallel_unit_num=cta_count,
        max_seq_len=max_seq_len,
        next_n=1,
    )
    logits = torch.full(
        (page_table.shape[0], max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=page_table.device,
    )
    return FP4DecodeWorkspace(page_table, cta_info, cta_count, logits)


def prepare_fp4_k_write_metadata(
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    out_loc: torch.Tensor,
    max_position: int,
    device: torch.device,
) -> FP4KWriteMetadata:
    """Build RoPE positions and physical cache slots from a compressor plan."""
    plan_words = plan[1].view(torch.int32)
    seq_lens = plan_words[:, 0].to(torch.int64)
    positions = seq_lens - plan.compress_ratio
    valid = (positions >= 0) & (positions < max_position)
    positions = torch.where(valid, positions, torch.zeros_like(positions))
    valid &= seq_lens % plan.compress_ratio == 0

    out_loc = out_loc.to(device=device, dtype=torch.int64)
    if plan.is_decode:
        slots = out_loc
    elif out_loc.shape[0] == 0:
        slots = torch.full_like(seq_lens, -1)
        valid.zero_()
    else:
        ragged_ids = plan_words[:, 1].bitwise_and(0xFFFF).to(torch.int64)
        valid &= ragged_ids < out_loc.shape[0]
        slots = out_loc[ragged_ids.clamp(max=out_loc.shape[0] - 1)]
    slots = torch.where(valid, slots, torch.full_like(slots, -1))
    return FP4KWriteMetadata(positions.contiguous(), slots.contiguous())


# ---------------------------------------------------------------------------
# Kernel interfaces
# ---------------------------------------------------------------------------


def aiter_q_indexer_fp4(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE and Hadamard rotation, then quantize indexer Q to FP4."""
    import aiter

    num_tokens = q.shape[0]
    positions = positions.to(device=q.device, dtype=torch.int64).contiguous()
    q_fp4 = torch.empty(
        (num_tokens, _HEADS, _HEAD_DIM // 2),
        dtype=aiter.dtypes.fp4x2,
        device=q.device,
    )
    q_scale = torch.empty(
        (num_tokens, *_Q_SCALE_SHAPE), dtype=torch.uint8, device=q.device
    )
    aiter.rope_rotate_activation(
        q_fp4,
        q,
        cos,
        sin,
        positions,
        rope_dim=_ROPE_DIM,
        out_scale=q_scale,
        group_size=_GROUP_SIZE,
        shuffle_scale=True,
        do_rotate_act=True,
    )
    return q_fp4, q_scale


def aiter_fp4_paged_mqa_logits(
    *,
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    weight_scale: float,
    is_decode: bool,
    decode_workspace: FP4DecodeWorkspace | None = None,
) -> torch.Tensor:
    """Compute FP4 Q/K indexer logits with the decode or prefill FlyDSL kernel."""
    from aiter.ops.flydsl import (
        flydsl_pa_mqa_logits_fp4,
        flydsl_pa_mqa_logits_fp4_prefill,
    )

    num_tokens = q_fp4.shape[0]
    c4_seq_lens = c4_seq_lens.reshape(-1).to(torch.int32).contiguous()
    if decode_workspace is None:
        page_table, max_seq_len = _guard_page_table(page_table)
    else:
        page_table = decode_workspace.guarded_page_table
        max_seq_len = decode_workspace.logits.shape[1]
    q_payload = q_fp4.view(torch.uint8)
    k_payload = k_payload.view(torch.uint8)
    common = {
        "weight_scale": weight_scale,
        "block_k": 256,
        "kv_block_size": _KV_BLOCK_SIZE,
        "num_warps": 4,
    }

    if is_decode:
        decode_kwargs = {}
        if decode_workspace is not None:
            decode_kwargs = {
                "out": decode_workspace.logits,
                "cta_info": decode_workspace.cta_info,
                "total_ctas": decode_workspace.cta_count,
            }
        logits = flydsl_pa_mqa_logits_fp4(
            q_payload.reshape(num_tokens, 1, _HEADS, _HEAD_DIM // 2),
            q_scale.reshape(num_tokens, 1, *_Q_SCALE_SHAPE),
            k_payload,
            k_scale,
            page_table,
            weights,
            c4_seq_lens,
            max_seq_len,
            next_n=1,
            parallel_unit_num=None,
            **decode_kwargs,
            **common,
        )
    else:
        row_to_batch = torch.arange(num_tokens, device=q_fp4.device, dtype=torch.int32)
        local_starts = torch.zeros(num_tokens, device=q_fp4.device, dtype=torch.int32)
        logits = flydsl_pa_mqa_logits_fp4_prefill(
            q_payload,
            q_scale,
            k_payload,
            k_scale,
            page_table,
            weights,
            row_to_batch,
            local_starts,
            c4_seq_lens,
            max_seq_len,
            parallel_unit_num=max(512, num_tokens),
            **common,
        )

    return logits


def aiter_k_indexer_fp4_cache_write(
    *,
    k: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_epsilon: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    out_loc: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    write_metadata: FP4KWriteMetadata | None = None,
) -> None:
    """Map compressed K rows to cache slots and run the fused AITER FP4 writer."""
    num_rows = k.shape[0]
    if num_rows == 0:
        return

    if write_metadata is None:
        positions, slots = prepare_fp4_k_write_metadata(
            plan, out_loc, cos.shape[0], k.device
        )
    else:
        positions, slots = write_metadata.positions, write_metadata.slots

    import aiter

    aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache(
        k_payload,
        k_scale,
        k.view(num_rows, 1, _HEAD_DIM),
        norm_weight.to(device=k.device, dtype=torch.bfloat16).contiguous(),
        cos,
        sin,
        positions.contiguous(),
        slots,
        norm_epsilon,
        rope_dim=_ROPE_DIM,
        kv_block_size=_KV_BLOCK_SIZE,
        group_size=_GROUP_SIZE,
        shuffle_scale=True,
        do_rotate_act=True,
    )
