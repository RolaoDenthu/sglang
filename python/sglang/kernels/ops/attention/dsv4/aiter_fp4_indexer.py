from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import torch

if TYPE_CHECKING:
    from sglang.kernels.ops.attention.dsv4.compress import (
        CompressorDecodePlan,
        CompressorPrefillPlan,
    )


_Q_HEADS = 64
_Q_HEAD_DIM = 128
_ROPE_DIM = 64
_GROUP_SIZE = 32
_KV_BLOCK_SIZE = 64
_Q_SCALE_SHAPE = (1, 4, 16, 4)
_BLOCK_K = 256
_MIN_PARALLEL_UNIT_NUM = 512


def prepare_aiter_fp4_indexer_cos_sin(
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    expected_rope_width = _ROPE_DIM // 2
    if (
        freqs_cis.ndim != 2
        or freqs_cis.shape[0] == 0
        or freqs_cis.shape[1] != expected_rope_width
        or not freqs_cis.is_complex()
    ):
        raise ValueError(
            "AITER FP4 C4Indexer requires complex freqs_cis with shape "
            f"[max_position, {expected_rope_width}]; got shape "
            f"{tuple(freqs_cis.shape)} and dtype {freqs_cis.dtype}"
        )

    cos = freqs_cis.real.to(dtype=torch.bfloat16).contiguous()
    sin = freqs_cis.imag.to(dtype=torch.bfloat16).contiguous()
    return cos, sin


def validate_aiter_fp4_indexer_cos_sin(
    cos: torch.Tensor, sin: torch.Tensor, device: torch.device
) -> None:
    expected_rope_width = _ROPE_DIM // 2
    for name, tensor in (("cos", cos), ("sin", sin)):
        if (
            tensor.ndim != 2
            or tensor.shape[0] == 0
            or tensor.shape[1] != expected_rope_width
        ):
            raise ValueError(
                "AITER FP4 C4Indexer requires precomputed cos/sin with shape "
                f"[max_position, {expected_rope_width}]; got {name} shape "
                f"{tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise ValueError(
                "AITER FP4 C4Indexer requires precomputed cos/sin dtype "
                f"torch.bfloat16; got {name} dtype {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"AITER FP4 C4Indexer requires contiguous {name}")
        if tensor.device != device:
            raise ValueError(
                "AITER FP4 C4Indexer requires q/k and precomputed cos/sin on "
                f"the same device; got q/k on {device} and {name} on "
                f"{tensor.device}"
            )
    if cos.shape != sin.shape:
        raise ValueError(
            "AITER FP4 C4Indexer requires matching precomputed cos/sin shapes; "
            f"got cos {tuple(cos.shape)} and sin {tuple(sin.shape)}"
        )


def aiter_q_indexer_rope_hadamard_fp4_quant(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = q.shape[0]
    positions = positions.to(device=q.device, dtype=torch.int64).contiguous()

    # AITER is intentionally imported only after the HIP FP4 path is selected.
    import aiter

    q_fp4 = torch.empty(
        (num_tokens, _Q_HEADS, _Q_HEAD_DIM // 2),
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


def prepare_aiter_fp4_prefill_plan(
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
    )

    num_tokens = page_table.shape[0]
    row_to_batch = torch.arange(
        num_tokens, device=page_table.device, dtype=torch.int32
    )
    local_starts = torch.zeros(
        num_tokens, device=page_table.device, dtype=torch.int32
    )
    padded_width = max(4, (page_table.shape[1] + 3) // 4 * 4)
    parallel_unit_num = max(_MIN_PARALLEL_UNIT_NUM, num_tokens)
    _, cta_info, n_ctas = compute_prefill_schedule(
        row_to_batch,
        local_starts,
        c4_seq_lens,
        _BLOCK_K,
        parallel_unit_num,
        padded_width * _KV_BLOCK_SIZE,
    )
    return row_to_batch, local_starts, cta_info, n_ctas


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
    prefill_plan: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None
    ) = None,
) -> torch.Tensor:
    num_tokens = q_fp4.shape[0]
    if page_table.ndim != 2 or page_table.shape[0] != num_tokens:
        raise ValueError(
            "AITER FP4 C4Indexer logits requires row-expanded page_table "
            f"shape [T, max_blocks]; got {tuple(page_table.shape)}"
        )

    page_table = page_table.to(dtype=torch.int32).contiguous()
    logical_page_table_width = page_table.shape[1]
    padded_page_table_width = max(4, (logical_page_table_width + 3) // 4 * 4)
    # FlyDSL pipelines one 256-token chunk ahead, so keep one zero page-id
    # chunk after the scheduled table width for its final speculative load.
    guarded_page_table_width = padded_page_table_width + 4
    padded_page_table = page_table.new_zeros(
        (num_tokens, guarded_page_table_width), dtype=torch.int32
    )
    padded_page_table[:, :logical_page_table_width].copy_(page_table)
    c4_seq_lens = c4_seq_lens.reshape(-1).to(dtype=torch.int32).contiguous()
    if c4_seq_lens.shape != (num_tokens,):
        raise ValueError(
            "AITER FP4 C4Indexer logits requires row-expanded c4_seq_lens "
            f"shape [T]; got {tuple(c4_seq_lens.shape)}"
        )
    max_seq_len = padded_page_table_width * _KV_BLOCK_SIZE

    # AITER FlyDSL is intentionally imported only after the HIP FP4 path is selected.
    from aiter.ops.flydsl import (
        flydsl_pa_mqa_logits_fp4,
        flydsl_pa_mqa_logits_fp4_prefill,
    )

    common_kwargs = {
        "weight_scale": weight_scale,
        "block_k": _BLOCK_K,
        "kv_block_size": _KV_BLOCK_SIZE,
        "num_warps": 4,
    }
    # FlyDSL models packed FP4 payloads as raw bytes. The AITER producer uses
    # torch's fp4x2 dtype for its pybind dispatch, so expose a zero-copy uint8
    # view at this boundary.
    q_payload = q_fp4.view(torch.uint8)
    k_payload_bytes = k_payload.view(torch.uint8)
    if is_decode:
        logits = flydsl_pa_mqa_logits_fp4(
            q_payload.reshape(num_tokens, 1, _Q_HEADS, _Q_HEAD_DIM // 2),
            q_scale.reshape(num_tokens, 1, *_Q_SCALE_SHAPE),
            k_payload_bytes,
            k_scale,
            padded_page_table,
            weights,
            c4_seq_lens,
            max_seq_len,
            next_n=1,
            parallel_unit_num=None,
            **common_kwargs,
        )
    else:
        if prefill_plan is None:
            prefill_plan = prepare_aiter_fp4_prefill_plan(
                page_table,
                c4_seq_lens,
            )
        row_to_batch, local_starts, cta_info, n_ctas = prefill_plan
        # This eager grid depends only on T, avoids sequence-value synchronization,
        # and guarantees at least one persistent unit for every query row.
        parallel_unit_num = max(_MIN_PARALLEL_UNIT_NUM, num_tokens)
        logits = flydsl_pa_mqa_logits_fp4_prefill(
            q_payload,
            q_scale,
            k_payload_bytes,
            k_scale,
            padded_page_table,
            weights,
            row_to_batch,
            local_starts,
            c4_seq_lens,
            max_seq_len,
            parallel_unit_num=parallel_unit_num,
            cta_info=cta_info,
            n_ctas=n_ctas,
            **common_kwargs,
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
) -> None:
    num_rows = k.shape[0]
    if num_rows == 0:
        return

    plan_words = plan[1].view(torch.int32)
    seq_lens = plan_words[:, 0].to(torch.int64)
    positions = seq_lens - plan.compress_ratio
    position_in_range = (positions >= 0) & (positions < cos.shape[0])
    positions = torch.where(position_in_range, positions, torch.zeros_like(positions))
    valid = position_in_range & (seq_lens % plan.compress_ratio == 0)

    out_loc_i64 = out_loc.to(device=k.device, dtype=torch.int64)
    if plan.is_decode:
        selected_slots = out_loc_i64
    elif out_loc_i64.shape[0] == 0:
        selected_slots = torch.full_like(seq_lens, -1)
        valid = torch.zeros_like(valid)
    else:
        ragged_ids = plan_words[:, 1].bitwise_and(0xFFFF).to(torch.int64)
        valid = valid & (ragged_ids < out_loc_i64.shape[0])
        safe_ragged_ids = ragged_ids.clamp(max=out_loc_i64.shape[0] - 1)
        selected_slots = out_loc_i64[safe_ragged_ids]

    slots = torch.where(valid, selected_slots, torch.full_like(selected_slots, -1))
    positions = positions.contiguous()
    slots = slots.contiguous()
    k_bf16 = k.to(dtype=torch.bfloat16).contiguous().view(num_rows, 1, _Q_HEAD_DIM)
    # Convert all 128 values per call so post-load weight mutations cannot stale a cache.
    norm_weight_bf16 = norm_weight.to(
        device=k.device, dtype=torch.bfloat16
    ).contiguous()

    # AITER is intentionally imported only after the HIP FP4 path is selected.
    import aiter

    aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache(
        k_payload,
        k_scale,
        k_bf16,
        norm_weight_bf16,
        cos,
        sin,
        positions,
        slots,
        norm_epsilon,
        rope_dim=_ROPE_DIM,
        kv_block_size=_KV_BLOCK_SIZE,
        group_size=_GROUP_SIZE,
        shuffle_scale=True,
        do_rotate_act=True,
    )
