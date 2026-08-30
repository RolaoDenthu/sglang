from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Tuple, Union

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
_DECODE_MIN_PARALLEL_UNITS = 1024
_DECODE_PARALLEL_UNITS_PER_QUERY = 4


class AiterFP4PagedMQADecodeMetadata(NamedTuple):
    padded_page_table: torch.Tensor
    cta_info: torch.Tensor
    total_ctas: int
    out: torch.Tensor
    logical_max_seq_len: int


class AiterFP4PagedMQAPrefillMetadata(NamedTuple):
    padded_page_table: torch.Tensor
    row_to_batch: torch.Tensor
    local_starts: torch.Tensor
    cta_info: torch.Tensor
    n_ctas: int
    out: torch.Tensor
    logical_max_seq_len: int


def _get_aiter_fp4_decode_parallel_unit_num(num_tokens: int, max_seq_len: int) -> int:
    chunks_per_seq = max(1, (max_seq_len + 255) // 256)
    auto_parallel_units = num_tokens * chunks_per_seq
    target_parallel_units = max(
        _DECODE_MIN_PARALLEL_UNITS,
        num_tokens * _DECODE_PARALLEL_UNITS_PER_QUERY,
    )
    return min(auto_parallel_units, target_parallel_units)


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


def _validate_cos_sin(
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


def validate_aiter_fp4_indexer_static_contract(
    *,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_weight: torch.Tensor,
) -> None:
    """Validate model-owned FP4 tensors once, before the forward hot path."""
    if (num_heads, head_dim, rope_dim) != (_Q_HEADS, _Q_HEAD_DIM, _ROPE_DIM):
        raise ValueError(
            "AITER FP4 C4Indexer requires 64 heads, head_dim=128, and rope_dim=64; "
            f"got heads={num_heads}, head_dim={head_dim}, rope_dim={rope_dim}"
        )
    _validate_cos_sin(cos, sin, cos.device)
    if norm_weight.shape != (_Q_HEAD_DIM,):
        raise ValueError(
            "AITER FP4 C4Indexer requires norm_weight shape [128]; "
            f"got {tuple(norm_weight.shape)}"
        )


def aiter_q_indexer_rope_hadamard_fp4_quant(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize Q after model-owned geometry and RoPE tables are validated."""
    if q.dtype != torch.bfloat16:
        raise ValueError(
            "AITER FP4 C4Indexer requires q dtype torch.bfloat16; " f"got {q.dtype}"
        )
    if not q.is_contiguous():
        raise ValueError("AITER FP4 C4Indexer requires contiguous q")

    num_tokens = q.shape[0]
    if positions.ndim != 1 or positions.shape[0] != num_tokens:
        raise ValueError(
            "AITER FP4 C4Indexer requires positions shape [T] matching q; "
            f"got positions {tuple(positions.shape)} for T={num_tokens}"
        )

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


def _prepare_aiter_fp4_guarded_page_table(
    page_table: torch.Tensor,
) -> Tuple[torch.Tensor, int, int]:
    page_table = page_table.to(dtype=torch.int32).contiguous()
    num_tokens, logical_page_table_width = page_table.shape
    padded_page_table_width = max(4, (logical_page_table_width + 3) // 4 * 4)
    # FlyDSL pipelines one 256-token chunk ahead, so keep one zero page-id
    # chunk after the scheduled table width for its final speculative load.
    guarded_page_table_width = padded_page_table_width + 4
    padded_page_table = page_table.new_zeros(
        (num_tokens, guarded_page_table_width), dtype=torch.int32
    )
    padded_page_table[:, :logical_page_table_width].copy_(page_table)
    return padded_page_table, logical_page_table_width, padded_page_table_width


def prepare_aiter_fp4_paged_mqa_decode_metadata(
    *,
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
) -> AiterFP4PagedMQADecodeMetadata:
    if page_table.ndim != 2:
        raise ValueError(
            "AITER FP4 C4Indexer requires row-expanded page_table; "
            f"got {tuple(page_table.shape)}"
        )
    num_tokens = page_table.shape[0]
    c4_seq_lens = c4_seq_lens.reshape(-1).to(dtype=torch.int32).contiguous()
    if c4_seq_lens.shape != (num_tokens,):
        raise ValueError(
            "AITER FP4 C4Indexer requires row-expanded c4_seq_lens "
            f"shape [T]; got {tuple(c4_seq_lens.shape)} for T={num_tokens}"
        )
    padded_page_table, logical_width, padded_width = (
        _prepare_aiter_fp4_guarded_page_table(page_table)
    )
    logical_max_seq_len = logical_width * _KV_BLOCK_SIZE
    max_seq_len = padded_width * _KV_BLOCK_SIZE

    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
        compute_varctx_schedule,
    )

    _, cta_info, total_ctas = compute_varctx_schedule(
        c4_seq_lens,
        block_k=256,
        parallel_unit_num=_get_aiter_fp4_decode_parallel_unit_num(
            num_tokens, max_seq_len
        ),
        max_seq_len=max_seq_len,
        next_n=1,
    )
    out = torch.full(
        (num_tokens, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=page_table.device,
    )
    return AiterFP4PagedMQADecodeMetadata(
        padded_page_table=padded_page_table,
        cta_info=cta_info,
        total_ctas=total_ctas,
        out=out,
        logical_max_seq_len=logical_max_seq_len,
    )


def prepare_aiter_fp4_paged_mqa_prefill_metadata(
    *,
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
) -> AiterFP4PagedMQAPrefillMetadata:
    if page_table.ndim != 2:
        raise ValueError(
            "AITER FP4 C4Indexer requires row-expanded page_table; "
            f"got {tuple(page_table.shape)}"
        )
    num_tokens = page_table.shape[0]
    c4_seq_lens = c4_seq_lens.reshape(-1).to(dtype=torch.int32).contiguous()
    if c4_seq_lens.shape != (num_tokens,):
        raise ValueError(
            "AITER FP4 C4Indexer requires row-expanded c4_seq_lens "
            f"shape [T]; got {tuple(c4_seq_lens.shape)} for T={num_tokens}"
        )
    padded_page_table, logical_width, padded_width = (
        _prepare_aiter_fp4_guarded_page_table(page_table)
    )
    logical_max_seq_len = logical_width * _KV_BLOCK_SIZE
    max_seq_len = padded_width * _KV_BLOCK_SIZE
    row_to_batch = torch.arange(num_tokens, device=page_table.device, dtype=torch.int32)
    local_starts = torch.zeros(num_tokens, device=page_table.device, dtype=torch.int32)
    parallel_unit_num = max(512, num_tokens)

    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
    )

    _, cta_info, n_ctas = compute_prefill_schedule(
        row_to_batch,
        local_starts,
        c4_seq_lens,
        block_k=256,
        parallel_unit_num=parallel_unit_num,
        max_seq_len=max_seq_len,
    )
    out = torch.full(
        (num_tokens, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=page_table.device,
    )
    return AiterFP4PagedMQAPrefillMetadata(
        padded_page_table=padded_page_table,
        row_to_batch=row_to_batch,
        local_starts=local_starts,
        cta_info=cta_info,
        n_ctas=n_ctas,
        out=out,
        logical_max_seq_len=logical_max_seq_len,
    )


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
    decode_metadata: AiterFP4PagedMQADecodeMetadata | None = None,
    prefill_metadata: AiterFP4PagedMQAPrefillMetadata | None = None,
) -> torch.Tensor:
    """Run logits with immutable Q/K layouts guaranteed by model and pool init."""
    num_tokens = q_fp4.shape[0]
    if (
        q_scale.shape[0] != num_tokens
        or weights.shape[0] != num_tokens
        or page_table.ndim != 2
        or page_table.shape[0] != num_tokens
    ):
        raise ValueError("AITER FP4 logits inputs must have the same row count")
    c4_seq_lens = c4_seq_lens.reshape(-1).to(dtype=torch.int32).contiguous()
    if c4_seq_lens.shape != (num_tokens,):
        raise ValueError(
            "AITER FP4 C4Indexer logits requires row-expanded c4_seq_lens "
            f"shape [T]; got {tuple(c4_seq_lens.shape)}"
        )
    if decode_metadata is not None and prefill_metadata is not None:
        raise ValueError("AITER FP4 logits accepts only one metadata mode")
    if decode_metadata is not None and not is_decode:
        raise ValueError("AITER FP4 prefill cannot use decode metadata")
    if prefill_metadata is not None and is_decode:
        raise ValueError("AITER FP4 decode cannot use prefill metadata")

    if decode_metadata is not None:
        padded_page_table = decode_metadata.padded_page_table
        logical_max_seq_len = page_table.shape[1] * _KV_BLOCK_SIZE
        max_seq_len = decode_metadata.out.shape[1]
        if (
            decode_metadata.logical_max_seq_len != logical_max_seq_len
            or padded_page_table.shape[0] != num_tokens
            or decode_metadata.out.shape[0] != num_tokens
        ):
            raise ValueError("AITER FP4 decode metadata does not match logits inputs")
    elif prefill_metadata is not None:
        padded_page_table = prefill_metadata.padded_page_table
        logical_max_seq_len = page_table.shape[1] * _KV_BLOCK_SIZE
        max_seq_len = prefill_metadata.out.shape[1]
        if (
            prefill_metadata.logical_max_seq_len != logical_max_seq_len
            or padded_page_table.shape[0] != num_tokens
            or prefill_metadata.row_to_batch.shape[0] != num_tokens
            or prefill_metadata.local_starts.shape[0] != num_tokens
            or prefill_metadata.out.shape[0] != num_tokens
        ):
            raise ValueError("AITER FP4 prefill metadata does not match logits inputs")
    else:
        padded_page_table, logical_page_table_width, padded_page_table_width = (
            _prepare_aiter_fp4_guarded_page_table(page_table)
        )
        logical_max_seq_len = logical_page_table_width * _KV_BLOCK_SIZE
        max_seq_len = padded_page_table_width * _KV_BLOCK_SIZE

    # AITER FlyDSL is intentionally imported only after the HIP FP4 path is selected.
    from aiter.ops.flydsl import (
        flydsl_pa_mqa_logits_fp4,
        flydsl_pa_mqa_logits_fp4_prefill,
    )

    common_kwargs = {
        "weight_scale": weight_scale,
        "block_k": 256,
        "kv_block_size": _KV_BLOCK_SIZE,
        "num_warps": 4,
    }
    # FlyDSL models packed FP4 payloads as raw bytes. The AITER producer uses
    # torch's fp4x2 dtype for its pybind dispatch, so expose a zero-copy uint8
    # view at this boundary.
    q_payload = q_fp4.view(torch.uint8)
    k_payload_bytes = k_payload.view(torch.uint8)
    if is_decode:
        decode_kwargs = {}
        if decode_metadata is not None:
            decode_kwargs = {
                "out": decode_metadata.out,
                "cta_info": decode_metadata.cta_info,
                "total_ctas": decode_metadata.total_ctas,
            }
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
            **decode_kwargs,
            **common_kwargs,
        )
    else:
        prefill_kwargs = {}
        if prefill_metadata is None:
            row_to_batch = torch.arange(
                num_tokens, device=q_fp4.device, dtype=torch.int32
            )
            local_starts = torch.zeros(
                num_tokens, device=q_fp4.device, dtype=torch.int32
            )
        else:
            row_to_batch = prefill_metadata.row_to_batch
            local_starts = prefill_metadata.local_starts
            prefill_kwargs = {
                "out": prefill_metadata.out,
                "cta_info": prefill_metadata.cta_info,
                "n_ctas": prefill_metadata.n_ctas,
            }
        # This eager grid depends only on T, avoids sequence-value synchronization,
        # and guarantees at least one persistent unit for every query row.
        parallel_unit_num = max(512, num_tokens)
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
            **prefill_kwargs,
            **common_kwargs,
        )

    return logits[:, :logical_max_seq_len]


def prepare_aiter_k_indexer_fp4_cache_write_metadata(
    *,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    out_loc: torch.Tensor,
    max_position: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    plan_words = plan[1].view(torch.int32)
    seq_lens = plan_words[:, 0].to(torch.int64)
    positions = seq_lens - plan.compress_ratio
    position_in_range = (positions >= 0) & (positions < max_position)
    positions = torch.where(position_in_range, positions, torch.zeros_like(positions))
    valid = position_in_range & (seq_lens % plan.compress_ratio == 0)

    out_loc_i64 = out_loc.to(device=device, dtype=torch.int64)
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
    return positions.contiguous(), slots.contiguous()


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
    write_metadata: Tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Write K after model-owned RoPE and cache layouts are validated."""
    if k.dtype != torch.bfloat16 or not k.is_contiguous():
        raise ValueError("AITER FP4 C4Indexer requires contiguous bfloat16 K")
    if norm_weight.dtype != torch.bfloat16 or not norm_weight.is_contiguous():
        raise ValueError("AITER FP4 C4Indexer requires contiguous bfloat16 norm_weight")
    num_rows = k.shape[0]
    if plan.compress_ratio != 4 or plan[1].shape != (num_rows, 16):
        raise ValueError(
            "AITER FP4 C4Indexer requires a C4 plan with one 16-byte row "
            f"per K row; got ratio={plan.compress_ratio}, "
            f"plan shape={tuple(plan[1].shape)}, N={num_rows}"
        )
    if out_loc.ndim != 1:
        raise ValueError(
            "AITER FP4 C4Indexer requires one-dimensional out_loc; "
            f"got {tuple(out_loc.shape)}"
        )
    if num_rows == 0:
        return
    if plan.is_decode and out_loc.shape[0] != num_rows:
        raise ValueError(
            "AITER FP4 C4Indexer decode requires out_loc length N; "
            f"got {out_loc.shape[0]} for N={num_rows}"
        )

    if write_metadata is None:
        positions, slots = prepare_aiter_k_indexer_fp4_cache_write_metadata(
            plan=plan,
            out_loc=out_loc,
            max_position=cos.shape[0],
            device=k.device,
        )
    else:
        positions, slots = write_metadata
        for name, tensor in (("positions", positions), ("slots", slots)):
            if (
                tensor.shape != (num_rows,)
                or tensor.dtype != torch.int64
                or tensor.device != k.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "AITER FP4 C4Indexer requires precomputed "
                    f"{name} shape [N], int64 dtype, and contiguous K device "
                    f"storage; got shape {tuple(tensor.shape)}, dtype "
                    f"{tensor.dtype}, device {tensor.device}, contiguous "
                    f"{tensor.is_contiguous()}"
                )
    k_bf16 = k.view(num_rows, 1, _Q_HEAD_DIM)

    # AITER is intentionally imported only after the HIP FP4 path is selected.
    import aiter

    aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache(
        k_payload,
        k_scale,
        k_bf16,
        norm_weight,
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
