# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Aiter MegaMoEV2 adapter for the SGLang MegaMoE hooks."""

from __future__ import annotations

import functools
import inspect
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.moe.fill_padded_rows import _fill_padded_rows
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_is_extend_in_batch,
)
from sglang.srt.layers.moe.topk import (
    megamoe_routing,
    megamoe_topk_masks_padded_rows,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.models.deepseek_common.amd.deepseek_v2_hip_moe import (
    batch_has_images,
    forward_gate,
)
from sglang.srt.multimodal.dsv41.vl_routing import vision_topk
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

logger = logging.getLogger(__name__)

_MORI_SHMEM_READY = False
_MEGA_MOE_INSTANCE: dict = {}
# Large-input protocol slots: one per layer_id; MTP layers take the margin.
_MEGA_LAYER_SLOT_MARGIN = 8
# Workaround for ROCm/aiter#7788: EP4 DeepSeek-V4.1-Flash large-input stage1 faults
# (illegal address) at MTPR 8192 and 16384; re-enable once that kernel is fixed.
_AITER_LARGE_INPUT_ROUTE = False
# (model_dim, inter_dim, experts) of DeepSeek-V4.1-Flash.
_FLASH_SHAPE = (5120, 2304, 384)


@functools.cache
def _import_aiter_megamoe():
    try:
        from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
    except ImportError as exc:
        raise ImportError(
            "Aiter MegaMoE requires a package containing MegaMoEV2 and the "
            f"A16W4 shuffle operators. Original import error: {exc}"
        ) from exc

    return SimpleNamespace(
        op=MegaMoEV2,
        shuffle_weight=shuffle_weight_a16w4,
        shuffle_scale=shuffle_scale_a16w4,
    )


def _mtpr() -> int:
    mtpr = int(envs.SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR.get())
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(
            f"SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR={mtpr} must be a positive power of two"
        )
    return mtpr


def _decode_mtpr() -> int:
    decode_mtpr = int(envs.SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR.get())
    if decode_mtpr == 0:
        return 0
    mtpr = _mtpr()
    if decode_mtpr < 0 or decode_mtpr > mtpr or decode_mtpr & (decode_mtpr - 1):
        raise ValueError(
            f"SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR={decode_mtpr} must be 0 (off) "
            "or a power of two no larger than "
            f"SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR={mtpr}"
        )
    if decode_mtpr == mtpr:
        _warn_decode_mtpr_merged(mtpr)
        return 0
    return decode_mtpr


@functools.cache
def _warn_decode_mtpr_merged(mtpr: int) -> None:
    logger.warning(
        "SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR equals "
        "SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=%d; running one MegaMoE capacity",
        mtpr,
    )


def _capacities() -> tuple[int, ...]:
    decode_mtpr = _decode_mtpr()
    return (decode_mtpr, _mtpr()) if decode_mtpr else (_mtpr(),)


def _is_decode_step_mode(forward_batch: ForwardBatch | None) -> bool:
    # IDLE also runs in prefill steps; _select_mtpr tells the two apart.
    if forward_batch is None:
        return False
    mode = forward_batch.forward_mode
    return (
        mode.is_decode_or_idle() or mode.is_target_verify() or mode.is_draft_extend_v2()
    )


def _select_mtpr(forward_batch: ForwardBatch | None) -> int:
    """The capacity this forward runs with; every EP rank must pick the same one.

    Local token counts, graph buckets and graph-vs-eager can differ across
    ranks, so only the step kind decides.
    """
    mtpr = _mtpr()
    decode_mtpr = _decode_mtpr()
    if not decode_mtpr or not _is_decode_step_mode(forward_batch):
        return mtpr
    if get_is_capture_mode():
        # Decode-step graphs replay only when no DP rank extends
        # (can_run_decode_cuda_graph is min-reduced over the group).
        return decode_mtpr
    # is_extend_in_batch is max-reduced over the DP group.
    return mtpr if get_is_extend_in_batch() else decode_mtpr


@functools.cache
def _forward_params(op_cls) -> frozenset[str]:
    return frozenset(inspect.signature(op_cls.forward).parameters)


def _routes_per_call(op_cls) -> bool:
    """Whether this MegaMoEv2 picks a staged, small- or large-input route per call."""
    return "max_tokens" in _forward_params(op_cls)


def _row_bound_kwargs(mega, shape, run_tokens: int, mtpr: int) -> dict[str, int]:
    """The source-row count of the widest EP rank, as MegaMoEv2 takes it.

    Every rank must plan a call alike, idle ranks with zero rows included, or
    the collective dispatch hangs. aiter takes the bound as ``max_tokens``,
    which picks the route (small-input, large-input or staged) for every
    shape, or in older builds as ``config_tokens``, which picks the
    DeepSeek-V4.1-Flash tuning bucket. Builds with neither, and prefill CP,
    whose DP counts are not per-rank rows, keep aiter's per-call default.
    ``max_tokens`` is the gathered DP max alone: a rank whose own rows exceed it
    makes aiter raise rather than plan a route its peers do not. A graph capture
    passes it only for the small-input route; the large-input route is never
    captured.
    """
    params = _forward_params(type(mega))
    if "max_tokens" in params:
        name = "max_tokens"
    elif "config_tokens" in params and shape == _FLASH_SHAPE:
        name = "config_tokens"
    else:
        return {}
    if is_dsa_enable_prefill_cp():
        return {}
    # The DP counts include graph padding; without them (attention TP) every
    # rank runs the same rows.
    global_num_tokens = get_dp_global_num_tokens()
    tokens = max(global_num_tokens) if global_num_tokens else run_tokens
    if name == "max_tokens":
        rows = min(tokens, mtpr)
        if rows <= 0 or (
            torch.cuda.is_current_stream_capturing() and rows > mega.small_m_mtpr
        ):
            return {}
        return {name: rows}
    return {name: min(max(tokens, run_tokens, 1), mtpr)}


def _select_layer_slot(mega, layer_id: int) -> None:
    """Point a shared instance's large-input route at this layer's protocol slot."""
    if mega.large_m_available:
        mega.select_layer_state(layer_id)


def _takes_fused_route(mega, forward_kwargs: dict[str, int]) -> bool:
    """Whether aiter routes this call to a small- or large-input kernel."""
    max_tokens = forward_kwargs.get("max_tokens")
    if max_tokens is None:
        return False
    # route_for: small capacities up to small_m_mtpr, the large route from 513 rows.
    return max_tokens <= mega.small_m_mtpr or mega.large_m_available


def _ep_rank_world():
    parallel = get_parallel()
    return parallel.moe_ep_rank, parallel.moe_ep_size


def _ensure_mori_shmem() -> None:
    global _MORI_SHMEM_READY
    if _MORI_SHMEM_READY:
        return

    import mori.shmem

    group_name = "megamoe_aiter"
    cpu_group = get_parallel().moe_ep_group.cpu_group
    try:
        torch._C._distributed_c10d._register_process_group(group_name, cpu_group)
    except Exception as exc:
        if "already registered" not in str(exc):
            raise
    mori.shmem.shmem_torch_process_group_init(group_name)
    _MORI_SHMEM_READY = True


def build_mega_moe_experts_weights(layer) -> None:
    if getattr(layer, "_mega_moe_weights_built", False):
        return

    backend = _import_aiter_megamoe()

    def scale_param(name):
        scale = getattr(layer, name, None)
        if scale is None:
            scale = getattr(layer, name + "_inv", None)
        if scale is None:
            raise AttributeError(f"Aiter MegaMoE weight scale {name} is missing")
        return scale

    def shuffle_existing(weight, scale, *, gate_up):
        experts = weight.shape[0]
        shuffled_weight = backend.shuffle_weight(weight, 16, gate_up).contiguous()
        shuffled_scale = backend.shuffle_scale(
            scale.view(-1, scale.shape[-1]), experts, gate_up
        ).contiguous()
        return (
            shuffled_weight.view(torch.uint8).contiguous(),
            shuffled_scale.view(torch.uint8).contiguous(),
        )

    w13_scale = scale_param("w13_weight_scale")
    w2_scale = scale_param("w2_weight_scale")
    layer._mega_w1, layer._mega_w1_scale = shuffle_existing(
        layer.w13_weight.data, w13_scale.data, gate_up=True
    )
    layer._mega_w2, layer._mega_w2_scale = shuffle_existing(
        layer.w2_weight.data, w2_scale.data, gate_up=False
    )

    experts = layer.w13_weight.shape[0]
    layer.w13_weight.data = layer._mega_w1.view(experts, -1)
    w13_scale.data = layer._mega_w1_scale.view(experts, -1)
    layer.w2_weight.data = layer._mega_w2.view(experts, -1)
    w2_scale.data = layer._mega_w2_scale.view(experts, -1)
    layer._mega_moe_weights_built = True


def _get_or_build_mega_moe(
    layer,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    quant: str,
    swiglu_limit: float,
    mtpr: int,
    num_layers: int,
):
    rank, world = _ep_rank_world()
    shape_key = (rank, world, model_dim, inter_dim, experts, topk, quant, swiglu_limit)
    mega = _MEGA_MOE_INSTANCE.get((*shape_key, mtpr))
    if mega is not None:
        return mega
    _ensure_mori_shmem()
    backend = _import_aiter_megamoe()
    extra = {}
    if _routes_per_call(backend.op):
        if _AITER_LARGE_INPUT_ROUTE:
            # Slots cost memory only on a capacity that builds the large-input route.
            extra["large_m_state_slots"] = num_layers + _MEGA_LAYER_SLOT_MARGIN
        else:
            extra["large_m_enabled"] = False
    # Construction synchronizes the device and barriers every EP rank, so all
    # capacities are built together on the first (eager) call, in the same order
    # on every rank: a capacity first used later could be inside graph capture.
    for capacity in _capacities():
        key = (*shape_key, capacity)
        if key in _MEGA_MOE_INSTANCE:
            continue
        _MEGA_MOE_INSTANCE[key] = backend.op(
            rank=rank,
            world_size=world,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant=quant,
            w1=layer._mega_w1,
            w1_scale=layer._mega_w1_scale,
            w2=layer._mega_w2,
            w2_scale=layer._mega_w2_scale,
            max_tok_per_rank=capacity,
            swiglu_limit=swiglu_limit,
            **extra,
        )
        # Load every route's kernels now, on every rank together: a first-use load
        # mid-serving stalls the peers already waiting inside a MegaMoE kernel.
        _MEGA_MOE_INSTANCE[key].preload_aot_bundles()
        logger.info("Aiter MegaMoE: built capacity max_tok_per_rank=%d", capacity)
    return _MEGA_MOE_INSTANCE[(*shape_key, mtpr)]


def _swap_layer_weights(mega, layer) -> None:
    mega._s1_w1 = layer._mega_w1
    mega._s1_w1_scale = layer._mega_w1_scale
    mega.w2 = layer._mega_w2
    mega.w2_scale = layer._mega_w2_scale


def should_use_mega_moe(moe: DeepseekV2MoE, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if get_is_capture_mode():
        return True
    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens and not is_dsa_enable_prefill_cp():
        max_tokens = max(global_num_tokens)
    else:
        max_tokens = hidden_states.shape[0]
    return max_tokens <= _mtpr()


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch | None = None,
    input_ids_global: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    mega, _, forward_kwargs = _mega_call_plan(moe, num_tokens, forward_batch)
    overlap = _should_overlap_shared_and_routed(
        moe, num_tokens, fused_route=_takes_fused_route(mega, forward_kwargs)
    )
    if overlap:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        stream_context = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        stream_context = nullcontext()

    with stream_context:
        output = _run_mega_routed(moe, hidden_states, forward_batch, input_ids_global)
    if overlap:
        current_stream.wait_stream(moe.alt_stream)
    if shared_output is not None:
        output.add_(shared_output)
    return output


def _should_overlap_shared_and_routed(
    moe: DeepseekV2MoE,
    num_tokens: int,
    *,
    fused_route: bool,
) -> bool:
    if envs.SGLANG_AITER_MEGA_RANK_SYNC.get():
        return False
    # The fused small-/large-input kernels keep every CTA resident and wait on the
    # peer ranks; a shared-expert kernel on alt_stream next to them deadlocks.
    if fused_route:
        return False
    return (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )


def _mega_call_plan(moe: DeepseekV2MoE, num_rows: int, forward_batch):
    """This call's instance, its MTPR and the route-bound kwargs of its forward."""
    hidden_size = moe.config.hidden_size
    inter_dim = moe.config.moe_intermediate_size
    experts = moe.experts.num_experts
    selected_mtpr = _select_mtpr(forward_batch)
    mega = _get_or_build_mega_moe(
        moe.experts,
        model_dim=hidden_size,
        inter_dim=inter_dim,
        experts=experts,
        topk=moe.config.num_experts_per_tok + moe.num_fused_shared_experts,
        quant=envs.SGLANG_AMD_FLYDSL_MEGA_QUANT.get() or "a8w4",
        swiglu_limit=float(getattr(moe.config, "swiglu_limit", 0.0) or 0.0),
        mtpr=selected_mtpr,
        num_layers=moe.config.num_hidden_layers,
    )
    forward_kwargs = _row_bound_kwargs(
        mega, (hidden_size, inter_dim, experts), num_rows, selected_mtpr
    )
    return mega, selected_mtpr, forward_kwargs


def _run_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch | None,
    input_ids_global: torch.Tensor | None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    topk = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    experts = moe.experts.num_experts
    if num_tokens:
        use_vision_topk = (
            not moe.is_hash
            and getattr(moe.gate, "e_score_correction_bias_vl", None) is not None
            and batch_has_images(forward_batch)
        )
        router_logits, router_logits_partials = forward_gate(
            moe, hidden_states, None, fused_gate=not use_vision_topk
        )
        if use_vision_topk:
            topk_output = vision_topk(moe, router_logits, input_ids_global)
            topk_masked_padded_rows = False
        else:
            topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
            if router_logits_partials is not None:
                topk_kwargs["router_logits_partials"] = router_logits_partials
            with megamoe_routing():
                topk_output = moe.topk(hidden_states, router_logits, **topk_kwargs)
                topk_masked_padded_rows = (
                    not moe.is_hash
                    and megamoe_topk_masks_padded_rows(moe.topk.topk_config)
                )
        x_in = hidden_states
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights.to(torch.float32)
        num_token_non_padded, _ = moe.topk.routing_inputs()
        if num_token_non_padded is not None and not topk_masked_padded_rows:
            _fill_padded_rows(
                topk_ids,
                num_token_non_padded,
                experts,
            )
    else:
        # An idle rank still enters the collective dispatch; MegaMoEv2 takes
        # zero source rows.
        x_in = hidden_states
        topk_ids = hidden_states.new_empty((0, topk), dtype=torch.int32)
        topk_weights = hidden_states.new_empty((0, topk), dtype=torch.float32)

    x_in = x_in.contiguous()
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    mega, selected_mtpr, forward_kwargs = _mega_call_plan(
        moe, x_in.shape[0], forward_batch
    )
    assert x_in.shape[0] <= selected_mtpr, (
        f"Aiter MegaMoE local tokens {x_in.shape[0]} exceed MTPR {selected_mtpr}"
    )
    _swap_layer_weights(mega, moe.experts)
    _select_layer_slot(mega, moe.layer_id)
    output = mega.forward(
        x_in, topk_weights, topk_ids, slice_output=False, **forward_kwargs
    )[:num_tokens]

    if not moe.routed_scaling_factor_fused_in_topk:
        output.mul_(moe.routed_scaling_factor)
    return output
