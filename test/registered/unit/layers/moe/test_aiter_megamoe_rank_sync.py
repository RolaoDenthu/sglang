import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.mega_moe_flydsl import (
    _config_tokens,
    _forward_accepts_config_tokens,
    _should_overlap_shared_and_routed,
)
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.managers.scheduler_components.dp_attn import (
    ForwardMode,
    _update_gather_batch,
)
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_rank_sync_forces_overlap_off(monkeypatch):
    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "1")
    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe_flydsl.get_is_capture_mode",
        lambda: True,
    )
    moe = SimpleNamespace(alt_stream=object(), num_fused_shared_experts=0)
    assert not _should_overlap_shared_and_routed(moe, 32)

    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "0")
    assert _should_overlap_shared_and_routed(moe, 32)


class _TakesConfigTokens:
    def forward(self, x, wts, ids, *, slice_output=True, config_tokens=None):
        pass


class _NoConfigTokens:
    def forward(self, x, wts, ids, *, slice_output=True):
        pass


_FLASH = (5120, 2304, 384)
_PRO = (7168, 3072, 384)


def test_flash_ranks_pick_the_config_of_the_widest_source_rank(monkeypatch):
    path = "sglang.srt.layers.moe.mega_moe_flydsl.get_dp_global_num_tokens"
    # Busy ranks, an idle rank with zero rows, and the MTPR clamp.
    monkeypatch.setattr(path, lambda: [16384, 10542, 0, 16384])
    for run_tokens in (16384, 10542, 0):
        assert _config_tokens(_TakesConfigTokens, _FLASH, run_tokens, 16384) == 16384
    assert _config_tokens(_TakesConfigTokens, _FLASH, 0, 8192) == 8192
    # Without synced DP counts every rank runs the same rows.
    monkeypatch.setattr(path, lambda: None)
    assert _config_tokens(_TakesConfigTokens, _FLASH, 64, 512) == 64


def test_other_shapes_and_older_aiter_keep_the_per_rank_config(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe_flydsl.get_dp_global_num_tokens",
        lambda: [300, 0],
    )
    assert _config_tokens(_TakesConfigTokens, _PRO, 0, 512) is None
    assert _config_tokens(_NoConfigTokens, _FLASH, 0, 512) is None
    assert _forward_accepts_config_tokens(_TakesConfigTokens)
    assert not _forward_accepts_config_tokens(_NoConfigTokens)


def test_idle_flash_rank_enters_with_the_widest_rank_config(monkeypatch):
    import sglang.srt.layers.moe.mega_moe_flydsl as mega

    seen = {}

    class Op:
        def forward(self, x, wts, ids, *, slice_output=True, config_tokens=None):
            seen.update(rows=x.shape[0], config_tokens=config_tokens)
            return x

    monkeypatch.setattr(mega, "get_dp_global_num_tokens", lambda: [16384, 0, 9000])
    monkeypatch.setattr(mega, "_select_mtpr", lambda forward_batch: 16384)
    monkeypatch.setattr(mega, "_get_or_build_mega_moe", lambda *a, **k: Op())
    monkeypatch.setattr(mega, "_swap_layer_weights", lambda op, layer: None)
    moe = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=_FLASH[0],
            num_experts_per_tok=6,
            moe_intermediate_size=_FLASH[1],
            swiglu_limit=0.0,
        ),
        num_fused_shared_experts=0,
        experts=SimpleNamespace(num_experts=_FLASH[2]),
        routed_scaling_factor=1.0,
        routed_scaling_factor_fused_in_topk=True,
    )
    mega._run_mega_routed(moe, torch.zeros(0, _FLASH[0]), None, None)
    assert seen == {"rows": 0, "config_tokens": 16384}


@pytest.mark.parametrize(
    "a2a_backend, flydsl, recorder_mode, remap, dropped",
    [
        # MegaMoE drops the out-of-range id 8 along with its weight.
        (MoeA2ABackend.MEGAMOE, True, None, False, True),
        # A remap table, the recorder or another consumer needs in-range ids.
        (MoeA2ABackend.MEGAMOE, True, "stat", False, False),
        (MoeA2ABackend.MEGAMOE, True, None, True, False),
        (MoeA2ABackend.MEGAMOE, False, None, False, False),
        (MoeA2ABackend.NONE, True, None, False, False),
    ],
)
def test_hip_megamoe_padded_rows_leave_topk_with_a_dropped_id(
    monkeypatch, a2a_backend, flydsl, recorder_mode, remap, dropped
):
    monkeypatch.setenv("SGLANG_AMD_USE_FLYDSL_MEGA_MOE", "1" if flydsl else "0")
    monkeypatch.setattr(topk_module, "_is_cuda", False)
    monkeypatch.setattr(topk_module, "_is_hip", True)
    monkeypatch.setattr(topk_module, "_skip_hip_pad_mask", False)
    monkeypatch.setattr(topk_module, "is_expert_remapping_enabled", lambda: remap)
    monkeypatch.setattr(topk_module, "topk_ids_logical_to_physical", lambda ids, _: ids)
    config = get_context().override_server_args(
        expert_distribution_recorder_mode=recorder_mode
    )
    config.install()
    ids = torch.tensor([[3, 1], [2, 5], [7, 6]], dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
    try:
        with get_flags().moe.override(a2a_backend=a2a_backend):
            out_ids, out_weights, _ = topk_module._post_process_topk_ids(
                ids.clone(),
                weights.clone(),
                topk_config=TopKConfig(top_k=2),
                router_logits=torch.zeros(3, 8),
                layer_id=0,
                num_token_non_padded=torch.tensor(1),
            )
    finally:
        config.restore()
    assert out_ids[:1].tolist() == ids[:1].tolist()
    torch.testing.assert_close(out_weights[:1], weights[:1])
    if dropped:
        assert out_ids[1:].tolist() == [[8, 8], [8, 8]]
        torch.testing.assert_close(out_weights[1:], weights[1:])
    else:
        assert out_ids[1:].tolist() == [[0, 0], [0, 0]]
        assert not out_weights[1:].any()


@pytest.mark.parametrize(
    "is_hash, topk_masks, filled",
    [(False, True, False), (False, False, True), (True, True, True)],
)
def test_mega_routed_fills_only_the_rows_topk_left_unmasked(
    monkeypatch, is_hash, topk_masks, filled
):
    import sglang.srt.layers.moe.mega_moe_flydsl as mega

    seen = {"fill": None}

    class Router:
        topk_config = TopKConfig(top_k=6)

        def __call__(self, hidden_states, router_logits, **kwargs):
            seen["topk_kwargs"] = kwargs
            return SimpleNamespace(
                topk_ids=torch.zeros(4, 6, dtype=torch.int32),
                topk_weights=torch.ones(4, 6),
            )

        def routing_inputs(self):
            return torch.tensor(2), None

    class Op:
        def forward(self, x, wts, ids, *, slice_output=True, config_tokens=None):
            return x

    monkeypatch.setattr(mega, "megamoe_topk_masks_padded_rows", lambda c: topk_masks)
    monkeypatch.setattr(
        mega, "_fill_padded_rows", lambda ids, n, value: seen.update(fill=value)
    )
    monkeypatch.setattr(mega, "get_dp_global_num_tokens", lambda: [4, 0])
    monkeypatch.setattr(mega, "_select_mtpr", lambda forward_batch: 512)
    monkeypatch.setattr(mega, "_get_or_build_mega_moe", lambda *a, **k: Op())
    monkeypatch.setattr(mega, "_swap_layer_weights", lambda op, layer: None)
    moe = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=_FLASH[0],
            num_experts_per_tok=6,
            moe_intermediate_size=_FLASH[1],
            swiglu_limit=0.0,
        ),
        num_fused_shared_experts=0,
        experts=SimpleNamespace(num_experts=_FLASH[2]),
        routed_scaling_factor=1.0,
        routed_scaling_factor_fused_in_topk=True,
        is_hash=is_hash,
        gate=lambda hidden_states, **kwargs: torch.zeros(
            hidden_states.shape[0], _FLASH[2]
        ),
        topk=Router(),
    )
    mega._run_mega_routed(moe, torch.zeros(4, _FLASH[0]), None, "ids")
    assert seen["fill"] == (_FLASH[2] if filled else None)


def test_mlp_sync_retains_full_vector_without_tp_gather():
    batch = SimpleNamespace()
    sync = SimpleNamespace(
        global_num_tokens=[32, 31, 7659, 256],
        global_num_tokens_for_logprob=[32, 31, 1, 1],
        num_tokens=32,
        num_tokens_for_logprob=32,
        is_extend_in_batch=True,
        tbo_split_seq_index=None,
        global_forward_mode=None,
        can_run_decode_cuda_graph=False,
        can_run_draft_cuda_graph=False,
        can_run_prefill_cuda_graph=False,
        prefill_cuda_graph_max_prefix_len=0,
        tp0_info_cpu=torch.tensor(
            [
                [32, 0, 0, 0, 0, ForwardMode.DECODE.value, 0],
                [31, 0, 0, 0, 0, ForwardMode.DECODE.value, 0],
                [7659, 0, 0, 1, 0, ForwardMode.EXTEND.value, 0],
                [256, 0, 0, 1, 0, ForwardMode.EXTEND.value, 0],
            ],
            dtype=torch.int64,
        ),
    )
    _update_gather_batch(batch, sync, require_mlp_tp_gather=False)
    assert batch.global_num_tokens == [32]
    assert batch.is_extend_in_batch


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
