import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.mega_moe_flydsl import _row_bound_kwargs
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.managers.scheduler_components.dp_attn import _update_gather_batch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _TakesMaxTokens:
    small_m_mtpr = 512
    large_m_available = False

    def forward(self, x, wts, ids, *, slice_output=True, max_tokens=None):
        pass


class _TakesConfigTokens:
    def forward(self, x, wts, ids, *, slice_output=True, config_tokens=None):
        pass


class _NoRowBound:
    def forward(self, x, wts, ids, *, slice_output=True):
        pass


_FLASH = (5120, 2304, 384)
_PRO = (7168, 3072, 384)
_MEGA = "sglang.srt.layers.moe.mega_moe_flydsl"


@pytest.fixture
def no_prefill_cp(monkeypatch):
    monkeypatch.setattr(f"{_MEGA}.is_dsa_enable_prefill_cp", lambda: False)


@pytest.mark.parametrize("shape", [_FLASH, _PRO])
def test_ranks_pass_the_widest_source_rank_as_max_tokens(
    monkeypatch, no_prefill_cp, shape
):
    path = f"{_MEGA}.get_dp_global_num_tokens"
    # Busy ranks, an idle rank with zero rows, and the MTPR clamp.
    monkeypatch.setattr(path, lambda: [16384, 10542, 0, 16384])
    for run_tokens in (16384, 10542, 0):
        assert _row_bound_kwargs(_TakesMaxTokens(), shape, run_tokens, 16384) == {
            "max_tokens": 16384
        }
    assert _row_bound_kwargs(_TakesMaxTokens(), shape, 0, 8192) == {"max_tokens": 8192}
    # Without synced DP counts every rank runs the same rows.
    monkeypatch.setattr(path, lambda: None)
    assert _row_bound_kwargs(_TakesMaxTokens(), shape, 64, 512) == {"max_tokens": 64}


def test_older_aiter_passes_config_tokens_for_flash_only(monkeypatch, no_prefill_cp):
    monkeypatch.setattr(f"{_MEGA}.get_dp_global_num_tokens", lambda: [300, 0])
    assert _row_bound_kwargs(_TakesConfigTokens(), _FLASH, 0, 512) == {
        "config_tokens": 300
    }
    assert _row_bound_kwargs(_TakesConfigTokens(), _PRO, 0, 512) == {}
    assert _row_bound_kwargs(_NoRowBound(), _FLASH, 0, 512) == {}


def test_prefill_cp_keeps_the_per_call_default(monkeypatch):
    monkeypatch.setattr(f"{_MEGA}.is_dsa_enable_prefill_cp", lambda: True)
    monkeypatch.setattr(f"{_MEGA}.get_dp_global_num_tokens", lambda: [300, 0])
    assert _row_bound_kwargs(_TakesMaxTokens(), _FLASH, 0, 512) == {}


def test_idle_flash_rank_enters_with_the_widest_rank_bound(monkeypatch, no_prefill_cp):
    import sglang.srt.layers.moe.mega_moe_flydsl as mega

    seen = {}

    class Op:
        small_m_mtpr = 512
        large_m_available = False

        def forward(self, x, wts, ids, *, slice_output=True, max_tokens=None):
            seen.update(rows=x.shape[0], max_tokens=max_tokens)
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
            num_hidden_layers=43,
        ),
        num_fused_shared_experts=0,
        experts=SimpleNamespace(num_experts=_FLASH[2]),
        routed_scaling_factor=1.0,
        routed_scaling_factor_fused_in_topk=True,
        layer_id=0,
    )
    mega._run_mega_routed(moe, torch.zeros(0, _FLASH[0]), None, None)
    assert seen == {"rows": 0, "max_tokens": 16384}


def test_graph_capture_passes_max_tokens_only_for_the_small_input_route(
    monkeypatch, no_prefill_cp
):
    monkeypatch.setattr(f"{_MEGA}.get_dp_global_num_tokens", lambda: [600, 0])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _row_bound_kwargs(_TakesMaxTokens(), _FLASH, 0, 8192) == {}
    monkeypatch.setattr(f"{_MEGA}.get_dp_global_num_tokens", lambda: [512, 0])
    assert _row_bound_kwargs(_TakesMaxTokens(), _FLASH, 0, 8192) == {"max_tokens": 512}


def test_all_idle_ranks_keep_the_staged_route(monkeypatch, no_prefill_cp):
    monkeypatch.setattr(f"{_MEGA}.get_dp_global_num_tokens", lambda: [0, 0])
    assert _row_bound_kwargs(_TakesMaxTokens(), _FLASH, 0, 8192) == {}


def test_each_layer_selects_its_layer_id_as_large_input_slot():
    import sglang.srt.layers.moe.mega_moe_flydsl as mega

    selected = []
    op = SimpleNamespace(large_m_available=True, select_layer_state=selected.append)
    for layer_id in (3, 7, 3):
        mega._select_layer_slot(op, layer_id)
    assert selected == [3, 7, 3]
    staged = SimpleNamespace(
        large_m_available=False,
        select_layer_state=lambda slot: pytest.fail("no large-input route"),
    )
    mega._select_layer_slot(staged, 7)


def test_max_tokens_is_the_dp_max_even_when_local_rows_exceed_it(
    monkeypatch, no_prefill_cp
):
    monkeypatch.setattr(f"{_MEGA}.get_dp_global_num_tokens", lambda: [300, 200])
    # Every rank passes the same bound; aiter refuses the inconsistent rank.
    assert _row_bound_kwargs(_TakesMaxTokens(), _FLASH, 400, 8192) == {
        "max_tokens": 300
    }


@pytest.mark.parametrize(
    "kwargs, small, large, fused",
    [
        ({}, 512, True, False),
        ({"max_tokens": 512}, 512, False, True),
        ({"max_tokens": 600}, 512, False, False),
        ({"max_tokens": 600}, 512, True, True),
    ],
)
def test_fused_route_detection(kwargs, small, large, fused):
    import sglang.srt.layers.moe.mega_moe_flydsl as mega

    op = SimpleNamespace(small_m_mtpr=small, large_m_available=large)
    assert mega._takes_fused_route(op, kwargs) is fused


@pytest.mark.parametrize(
    "op, routed, large",
    [
        (_TakesMaxTokens, True, True),
        (_TakesMaxTokens, True, False),
        (_NoRowBound, False, False),
    ],
)
def test_routed_aiter_builds_layer_slots_and_preloads(monkeypatch, op, routed, large):
    import sglang.srt.layers.moe.mega_moe_flydsl as mega

    monkeypatch.setattr(mega, "_AITER_LARGE_INPUT_ROUTE", large)

    built, preloaded = [], []

    class Op(op):
        def __init__(self, **kwargs):
            built.append(kwargs)

        def preload_aot_bundles(self):
            preloaded.append(self)

    backend = SimpleNamespace(op=Op)
    monkeypatch.setattr(mega, "_MEGA_MOE_INSTANCE", {})
    monkeypatch.setattr(mega, "_import_aiter_megamoe", lambda: backend)
    monkeypatch.setattr(mega, "_ensure_mori_shmem", lambda: None)
    monkeypatch.setattr(mega, "_ep_rank_world", lambda: (0, 4))
    monkeypatch.setattr(mega, "_capacities", lambda: (8192,))
    monkeypatch.setattr(mega, "get_is_capture_mode", lambda: True)
    layer = SimpleNamespace(
        _mega_w1=None, _mega_w1_scale=None, _mega_w2=None, _mega_w2_scale=None
    )
    mega._get_or_build_mega_moe(
        layer,
        model_dim=_FLASH[0],
        inter_dim=_FLASH[1],
        experts=_FLASH[2],
        topk=6,
        quant="a8w4",
        swiglu_limit=0.0,
        mtpr=8192,
        num_layers=43,
    )
    assert built[0].get("large_m_state_slots") == (43 + 8 if routed and large else None)
    # With the large-input route off, a routed build leaves it out explicitly.
    assert built[0].get("large_m_enabled") == (False if routed and not large else None)
    # Every built instance loads all its routes' kernels before serving.
    assert len(preloaded) == len(built) == 1
    moe = SimpleNamespace(alt_stream=object(), num_fused_shared_experts=0)
    monkeypatch.setattr(mega.envs.SGLANG_AITER_MEGA_RANK_SYNC, "get", lambda: False)
    for fused in (True, False):
        assert (
            mega._should_overlap_shared_and_routed(moe, 32, fused_route=fused)
            is not fused
        )


@pytest.mark.parametrize(
    "a2a_backend, in_mega, recorder_mode, remap, gate_masked, padded",
    [
        # MegaMoE drops the out-of-range id 8; a gate that wrote it leaves nothing.
        (MoeA2ABackend.MEGAMOE, True, None, False, False, "dropped"),
        (MoeA2ABackend.MEGAMOE, True, None, False, True, "kept"),
        # A remap table, the recorder or another consumer needs in-range ids.
        (MoeA2ABackend.MEGAMOE, True, "stat", False, False, "zeroed"),
        (MoeA2ABackend.MEGAMOE, True, None, True, False, "zeroed"),
        (MoeA2ABackend.NONE, True, None, False, False, "zeroed"),
        # A megamoe-backend layer falling back to the fused MoE.
        (MoeA2ABackend.MEGAMOE, False, None, False, False, "zeroed"),
    ],
)
def test_hip_megamoe_padded_rows_leave_topk_with_a_dropped_id(
    monkeypatch, a2a_backend, in_mega, recorder_mode, remap, gate_masked, padded
):
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
        with (
            get_flags().moe.override(a2a_backend=a2a_backend),
            topk_module.megamoe_routing() if in_mega else nullcontext(),
        ):
            out_ids, out_weights, _ = topk_module._post_process_topk_ids(
                ids.clone(),
                weights.clone(),
                topk_config=TopKConfig(top_k=2),
                router_logits=torch.zeros(3, 8),
                layer_id=0,
                num_token_non_padded=torch.tensor(1),
                padded_rows_masked=gate_masked,
            )
    finally:
        config.restore()
    assert out_ids[:1].tolist() == ids[:1].tolist()
    torch.testing.assert_close(out_weights[:1], weights[:1])
    if padded == "zeroed":
        assert out_ids[1:].tolist() == [[0, 0], [0, 0]]
        assert not out_weights[1:].any()
        return
    expected_ids = [[8, 8], [8, 8]] if padded == "dropped" else ids[1:].tolist()
    assert out_ids[1:].tolist() == expected_ids
    torch.testing.assert_close(out_weights[1:], weights[1:])


@pytest.mark.parametrize(
    "is_hash, topk_masks, filled",
    [(False, True, False), (False, False, True), (True, True, True)],
)
def test_mega_routed_takes_the_decode_router_and_fills_only_unmasked_rows(
    monkeypatch, no_prefill_cp, is_hash, topk_masks, filled
):
    import sglang.srt.layers.moe.mega_moe_flydsl as mega

    partials = torch.zeros(2, 4, _FLASH[2])
    seen = {"fill": None}

    def forward_gate(moe, hidden_states, allocator, *, fused_gate):
        seen["fused_gate"] = fused_gate
        return torch.zeros(hidden_states.shape[0], _FLASH[2]), partials

    class Router:
        topk_config = TopKConfig(top_k=6)

        def __call__(self, hidden_states, router_logits, **kwargs):
            seen["topk_kwargs"] = kwargs
            seen["in_mega"] = topk_module._megamoe_routing_active
            return SimpleNamespace(
                topk_ids=torch.zeros(4, 6, dtype=torch.int32),
                topk_weights=torch.ones(4, 6),
            )

        def routing_inputs(self):
            return torch.tensor(2), None

    class Op:
        small_m_mtpr = 512
        large_m_available = False

        def forward(self, x, wts, ids, *, slice_output=True, max_tokens=None):
            return x

    monkeypatch.setattr(mega, "forward_gate", forward_gate)
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
            num_hidden_layers=43,
        ),
        num_fused_shared_experts=0,
        experts=SimpleNamespace(num_experts=_FLASH[2]),
        routed_scaling_factor=1.0,
        routed_scaling_factor_fused_in_topk=True,
        layer_id=0,
        is_hash=is_hash,
        gate=SimpleNamespace(),
        topk=Router(),
    )
    mega._run_mega_routed(moe, torch.zeros(4, _FLASH[0]), None, "ids")
    assert seen["fused_gate"]
    assert seen["in_mega"] and not topk_module._megamoe_routing_active
    assert seen["topk_kwargs"]["router_logits_partials"] is partials
    assert seen["fill"] == (_FLASH[2] if filled else None)


@pytest.mark.parametrize("hip, expected", [(True, True), (False, False)])
def test_hip_megamoe_syncs_every_dp_count_under_dp_lm_head(monkeypatch, hip, expected):
    """Without the full DP vector, each aiter MegaMoE rank would pass its own row
    count as max_tokens and ranks would plan different routes."""
    from sglang.srt.layers import dp_attention

    parallel = SimpleNamespace(
        attn_dp_enabled=True,
        num_dp_ranks=4,
        moe_dense_tp_size=1,
        enable_dp_lm_head=True,
        tp_size=4,
    )
    monkeypatch.setattr(dp_attention, "_is_hip", hip)
    monkeypatch.setattr(dp_attention, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        dp_attention,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=None)),
    )
    assert (
        dp_attention.require_mlp_tp_gather(moe_a2a_backend=MoeA2ABackend.MEGAMOE)
        is expected
    )
    assert not dp_attention.require_mlp_tp_gather(moe_a2a_backend=MoeA2ABackend.DEEPEP)


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
