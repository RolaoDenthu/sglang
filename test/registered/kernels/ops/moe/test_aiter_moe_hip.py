"""The ROCm decode router gate must reproduce aiter's topk_gating bit for bit."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-kernel-test-1-gpu-amd-mi35x")


try:
    from aiter import topk_gating as aiter_topk_gating
except ImportError:  # aiter is absent off ROCm
    aiter_topk_gating = None


NUM_EXPERTS = 384
HIDDEN = 5120
TOPK = 6
ROUTED_SCALING = 1.5


def _aiter_gate(logits, bias, topk, renorm, rsf):
    weights = torch.empty(
        logits.shape[0], topk, dtype=torch.float32, device=logits.device
    )
    ids = torch.empty(logits.shape[0], topk, dtype=torch.int32, device=logits.device)
    aiter_topk_gating(
        weights, ids, logits, bias, renorm, rsf, score_func="sqrtsoftplus"
    )
    return weights, ids


@unittest.skipUnless(
    is_hip() and is_gfx95_supported() and aiter_topk_gating is not None,
    "ROCm gfx95 + aiter only",
)
class TestRocmRouterGate(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.moe.rocm_router_gate import (
            ROCM_ROUTER_MAX_TOKENS,
            rocm_router_gate,
            rocm_router_gemv_split_k,
            rocm_router_max_tokens,
            rocm_router_reduce_partials,
        )

        self.max_tokens = ROCM_ROUTER_MAX_TOKENS
        self.gate = rocm_router_gate
        self.gemv = rocm_router_gemv_split_k
        self.max_tokens_for = rocm_router_max_tokens
        self.reduce = rocm_router_reduce_partials
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device=self.device).manual_seed(0)
        self.bias_bf16 = (
            torch.randn(NUM_EXPERTS, device=self.device, generator=self.gen) * 0.5
        ).to(torch.bfloat16)

    def _randn(self, *shape, scale=1.0):
        return torch.randn(*shape, device=self.device, generator=self.gen) * scale

    def _assert_same_gate(self, logits, bias, renorm=True, rsf=ROUTED_SCALING, msg=""):
        for topk in (TOPK,):
            ref_w, ref_i = _aiter_gate(logits, bias, topk, renorm, rsf)
            out_w, out_i = self.gate(logits, bias, topk, renorm, rsf)
            self.assertTrue(torch.equal(ref_i, out_i), f"ids {msg} topk {topk}")
            self.assertTrue(torch.equal(ref_w, out_w), f"weights {msg} topk {topk}")

    def test_gate_matches_aiter_on_ties(self):
        zero_bias = torch.zeros(NUM_EXPERTS, device=self.device, dtype=torch.bfloat16)
        for num_tokens in (512,):
            levels = torch.randint(
                0, 8, (num_tokens, NUM_EXPERTS), device=self.device, generator=self.gen
            ).float()
            self._assert_same_gate(levels - 3, self.bias_bf16, msg="8 levels")
            self._assert_same_gate(
                torch.zeros(num_tokens, NUM_EXPERTS, device=self.device),
                zero_bias,
                msg="all equal",
            )
            few = torch.full(
                (num_tokens, NUM_EXPERTS), float("-inf"), device=self.device
            )
            few[:, :3] = 1.0
            self._assert_same_gate(few, zero_bias, msg="3 finite experts")

    def test_gate_matches_aiter_non_finite(self):
        for num_tokens in (16,):
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[logits < 0] = float("-inf")
            self._assert_same_gate(logits, self.bias_bf16, msg="-inf")
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[:, ::7] = float("nan")
            self._assert_same_gate(logits, self.bias_bf16, msg="nan")
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[:, 5] = float("inf")
            self._assert_same_gate(logits, self.bias_bf16, msg="+inf")

    def test_gemv_accuracy_batch_invariance_and_repeatability(self):
        weight = (self._randn(NUM_EXPERTS, HIDDEN) * 0.02).to(torch.bfloat16)
        x = self._randn(self.max_tokens, HIDDEN).to(torch.bfloat16)
        ref = (x.double() @ weight.double().T).float()
        full = torch.empty(self.max_tokens, NUM_EXPERTS, device=self.device)
        self.reduce(self.gemv(x, weight), full)
        self.assertTrue(torch.allclose(full, ref, atol=2e-3, rtol=1e-4))
        for num_tokens in (1, 17, 64):
            rows = x[:num_tokens]
            out = torch.empty(num_tokens, NUM_EXPERTS, device=self.device)
            self.reduce(self.gemv(rows, weight), out)
            self.assertTrue(
                torch.equal(out, full[:num_tokens]), f"batch of {num_tokens}"
            )
        # The same rows moved to other positions of the batch.
        shifted = torch.roll(x, shifts=17, dims=0)
        out_shifted = torch.empty_like(full)
        self.reduce(self.gemv(shifted, weight), out_shifted)
        self.assertTrue(torch.equal(out_shifted, torch.roll(full, 17, 0)))

    def test_gate_shared_append_matches_aiter_plus_append(self):
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            fused_append_shared_experts,
        )

        weight = (self._randn(NUM_EXPERTS, HIDDEN) * 0.02).to(torch.bfloat16)
        for num_tokens in (1, 6, 64):
            x = self._randn(num_tokens, HIDDEN).to(torch.bfloat16)
            partials = self.gemv(x, weight)
            logits = torch.empty(num_tokens, NUM_EXPERTS, device=self.device)
            self.reduce(partials, logits)
            ref_w, ref_i = _aiter_gate(logits, self.bias_bf16, TOPK, True, 1.5)
            ref_i, ref_w = fused_append_shared_experts(
                ref_i, ref_w, 1, 1.0, NUM_EXPERTS
            )
            fused_logits = torch.empty_like(logits)
            out_w, out_i = self.gate(
                fused_logits,
                self.bias_bf16,
                TOPK,
                True,
                1.5,
                partials=partials,
                num_shared=1,
            )
            self.assertTrue(torch.equal(ref_i, out_i), f"ids M={num_tokens}")
            self.assertTrue(torch.equal(ref_w, out_w), f"weights M={num_tokens}")
            self.assertTrue(torch.equal(fused_logits, logits), f"logits M={num_tokens}")

    def test_select_experts_partials_with_fused_shared_expert(self):
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        weight = (self._randn(NUM_EXPERTS, HIDDEN) * 0.02).to(torch.bfloat16)
        for scaling in (None, 0.5):
            config = TopKConfig(
                top_k=TOPK + 1,
                renormalize=True,
                num_fused_shared_experts=1,
                correction_bias=self.bias_bf16,
                routed_scaling_factor=1.5,
                fused_shared_experts_scaling_factor=scaling,
                scoring_func="sqrtsoftplus",
            )
            for num_tokens in (1, 6, 64):
                x = self._randn(num_tokens, HIDDEN).to(torch.bfloat16)
                partials = self.gemv(x, weight)
                logits = torch.empty(num_tokens, NUM_EXPERTS, device=self.device)
                self.reduce(partials, logits)
                ref = select_experts(x, logits.clone(), config)
                out = select_experts(
                    x,
                    torch.empty_like(logits),
                    config,
                    router_logits_partials=partials,
                )
                msg = f"M={num_tokens} scaling={scaling}"
                self.assertEqual(out.topk_ids.shape, (num_tokens, TOPK + 1), msg)
                self.assertTrue(torch.equal(ref.topk_ids, out.topk_ids), msg)
                self.assertTrue(torch.equal(ref.topk_weights, out.topk_weights), msg)
                self.assertTrue(torch.equal(ref.router_logits, out.router_logits), msg)


@unittest.skipUnless(
    is_hip() and is_gfx95_supported() and aiter_topk_gating is not None,
    "ROCm gfx95 + aiter only",
)
class TestAiterMoeSortingDispatchPolicy(CustomTestCase):
    """SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY=2 must sort exactly like aiter's auto policy."""

    def _sort(self, ids, weights, num_experts, policy, block_m=32, hidden=HIDDEN):
        import aiter

        m, topk = ids.shape
        padded = ids.numel() + num_experts * block_m - topk
        blocks = (padded + block_m - 1) // block_m
        sorted_ids = torch.empty(blocks * block_m, dtype=torch.int32, device="cuda")
        sorted_weights = torch.empty(blocks * block_m, device="cuda")
        sorted_experts = torch.empty(blocks, dtype=torch.int32, device="cuda")
        num_valid = torch.empty(2, dtype=torch.int32, device="cuda")
        moe_buf = torch.ones(m, hidden, dtype=torch.bfloat16, device="cuda")
        ws = aiter.moe_sorting_opus_get_workspace_size(m, num_experts, topk, policy)
        workspace = (
            torch.empty(ws, dtype=torch.uint8, device="cuda") if ws > 0 else None
        )
        aiter.moe_sorting_opus_fwd(
            ids,
            weights,
            sorted_ids,
            sorted_weights,
            sorted_experts,
            num_valid,
            moe_buf,
            num_experts,
            block_m,
            None,
            None,
            workspace,
            policy,
            None,
            None,
            None,
        )
        n = int(num_valid[0])
        return (
            num_valid.clone(),
            sorted_ids[:n].clone(),
            sorted_weights[:n].clone(),
            sorted_experts[: n // block_m].clone(),
            bool((moe_buf == 0).all()),
        )

    def test_multi_phase_matches_auto(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for num_experts, topk in ((385, 7), (129, 4)):
            routed = num_experts - 1
            for m in (1, 6, 16, 64, 1024):
                ids = torch.argsort(
                    torch.rand(m, routed, device="cuda", generator=gen), dim=1
                )[:, : topk - 1].int()
                shared = torch.full((m, 1), routed, dtype=torch.int32, device="cuda")
                ids = torch.cat([ids, shared], dim=1).contiguous()
                weights = torch.rand(m, topk, device="cuda", generator=gen)
                auto = self._sort(ids, weights, num_experts, policy=0)
                multi_phase = self._sort(ids, weights, num_experts, policy=2)
                msg = f"E={num_experts} topk={topk} M={m}"
                self.assertTrue(auto[4] and multi_phase[4], f"moe_buf zeroed {msg}")
                for a, b in zip(auto[:4], multi_phase[:4]):
                    self.assertTrue(torch.equal(a, b), msg)


if __name__ == "__main__":
    unittest.main()
