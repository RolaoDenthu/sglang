"""DeepSeek-V4.1 vision routing on the aiter MegaMoEv2 (ROCm) path."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import MoeA2ABackend, mega_moe_flydsl
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.runtime_context import get_flags, get_parallel

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

IMAGE_TOKEN_ID = 99
NUM_ROUTED_EXPERTS = 4
# Rows past this bound are padding.
NON_PADDED_ROWS = 2


def _fill_padded_rows(x, num_token_non_padded, fill_value):
    x[int(num_token_non_padded) :] = fill_value


class _FakeMega:
    """Routed experts multiply x by their physical ID + 1; the rank-1 shared slot
    (ID 5) is an identity expert, and the out-of-range fill ID contributes nothing."""

    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.calls = []

    def forward(self, x, weights, topk_ids, slice_output=True):
        self.calls.append((topk_ids.clone(), weights.clone()))
        valid = (topk_ids >= 0) & (topk_ids < self.num_experts)
        factors = (topk_ids + 1).float()
        factors[topk_ids == 5] = 1.0
        factors[~valid] = 0.0
        return x * (factors * weights).sum(-1, keepdim=True).to(x.dtype)


class TestAiterMegaMoeVisionRouting(CustomTestCase):
    @staticmethod
    def _v41_moe(fused_shared, scale_in_topk, topk=None):
        return SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                image_token_id=IMAGE_TOKEN_ID,
                swiglu_limit=None,
            ),
            experts=SimpleNamespace(
                num_experts=NUM_ROUTED_EXPERTS + 2 * fused_shared,
                uses_expert_location=True,
            ),
            gate=MagicMock(
                return_value=torch.zeros((3, 4)),
                e_score_correction_bias=torch.tensor([4.0, 3.0, 0.0, 0.0]),
                e_score_correction_bias_vl=torch.tensor([0.0, 0.0, 3.0, 4.0]),
            ),
            topk=topk
            or SimpleNamespace(
                topk_config=TopKConfig(
                    top_k=2 + fused_shared,
                    shared_experts=SharedExpertsConfig(
                        num_fused_shared_experts=fused_shared
                    ),
                    renormalize=True,
                    routed_scaling_factor=2.0,
                    apply_routed_scaling_factor_on_output=scale_in_topk,
                ),
                routing_inputs=lambda: (
                    torch.tensor(NON_PADDED_ROWS, dtype=torch.int32),
                    None,
                ),
            ),
            is_hash=False,
            num_fused_shared_experts=fused_shared,
            layer_id=0,
            routed_scaling_factor=2.0,
            routed_scaling_factor_fused_in_topk=scale_in_topk,
        )

    @staticmethod
    def _batch(*, extend=True, images=True):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: extend),
            contains_image_inputs=lambda: images,
        )

    def _run(self, moe, batch, ids):
        mega = _FakeMega(moe.experts.num_experts)
        with (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
            get_parallel().override(moe_ep_size=2, moe_ep_rank=1),
            patch.object(mega_moe_flydsl, "_get_or_build_mega_moe", return_value=mega),
            patch.object(mega_moe_flydsl, "_swap_layer_weights"),
            patch.object(mega_moe_flydsl, "_select_mtpr", return_value=8),
            # The padded-row fill is a Triton kernel.
            patch.object(mega_moe_flydsl, "_fill_padded_rows", _fill_padded_rows),
            patch(
                "sglang.srt.multimodal.dsv41.vl_routing.is_cuda",
                return_value=False,
            ),
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_can_fuse_padded_region", return_value=False),
        ):
            out = mega_moe_flydsl._run_mega_routed(
                moe, torch.ones((3, 4), dtype=torch.bfloat16), batch, ids
            )
        return out, mega

    def test_image_rows_use_vl_bias(self):
        """Equal logits make the two biases select disjoint experts, so an image
        row routed as text lands on the wrong expert boundary. Padded rows must
        contribute nothing, and the routed scale must be applied exactly once."""
        for fused_shared in (0, 1):
            for scale_in_topk in (False, True):
                with self.subTest(
                    fused_shared=fused_shared, scale_in_topk=scale_in_topk
                ):
                    moe = self._v41_moe(fused_shared, scale_in_topk)
                    ids = torch.tensor([7, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID])
                    out, mega = self._run(moe, self._batch(), ids)
                    # Text selects logical 0/1; image selects logical 2/3. With
                    # a shared slot, rank-1 routed IDs shift by one (2/3 -> 3/4).
                    expected = torch.tensor(
                        [3.0 + fused_shared, 7.0 + 3.0 * fused_shared, 0.0]
                    )[:, None].expand(3, 4)
                    torch.testing.assert_close(out.float(), expected)
                    topk_ids, weights = mega.calls[0]
                    self.assertEqual(topk_ids.dtype, torch.int32)
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertTrue((topk_ids[2] == moe.experts.num_experts).all())

    def test_batches_without_images_keep_regular_topk(self):
        """Decode, text-only extend and hash layers never read the VL bias."""
        for name, batch, is_hash in (
            ("decode", self._batch(extend=False), False),
            ("text_extend", self._batch(images=False), False),
            ("hash_layer", self._batch(), True),
        ):
            with self.subTest(name):
                topk = MagicMock(
                    return_value=SimpleNamespace(
                        topk_ids=torch.tensor([[0, 1], [0, 1], [0, 1]]),
                        topk_weights=torch.full((3, 2), 0.5),
                    )
                )
                topk.routing_inputs.return_value = (None, None)
                moe = self._v41_moe(0, True, topk=topk)
                moe.is_hash = is_hash
                ids = torch.tensor([7, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID])
                with patch.object(mega_moe_flydsl, "vision_topk") as vision_topk:
                    self._run(moe, batch, ids)
                vision_topk.assert_not_called()
                topk.assert_called_once()
                if is_hash:
                    self.assertIs(topk.call_args.kwargs["input_ids"], ids)
                else:
                    self.assertNotIn("input_ids", topk.call_args.kwargs)


if __name__ == "__main__":
    sys.exit(unittest.main())
