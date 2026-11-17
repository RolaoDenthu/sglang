"""Ulysses output-projection input in one pass: norm-gate + per-row FP8 quantization
(FusedRMSNormGate.forward_quant_fp8_heads) against the two launches it replaces."""

import types
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.gemm.fp8_prefill_gemm import quantize_head_rows_fp8
from sglang.srt.rocm_muse_spark import fused_norm_gate
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="base-b-test-1-gpu-small-amd-mi35x")


def _op():
    # The Ulysses output norm: no weight, eps of the checkpoint's o_norm.
    o_norm = types.SimpleNamespace(eps=1e-6, has_weight=False)
    return fused_norm_gate.FusedRMSNormGate(o_norm, use_o_gate=True, head_dim=64)


class TestUlyssesNormGateQuant(unittest.TestCase):
    def test_applies_only_to_the_replayed_layout(self):
        op = _op()
        heads = torch.randn(4, 8, 2048).to(torch.bfloat16)
        gate = torch.randn(4, 8, 2048).to(torch.bfloat16)
        with mock.patch.object(
            op, "_resolve_backend", return_value=fused_norm_gate.FusedOpBackend.TRITON
        ):
            self.assertTrue(op.quant_fp8_heads_applies(heads, gate))
            self.assertFalse(
                op.quant_fp8_heads_applies(heads[:, :, :1024], gate[:, :, :1024])
            )  # not contiguous
            self.assertFalse(op.quant_fp8_heads_applies(heads.float(), gate))
            self.assertFalse(
                op.quant_fp8_heads_applies(heads[:3], gate[:3])
            )  # 96 heads per row
            op._effective_weight = torch.ones(64)
            self.assertFalse(op.quant_fp8_heads_applies(heads, gate))
        op._effective_weight = None
        with mock.patch.object(
            op, "_resolve_backend", return_value=fused_norm_gate.FusedOpBackend.NATIVE
        ):
            self.assertFalse(op.quant_fp8_heads_applies(heads, gate))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_bitwise_the_norm_gate_then_row_quant(self):
        # The two-launch reference: the norm-gate at its default config, then quantize_head_rows_fp8.
        torch.manual_seed(0)
        op = _op()
        for world, hw in ((4, 2048), (8, 1024)):
            for rows in (1, 2019, 4096):
                heads = (torch.randn(world, rows, hw, device="cuda") * 4).to(
                    torch.bfloat16
                )
                big = torch.randn(world, rows + 7, hw, device="cuda").to(torch.bfloat16)
                gate = big[:, 3 : 3 + rows]
                self.assertTrue(op.quant_fp8_heads_applies(heads, gate))
                with mock.patch.object(
                    fused_norm_gate.fused_op_tuning,
                    "resolve",
                    lambda op_name, defaults, **_: dict(defaults),
                ):
                    gated = op.forward_triton(
                        heads.view(world * rows, hw), gate.reshape(world * rows, hw)
                    )
                ref_q, ref_s = quantize_head_rows_fp8(gated.view(world, rows, hw))
                xq, xs = op.forward_quant_fp8_heads(heads, gate)
                self.assertTrue(torch.equal(xs, ref_s), (world, rows))
                self.assertTrue(
                    torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8)),
                    (world, rows),
                )


if __name__ == "__main__":
    unittest.main()
