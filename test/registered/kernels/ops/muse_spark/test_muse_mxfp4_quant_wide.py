"""MXFP4 row quant on the wide tile (mxfp4_quant_rows) against the row-block kernel it replaces."""

import unittest

import torch
import triton

from sglang.kernels.ops.quantization import mxfp4_quant_rows as mqr
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="base-b-test-1-gpu-small-amd-mi35x")


def _narrow_quant(x):
    """The row-block kernel mxfp4_quant_rows ran before the wide tile (unshuffled aiter layout)."""
    m, n = x.shape
    fp4 = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    bs = torch.empty((m, n // 32), dtype=torch.uint8, device=x.device)
    mqr._mxfp4_quant_rows[(triton.cdiv(m, 128), n // 32)](
        x, fp4, bs, *x.stride(), *fp4.stride(), *bs.stride(), M=m, N=n, scaleN=n // 32, scaleM_pad=m,
        scaleN_pad=n // 32, BLOCK_SIZE=128, MXFP4_QUANT_BLOCK_SIZE=32, SCALING_MODE=0, SHUFFLE=False,
    )  # fmt: skip
    return fp4, bs


def _rows(m, n=4096):
    # Magnitudes spanning the E8M0 range, all-zero blocks, signed zeros and FP4 saturation.
    x = torch.randn(m, n, device="cuda") * torch.logspace(-30, 30, n, device="cuda")
    x[0, :64] = 0
    x[-1, 32:64] = -0.0
    return x.to(torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestMxfp4QuantWideTile(unittest.TestCase):
    def test_wide_tile_quant_is_bitwise_the_row_block_kernel(self):
        torch.manual_seed(0)
        for m in (1, 15, 16, 17, 1712, 4096, 4097, 16384):
            x = _rows(m)
            fp4, bs = mqr.mxfp4_quant_rows(x)
            ref_fp4, ref_bs = _narrow_quant(x)
            self.assertTrue(torch.equal(fp4.view(torch.uint8), ref_fp4), m)
            self.assertTrue(torch.equal(bs.view(torch.uint8), ref_bs), m)


if __name__ == "__main__":
    unittest.main()
