"""Muse BF16 expert layers on FlyDSL (gfx950): the verify-width tile _pick_flydsl_config picks above 128 rows gives the
same bits as the prefill tile it replaced there (split-K 1: each output is one K-ordered MFMA chain), under uniform
and skewed EP4 routing.

One GPU: python3 -m pytest -q test/registered/kernels/ops/muse_spark/test_muse_moe_flydsl_bf16_tiles.py
"""

import pytest
import torch

from sglang.srt.rocm_muse_spark import muse_moe_flydsl as F
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, suite="base-b-test-1-gpu-small-amd-mi35x")

E, START, TOPK = 64, 64, 8
# The tile rows above 128 used before (M * topk > 1024 fell through to it).
PREFILL_TILE = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, FLYDSL_K_UNROLL=1)

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    ),
    reason="Muse FlyDSL MFMA kernels need gfx950",
)


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device="cuda").manual_seed(0)
    w13 = (torch.randn(E, 8192, 4096, device="cuda", generator=g) * 0.02).bfloat16()
    w2 = (torch.randn(E, 4096, 4096, device="cuda", generator=g) * 0.02).bfloat16()
    return w13, w2


@torch.no_grad()
@pytest.mark.parametrize("m", [136, 192, 256, 320])
@pytest.mark.parametrize("sigma", [0.0, 0.3])
def test_verify_tile_bit_identical(weights, m, sigma):
    assert F._pick_flydsl_config(m, TOPK, E, False)["BLOCK_M"] == 32
    g = torch.Generator(device="cuda").manual_seed(m)
    x = (torch.randn(m, 4096, device="cuda", generator=g) * 0.5).bfloat16()
    logits = (
        torch.rand(m, 256, device="cuda", generator=g)
        + torch.randn(256, device="cuda", generator=g) * sigma
    )
    scores, ids = torch.topk(logits, TOPK, dim=1)
    scores = (scores / scores.sum(1, keepdim=True)).float()
    norm_w = (torch.randn(4096, device="cuda", generator=g) * 0.1).bfloat16()
    w13, w2 = weights
    args = (x, scores, ids.int(), w13, w2, START, norm_w, 1e-8)
    picked = F.muse_moe_flydsl(*args, post_norm_gain=1.0)
    before = F.muse_moe_flydsl(*args, cfg=PREFILL_TILE, post_norm_gain=1.0)
    assert torch.equal(picked, before)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
