"""The FFN-boundary kernel fed the MoE unpooler's two FP32 split-K partials writes the bits it writes when fed the
unpooler's BF16 output (the row-major GEMM followed by its _sum_partials), for every flag combination the boundary
uses, at the widths where that GEMM splits K two ways (65-128 rows); the torch fallback sums the partials the same
way. Covers the FP8-copy output the decode path asks for."""

import itertools

import pytest
import torch
import triton

from sglang.kernels.ops.gemm.bf16_tiled_small_m_gemm import (
    rowmajor_gemm,
    rowmajor_gemm_partials,
    rowmajor_gemm_partials_split,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="base-b-test-1-gpu-small-amd-mi35x")

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel")

H, K = 8192, 4096


def _boundary(branch, stream, post_w, pre_w, alpha, beta, flags, splits):
    from sglang.srt.rocm_muse_spark.fused_norm_residual import (
        _fused_norm_residual_triton_kernel,
    )

    input_scale, gate, cw, pre_has_w, fp8 = flags
    m = stream.shape[0]
    dev = stream.device
    out = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    ns = torch.empty(m, H, dtype=torch.float32, device=dev)
    empty = torch.empty(0, dtype=torch.float32, device=dev)
    of8 = torch.empty(m, H, dtype=torch.float8_e4m3fn, device=dev) if fp8 else empty
    osc = torch.empty(m, dtype=torch.float32, device=dev) if fp8 else empty
    _fused_norm_residual_triton_kernel[(m,)](
        branch, stream, out, ns, empty, of8, osc, post_w, pre_w if pre_has_w else empty,
        alpha if gate else empty, beta if gate else empty, 1e-6, 1e-6, m * H, H, triton.next_power_of_2(H),
        input_scale, gate, cw, True, pre_has_w, False, False, 0, fp8, splits, num_warps=8, num_stages=1,
    )  # fmt: skip
    return [out, ns] + ([of8.view(torch.uint8), osc] if fp8 else [])


def test_two_way_split_at_65_to_128_rows():
    assert all(rowmajor_gemm_partials_split(m, H, K) == 2 for m in (65, 96, 128))
    assert rowmajor_gemm_partials_split(64, H, K) == 1


@needs_gpu
@pytest.mark.parametrize("m", [65, 104, 128])
def test_partials_match_summed_branch(m):
    torch.manual_seed(m)
    x = (torch.randn(m, K, device="cuda") * 0.5).bfloat16()
    w = (torch.randn(H, K, device="cuda") * 0.02).bfloat16()
    partials = rowmajor_gemm_partials(x, w)
    assert partials.shape == (2, m, H) and partials.dtype == torch.float32
    branch = rowmajor_gemm(x, w)
    stream = torch.randn(m, H, device="cuda") * 10
    post_w = (torch.rand(H, device="cuda") + 0.5).bfloat16()
    pre_w = (torch.rand(H, device="cuda") + 0.5).bfloat16()
    for gate_mode, input_scale, pre_has_w, fp8 in itertools.product(
        ((False, False), (True, False), (True, True)),
        (False, True),
        (False, True),
        (False, True),
    ):
        gate, cw = gate_mode
        alpha = torch.rand(H if cw else 1, device="cuda") + 0.5
        beta = torch.rand(H if cw else 1, device="cuda") + 0.5
        flags = (input_scale, gate, cw, pre_has_w, fp8)
        ref = _boundary(branch, stream, post_w, pre_w, alpha, beta, flags, 1)
        got = _boundary(partials, stream, post_w, pre_w, alpha, beta, flags, 2)
        for a, b in zip(ref, got):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), flags


@needs_gpu
def test_native_fallback_sums_like_the_gemm():
    torch.manual_seed(0)
    x = (torch.randn(96, K, device="cuda") * 0.5).bfloat16()
    w = (torch.randn(H, K, device="cuda") * 0.02).bfloat16()
    p = rowmajor_gemm_partials(x, w)
    assert torch.equal((p[0] + p[1]).to(torch.bfloat16), rowmajor_gemm(x, w))
