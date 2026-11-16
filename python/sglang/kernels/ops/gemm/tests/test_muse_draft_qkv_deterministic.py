"""The DFlash draft QKV projection [2560, 8192] at 1..16 rows (one request's draft block) is bitwise stable run to
run: it runs the row-major split-K GEMM (FP32 partials, ordered sum), not aiter's FlyDSL split-K kernel with bf16
atomic adds."""

import hashlib
import os

import pytest
import torch

os.environ.setdefault("SGLANG_USE_AITER", "1")

from sglang.kernels.ops.gemm import bf16_tiled_small_m_gemm as rm

N, K = 2560, 8192
ROWS = (1, 2, 4, 7, 8, 12, 16)


def test_draft_qkv_has_a_row_major_launch_down_to_one_row():
    assert all((N, K, rm._bm(m)) in rm.ROW_MAJOR for m in ROWS)


def _digest(t):
    return hashlib.sha1(
        t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def _on_gfx950():
    if not (torch.cuda.is_available() and torch.version.hip):
        return False
    from sglang.srt.utils import is_gfx95_supported

    return is_gfx95_supported()


@torch.no_grad()
@pytest.mark.skipif(not _on_gfx950(), reason="needs gfx950")
@pytest.mark.parametrize("m", ROWS)
def test_draft_qkv_is_bitwise_stable_eager_and_in_a_graph(m):
    from sglang.srt.layers.quantization import unquant
    from sglang.srt.models import dflash

    # The registration the DFlash draft makes before its weights load.
    unquant.register_gfx950_unquantized_gemm_shapes(
        rowmajor_weights=dflash._GFX950_ROWMAJOR_WEIGHTS
    )
    method = unquant.UnquantizedLinearMethod()
    layer = torch.nn.Module()
    g = torch.Generator(device="cuda").manual_seed(m)
    layer.weight = torch.nn.Parameter(
        (0.02 * torch.randn(N, K, device="cuda", generator=g)).bfloat16(),
        requires_grad=False,
    )
    method.process_weights_after_loading(layer)
    x = torch.randn(m, K, device="cuda", generator=g).bfloat16()
    assert rm.rowmajor_gemm_applies(x, layer.weight)
    y = method.apply(layer, x)
    assert len({_digest(method.apply(layer, x)) for _ in range(64)}) == 1
    assert _digest(y) == _digest(method.apply(layer, x))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        yg = method.apply(layer, x)
    for _ in range(16):
        graph.replay()
        torch.cuda.synchronize()
        assert _digest(yg) == _digest(y)
    ref = x.double() @ layer.weight.double().T
    # FP32 split sums and one BF16 rounding: within one BF16 ulp-scale of the exact product.
    assert (y.double() - ref).abs().max() <= 2**-7 * ref.abs().max() + 1e-3
