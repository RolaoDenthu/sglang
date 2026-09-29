"""gfx950 TLX KDA backend against the Triton KDA backend it replaces.

Both kernels receive the tensors ``KDABackend`` hands them (split views of one
packed qkv, the gate/beta layouts ``kimi_linear`` produces, a shuffled slot
pool), so ``prepare_kda_inputs``, the TLX kernels and ``TlxKDAKernel``'s pool
plumbing are all covered together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from sglang.kernels.ops.kimi_k3 import tlx as tlx_kernels
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gfx950_tlx_available() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return arch.startswith("gfx950") and tlx_kernels.is_available()


pytestmark = pytest.mark.skipif(
    not _gfx950_tlx_available(), reason="gfx950 with a TLX-capable triton required"
)

requires_tlx_prefill = pytest.mark.skipif(
    not _gfx950_tlx_available() or not tlx_kernels.is_prefill_available(),
    reason="TLX prefill ops missing: " + ", ".join(tlx_kernels.missing_prefill_ops()),
)

if _gfx950_tlx_available():
    from sglang.kernels.ops.kimi_k3.tlx.kimi_k3_kda_prepare import prepare_kda_inputs
    from sglang.srt.layers.attention.linear.kernels.kda_tlx import TlxKDAKernel
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

_DEVICE = torch.device("cuda")
_HEADS = 12
_DIM = 128
_LOWER_BOUND = -5.0

# TLX decode rounds the log decay to bf16 before exp(); Triton keeps it fp32.
_DECODE_TOL = 5e-3
_DECODE_MULTI_STEP_TOL = 1e-2
# triton-ext#142 measures the TLX prefill at 4.6e-3 .. 6.9e-3 against FLA.
_EXTEND_TOL = 1e-2


def _relative_rmse(reference: torch.Tensor, actual: torch.Tensor) -> float:
    delta = actual.float() - reference.float()
    return float(
        delta.square().mean().sqrt() / (reference.float().square().mean().sqrt() + 1e-8)
    )


def _split_qkv(packed: torch.Tensor):
    q, k, v = packed.split(_HEADS * _DIM, dim=-1)
    return tuple(t.unflatten(-1, (_HEADS, _DIM)).unsqueeze(0) for t in (q, k, v))


@dataclass
class Case:
    qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    pool: torch.Tensor
    cache_indices: torch.Tensor
    query_start_loc: torch.Tensor


def _make_case(
    seq_lens: list[int],
    *,
    decode: bool,
    beta_is_raw: bool = True,
    a_log_mean: float = -4.0,
    seed: int = 20260929,
) -> Case:
    gen = torch.Generator(device=_DEVICE).manual_seed(seed + sum(seq_lens))
    tokens = sum(seq_lens)
    width = _HEADS * _DIM

    def randn(*shape, dtype=torch.bfloat16, scale=1.0):
        return scale * torch.randn(shape, dtype=dtype, device=_DEVICE, generator=gen)

    if decode:
        # forward_decode: flat raw gate [T, H*K], raw beta [1, T, H].
        a = randn(tokens, width)
        b = randn(1, tokens, _HEADS)
    else:
        # forward_extend: gate unflattened to [1, T, H, K]; beta either
        # pre-sigmoided fp32 (kimi_linear) or raw (flat-gate callers).
        a = randn(1, tokens, _HEADS, _DIM)
        b = randn(1, tokens, _HEADS, dtype=torch.float32)
        if not beta_is_raw:
            b = b.sigmoid()

    # Slot 0 is kept out of the batch so untouched-slot checks cover it too.
    slots = len(seq_lens) + 3
    cache_indices = (
        torch.randperm(slots - 1, device=_DEVICE, generator=gen)[: len(seq_lens)] + 1
    ).to(torch.int32)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=_DEVICE)
    query_start_loc = torch.nn.functional.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))

    return Case(
        qkv=randn(tokens, 3 * width),
        a=a,
        b=b,
        # The TLX prefill exponentiates differences of cumulative log decay, so
        # A_log ~ N(0, 1) overflows to NaN (triton-ext#142); keep it physical.
        A_log=a_log_mean + randn(_HEADS, dtype=torch.float32, scale=0.1),
        dt_bias=randn(width, dtype=torch.float32, scale=0.1),
        pool=randn(slots, _HEADS, _DIM, _DIM, dtype=torch.float32, scale=0.02),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )


def _run_decode(kernel, case: Case, pool: torch.Tensor, lower_bound: Optional[float]):
    q, k, v = _split_qkv(case.qkv.clone())
    return kernel.decode(
        q,
        k,
        v,
        case.a.clone(),
        case.b.clone(),
        A_log=case.A_log,
        dt_bias=case.dt_bias,
        ssm_states=pool,
        cache_indices=case.cache_indices,
        query_start_loc=case.query_start_loc,
        lower_bound=lower_bound,
    )


def _run_extend(
    kernel,
    case: Case,
    pool: torch.Tensor,
    lower_bound: Optional[float],
    beta_is_raw: bool,
):
    # chunk_kda writes its output into v, so every run gets its own qkv.
    q, k, v = _split_qkv(case.qkv.clone())
    return kernel.extend(
        q,
        k,
        v,
        case.a.clone(),
        case.b.clone(),
        ssm_states=pool,
        cache_indices=case.cache_indices,
        query_start_loc=case.query_start_loc,
        A_log=case.A_log,
        dt_bias=case.dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=beta_is_raw,
    )


def _assert_untouched(before: torch.Tensor, after: torch.Tensor, used: torch.Tensor):
    untouched = torch.ones(before.shape[0], dtype=torch.bool, device=before.device)
    untouched[used[used >= 0].long()] = False
    assert torch.equal(before[untouched], after[untouched])


@pytest.mark.parametrize("lower_bound", [None, _LOWER_BOUND])
@pytest.mark.parametrize("batch", [1, 8, 32])
def test_decode_matches_triton(batch: int, lower_bound: Optional[float]) -> None:
    case = _make_case([1] * batch, decode=True)
    ref_pool, tlx_pool = case.pool.clone(), case.pool.clone()

    ref = _run_decode(TritonKDAKernel(), case, ref_pool, lower_bound)
    out = _run_decode(TlxKDAKernel(), case, tlx_pool, lower_bound)
    torch.cuda.synchronize()

    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert not torch.isnan(out).any()
    assert _relative_rmse(ref, out) < _DECODE_TOL
    idx = case.cache_indices.long()
    assert _relative_rmse(ref_pool[idx], tlx_pool[idx]) < _DECODE_TOL
    _assert_untouched(case.pool, tlx_pool, case.cache_indices)


@pytest.mark.parametrize("lower_bound", [None, _LOWER_BOUND])
def test_decode_recurrence_stays_close(lower_bound: Optional[float]) -> None:
    ref_kernel, tlx_kernel = TritonKDAKernel(), TlxKDAKernel()
    case = _make_case([1] * 4, decode=True)
    ref_pool, tlx_pool = case.pool.clone(), case.pool.clone()

    for step in range(32):
        step_case = _make_case([1] * 4, decode=True, seed=step)
        step_case.pool = case.pool
        step_case.cache_indices = case.cache_indices
        ref = _run_decode(ref_kernel, step_case, ref_pool, lower_bound)
        out = _run_decode(tlx_kernel, step_case, tlx_pool, lower_bound)
    torch.cuda.synchronize()

    assert _relative_rmse(ref, out) < _DECODE_MULTI_STEP_TOL
    idx = case.cache_indices.long()
    assert _relative_rmse(ref_pool[idx], tlx_pool[idx]) < _DECODE_MULTI_STEP_TOL


def test_decode_padded_rows_do_not_touch_pool() -> None:
    # CUDA-graph padding rows carry slot -1.
    case = _make_case([1] * 4, decode=True)
    case.cache_indices[1] = -1
    case.cache_indices[3] = -1
    ref_pool, tlx_pool = case.pool.clone(), case.pool.clone()

    ref = _run_decode(TritonKDAKernel(), case, ref_pool, None)
    out = _run_decode(TlxKDAKernel(), case, tlx_pool, None)
    torch.cuda.synchronize()

    _assert_untouched(case.pool, tlx_pool, case.cache_indices)
    valid = case.cache_indices >= 0
    assert _relative_rmse(ref[:, valid], out[:, valid]) < _DECODE_TOL
    idx = case.cache_indices[valid].long()
    assert _relative_rmse(ref_pool[idx], tlx_pool[idx]) < _DECODE_TOL


@requires_tlx_prefill
@pytest.mark.parametrize("beta_is_raw", [False, True])
@pytest.mark.parametrize("lower_bound", [None, _LOWER_BOUND])
@pytest.mark.parametrize(
    "seq_lens", [[1], [64], [37, 300, 129], [2048], [5, 1024, 63, 65]]
)
def test_extend_matches_triton(
    seq_lens: list[int], lower_bound: Optional[float], beta_is_raw: bool
) -> None:
    case = _make_case(seq_lens, decode=False, beta_is_raw=beta_is_raw)
    ref_pool, tlx_pool = case.pool.clone(), case.pool.clone()

    ref = _run_extend(TritonKDAKernel(), case, ref_pool, lower_bound, beta_is_raw)
    out = _run_extend(TlxKDAKernel(), case, tlx_pool, lower_bound, beta_is_raw)
    torch.cuda.synchronize()

    # forward_extend uses the result directly as core_attn_out.
    assert isinstance(out, torch.Tensor), type(out)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert not torch.isnan(out).any()
    assert _relative_rmse(ref, out) < _EXTEND_TOL
    idx = case.cache_indices.long()
    assert _relative_rmse(ref_pool[idx], tlx_pool[idx]) < _EXTEND_TOL
    _assert_untouched(case.pool, tlx_pool, case.cache_indices)


@requires_tlx_prefill
@pytest.mark.parametrize("seq_lens", [[256], [64, 1000]])
def test_extend_saturated_safe_gate_stays_finite(seq_lens: list[int]) -> None:
    # Large A_log saturates the sigmoid, so most channels sit at the -5 per-token
    # floor: the steepest cumulative decay a safe-gate model can produce.
    case = _make_case(seq_lens, decode=False, beta_is_raw=False, a_log_mean=2.0)
    ref_pool, tlx_pool = case.pool.clone(), case.pool.clone()

    ref = _run_extend(TritonKDAKernel(), case, ref_pool, _LOWER_BOUND, False)
    out = _run_extend(TlxKDAKernel(), case, tlx_pool, _LOWER_BOUND, False)
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    assert torch.isfinite(tlx_pool).all()
    assert _relative_rmse(ref, out) < _EXTEND_TOL
    idx = case.cache_indices.long()
    assert _relative_rmse(ref_pool[idx], tlx_pool[idx]) < _EXTEND_TOL


def _prepare_reference(q, k, a, b, A_log, dt_bias, lower_bound, sigmoid_beta):
    tokens = a.numel() // (_HEADS * _DIM)
    q, k = (x.reshape(tokens, _HEADS, _DIM).float() for x in (q, k))
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    x = a.reshape(tokens, _HEADS, _DIM).float() + dt_bias.view(_HEADS, _DIM)
    A = A_log.exp().view(1, _HEADS, 1)
    if lower_bound is None:
        g = -A * torch.nn.functional.softplus(x, threshold=20.0)
    else:
        g = lower_bound * torch.sigmoid(A * x)
    beta = b.reshape(tokens, _HEADS).float()
    if sigmoid_beta:
        beta = beta.sigmoid()
    return q.bfloat16(), k.bfloat16(), g, beta


# 341 * 12 = 4092 rows and 342 * 12 = 4104 rows sit on either side of the
# small/large tile switch; 37 and 1000 leave a partial last tile.
@pytest.mark.parametrize("sigmoid_beta", [True, False])
@pytest.mark.parametrize("lower_bound", [None, _LOWER_BOUND])
@pytest.mark.parametrize("strided", [True, False])
@pytest.mark.parametrize("tokens", [1, 37, 341, 342, 1000])
def test_prepare_matches_reference(
    tokens: int, strided: bool, lower_bound: Optional[float], sigmoid_beta: bool
) -> None:
    case = _make_case([tokens], decode=True)
    if strided:
        q, k, _ = _split_qkv(case.qkv)
    else:
        q, k, _ = (t.contiguous() for t in _split_qkv(case.qkv))
    args = (q, k, case.a, case.b, case.A_log, case.dt_bias)

    qn, kn, g, beta = prepare_kda_inputs(
        *args, num_heads=_HEADS, head_dim=_DIM,
        lower_bound=lower_bound, sigmoid_beta=sigmoid_beta,
    )
    ref = _prepare_reference(*args, lower_bound, sigmoid_beta)

    assert qn.shape == kn.shape == g.shape == (1, tokens, _HEADS, _DIM)
    assert beta.shape == (1, tokens, _HEADS)
    # q/k are rounded to bf16 after a reduction whose order differs from torch,
    # so they get bf16 tolerances; the fp32 gate and beta must match tightly.
    for out, expected in zip((qn, kn), ref[:2]):
        torch.testing.assert_close(out[0], expected)
    for out, expected in zip((g, beta), ref[2:]):
        torch.testing.assert_close(out[0], expected, rtol=1e-5, atol=1e-5)


@requires_tlx_prefill
def test_extend_spec_decode_falls_back_to_triton() -> None:
    # draft_extend_v2 must stay on Triton, so TLX has to match it exactly.
    case = _make_case([37, 300], decode=False, beta_is_raw=False)
    ref_pool, tlx_pool = case.pool.clone(), case.pool.clone()

    def run(kernel, pool):
        q, k, v = _split_qkv(case.qkv.clone())
        return kernel.extend(
            q, k, v, case.a.clone(), case.b.clone(),
            ssm_states=pool, cache_indices=case.cache_indices,
            query_start_loc=case.query_start_loc, A_log=case.A_log,
            dt_bias=case.dt_bias, lower_bound=_LOWER_BOUND,
            beta_is_raw=False, is_spec_decode=True,
        )

    ref = run(TritonKDAKernel(), ref_pool)
    out = run(TlxKDAKernel(), tlx_pool)
    torch.cuda.synchronize()

    assert torch.equal(ref, out)
    assert torch.equal(ref_pool, tlx_pool)


@requires_tlx_prefill
def test_extend_then_decode_continues_from_prefill_state() -> None:
    seq_lens = [300, 17]
    ref_kernel, tlx_kernel = TritonKDAKernel(), TlxKDAKernel()
    prefill = _make_case(seq_lens, decode=False)
    ref_pool, tlx_pool = prefill.pool.clone(), prefill.pool.clone()
    _run_extend(ref_kernel, prefill, ref_pool, _LOWER_BOUND, False)
    _run_extend(tlx_kernel, prefill, tlx_pool, _LOWER_BOUND, False)

    decode = _make_case([1] * len(seq_lens), decode=True)
    decode.pool = prefill.pool
    decode.cache_indices = prefill.cache_indices
    ref = _run_decode(ref_kernel, decode, ref_pool, _LOWER_BOUND)
    out = _run_decode(tlx_kernel, decode, tlx_pool, _LOWER_BOUND)
    torch.cuda.synchronize()

    assert _relative_rmse(ref, out) < _DECODE_MULTI_STEP_TOL
    idx = prefill.cache_indices.long()
    assert _relative_rmse(ref_pool[idx], tlx_pool[idx]) < _DECODE_MULTI_STEP_TOL


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
