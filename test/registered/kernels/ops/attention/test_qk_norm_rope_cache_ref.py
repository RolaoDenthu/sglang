"""kernels/ops/attention/qk_norm_rope_cache.py against an in-tree torch reference (_qk_norm_rope_ref.py).

Entry points: ``qk_norm_rope_cache`` (decode / verify) and ``qkv_proj_qk_norm_rope_cache`` (the BF16 projected
path). The operands come from dyadic grids that make every sum exact in FP32 (see the reference module). Stage 1: the
kernel's norm-only output (RoPE off, query scale 1.0) must equal the reference norm byte for byte, except at BF16
rounding near-ties, where the kernel's and torch's rsqrt (one FP32 ulp apart for a few heads) may land one BF16 ulp
apart. Stage 2: from that checked norm output, the reference's query scale, RoPE and FP8 cache store must reproduce
the kernel's Q / K (and V / gate) and the full FP8 K / V caches byte for byte. Each case also repeats the kernel and
requires identical bytes. Two query scales run: a dyadic one (0.125, exact multiply) and the model's 5.4730064863875,
which exercises the scale's own BF16 rounding.

Callers: in this tree only the Muse Spark ROCm model (srt/rocm_muse_spark/attention_kv_write.py and model.py) imports
this module. The shapes cover its contract (head_dim 64, FP8 E4M3 cache) at TP4 (32 Q / 4 KV heads, QKV + gate N 4608)
and TP8 (16 / 2, N 2304), with and without RoPE tables, the full -> SWA slot mapping (with -1 entries), skipped (-1)
locations and K / V scales.

MI355X (gfx950, Triton): python3 -m pytest -q test/registered/kernels/ops/attention/test_qk_norm_rope_cache_ref.py
"""

import os
import sys

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _qk_norm_rope_ref as R  # noqa: E402

register_amd_ci(est_time=20, suite="base-b-test-1-gpu-small-amd-mi35x")

D = R.D
EPS = 1e-6
QSCALES = (
    0.125,
    5.4730064863875,
)  # dyadic; the model's (non-dyadic: its BF16 round trip is exercised)
needs_gfx950 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx95_supported()),
    reason="the FP8 E4M3 cache kernels serve the Muse Spark ROCm model on gfx950",
)


def _table():
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    ang = torch.arange(8192, device="cuda").float()[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1).contiguous()


def _common(m, hk, rope, mapped, scaled, gen, slots=4096):
    positions = torch.randint(
        0, 8192, (m,), device="cuda", generator=gen, dtype=torch.int64
    )
    cos_sin = _table() if rope else None
    locations = torch.randperm(slots, device="cuda", generator=gen)[:m].to(torch.int64)
    if m > 2:
        locations[1] = -1  # a padded row: no cache store
    mapping = None
    if mapped:
        mapping = torch.randperm(slots, device="cuda", generator=gen).to(torch.int64)
        mapping[locations[0]] = -1  # a token outside the window
    ks = torch.tensor([0.75], device="cuda") if scaled else None
    vs = torch.tensor([1.25], device="cuda") if scaled else None
    cache = (torch.randn(slots, hk, D, device="cuda", generator=gen) * 2).to(
        torch.float8_e4m3fn
    )
    return positions, cos_sin, locations, mapping, ks, vs, cache


CASES = [
    (m, hq, hk, rope, mapped, scaled)
    for m in (1, 8, 33)
    for hq, hk in ((32, 4), (16, 2))
    for rope, mapped, scaled in (
        (True, False, False),
        (True, True, True),
        (False, False, True),
        (True, True, False),
    )
]


def _check(fn, qscale, q, k, v, pos, cs, loc, mp, ks, vs, cache):
    """Stage 1: the kernel's norm-only Q / K (RoPE off, scale 1.0) vs norm_ref (byte-equal except BF16 near-ties).
    Stage 2: the kernel (twice) vs finish_ref / cache_ref built on that norm output; Q, K and both caches byte-equal."""
    qn, kn = fn(
        q, k, v, pos, None, cache.clone(), cache.clone(), loc, mp, EPS, 1.0, ks, vs
    )
    R.check_norm(qn, q, EPS)
    R.check_norm(kn, k, EPS)
    want_q = R.finish_ref(qn, pos, cs, qscale)
    want_k = R.finish_ref(kn, pos, cs)
    rk, rv = cache.clone(), cache.clone()
    R.cache_ref(rk, rv, want_k, v, loc, mp, ks, vs)
    for _ in range(2):
        ck, cv = cache.clone(), cache.clone()
        qo, ko = fn(q, k, v, pos, cs, ck, cv, loc, mp, EPS, qscale, ks, vs)
        for name, got, want in (
            ("q", qo, want_q),
            ("k", ko, want_k),
            ("cache_k", ck, rk),
            ("cache_v", cv, rv),
        ):
            assert R.same_bytes(got, want), f"{name} differs from the torch reference"


@needs_gfx950
@pytest.mark.parametrize("qscale", QSCALES)
@pytest.mark.parametrize("m,hq,hk,rope,mapped,scaled", CASES)
def test_qk_norm_rope_cache_matches_reference(m, hq, hk, rope, mapped, scaled, qscale):
    from sglang.kernels.ops.attention import qk_norm_rope_cache as cur

    gen = torch.Generator(device="cuda").manual_seed(100 + m + hq)
    q, k, v = (
        R.exact_heads(m, hq, gen),
        R.exact_heads(m, hk, gen),
        R.exact_heads(m, hk, gen),
    )
    _check(
        cur.qk_norm_rope_cache,
        qscale,
        q,
        k,
        v,
        *_common(m, hk, rope, mapped, scaled, gen),
    )


@needs_gfx950
@pytest.mark.parametrize("qscale", QSCALES)
@pytest.mark.parametrize("m", (1, 8, 16, 17, 32))
@pytest.mark.parametrize("hq,hk", ((32, 4), (16, 2)))
@pytest.mark.parametrize(
    "mapped,scales", ((False, (0.75, 1.25)), (True, (0.75, 1.25)), (True, (1.0, 1.0)))
)
def test_qkv_proj_qk_norm_rope_cache_matches_reference(
    m, hq, hk, mapped, scales, qscale
):
    """The BF16 projected path (SCALED off; it always divides by the K / V scales): the epilogue's XS / WS operands
    are unused placeholders there. Q / K / V / gate and both caches byte-equal to the reference projection + norm."""
    from sglang.kernels.ops.attention import qk_norm_rope_cache as cur

    gen = torch.Generator(device="cuda").manual_seed(300 + m + hq)
    n = (2 * hq + 2 * hk) * D
    x, w = R.exact_proj_operands(m, n, 8192, gen)
    pos, cs, loc, mp, _, _, cache = _common(m, hk, True, mapped, False, gen)
    ks, vs = (
        torch.tensor([scales[0]], device="cuda"),
        torch.tensor([scales[1]], device="cuda"),
    )
    y = R.proj_ref(x, w)
    q, k, v, gate = (
        t.contiguous() for t in y.split([hq * D, hk * D, hk * D, hq * D], dim=-1)
    )
    qn, kn, _, _ = cur.qkv_proj_qk_norm_rope_cache(
        x,
        w,
        pos,
        None,
        cache.clone(),
        cache.clone(),
        loc,
        mp,
        EPS,
        1.0,
        ks,
        vs,
        num_heads=hq,
        num_kv_heads=hk,
    )
    R.check_norm(qn, q, EPS)
    R.check_norm(kn, k, EPS)
    want_q, want_k = R.finish_ref(qn, pos, cs, qscale), R.finish_ref(kn, pos, cs)
    rk, rv = cache.clone(), cache.clone()
    R.cache_ref(rk, rv, want_k, v, loc, mp, ks, vs)
    for _ in range(2):
        ck, cv = cache.clone(), cache.clone()
        qo, ko, vo, go = cur.qkv_proj_qk_norm_rope_cache(
            x,
            w,
            pos,
            cs,
            ck,
            cv,
            loc,
            mp,
            EPS,
            qscale,
            ks,
            vs,
            num_heads=hq,
            num_kv_heads=hk,
        )
        for name, got, want in (
            ("q", qo, want_q),
            ("k", ko, want_k),
            ("v", vo, v),
            ("gate", go, gate),
            ("cache_k", ck, rk),
            ("cache_v", cv, rv),
        ):
            assert R.same_bytes(got, want), (
                f"qkv_proj {name} differs from the torch reference"
            )
