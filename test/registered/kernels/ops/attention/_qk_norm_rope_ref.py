"""Torch reference for kernels/ops/attention/qk_norm_rope_cache.py (Muse Spark, head_dim 64).

It mirrors the kernels' arithmetic step by step: per-head RMS norm in FP32 with the BF16 round trip of the norm output,
the query scale with its own BF16 round trip, interleaved RoPE from a [positions, 64] FP32 table (cos | sin halves,
loaded as BF16) with both products rounded to BF16 before the FP32 add, and the FP8 E4M3 cache store of K / V
(divided by the K / V scales when given) at ``locations``, optionally through the full -> SWA slot mapping, skipping
-1 at either level.

Exactness. The only reduction in these kernels is the per-head sum of squares (and, on the projected path, the split-K
sum of the QKV + gate GEMM). ``exact_*`` below draw operands from small dyadic grids so that every product and every
partial sum is exactly representable in FP32: the result then does not depend on the summation order (RMS input sums
are computed here in FP64 and are exact in FP32 by construction). The one step that is not bit-reproducible across
implementations is rsqrt (one FP32 ulp apart for a few heads on gfx950), so the check runs in two stages: stage 1
compares a kernel's norm-only output (RoPE off, query scale 1.0) with ``norm_ref`` byte for byte except at BF16
rounding near-ties (``check_norm``); stage 2 takes that checked norm output through ``finish_ref`` (query scale,
RoPE) and ``cache_ref`` (FP8 store) and requires the kernel's full outputs and caches to be byte-equal.
"""

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(
    est_time=0,
    suite="base-b-test-1-gpu-small-amd-mi35x",
    disabled="torch reference helper for test_qk_norm_rope_cache_ref.py",
)

D = 64


def exact_heads(m, heads, gen, device="cuda"):
    """[m, heads * 64] BF16 with entries k / 32, |k| <= 128: squares are multiples of 2^-10 below 16, so a head's
    sum of 64 squares (< 1024) is exact in FP32 in any order."""
    k = torch.randint(-128, 129, (m, heads * D), generator=gen, device=device)
    return (k.float() / 32).bfloat16()


def exact_proj_operands(m, n, k_dim, gen, device="cuda"):
    """x [m, k_dim] with 4 nonzeros per row in {+-1, +-0.5, +-0.25} and w [n, k_dim] = j / 16, |j| <= 16 (BF16):
    every projected value is a sum of <= 4 multiples of 2^-6 below 1 in magnitude, so the split-K sums are exact in
    FP32, the values are exact in BF16, and the RMS sums over 64 of them stay exact (multiples of 2^-12 below 1024)."""
    x = torch.zeros(m, k_dim, device=device)
    vals = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25, -0.25], device=device)
    for r in range(m):
        cols = torch.randperm(k_dim, generator=gen, device=device)[:4]
        x[r, cols] = vals[
            torch.randint(0, len(vals), (4,), generator=gen, device=device)
        ]
    w = torch.randint(-16, 17, (n, k_dim), generator=gen, device=device).float() / 16
    return x.bfloat16(), w.bfloat16()


def proj_ref(x, w):
    """x @ w.T exactly (FP64 on the host; exact by construction of the operands), as BF16."""
    y = x.double().cpu() @ w.double().cpu().T
    return y.float().to(x.device).bfloat16()


def _rope(value, positions, cos_sin):
    """Interleaved RoPE of value [m, heads, 64] (FP32, BF16-exact) as the kernels compute it."""
    idx = torch.arange(D, device=value.device) // 2
    table = cos_sin[positions.long()]  # [m, 64] FP32: cos | sin
    cosine = table[:, idx].bfloat16().float()[:, None, :]
    sine = table[:, D // 2 + idx].bfloat16().float()[:, None, :]
    pairs = value.reshape(*value.shape[:-1], D // 2, 2)
    other = pairs.flip(-1).reshape(value.shape)
    even = torch.arange(D, device=value.device) % 2 == 0
    rotated = torch.where(even, -other, other)
    first = (value * cosine).bfloat16().float()
    second = (rotated * sine).bfloat16().float()
    return first + second


def norm_rope_ref(x, positions, cos_sin, eps, scale=None):
    """RMS norm per 64-wide head (+ the query scale when ``scale`` is given) + RoPE when ``cos_sin`` is given; BF16."""
    m, width = x.shape
    f = x.float().reshape(m, width // D, D)
    ss = (
        (f.double() * f.double()).sum(-1, keepdim=True).float()
    )  # exact in FP32 for exact_* operands
    inv = torch.rsqrt(ss / D + eps)
    value = (f * inv).bfloat16().float()
    if scale is not None:
        value = (
            (value * torch.tensor(float(scale), dtype=torch.float32, device=x.device))
            .bfloat16()
            .float()
        )
    if cos_sin is not None:
        value = _rope(value, positions, cos_sin)
    return value.bfloat16().reshape(m, width)


def norm_ref(x, eps):
    """The per-head RMS norm output (BF16) and the mask of near-tie elements. The kernels' rsqrt and torch's may differ
    by one FP32 ulp for some heads (measured on gfx950: 2 of 9,600 heads in one 300-row case), which can move the FP32 product
    x * rsqrt across a BF16 rounding tie; on these dyadic operands that happens only where the reference product lies
    within two FP32 ulps of a tie (a one-ulp rsqrt change moves the product by at most two ulps; low 16 bits in
    0x7ffe..0x8002), so those elements may differ by one BF16 ulp."""
    m, width = x.shape
    f = x.float().reshape(m, width // D, D)
    ss = (f.double() * f.double()).sum(-1, keepdim=True).float()
    prod = f * torch.rsqrt(ss / D + eps)
    low = prod.reshape(m, width).view(torch.int32) & 0xFFFF
    return prod.bfloat16().reshape(m, width), (low >= 0x7FFE) & (low <= 0x8002)


def check_norm(got, x, eps):
    """Stage 1: a kernel's norm-only output (RoPE off, query scale 1.0) equals norm_ref byte for byte, except near-tie
    elements, which may differ by one BF16 ulp. Returns the number of such one-ulp elements."""
    ref, near_tie = norm_ref(x, eps)
    d = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
    assert bool(torch.all(d[~near_tie] == 0)), (
        "norm output differs away from a BF16 tie"
    )
    assert bool(torch.all(d[near_tie] <= 1)), (
        "norm output differs by more than one BF16 ulp at a tie"
    )
    return int((d > 0).sum())


def finish_ref(norm_bf16, positions, cos_sin, scale=None):
    """Stage 2: from a (checked) norm output, the query scale with its BF16 round trip and RoPE, as the kernels do."""
    m, width = norm_bf16.shape
    value = norm_bf16.float().reshape(m, width // D, D)
    if scale is not None:
        value = (
            (
                value
                * torch.tensor(float(scale), dtype=torch.float32, device=value.device)
            )
            .bfloat16()
            .float()
        )
    if cos_sin is not None:
        value = _rope(value, positions, cos_sin)
    return value.bfloat16().reshape(m, width)


def cache_ref(cache_k, cache_v, k, v, locations, swa_mapping, k_scale, v_scale):
    """Store K / V [m, hk * 64] (BF16) into FP8 caches [slots, hk, 64] in place, as the kernels do."""
    m = k.shape[0]
    hk = k.shape[1] // D
    for t in range(m):
        slot = int(locations[t])
        if slot < 0:
            continue
        if swa_mapping is not None:
            slot = int(swa_mapping[slot])
            if slot < 0:
                continue
        kv = k[t].float().reshape(hk, D)
        vv = v[t].float().reshape(hk, D)
        if k_scale is not None:
            kv = kv / k_scale.float()
            vv = vv / v_scale.float()
        cache_k[slot] = kv.to(torch.float8_e4m3fn)
        cache_v[slot] = vv.to(torch.float8_e4m3fn)


def same_bytes(a, b):
    if a.dtype == torch.float8_e4m3fn:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a, b)
