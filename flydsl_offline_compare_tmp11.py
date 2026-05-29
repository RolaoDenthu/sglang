#!/usr/bin/env python3
"""TEMP (#11 real-data cross-check) — offline comparison harness.

Loads /tmp/flydsl_capture.pt (captured from a REAL DeepSeek-V4 prefill via the
SGLANG_FLYDSL_CAPTURE instrumentation), reconstructs the EXACT attention inputs
on GPU with their captured dtypes, and compares three outputs on identical real
tensors:

  * o_prod   : the PRODUCTION (FlashMLA/Triton) output captured in the server.
  * o_ref    : our pure-PyTorch reference `_torch_reference_dual_scope`.
  * o_kernel : our FlyDSL kernel launcher `_flydsl_dual_scope_kernel_impl`.

Pairings reported (cosine / max_abs / max_rel):
  o_ref    vs o_prod    <- THE foundation check (is our understanding correct?)
  o_kernel vs o_prod
  o_kernel vs o_ref

OOM guard: the reference builds a dense [T, topk, 512] gather, which can OOM at
real T (up to 8192) x real topk (~2048). So the reference + comparison run on a
SUBSET of the first N query tokens (default 256). The kernel runs on the FULL
set (cheap) and is sliced to the same subset for comparison.

Run in-container:
    PYTHONPATH=/sgl-workspace/squidward/python python flydsl_offline_compare_tmp11.py
"""

from __future__ import annotations

import os
import sys

import torch

CAPTURE_PATH = os.environ.get("FLYDSL_CAPTURE_PATH", "/tmp/flydsl_capture.pt")
SUBSET = int(os.environ.get("FLYDSL_COMPARE_SUBSET", "256"))


def _metrics(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> dict:
    """Cosine + max_abs + max_rel between two tensors (float32, flattened)."""
    af = a.detach().to(torch.float32).reshape(-1)
    bf = b.detach().to(torch.float32).reshape(-1)
    assert af.shape == bf.shape, f"shape mismatch {af.shape} vs {bf.shape}"
    cos = torch.nn.functional.cosine_similarity(
        af.unsqueeze(0), bf.unsqueeze(0), dim=1
    ).item()
    diff = (af - bf).abs()
    max_abs = diff.max().item()
    denom = bf.abs().clamp_min(eps)
    max_rel = (diff / denom).max().item()
    return {
        "cosine": cos,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "a_mean_abs": af.abs().mean().item(),
        "b_mean_abs": bf.abs().mean().item(),
    }


def _report(name: str, m: dict) -> None:
    print(
        f"  {name:24s} cosine={m['cosine']:.6f}  max_abs={m['max_abs']:.4e}  "
        f"max_rel={m['max_rel']:.4e}  |a|~{m['a_mean_abs']:.4e}  |b|~{m['b_mean_abs']:.4e}"
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; this must run in the gfx950 container.")
        return 2

    dev = torch.device("cuda")
    print(f"[load] {CAPTURE_PATH}")
    cap = torch.load(CAPTURE_PATH, map_location="cpu")

    from sglang.srt.layers.attention.nsa.flydsl_dual_scope_prefill import (
        _flydsl_dual_scope_kernel_impl,
        _torch_reference_dual_scope,
    )

    def g(key):
        v = cap.get(key)
        if isinstance(v, torch.Tensor):
            return v.to(dev)
        return v

    q = g("q")
    swa_k_cache = g("swa_k_cache")
    swa_indices = g("swa_indices")
    swa_topk_length = g("swa_topk_length")
    extra_k_cache = g("extra_k_cache")
    extra_indices = g("extra_indices")
    extra_topk_length = g("extra_topk_length")
    compress_ratio = int(cap["compress_ratio"])
    softmax_scale = float(cap["softmax_scale"])
    attn_sink = g("attn_sink")
    head_dim_v = int(cap["head_dim_v"])
    o_prod = g("o_prod")

    # --- T (token count) from q: q is [T, 1, H, D] or [T, H, D]. ---
    T_full = q.shape[0]
    H = q.shape[-2]

    def shp(t):
        return tuple(t.shape) if isinstance(t, torch.Tensor) else t

    def dt(t):
        return str(t.dtype) if isinstance(t, torch.Tensor) else type(t).__name__

    print("\n=== CAPTURE SHAPES / DTYPES ===")
    print(f"  q                : {shp(q)}  {dt(q)}")
    print(f"  swa_k_cache      : {shp(swa_k_cache)}  {dt(swa_k_cache)}")
    print(f"  swa_indices      : {shp(swa_indices)}  {dt(swa_indices)}")
    print(f"  swa_topk_length  : {shp(swa_topk_length)}  {dt(swa_topk_length)}")
    print(f"  extra_k_cache    : {shp(extra_k_cache)}  {dt(extra_k_cache)}")
    print(f"  extra_indices    : {shp(extra_indices)}  {dt(extra_indices)}")
    print(f"  extra_topk_length: {shp(extra_topk_length)}  {dt(extra_topk_length)}")
    print(f"  attn_sink        : {shp(attn_sink)}  {dt(attn_sink)}")
    print(f"  compress_ratio   : {compress_ratio}")
    print(f"  softmax_scale    : {softmax_scale}")
    print(f"  head_dim_v       : {head_dim_v}")
    print(f"  o_prod           : {shp(o_prod)}  {dt(o_prod)}")
    print(f"  T_full={T_full}  H={H}")
    if isinstance(swa_topk_length, torch.Tensor):
        print(
            f"  swa_topk_length stats: min={int(swa_topk_length.min())} "
            f"max={int(swa_topk_length.max())} shape0={swa_topk_length.shape[0]} "
            f"(==T_full? {swa_topk_length.shape[0] == T_full})"
        )
    if isinstance(extra_topk_length, torch.Tensor):
        print(
            f"  extra_topk_length stats: min={int(extra_topk_length.min())} "
            f"max={int(extra_topk_length.max())} shape0={extra_topk_length.shape[0]}"
        )

    # --- align o_prod to q's T (production output may be padded). ---
    if o_prod.shape[0] > T_full:
        print(f"\n[align] slicing o_prod rows {o_prod.shape[0]} -> {T_full}")
        o_prod = o_prod[:T_full]

    # --- subset slicing for the reference (OOM guard). ---
    def slice_tok(t, n):
        """Slice token dim 0 only when the tensor is per-token (dim0 == T_full)."""
        if isinstance(t, torch.Tensor) and t.shape[0] == T_full:
            return t[:n].contiguous()
        return t

    n = min(SUBSET, T_full)
    print(f"\n[subset] using first N={n} of T_full={T_full} query tokens for the reference\n")

    q_s = slice_tok(q, n)
    swa_indices_s = slice_tok(swa_indices, n)
    swa_topk_length_s = slice_tok(swa_topk_length, n)
    extra_indices_s = slice_tok(extra_indices, n)
    extra_topk_length_s = slice_tok(extra_topk_length, n)
    o_prod_s = o_prod[:n].contiguous()

    common = dict(
        swa_k_cache=swa_k_cache,
        compress_ratio=compress_ratio,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        head_dim_v=head_dim_v,
        extra_k_cache=extra_k_cache,
    )

    # ---- reference on subset ----
    print("[run] _torch_reference_dual_scope (subset)")
    o_ref = None
    try:
        with torch.no_grad():
            o_ref = _torch_reference_dual_scope(
                q=q_s,
                swa_indices=swa_indices_s,
                swa_topk_length=swa_topk_length_s,
                extra_indices=extra_indices_s,
                extra_topk_length=extra_topk_length_s,
                **common,
            )
        o_ref = o_ref.reshape(n, H, head_dim_v)
        print(f"       o_ref shape={tuple(o_ref.shape)} dtype={o_ref.dtype}")
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        print(f"       reference OOM at N={n}: {e}\n       retry with smaller FLYDSL_COMPARE_SUBSET")
        return 3
    except Exception as e:
        import traceback

        print(f"       reference FAILED: {e}\n{traceback.format_exc()}")

    # ---- kernel on FULL set, sliced to subset ----
    print("[run] _flydsl_dual_scope_kernel_impl (full set)")
    o_kernel_s = None
    try:
        with torch.no_grad():
            o_kernel = _flydsl_dual_scope_kernel_impl(
                q=q,
                swa_indices=swa_indices,
                swa_topk_length=swa_topk_length,
                extra_indices=extra_indices,
                extra_topk_length=extra_topk_length,
                **common,
            )
        o_kernel = o_kernel.reshape(T_full, H, head_dim_v)
        o_kernel_s = o_kernel[:n].contiguous()
        print(
            f"       o_kernel shape={tuple(o_kernel.shape)} dtype={o_kernel.dtype} "
            f"(nonzero frac={ (o_kernel.float().abs() > 0).float().mean().item():.4f })"
        )
    except Exception as e:
        import traceback

        print(f"       kernel FAILED: {e}\n{traceback.format_exc()}")

    # ---- comparisons ----
    print("\n=== COMPARISON (subset N={}) ===".format(n))
    if o_ref is not None:
        _report("o_ref    vs o_prod", _metrics(o_ref, o_prod_s))
    if o_kernel_s is not None:
        _report("o_kernel vs o_prod", _metrics(o_kernel_s, o_prod_s))
    if o_ref is not None and o_kernel_s is not None:
        _report("o_kernel vs o_ref", _metrics(o_kernel_s, o_ref))

    # ---- per-row scale/offset probe (o_ref vs o_prod), a layout sanity check. ----
    if o_ref is not None:
        rf = o_ref.to(torch.float32).reshape(n, -1)
        pf = o_prod_s.to(torch.float32).reshape(n, -1)
        # per-row best-fit scale a = <r,p>/<p,p>
        num = (rf * pf).sum(dim=1)
        den = (pf * pf).sum(dim=1).clamp_min(1e-12)
        a = (num / den)
        print(
            f"\n[probe] per-row best-fit scale o_ref~a*o_prod: a mean={a.mean().item():.4f} "
            f"std={a.std().item():.4f} min={a.min().item():.4f} max={a.max().item():.4f}"
        )
        zero_ref = (rf.abs().sum(dim=1) == 0)
        zero_prod = (pf.abs().sum(dim=1) == 0)
        print(
            f"[probe] zero rows: o_ref={int(zero_ref.sum())} o_prod={int(zero_prod.sum())} "
            f"(of {n})"
        )

    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
