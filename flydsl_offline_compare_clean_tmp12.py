#!/usr/bin/env python3
"""TEMP (#12 clean capture) — pristine 3-way per-head comparison harness.

Loads the CLEAN warm C4 capture (default /tmp/flydsl_capture_c4_clean.pt) and
compares three outputs on identical real tensors:

  * o_prod   : the PRODUCTION (FlashMLA/Triton) WARM output captured in server.
  * o_ref    : our pure-PyTorch reference `_torch_reference_dual_scope`.
  * o_kernel : our FlyDSL kernel launcher `_flydsl_dual_scope_kernel_impl`.

For each pairing reports overall cosine / max_abs AND per-head min cosine across
ALL heads, so we can confirm the prior [16:32]u[48:64] 32-head corruption is gone.

OOM guard: the reference builds a dense gather, so it runs on a SUBSET of query
tokens. We pick the tokens with the LARGEST extra_topk_length (C4 dominates).

Run in-container:
    PYTHONPATH=/sgl-workspace/squidward/python python flydsl_offline_compare_clean_tmp12.py
"""

from __future__ import annotations

import os
import sys

import torch

CAPTURE_PATH = os.environ.get("FLYDSL_CAPTURE_PATH", "/tmp/flydsl_capture_c4_clean.pt")
SUBSET = int(os.environ.get("FLYDSL_COMPARE_SUBSET", "256"))


def _overall(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> dict:
    af = a.detach().to(torch.float32).reshape(-1)
    bf = b.detach().to(torch.float32).reshape(-1)
    assert af.shape == bf.shape, f"shape mismatch {af.shape} vs {bf.shape}"
    cos = torch.nn.functional.cosine_similarity(
        af.unsqueeze(0), bf.unsqueeze(0), dim=1
    ).item()
    diff = (af - bf).abs()
    return {"cosine": cos, "max_abs": diff.max().item()}


def _per_head(a: torch.Tensor, b: torch.Tensor):
    """a,b: [N, H, Dv]. Returns (per_head_cos[H], min_cos, n_bad, bad_heads)."""
    af = a.detach().to(torch.float32)
    bf = b.detach().to(torch.float32)
    H = af.shape[1]
    afh = af.permute(1, 0, 2).reshape(H, -1)
    bfh = bf.permute(1, 0, 2).reshape(H, -1)
    cos = torch.nn.functional.cosine_similarity(afh, bfh, dim=1)
    bad = torch.nonzero(cos < 0.999).flatten().tolist()
    return cos, float(cos.min()), len(bad), bad


def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    ov = _overall(a, b)
    cos, mn, nbad, bad = _per_head(a, b)
    print(
        f"  {name:20s} overall_cos={ov['cosine']:.6f}  max_abs={ov['max_abs']:.4e}  "
        f"per_head_min_cos={mn:.6f}  n_bad_heads(<0.999)={nbad}/{cos.shape[0]}"
    )
    if nbad:
        print(f"      bad heads: {bad}")


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

    T_full = q.shape[0]
    H = q.shape[-2]

    def shp(t):
        return tuple(t.shape) if isinstance(t, torch.Tensor) else t

    def dt(t):
        return str(t.dtype) if isinstance(t, torch.Tensor) else type(t).__name__

    print("\n=== CLEAN CAPTURE SHAPES / DTYPES ===")
    for nm, t in [
        ("q", q), ("swa_k_cache", swa_k_cache), ("swa_indices", swa_indices),
        ("swa_topk_length", swa_topk_length), ("extra_k_cache", extra_k_cache),
        ("extra_indices", extra_indices), ("extra_topk_length", extra_topk_length),
        ("attn_sink", attn_sink), ("o_prod", o_prod),
    ]:
        print(f"  {nm:18s}: {shp(t)}  {dt(t)}")
    print(f"  compress_ratio={compress_ratio}  softmax_scale={softmax_scale}  "
          f"head_dim_v={head_dim_v}  T_full={T_full}  H={H}")
    print(f"  capture meta: match_idx={cap.get('capture_match_idx')} "
          f"skip={cap.get('capture_skip')} "
          f"verify_min_head_cos={cap.get('verify_min_head_cos')}")
    if isinstance(extra_topk_length, torch.Tensor):
        print(f"  extra_topk_length: min={int(extra_topk_length.min())} "
              f"max={int(extra_topk_length.max())} shape0={extra_topk_length.shape[0]}")
    if isinstance(swa_topk_length, torch.Tensor):
        print(f"  swa_topk_length:   min={int(swa_topk_length.min())} "
              f"max={int(swa_topk_length.max())} shape0={swa_topk_length.shape[0]}")

    if o_prod.shape[0] > T_full:
        o_prod = o_prod[:T_full]

    # pick the SUBSET tokens with the largest extra_topk_length (C4 dominates).
    n = min(SUBSET, T_full)
    if isinstance(extra_topk_length, torch.Tensor) and extra_topk_length.shape[0] == T_full:
        order = torch.argsort(extra_topk_length.to(torch.int64), descending=True)
        sel = order[:n].sort().values.to(dev)
        print(f"\n[subset] selecting {n} tokens with largest extra_topk_length "
              f"(range {int(extra_topk_length[sel.cpu()].min())}.."
              f"{int(extra_topk_length[sel.cpu()].max())})")
    else:
        sel = torch.arange(n, device=dev)
        print(f"\n[subset] first {n} tokens (no per-token extra_topk_length)")

    def take(t):
        if isinstance(t, torch.Tensor) and t.shape[0] == T_full:
            return t.index_select(0, sel).contiguous()
        return t

    q_s = take(q)
    swa_indices_s = take(swa_indices)
    swa_topk_length_s = take(swa_topk_length)
    extra_indices_s = take(extra_indices)
    extra_topk_length_s = take(extra_topk_length)
    o_prod_s = o_prod.index_select(0, sel).contiguous().reshape(n, H, head_dim_v)

    common = dict(
        swa_k_cache=swa_k_cache,
        compress_ratio=compress_ratio,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        head_dim_v=head_dim_v,
        extra_k_cache=extra_k_cache,
    )

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
            ).reshape(n, H, head_dim_v)
        print(f"       o_ref {tuple(o_ref.shape)} {o_ref.dtype}")
    except Exception as e:
        import traceback
        print(f"       reference FAILED: {e}\n{traceback.format_exc()}")

    print("[run] _flydsl_dual_scope_kernel_impl (full set, sliced)")
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
            ).reshape(T_full, H, head_dim_v)
        o_kernel_s = o_kernel.index_select(0, sel).contiguous()
        print(f"       o_kernel {tuple(o_kernel.shape)} {o_kernel.dtype}")
    except Exception as e:
        import traceback
        print(f"       kernel FAILED: {e}\n{traceback.format_exc()}")

    print(f"\n=== 3-WAY COMPARISON (subset N={n}, all {H} heads) ===")
    if o_ref is not None:
        _report("o_ref    vs o_prod", o_ref, o_prod_s)
    if o_kernel_s is not None:
        _report("o_kernel vs o_prod", o_kernel_s, o_prod_s)
    if o_ref is not None and o_kernel_s is not None:
        _report("o_kernel vs o_ref ", o_kernel_s, o_ref)

    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
