#!/usr/bin/env python3
"""TEMP (#12 clean capture) — pristine 3-way offline comparison harness.

Loads the CLEAN capture written by the hardened SGLANG_FLYDSL_CAPTURE hook
(recompute-verify-before-save, all-128-head self-consistent) and compares three
outputs on identical real tensors:

  * o_prod   : clean PRODUCTION (FlashMLA/Triton) output captured in the server.
  * o_ref    : our pure-PyTorch reference `_torch_reference_dual_scope`.
  * o_kernel : our FlyDSL kernel launcher `_flydsl_dual_scope_kernel_impl`.

Reports, for each pairing, OVERALL cosine + max_abs AND PER-HEAD min cosine
across ALL 128 heads, plus the list of any heads below threshold (to confirm the
prior [16:32]u[48:64] corruption is gone).

OOM guard: the reference builds a dense gather, so it runs on a SUBSET of query
tokens. We pick the tokens with the LARGEST extra_topk_length so the C4 (extra)
scope dominates the comparison.

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


def _per_head(a: torch.Tensor, b: torch.Tensor, thr: float = 0.999) -> dict:
    """a,b are [N, H, Dv]. Per-head cosine over (N*Dv) for each head."""
    H = a.shape[1]
    af = a.detach().to(torch.float32).permute(1, 0, 2).reshape(H, -1)
    bf = b.detach().to(torch.float32).permute(1, 0, 2).reshape(H, -1)
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=1)
    bad = torch.nonzero(cos < thr).flatten().tolist()
    return {
        "min_head_cos": float(cos.min()),
        "argmin_head": int(cos.argmin()),
        "n_bad": len(bad),
        "bad_heads": bad,
        "H": H,
    }


def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    o = _overall(a, b)
    ph = _per_head(a, b)
    print(
        f"  {name:20s} cosine={o['cosine']:.6f}  max_abs={o['max_abs']:.4e}  "
        f"per_head_min_cos={ph['min_head_cos']:.6f}@h{ph['argmin_head']}  "
        f"bad_heads(<0.999)={ph['n_bad']}/{ph['H']}"
    )
    if ph["n_bad"]:
        print(f"      bad_heads list: {ph['bad_heads']}")


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
        return v.to(dev) if isinstance(v, torch.Tensor) else v

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
    print(f"  compress_ratio   : {compress_ratio}")
    print(f"  softmax_scale    : {softmax_scale}")
    print(f"  head_dim_v       : {head_dim_v}")
    print(f"  T_full={T_full}  H={H}")
    print(f"  verify_min_head_cos (capture-time): {cap.get('verify_min_head_cos')}")
    print(f"  capture_match_idx: {cap.get('capture_match_idx')}  skip={cap.get('capture_skip')}")
    if isinstance(extra_topk_length, torch.Tensor):
        print(
            f"  extra_topk_length: min={int(extra_topk_length.min())} "
            f"max={int(extra_topk_length.max())} mean={float(extra_topk_length.float().mean()):.1f}"
        )
    if isinstance(swa_topk_length, torch.Tensor):
        print(
            f"  swa_topk_length  : min={int(swa_topk_length.min())} "
            f"max={int(swa_topk_length.max())}"
        )

    if o_prod.shape[0] > T_full:
        o_prod = o_prod[:T_full]

    # --- choose subset: tokens with LARGEST extra_topk_length (C4 dominates). ---
    n = min(SUBSET, T_full)
    if isinstance(extra_topk_length, torch.Tensor) and extra_topk_length.shape[0] == T_full:
        order = torch.argsort(extra_topk_length, descending=True)
        sel = order[:n].sort().values  # keep ascending token order for readability
    else:
        sel = torch.arange(n, device=dev)
    sel = sel.to(dev)
    print(f"\n[subset] N={n} of T_full={T_full}, tokens chosen by largest extra_topk_length")
    if isinstance(extra_topk_length, torch.Tensor) and extra_topk_length.shape[0] == T_full:
        print(f"         selected extra_topk_length: min={int(extra_topk_length[sel].min())} "
              f"max={int(extra_topk_length[sel].max())}")

    def sel_tok(t):
        if isinstance(t, torch.Tensor) and t.shape[0] == T_full:
            return t.index_select(0, sel).contiguous()
        return t

    q_s = sel_tok(q)
    swa_indices_s = sel_tok(swa_indices)
    swa_topk_length_s = sel_tok(swa_topk_length)
    extra_indices_s = sel_tok(extra_indices)
    extra_topk_length_s = sel_tok(extra_topk_length)
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
        print(f"       o_ref shape={tuple(o_ref.shape)} dtype={o_ref.dtype}")
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        print(f"       reference OOM at N={n}: {e}\n       retry smaller FLYDSL_COMPARE_SUBSET")
        return 3
    except Exception as e:
        import traceback
        print(f"       reference FAILED: {e}\n{traceback.format_exc()}")

    print("[run] _flydsl_dual_scope_kernel_impl (full set, sliced to subset)")
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
        print(f"       o_kernel shape={tuple(o_kernel.shape)} dtype={o_kernel.dtype}")
    except Exception as e:
        import traceback
        print(f"       kernel FAILED: {e}\n{traceback.format_exc()}")

    print(f"\n=== PRISTINE 3-WAY COMPARISON (subset N={n}, all {H} heads) ===")
    if o_ref is not None:
        _report("o_ref    vs o_prod", o_ref, o_prod_s)
    if o_kernel_s is not None:
        _report("o_kernel vs o_prod", o_kernel_s, o_prod_s)
    if o_ref is not None and o_kernel_s is not None:
        _report("o_kernel vs o_ref", o_kernel_s, o_ref)

    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
