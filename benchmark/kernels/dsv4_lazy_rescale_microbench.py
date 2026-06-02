"""Micro-benchmark: D4 lazy online-softmax O-rescale OFF vs ON (non-split kernel).

Measures the Triton NON-split dual-scope kernel
(`_fused_gather_attn_dsv4_dual_scope_kernel`, the path that actually runs for
prefill/large-batch because the dispatcher forces `force_no_splitk=True`) with
the deferred / lazy O-accumulator rescale (D4) toggled via env
`SGLANG_TRITON_LAZY_RESCALE`.

The flag is a compile-time constant baked into the Triton kernel at trace time,
so OFF vs ON must be compared across two *separate* processes (one per env
value).  Inputs are fully seeded, so the two processes build identical inputs;
the OFF run dumps its outputs and the ON run loads them to report cosine.

Run on the box (inside the dsv4 container), from /sgl-workspace/squidward:
    # OFF reference + timings
    SGLANG_TRITON_LAZY_RESCALE=0 python3 benchmark/kernels/dsv4_lazy_rescale_microbench.py \
        --dump /tmp/d4_off.pt
    # ON timings + cosine vs OFF
    SGLANG_TRITON_LAZY_RESCALE=1 python3 benchmark/kernels/dsv4_lazy_rescale_microbench.py \
        --ref /tmp/d4_off.pt
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import torch

from sglang.srt.layers.attention.nsa.triton_decode import (
    triton_mla_kernels_decode_fused as fused,
)
from sglang.srt.layers.attention.nsa.triton_decode.triton_mla_kernels_decode_fused import (
    fused_gather_attn_decode_dsv4_dual_scope,
)
from sglang.srt.layers.attention.nsa.test_flydsl_dual_scope_prefill import (
    build_synthetic_dual_scope,
)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0
    return float((a @ b) / denom)


def _build_inputs(
    T: int,
    H: int,
    topk_main: int,
    topk_extra: int,
    block_size_main: int,
    block_size_extra: int,
    num_blocks_main: int,
    num_blocks_extra: int,
    full_topk: bool,
    device: str,
):
    s = build_synthetic_dual_scope(
        T=T,
        H=H,
        topk_main=topk_main,
        topk_extra=topk_extra,
        block_size_main=block_size_main,
        block_size_extra=block_size_extra,
        num_blocks_main=num_blocks_main,
        num_blocks_extra=num_blocks_extra,
        compress_ratio=4,
        with_extra=True,
        with_attn_sink=True,
        with_topk_length=not full_topk,
        device=device,
        seed=0,
    )
    q = s.q.squeeze(1).contiguous()
    indices_main = s.swa_indices.squeeze(1).contiguous()
    indices_extra = s.extra_indices.squeeze(1).contiguous()
    return dict(
        q=q,
        kv_cache_main=s.swa_k_cache,
        indices_main=indices_main,
        block_size_main=block_size_main,
        kv_cache_extra=s.extra_k_cache,
        indices_extra=indices_extra,
        block_size_extra=block_size_extra,
        sm_scale=s.softmax_scale,
        topk_length_main=s.swa_topk_length,
        topk_length_extra=s.extra_topk_length,
        attn_sink=s.attn_sink,
    )


def _run(inp: dict) -> torch.Tensor:
    out, _ = fused_gather_attn_decode_dsv4_dual_scope(
        inp["q"],
        inp["kv_cache_main"],
        inp["indices_main"],
        inp["block_size_main"],
        inp["kv_cache_extra"],
        inp["indices_extra"],
        inp["block_size_extra"],
        inp["sm_scale"],
        topk_length_main=inp["topk_length_main"],
        topk_length_extra=inp["topk_length_extra"],
        attn_sink=inp["attn_sink"],
        s_q=1,
        force_no_splitk=True,
    )
    return out


def _time(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokens", type=int, nargs="+", default=[8, 32, 512, 1024, 4096]
    )
    ap.add_argument("--topk-main", type=int, default=256)
    ap.add_argument("--topk-extra", type=int, default=512)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--full-topk", action="store_true")
    ap.add_argument("--block-size-main", type=int, default=256)
    ap.add_argument("--block-size-extra", type=int, default=64)
    ap.add_argument("--num-blocks-main", type=int, default=64)
    ap.add_argument("--num-blocks-extra", type=int, default=512)
    ap.add_argument("--dump", type=str, default=None, help="save outputs to path")
    ap.add_argument("--ref", type=str, default=None, help="load ref outputs for cosine")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required (run inside the dsv4 container)")

    state = "ON" if fused._TRITON_LAZY_RESCALE else "OFF"
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"D4 lazy rescale (SGLANG_TRITON_LAZY_RESCALE): {state}")
    print(
        f"sweep: tokens={args.tokens}  topk={args.topk_main}+{args.topk_extra}  "
        f"iters={args.iters}  full_topk={args.full_topk}"
    )

    H = 128
    ref_store = {}
    if args.ref:
        ref_store = torch.load(args.ref)

    dump_store = {}
    print(f"\n{'T':>6}  {'ms':>9}  {'tok/s':>11}  {'cosine_vs_OFF':>14}")
    for T in args.tokens:
        inp = _build_inputs(
            T, H, args.topk_main, args.topk_extra,
            args.block_size_main, args.block_size_extra,
            args.num_blocks_main, args.num_blocks_extra,
            args.full_topk, "cuda",
        )
        out = _run(inp)
        ms = _time(lambda: _run(inp), args.iters, args.warmup)
        cos_str = ""
        if args.ref and T in ref_store:
            cos_str = f"{_cosine(out, ref_store[T].cuda()):.6f}"
        if args.dump:
            dump_store[T] = out.detach().cpu()
        print(f"{T:>6}  {ms:9.4f}  {1e3 / ms * T:11.1f}  {cos_str:>14}")

    if args.dump:
        torch.save(dump_store, args.dump)
        print(f"\nsaved outputs to {args.dump}")


if __name__ == "__main__":
    main()
