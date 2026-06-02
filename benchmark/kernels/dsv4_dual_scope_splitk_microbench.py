"""Standalone micro-benchmark: non-split vs split-K dual-scope DSV4 attention.

Goal
----
The MI355 prefill attention runs the Triton NON-split dual-scope kernel
(`_fused_gather_attn_dsv4_dual_scope_kernel`), because the dispatcher forces
`force_no_splitk=True` for large batches.  B200 prefill uses a split-KV MLA
flash kernel and is ~2.3x faster.  This script measures whether the *existing*
Triton split-K dual-scope path (`_fused_gather_attn_dsv4_dual_scope_splitk_kernel`
+ combine) is actually faster than the non-split kernel for prefill shapes,
*before* touching the dispatcher / routing or porting split-K into FlyDSL.

It reuses `build_synthetic_dual_scope` from the dual-scope prefill test, which
emits inputs in the exact paged-pool byte layout the Triton entrypoint expects.

Each configuration times:
  * non-split   (force_no_splitk=True)
  * split_k=2/4/8 (force_no_splitk=False, with `_decide_splitk_dual_scope`
    monkeypatched to the requested split count so we can sweep)
  * the production auto decision (force_no_splitk=False, untouched heuristic)

Correctness: every split-K output is cosine-compared against the non-split
output (the established reference path).  ~1.0 means split-K is numerically
equivalent.

Run on the box (inside the dsv4 container):
    python benchmark/kernels/dsv4_dual_scope_splitk_microbench.py
    python benchmark/kernels/dsv4_dual_scope_splitk_microbench.py \
        --tokens 1024 2048 4096 --topk-main 256 --topk-extra 512 --iters 50
"""

from __future__ import annotations

import argparse
import contextlib
from typing import List, Optional, Tuple

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


@contextlib.contextmanager
def _forced_split_k(value: Optional[int]):
    """Temporarily pin `_decide_splitk_dual_scope` to a constant split count.

    value=None leaves the production heuristic untouched.
    """
    if value is None:
        yield
        return
    orig = fused._decide_splitk_dual_scope
    fused._decide_splitk_dual_scope = lambda *a, **k: value
    try:
        yield
    finally:
        fused._decide_splitk_dual_scope = orig


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
    )
    # Triton entrypoint layout: q [T, H, d_qk], indices [T, topk].
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


def _run(inp: dict, force_no_splitk: bool) -> torch.Tensor:
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
        force_no_splitk=force_no_splitk,
    )
    return out


def _time(fn, iters: int, warmup: int) -> float:
    """Return median ms over `iters` timed runs after `warmup` runs."""
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


def _bench_config(
    T: int,
    topk_main: int,
    topk_extra: int,
    splits: List[int],
    iters: int,
    warmup: int,
    full_topk: bool,
    block_size_main: int,
    block_size_extra: int,
    num_blocks_main: int,
    num_blocks_extra: int,
) -> None:
    H = 128
    total_topk = topk_main + topk_extra
    inp = _build_inputs(
        T, H, topk_main, topk_extra, block_size_main, block_size_extra,
        num_blocks_main, num_blocks_extra, full_topk, "cuda",
    )

    print(
        f"\n=== T={T}  H={H}  topk={topk_main}+{topk_extra}={total_topk}  "
        f"full_topk={full_topk} ==="
    )
    auto_decision = fused._decide_splitk_dual_scope(T, H, total_topk)
    print(f"  production heuristic would pick split_k = {auto_decision}")

    # Reference: non-split path.
    ref = _run(inp, force_no_splitk=True)
    ns_ms = _time(lambda: _run(inp, force_no_splitk=True), iters, warmup)
    print(
        f"  {'non-split':<16}  {ns_ms:8.3f} ms   "
        f"{1e3 / ns_ms * T:10.1f} tok/s   (reference)"
    )

    for sk in splits:
        try:
            with _forced_split_k(sk):
                out = _run(inp, force_no_splitk=False)
                ms = _time(
                    lambda: _run(inp, force_no_splitk=False), iters, warmup
                )
        except Exception as e:  # combine kernel may reject some split values
            print(f"  split_k={sk:<9}  FAILED: {type(e).__name__}: {e}")
            continue
        cos = _cosine(out, ref)
        speedup = ns_ms / ms
        partial_gib = sk * T * 128 * 512 * 4 / (1024**3)
        print(
            f"  split_k={sk:<9}  {ms:8.3f} ms   {1e3 / ms * T:10.1f} tok/s   "
            f"cos={cos:.5f}   {speedup:5.2f}x   partial={partial_gib:5.2f}GiB"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--topk-main", type=int, default=256)
    ap.add_argument("--topk-extra", type=int, default=512)
    ap.add_argument("--splits", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--full-topk",
        action="store_true",
        help="ignore topk_length (every slot valid) for worst-case work",
    )
    ap.add_argument("--block-size-main", type=int, default=256)
    ap.add_argument("--block-size-extra", type=int, default=64)
    ap.add_argument("--num-blocks-main", type=int, default=64)
    ap.add_argument("--num-blocks-extra", type=int, default=512)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required (run inside the dsv4 container)")

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"sweep: tokens={args.tokens}  topk={args.topk_main}+{args.topk_extra}  "
        f"splits={args.splits}  iters={args.iters}  full_topk={args.full_topk}"
    )

    for T in args.tokens:
        _bench_config(
            T,
            args.topk_main,
            args.topk_extra,
            args.splits,
            args.iters,
            args.warmup,
            args.full_topk,
            args.block_size_main,
            args.block_size_extra,
            args.num_blocks_main,
            args.num_blocks_extra,
        )


if __name__ == "__main__":
    main()
