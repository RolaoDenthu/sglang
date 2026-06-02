# FlyDSL dual-scope prefill — E2E serving sweep

DeepSeek-V4-Pro FP8 · tp8 · MI350X (gfx950) · input 8192 / output 1024 · `bench_serving` random.
Same-session A/B: **baseline** = `amd/deepseek_v4` branch (no FlyDSL); **flydsl** = branch `6cf5`
(specialized C128 `(128,64,8)` + transpose-read on).

## Baseline (`amd/deepseek_v4`, no FlyDSL)

| conc | TTFT med (ms) | TPOT med (ms) | ITL med (ms) | out tput (tok/s) |
|---:|---:|---:|---:|---:|
| 2  | 291.95  | 19.17 | 19.18 | 102.87 |
| 4  | 1645.41 | 19.15 | 19.16 | 141.79 |
| 8  | 3044.85 | 20.68 | 20.65 | 328.87 |
| 16 | 3668.75 | 23.02 | 22.93 | 552.75 |
| 32 | 10126.18 | 36.59 | 26.95 | 692.19 |
| 64 | 20908.94 | 48.36 | 32.58 | 762.29 |

## FlyDSL branch (`6cf5`)

| conc | TTFT med (ms) | TPOT med (ms) | ITL med (ms) | out tput (tok/s) |
|---:|---:|---:|---:|---:|
| 2  | 290.72   | 19.18 | 19.19 | 102.84 |
| 4  | 516.43   | 19.16 | 19.16 | 194.38 |
| 8  | 499.13   | 20.66 | 20.62 | 343.21 |
| 16 | 746.41   | 22.99 | 22.88 | 564.95 |
| 32 | 10954.32 | 36.35 | 26.91 | 680.47 |
| 64 | 24482.82 | 49.33 | 32.56 | 726.45 |

## TTFT comparison (baseline vs flydsl)

| conc | baseline TTFT (ms) | flydsl TTFT (ms) | delta |
|---:|---:|---:|---:|
| 2  | 291.95   | 290.72   | -0.4 % |
| 4  | 1645.41  | 516.43   | -68.6 % |
| 8  | 3044.85  | 499.13   | -83.6 % |
| 16 | 3668.75  | 746.41   | -79.7 % |
| 32 | 10126.18 | 10954.32 | +8.2 % |
| 64 | 20908.94 | 24482.82 | +17.1 % |

Negative = flydsl faster.

## Reading the sweep

FlyDSL wins decisively at **c = 2…16** (TTFT 3–6× lower, throughput equal-or-better),
then **regresses at c = 32 / 64** (TTFT +8 / +17 %, throughput −2 / −5 %). The crossover is
the latency-vs-throughput tradeoff baked into the kernel, confirmed by standalone profiling
below.

## Throughput-ceiling diagnosis (`dual_scope_prefill_kernel_c4_0`)

Standalone profile of the dominant high-topk path (C4: H=128, topk 256+512) on a free GPU,
`rocprofv3` kernel-trace + PMC, MI350X (256 CU · 160 KB LDS/CU · 32 waves/CU max).

**Per-token time is flat from T=512 → T=8192 — the kernel hits its throughput floor almost
immediately and a bigger prefill batch buys zero extra throughput:**

| T (tokens) | median | us/token | ktok/s |
|---:|---:|---:|---:|
| 256  | 0.32 ms | 1.25 | 800 |
| 512  | 0.56 ms | 1.09 | 914 |
| 1024 | 1.11 ms | 1.09 | 921 |
| 2048 | 2.22 ms | 1.08 | 924 |
| 4096 | 4.39 ms | 1.07 | 933 |
| 8192 | 8.82 ms | 1.08 | 929 |

That flat curve is a low-occupancy / latency-bound signature, not compute saturation. The
kernel resources explain why:

| Resource | Value | Occupancy cap |
|---|---|---|
| LDS / workgroup | **100 KB** | 160 / 100 → **1 CTA/CU** → 4 waves/CU = **12.5 %** (binding) |
| VGPR / wave | **256** (arch max) | 512 / 256 → 2 waves/SIMD = 25 % |
| Scratch / wave | 432 B | register spill present |
| Launch shape | 256-thread WG (4 waves), **1 CTA per token** | — |

- **Achieved occupancy ≈ 12.5 %** (4 of 32 waves/CU), pinned by the 100 KB LDS footprint. With
  only one CTA resident per CU there is no second CTA to hide its memory/LDS latency behind:
  `SQ_WAIT_INST_LDS` ≈ 4.5e7 stall cycles.
- **Compute is 86 % non-MFMA**: `SQ_INSTS_MFMA` = 5.0e7 vs `SQ_INSTS_VALU` = 3.0e8. The matrix
  units (the throughput engine) sit mostly idle while the wave grinds through
  softmax / index-gather / LDS traffic.
- GPU is time-active 100 % (`GRBM_GUI_ACTIVE == GRBM_COUNT`), but that only means a thin set of
  latency-bound waves is always spinning — not that throughput is saturated.

**Mechanism:** one CTA/token + 100 KB LDS + max VGPR give excellent single-request TTFT at
low/moderate concurrency, but at saturation the 12.5 %-occupancy wall caps the device at
~930 ktok/s for C4, while the baseline batched-GEMM path keeps MFMA fed and climbs past it —
exactly the c=32/64 result above.

## Occupancy experiments (kernel microbench, `dual_scope_prefill_kernel_c4_0`, T=4096)

Branch `amd/deepseek_v4-flydsl-prefill-highconc`. All flag-gated, OFF by default.

| variant | VGPR | scratch (B) | LDS/WG | cosine | ktok/s | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline bn64, G=1 (default) | 256 | 432 | 100 KB | 1.000000 | 931 | — |
| split-D bn64, G=2 | 196 | 0 | 100 KB | 0.999997 | <931 | regress |
| split-D bn64, G=4 | 196 | 0 | 100 KB | 0.999997 | <931 | regress |
| Path B: bn32 + split-D | — | 0 | 51 KB | bit-exact | 604 | −35 % |

Stall breakdown (standalone profile): barriers ≈ 70 % of stall cycles, LDS bank conflicts ≈ 39 %.

## Latency experiments (kernel microbench T=4096)

Done on branch `amd/deepseek_v4-flydsl-prefill-highconc` (deleted as a failed
experiment; all variants below were bit-exact but slower, so none landed in
mainline). Baseline to beat: bn64 G=1 f32 = 4.41 ms / 929 ktok/s, cos 0.999998.

| variant | flags | cos | VGPR | scratch | LDS | occupancy | MfmaUtil | ms | ktok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | — | 0.999998 | 256 | 432 | 100 KB | 3.83 w | 8.8% | 4.41 | 929 |
| bf16 P | `C4_P_BF16=1` | 0.999998 | 256 | 432 | 83 KB | — | — | 5.08 | 807 |
| bf16 P pad0 | `C4_P_BF16=1 KV_PAD=0 P_PAD=0` | 0.999998 | 256 | 432 | 80 KB | 3.84 w | — | 5.88 | 696 |
| barrier-merge | `C4_PMERGE=1` | 0.999998 | 256 | 868 | 100 KB | — | — | 6.07 | 674 |
| split-D + bf16P pad0 | `DSPLIT=2 C4_P_BF16=1 KV_PAD=0 P_PAD=0` | 0.999998 | 236 | 0 | 80 KB | 3.95 w | 7.5% | 7.51 | 545 |
| wpe2 + bf16P pad0 + split-D | `TILING=128,64,4,2 DSPLIT=2 C4_P_BF16=1 KV_PAD=0 P_PAD=0` | 0.999998 | 128 | 740 | 80 KB | **7.43 w** | 7.1% | 7.89 | 519 |

LDS budget = 160 KB/CU (rocminfo). Occupancy can be doubled to 2 CTAs/CU (7.43 waves) but is **slower** and MfmaUtil stays ~7% — the kernel is not occupancy/register/LDS-size bound; every restructuring is bit-exact but regresses. bn64 G=1 f32 (4.41 ms) is the floor.

## Follow-up

The c ≥ 32 gap is **latency**, not throughput-occupancy. Freeing VGPR (split-D) kills the
spill but not the latency, and adding CTAs (bn32 Path B) adds proportional barrier/gather
cost (−35 %). Direction: cut exposed per-tile barriers (double-buffer KV/P), then retune LDS
bank-conflict layout. Split-D / bn32 stay in-tree as reference (`SGLANG_C4_DSPLIT`,
`SGLANG_C4_TILING`), OFF by default. The c = 2…16 wins land as-is.