# TLX KDA kernels: uTLX vs fbtriton

Kernel-level speed of the gfx950 TLX KDA kernels in
`python/sglang/kernels/ops/kimi_k3/tlx/` under two compilers. Both sides run the
same kernel source; only triton, and who provides `tlx`, differs:

| Env | Python | triton | `tlx` |
|---|---|---|---|
| uTLX | `/opt/venv/bin/python` | ROCm Triton 3.8 (`4cff872c`) rebuilt with `TRITON_EXT_ENABLED=ON` | `utlx_plugin` (triton-ext#142) |
| fbtriton | `/sgl-workspace/fbtriton-venv/bin/python` | fbtriton `3.8.0.dev20260923` | built in |

Shapes are Kimi-K3 per rank at TP8: 12 heads, K=V=128, bf16 inputs, fp32
recurrent state, `lower_bound=-5`. Inputs come from
`test/registered/kernels/ops/attention/kda_tlx/test_kimi_k3_kda_tlx.py`.

## Setup

Needs an MI355X (gfx950).

1. Build the uTLX image from the repo root and start a container:

   ```bash
   docker build --network=host -f docker/rocm-triton-ext.Dockerfile -t sglang-rocm:triton-ext .
   docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
     --security-opt seccomp=unconfined sglang-rocm:triton-ext bash
   ```

2. Inside the container, put the image's sglang checkout on this branch (the
   image installs `/sgl-workspace/sglang` as editable):

   ```bash
   cd /sgl-workspace/sglang
   git fetch https://github.com/RolaoDenthu/sglang.git tlx/kimi-k3-kda && git checkout FETCH_HEAD
   ```

3. Create the fbtriton venv next to `/opt/venv`:

   ```bash
   ./tlx/setup_fbtriton_venv.sh
   ```

   The venv inherits `/opt/venv` and replaces only triton. The script writes a
   `zz_opt_venv.pth` into the venv that:
   - adds `/opt/venv`'s site-packages, so torch (ROCm), sglang, aiter and the rest
     come from `/opt/venv` unchanged;
   - leaves the venv's own site-packages first on `sys.path`, so `import triton`
     resolves to fbtriton and shadows ROCm Triton;
   - sets `UTLX_NO_AUTOREGISTER=1` and clears `TRITON_PLUGIN_PATHS`, because
     `/opt/venv`'s `utlx_plugin.pth` would otherwise register uTLX with fbtriton.

   `--system-site-packages` would not work here: `/opt/venv` is itself a venv,
   so that flag would inherit the system Python's packages instead.

   It installs `fbtriton==3.8.0.dev20260923` from fbtriton's nightly index, which
   keeps wheels for about 30 days. Once it is gone, pass a saved wheel with
   `FBTRITON_WHEEL=/path/to/wheel`, or pick another version with `FBTRITON_VERSION=`.

## Run

On an idle GPU:

```bash
cd /sgl-workspace/sglang/tlx
./run_bench_tlx_kda.sh            # -> /tmp/tlx_bench_<time>/compare.md
```

Each side runs in its serving configuration: fbtriton with
`TRITON_HIP_USE_ASYNC_COPY=0` (see below), uTLX on its default.

| Script | What it does |
|---|---|
| `setup_fbtriton_venv.sh` | Creates the fbtriton venv (`VENV=`, `FBTRITON_VERSION=`, `FBTRITON_WHEEL=`) |
| `run_bench_tlx_kda.sh [OUT]` | Runs `bench_tlx_kda.py` under uTLX, fbtriton, then uTLX again as a noise check; writes `compare.md`. `GPU=`, `FB_ASYNC_COPY=`, `UTLX_PY=`, `FB_PY=` |
| `bench_tlx_kda.py` | Times the TLX core (`kda_recurrent_decode` / `kda_paged_prefill`), `prepare_kda_inputs`, and the full `TlxKDAKernel` call. Decode batch 1–256 in a CUDA graph; prefill M <= 16384 in graph and eager. `marker.do_bench`, median, L2 and the 256 MB Infinity Cache flushed |

## fbtriton needs async copy off

Set `TRITON_HIP_USE_ASYNC_COPY=0` whenever sglang runs on fbtriton. fbtriton
turns async copy on by default on gfx950, and with it sglang's FLA Triton KDA
kernels (`python/sglang/kernels/ops/attention/fla/`) produce NaN. The TLX
backend still falls back to those kernels for the prefix-cache state snapshot,
speculative decoding and ReplaySSM decode, so end-to-end accuracy drops to about 0.

| | async copy on | `TRITON_HIP_USE_ASYNC_COPY=0` |
|---|---|---|
| `test_kimi_k3_kda_tlx.py` | 18 of 73 fail (all extend cases, NaN in the Triton reference) | 73 passed |
| Kimi-K3 end-to-end | accuracy ≈ 0 | gsm8k 0.990 |
| TLX kernels | correct (bit-identical to uTLX), same speed | correct, same speed |

The two kernels that break:

- `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (`chunk_delta_h.py`): the final
  state is NaN whenever the last chunk is shorter than 64 tokens. Its
  `boundary_check` loads have no `padding_option`, so out-of-bounds values are
  undefined; adding `padding_option="zero"` fixes it.
- `_recompute_w_u_fwd_kernel` (`kda.py`): `u` is NaN only for the autotune configs
  with `BV=64, num_stages=4`. There is no out-of-bounds access, and the same input
  is correct on ROCm Triton and on fbtriton with async copy off, so this looks like
  an fbtriton pipeliner / async-copy bug.

ROCm Triton + uTLX is correct with async copy on, so uTLX keeps its default.
To check on fbtriton:

```bash
cd /sgl-workspace/sglang
T=test/registered/kernels/ops/attention/kda_tlx/test_kimi_k3_kda_tlx.py
TRITON_HIP_USE_ASYNC_COPY=0 /sgl-workspace/fbtriton-venv/bin/python -m pytest -q $T   # 73 passed
TRITON_HIP_USE_ASYNC_COPY=1 /sgl-workspace/fbtriton-venv/bin/python -m pytest -q $T   # 18 failed
```

## Results

One MI355X (GPU 0), 2026-10-07, following the setup above:

- uTLX: `triton 3.8.0+git4cff872c.rocm10.0.0`, `triton-utlx 3.8.0.post1`
- fbtriton: `3.8.0.dev20260923+fb.gitd19a2e59`, venv from `VENV=/tmp/fbtriton-venv-check ./setup_fbtriton_venv.sh`
- `FB_PY=/tmp/fbtriton-venv-check/bin/python ./run_bench_tlx_kda.sh`
- Outputs agree: max relative checksum difference 5e-5 between uTLX and fbtriton, 0 for the uTLX rerun.

GPU time per call in us; decode in a CUDA graph, prefill shown for the graph
mode. Ratio = fbtriton / uTLX, so above 1 means fbtriton is slower. The uTLX
rerun stays within 1% for the prefill TLX core and within 5% for decode at
batch 128–256.

| Case (seqs × tokens) | uTLX core | fbtriton core | core ratio | uTLX total | fbtriton total | total ratio |
|---|---|---|---|---|---|---|
| decode bs=1 | 4.4 | 3.8 | 0.86 | 7.3 | 6.9 | 0.94 |
| decode bs=4 | 5.6 | 4.9 | 0.88 | 8.5 | 8.0 | 0.94 |
| decode bs=16 | 9.6 | 8.6 | 0.90 | 12.6 | 12.0 | 0.95 |
| decode bs=32 | 14.0 | 13.4 | 0.96 | 17.3 | 17.0 | 0.98 |
| decode bs=64 | 23.5 | 22.9 | 0.97 | 26.7 | 26.2 | 0.98 |
| decode bs=128 | 41.0 | 40.7 | 0.99 | 44.3 | 44.5 | 1.00 |
| decode bs=256 | 80.7 | 80.2 | 0.99 | 84.1 | 83.8 | 1.00 |
| prefill 1×512 | 64.7 | 76.2 | 1.18 | 83.8 | 95.6 | 1.14 |
| prefill 1×1024 | 81.6 | 102.9 | 1.26 | 103.8 | 125.5 | 1.21 |
| prefill 1×2048 | 123.1 | 166.3 | 1.35 | 148.4 | 191.9 | 1.29 |
| prefill 1×4096 | 213.4 | 295.2 | 1.38 | 246.7 | 328.2 | 1.33 |
| prefill 1×8192 | 428.2 | 585.9 | 1.37 | 489.9 | 651.8 | 1.33 |
| prefill 1×16384 | 883.2 | 1214.2 | 1.37 | 1009.5 | 1304.9 | 1.29 |
| prefill 2×8192 | 630.7 | 792.5 | 1.26 | 735.2 | 888.7 | 1.21 |
| prefill 4×4096 | 509.8 | 616.6 | 1.21 | 606.4 | 701.8 | 1.16 |
| prefill 16×1024 | 474.6 | 559.5 | 1.18 | 587.1 | 661.3 | 1.13 |

<details>
<summary>Full <code>compare.md</code> (prepare, eager mode, uTLX rerun)</summary>


| case | part | utlx (us) | fbtriton_async0 (us) | utlx_rerun (us) | fbtriton_async0 / utlx | utlx_rerun / utlx |
|---|---|---|---|---|---|---|
| decode bs=1 | prepare | 3.0 | 3.1 | 3.1 | 1.01 | 1.02 |
| decode bs=1 | tlx core | 4.4 | 3.8 | 4.4 | 0.86 | 1.00 |
| decode bs=1 | total | 7.3 | 6.9 | 7.4 | 0.94 | 1.00 |
| decode bs=4 | prepare | 3.2 | 3.2 | 3.2 | 0.99 | 0.99 |
| decode bs=4 | tlx core | 5.6 | 4.9 | 5.5 | 0.88 | 1.00 |
| decode bs=4 | total | 8.5 | 8.0 | 8.5 | 0.94 | 1.00 |
| decode bs=16 | prepare | 3.2 | 3.3 | 3.3 | 1.01 | 1.01 |
| decode bs=16 | tlx core | 9.6 | 8.6 | 9.5 | 0.90 | 1.00 |
| decode bs=16 | total | 12.6 | 12.0 | 12.6 | 0.95 | 1.00 |
| decode bs=32 | prepare | 3.4 | 3.4 | 3.4 | 1.00 | 1.00 |
| decode bs=32 | tlx core | 14.0 | 13.4 | 14.0 | 0.96 | 1.00 |
| decode bs=32 | total | 17.3 | 17.0 | 17.2 | 0.98 | 0.99 |
| decode bs=64 | prepare | 3.5 | 3.5 | 3.4 | 1.01 | 1.00 |
| decode bs=64 | tlx core | 23.5 | 22.9 | 23.3 | 0.97 | 0.99 |
| decode bs=64 | total | 26.7 | 26.2 | 26.5 | 0.98 | 0.99 |
| decode bs=128 | prepare | 3.8 | 3.7 | 3.8 | 0.98 | 1.01 |
| decode bs=128 | tlx core | 41.0 | 40.7 | 42.6 | 0.99 | 1.04 |
| decode bs=128 | total | 44.3 | 44.5 | 46.0 | 1.00 | 1.04 |
| decode bs=256 | prepare | 4.5 | 4.5 | 4.3 | 1.01 | 0.95 |
| decode bs=256 | tlx core | 80.7 | 80.2 | 76.6 | 0.99 | 0.95 |
| decode bs=256 | total | 84.1 | 83.8 | 80.2 | 1.00 | 0.95 |
| prefill 1x512 | prepare graph | 6.1 | 6.4 | 6.1 | 1.05 | 1.00 |
| prefill 1x512 | tlx core graph | 64.7 | 76.2 | 64.7 | 1.18 | 1.00 |
| prefill 1x512 | total graph | 83.8 | 95.6 | 83.6 | 1.14 | 1.00 |
| prefill 1x512 | prepare eager | 8.2 | 8.8 | 9.1 | 1.06 | 1.10 |
| prefill 1x512 | tlx core eager | 70.3 | 82.3 | 70.4 | 1.17 | 1.00 |
| prefill 1x512 | total eager | 91.9 | 104.2 | 92.0 | 1.13 | 1.00 |
| prefill 1x1024 | prepare graph | 8.2 | 8.5 | 8.3 | 1.04 | 1.01 |
| prefill 1x1024 | tlx core graph | 81.6 | 102.9 | 81.5 | 1.26 | 1.00 |
| prefill 1x1024 | total graph | 103.8 | 125.5 | 103.5 | 1.21 | 1.00 |
| prefill 1x1024 | prepare eager | 12.5 | 12.6 | 11.4 | 1.00 | 0.91 |
| prefill 1x1024 | tlx core eager | 88.2 | 110.7 | 88.9 | 1.26 | 1.01 |
| prefill 1x1024 | total eager | 114.9 | 135.4 | 114.6 | 1.18 | 1.00 |
| prefill 1x2048 | prepare graph | 11.7 | 12.2 | 11.7 | 1.04 | 0.99 |
| prefill 1x2048 | tlx core graph | 123.1 | 166.3 | 123.0 | 1.35 | 1.00 |
| prefill 1x2048 | total graph | 148.4 | 191.9 | 147.7 | 1.29 | 1.00 |
| prefill 1x2048 | prepare eager | 16.0 | 16.8 | 17.1 | 1.05 | 1.07 |
| prefill 1x2048 | tlx core eager | 134.6 | 178.0 | 134.3 | 1.32 | 1.00 |
| prefill 1x2048 | total eager | 161.2 | 205.4 | 161.2 | 1.27 | 1.00 |
| prefill 1x4096 | prepare graph | 19.6 | 20.2 | 18.7 | 1.03 | 0.95 |
| prefill 1x4096 | tlx core graph | 213.4 | 295.2 | 213.2 | 1.38 | 1.00 |
| prefill 1x4096 | total graph | 246.7 | 328.2 | 245.5 | 1.33 | 1.00 |
| prefill 1x4096 | prepare eager | 26.9 | 29.1 | 26.2 | 1.08 | 0.97 |
| prefill 1x4096 | tlx core eager | 229.6 | 314.0 | 228.5 | 1.37 | 1.00 |
| prefill 1x4096 | total eager | 267.2 | 348.5 | 264.0 | 1.30 | 0.99 |
| prefill 1x8192 | prepare graph | 32.1 | 34.1 | 32.3 | 1.06 | 1.01 |
| prefill 1x8192 | tlx core graph | 428.2 | 585.9 | 429.7 | 1.37 | 1.00 |
| prefill 1x8192 | total graph | 489.9 | 651.8 | 490.8 | 1.33 | 1.00 |
| prefill 1x8192 | prepare eager | 50.5 | 51.8 | 51.1 | 1.02 | 1.01 |
| prefill 1x8192 | tlx core eager | 477.6 | 628.4 | 461.9 | 1.32 | 0.97 |
| prefill 1x8192 | total eager | 534.0 | 682.6 | 537.2 | 1.28 | 1.01 |
| prefill 1x16384 | prepare graph | 65.5 | 69.0 | 65.1 | 1.05 | 0.99 |
| prefill 1x16384 | tlx core graph | 883.2 | 1214.2 | 891.4 | 1.37 | 1.01 |
| prefill 1x16384 | total graph | 1009.5 | 1304.9 | 1001.1 | 1.29 | 0.99 |
| prefill 1x16384 | prepare eager | 88.7 | 88.6 | 80.9 | 1.00 | 0.91 |
| prefill 1x16384 | tlx core eager | 970.0 | 1253.0 | 952.6 | 1.29 | 0.98 |
| prefill 1x16384 | total eager | 1039.1 | 1332.2 | 1028.6 | 1.28 | 0.99 |
| prefill 4x4096 | prepare graph | 64.2 | 67.7 | 65.4 | 1.05 | 1.02 |
| prefill 4x4096 | tlx core graph | 509.8 | 616.6 | 506.9 | 1.21 | 0.99 |
| prefill 4x4096 | total graph | 606.4 | 701.8 | 608.8 | 1.16 | 1.00 |
| prefill 4x4096 | prepare eager | 80.2 | 90.3 | 89.9 | 1.13 | 1.12 |
| prefill 4x4096 | tlx core eager | 550.8 | 648.0 | 555.0 | 1.18 | 1.01 |
| prefill 4x4096 | total eager | 625.9 | 736.2 | 636.6 | 1.18 | 1.02 |
| prefill 2x8192 | prepare graph | 65.7 | 67.9 | 65.4 | 1.03 | 0.99 |
| prefill 2x8192 | tlx core graph | 630.7 | 792.5 | 636.2 | 1.26 | 1.01 |
| prefill 2x8192 | total graph | 735.2 | 888.7 | 748.4 | 1.21 | 1.02 |
| prefill 2x8192 | prepare eager | 88.0 | 89.5 | 81.4 | 1.02 | 0.93 |
| prefill 2x8192 | tlx core eager | 696.9 | 849.8 | 682.2 | 1.22 | 0.98 |
| prefill 2x8192 | total eager | 781.6 | 939.7 | 766.4 | 1.20 | 0.98 |
| prefill 16x1024 | prepare graph | 65.7 | 68.0 | 65.7 | 1.03 | 1.00 |
| prefill 16x1024 | tlx core graph | 474.6 | 559.5 | 477.3 | 1.18 | 1.01 |
| prefill 16x1024 | total graph | 587.1 | 661.3 | 591.2 | 1.13 | 1.01 |
| prefill 16x1024 | prepare eager | 80.2 | 89.0 | 90.4 | 1.11 | 1.13 |
| prefill 16x1024 | tlx core eager | 516.6 | 597.8 | 521.4 | 1.16 | 1.01 |
| prefill 16x1024 | total eager | 611.6 | 702.8 | 625.3 | 1.15 | 1.02 |

</details>
