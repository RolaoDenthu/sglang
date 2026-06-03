# DSV4 FlyDSL vs Baseline — Concurrency Sweep (2026-06-03)

- FlyDSL: `amd/deepseek_v4-34-flydsl-nsa-prefill` @ `57b84575` (host-prep cut + fp8 act-quant fusion, INT8 quickreduce default)
- Baseline: `amd/deepseek_v4` @ `0b9a3a12e`
- Image: `rocm/sgl-dev:v0.5.12.post1-rocm720-mi35x-20260529`
- Server: `sglang.launch_server --tp 8 --attention-backend compressed --max-running-request 256 --page-size 256 --chunked-prefill-size 8192 --disable-shared-experts-fusion`
- Bench: `sglang.bench_serving --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 --num-prompts $((C*4)) --max-concurrency C`
- Metric: median TTFT (ms), single-pass; c=2 warm rerun

| Conc | Baseline ms | FlyDSL ms | Change % |
|------|-------------|-----------|----------|
| 2    | 293.58      | 285.43    | -2.8     |
| 4    | 1639.68     | 480.17    | -70.7    |
| 8    | 3030.52     | 456.51    | -84.9    |
| 16   | 3626.58     | 722.86    | -80.1    |
| 32   | 9361.74     | 9661.45   | +3.2     |
| 64   | 20879.54    | 21586.36  | +3.4     |
