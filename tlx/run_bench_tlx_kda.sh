#!/usr/bin/env bash
# TLX KDA kernel speed, uTLX (/opt/venv) vs fbtriton (fbtriton-venv), on one GPU.
# Runs uTLX, fbtriton, then uTLX again (the rerun shows measurement noise) and
# writes compare.md. Needs sglang on the tlx/kimi-k3-kda branch.
#
#   ./run_bench_tlx_kda.sh [OUT_DIR]
#   GPU=3 FB_ASYNC_COPY=1 ./run_bench_tlx_kda.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/tmp/tlx_bench_$(date +%Y%m%d_%H%M%S)}
GPU=${GPU:-0}
FB_ASYNC_COPY=${FB_ASYNC_COPY:-0}
UTLX_PY=${UTLX_PY:-/opt/venv/bin/python}
FB_PY=${FB_PY:-/sgl-workspace/fbtriton-venv/bin/python}
BENCH="$HERE/bench_tlx_kda.py"

mkdir -p "$OUT"
quiet() {
    grep --line-buffered -v -E "'-x c' after|UserWarning|kernel = self.fn.run|warn\(|make_block_ptr is deprecated" || true
}

HIP_VISIBLE_DEVICES=$GPU "$UTLX_PY" "$BENCH" run --label utlx --out "$OUT/utlx.json" 2>&1 | quiet
HIP_VISIBLE_DEVICES=$GPU TRITON_HIP_USE_ASYNC_COPY=$FB_ASYNC_COPY \
    TRITON_CACHE_DIR="$HOME/.triton/cache-fbtriton-async$FB_ASYNC_COPY" \
    "$FB_PY" "$BENCH" run --label "fbtriton_async$FB_ASYNC_COPY" --out "$OUT/fbtriton.json" 2>&1 | quiet
HIP_VISIBLE_DEVICES=$GPU "$UTLX_PY" "$BENCH" run --label utlx_rerun --out "$OUT/utlx_rerun.json" 2>&1 | quiet

"$UTLX_PY" "$BENCH" compare "$OUT/utlx.json" "$OUT/fbtriton.json" "$OUT/utlx_rerun.json" 2>&1 | quiet \
    | tee "$OUT/compare.md"
