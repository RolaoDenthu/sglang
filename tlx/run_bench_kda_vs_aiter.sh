#!/usr/bin/env bash
# Kimi-K3 KDA kernel speed on one GPU: sglang Triton and TLX (uTLX, /opt/venv) vs
# aiter's Gluon KDA kernels from an upstream aiter checkout. Writes compare.md.
#
#   ./run_bench_kda_vs_aiter.sh [OUT_DIR]
#   GPU=3 AITER_MAIN=/path/to/aiter-main ./run_bench_kda_vs_aiter.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/tmp/kda_vs_aiter_$(date +%Y%m%d_%H%M%S)}
GPU=${GPU:-0}
PY=${PY:-/opt/venv/bin/python}
AITER_MAIN=${AITER_MAIN:-/sgl-workspace/aiter-main}
BENCH="$HERE/bench_kda_vs_aiter.py"

if [ ! -f "$AITER_MAIN/aiter/ops/triton/attention/chunk_kda.py" ]; then
    echo "no upstream aiter with chunk_kda at $AITER_MAIN; create it with:" >&2
    echo "  git -C /sgl-workspace/aiter fetch origin main" >&2
    echo "  git -C /sgl-workspace/aiter worktree add --detach $AITER_MAIN origin/main" >&2
    exit 1
fi

mkdir -p "$OUT"
quiet() {
    grep --line-buffered -v -E "'-x c' after|UserWarning|_warn_once|kernel = self.fn.run|warn\(|make_block_ptr is deprecated|^\[aiter\]" || true
}

HIP_VISIBLE_DEVICES=$GPU "$PY" "$BENCH" sglang --out "$OUT" 2>&1 | quiet
# aiter's kernels use no TLX; keep their cache apart from the uTLX-compiled one.
HIP_VISIBLE_DEVICES=$GPU UTLX_NO_AUTOREGISTER=1 TRITON_CACHE_DIR="$HOME/.triton/cache-aiter-main" \
    "$PY" "$BENCH" aiter-main --out "$OUT" --aiter-main "$AITER_MAIN" 2>&1 | quiet
"$PY" "$BENCH" report --out "$OUT" 2>&1 | quiet
