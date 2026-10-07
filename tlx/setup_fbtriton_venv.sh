#!/usr/bin/env bash
# Create the fbtriton venv the benchmarks compare against, inside the uTLX image
# (docker/rocm-triton-ext.Dockerfile). The venv reuses /opt/venv's packages
# (torch, sglang, aiter, ...) and only swaps in fbtriton as `triton`;
# /opt/venv is left untouched.
#
#   ./setup_fbtriton_venv.sh
#   FBTRITON_WHEEL=/path/to/fbtriton-....whl ./setup_fbtriton_venv.sh
#   VENV=/tmp/fbtriton-venv FBTRITON_VERSION=3.8.0.dev20260923 ./setup_fbtriton_venv.sh
set -euo pipefail

VENV=${VENV:-/sgl-workspace/fbtriton-venv}
BASE_PY=${BASE_PY:-/opt/venv/bin/python}
FBTRITON_VERSION=${FBTRITON_VERSION:-3.8.0.dev20260923}
INDEX=https://facebookexperimental.github.io/triton/nightly/simple/

if [[ -e "$VENV" ]]; then
    echo "$VENV already exists; remove it or set VENV=..." >&2
    exit 1
fi

BASE_SITE=$("$BASE_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
"$BASE_PY" -m venv "$VENV"
VENV_SITE=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')

# /opt/venv's utlx_plugin.pth would register libutlx.so with whatever triton is
# imported, fbtriton included; opt out before its site dir is added.
cat > "$VENV_SITE/zz_opt_venv.pth" <<EOF
import os, site; os.environ.setdefault("UTLX_NO_AUTOREGISTER", "1"); os.environ.pop("TRITON_PLUGIN_PATHS", None); site.addsitedir("$BASE_SITE")
EOF

if [[ -n "${FBTRITON_WHEEL:-}" ]]; then
    "$VENV/bin/pip" install --no-deps "$FBTRITON_WHEEL"
else
    "$VENV/bin/pip" install --pre --no-deps --index-url "$INDEX" "fbtriton==$FBTRITON_VERSION"
fi

"$VENV/bin/python" -c '
import sys, torch, triton
from triton.language.extra import tlx
assert "utlx_plugin" not in sys.modules
print("triton", triton.__version__, triton.__file__)
print("tlx   ", tlx.__file__)
'
