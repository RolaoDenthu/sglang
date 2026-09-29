# SGLang ROCm image with Triton rebuilt with TRITON_EXT_ENABLED=ON, so Triton
# plugins such as triton-utlx (standalone TLX) can be loaded via
# TRITON_PLUGIN_PATHS. The stock Triton in the base image is built without it
# and skips every plugin with a "not built with TRITON_EXT_ENABLED" warning.
#
# Triton is rebuilt from the same commit and with the same version string as
# the stock wheel, because torch pins triton==3.8.0+git4cff872c.rocm10.0.0.
#
# uTLX is compiled from source against that Triton. The PyPI triton-utlx wheels
# are built against upstream triton-lang commits, and libutlx.so only works with
# the exact Triton headers it was compiled against.
#
# Build (see docker/rocm-triton-ext.md for build args, validation and troubleshooting):
#   docker build -f docker/rocm-triton-ext.Dockerfile -t sglang-rocm:triton-ext .
#
# uTLX installs a .pth that adds libutlx.so to TRITON_PLUGIN_PATHS in every
# interpreter. Set UTLX_NO_AUTOREGISTER=1 to opt out.
#
# The uTLX examples and tests are kept in /sgl-workspace/utlx; on a gfx950 GPU:
#   python /sgl-workspace/utlx/examples/kda_prefill/run.py

ARG BASE_IMAGE="rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260928"
FROM ${BASE_IMAGE}

# The commit lives on ROCm/triton release/internal/3.8.x, not on triton-lang/triton.
# Not named TRITON_COMMIT: the base image sets ENV TRITON_COMMIT="", and an
# inherited ENV takes precedence over an ARG of the same name.
ARG TRITON_REPO="https://github.com/ROCm/triton.git"
ARG TRITON_EXT_COMMIT="4cff872ced001ea92d9fcf05b3f6517e2b486d19"
# A detached checkout yields "+git4cff872c"; this suffix completes the version torch pins.
ARG TRITON_WHEEL_VERSION_SUFFIX=".rocm10.0.0"
ARG MAX_JOBS="64"
ARG UTLX_REPO="https://github.com/triton-lang/triton-ext.git"
# Head of triton-ext#142 (AMD MFMA register layouts; KDA prefill on gfx950), not yet merged.
ARG UTLX_COMMIT="1afa671b54e138eff8f29de1a0744c04e129889d"

COPY docker/patches/ /tmp/patches/

# 1. Triton. CPATH/LIBRARY_PATH point at the ROCm SDK and must not leak into its LLVM build.
# 2. Triton's C++ headers, which uTLX compiles against. setup.py does not ship them, so they
#    come from the cmake build tree. The AMD dialect headers are not part of that install and
#    are included both as "Dialect/..." and as "amd/include/...", hence the two copies.
# 3. uTLX, against the LLVM that Triton downloaded and linked into libtriton.
RUN set -eux; \
    git init -q /tmp/triton && cd /tmp/triton; \
    git fetch -q --depth 1 "${TRITON_REPO}" "${TRITON_EXT_COMMIT}" && git checkout -q FETCH_HEAD; \
    git apply /tmp/patches/triton-4cff872c-ext.patch; \
    env -u CPATH -u LIBRARY_PATH TRITON_EXT_ENABLED=ON MAX_JOBS="${MAX_JOBS}" \
        TRITON_WHEEL_VERSION_SUFFIX="${TRITON_WHEEL_VERSION_SUFFIX}" \
        pip install --no-build-isolation --no-deps --force-reinstall .; \
    \
    triton_dir="$(python -c 'import triton, os; print(os.path.dirname(triton.__file__))')"; \
    build=$(echo /tmp/triton/build/cmake.*); \
    amd=/tmp/triton/third_party/amd/include; \
    cmake --install "${build}" --component headers --prefix "${triton_dir}"; \
    mkdir -p "${triton_dir}/include/amd/include"; \
    cp -r "${amd}/." "${build}/third_party/amd/include/." "${triton_dir}/include/"; \
    cp -r "${amd}/." "${build}/third_party/amd/include/." "${triton_dir}/include/amd/include/"; \
    \
    git init -q /tmp/triton-ext && cd /tmp/triton-ext; \
    git fetch -q --depth 1 "${UTLX_REPO}" "${UTLX_COMMIT}" && git checkout -q FETCH_HEAD; \
    git apply /tmp/patches/utlx-pr142-triton-3.8.patch; \
    env -u CPATH -u LIBRARY_PATH TRITON_WHEEL_DIR="${triton_dir}" \
        LLVM_INSTALL_DIR="$(readlink -f /root/.triton/llvm/llvm-ubuntu-x64)" \
        pip install --no-build-isolation --no-deps ./extensions/utlx; \
    mkdir -p /sgl-workspace/utlx; \
    cp -r extensions/utlx/examples extensions/utlx/test /sgl-workspace/utlx/; \
    \
    rm -rf /tmp/triton /tmp/triton-ext /tmp/patches /root/.triton /root/.cache/pip

# SYMBOLIC is only set when TRITON_EXT_ENABLED and the Triton patch both took effect.
# Importing torch first is the order that breaks without the patch; importing triton
# also loads libutlx.so through the .pth.
RUN set -eux; \
    pip show triton | grep -x 'Version: 3.8.0+git4cff872c.rocm10.0.0'; \
    readelf -d "$(python -c 'import triton, os; print(os.path.dirname(triton.__file__))')/_C/libtriton.so" | grep -q SYMBOLIC; \
    python -c "import torch, triton, utlx_plugin"; \
    cd /sgl-workspace/utlx/examples && \
        python -c "import utlx_plugin, kda_prefill as k; assert k.is_prefill_available(), k.missing_prefill_ops()"
