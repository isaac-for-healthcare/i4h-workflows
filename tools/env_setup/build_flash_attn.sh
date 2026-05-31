#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Build a flash-attn wheel from source under tools/env_setup/wheels/, consumed
# as a path source by policy/ and training/. No public wheel exists for
# cu130 + torch 2.9 + cp311 on x86_64 or aarch64.
#
# Default build is a fat wheel covering every supported SM in the local arch
# family (one aarch64 wheel runs on both Spark sm_121 and Thor sm_110; one
# x86 wheel covers Ampere through consumer Blackwell). Override with
# FLASH_ATTN_CUDA_ARCHS, PYTHON_VERSION, or FLASH_ATTN_VERSION via env.
#
# Idempotent. Pass --force to rebuild even if a matching wheel exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="$SCRIPT_DIR/wheels"

# Keep in lockstep with policy/ + training/ pyproject pins.
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) echo "[build-flash-attn] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|aarch64) ;;
    *) echo "[build-flash-attn] unsupported arch: $ARCH" >&2; exit 2 ;;
esac

mkdir -p "$WHEELS_DIR"

# pyprojects reference the manylinux-tagged wheel.
PYTHON_TAG="cp${PYTHON_VERSION//./}"
RETAGGED_WHEEL="$(ls "$WHEELS_DIR"/flash_attn-${FLASH_ATTN_VERSION}-${PYTHON_TAG}-${PYTHON_TAG}-manylinux_2_35_${ARCH}.whl 2>/dev/null | head -1 || true)"
if [ -n "$RETAGGED_WHEEL" ] && [ "$FORCE" -ne 1 ]; then
    echo "[build-flash-attn] reusing existing wheel: $RETAGGED_WHEEL"
    echo "[build-flash-attn] pass --force to rebuild."
    exit 0
fi

STALE_LINUX_WHEEL="$(ls "$WHEELS_DIR"/flash_attn-${FLASH_ATTN_VERSION}-${PYTHON_TAG}-${PYTHON_TAG}-linux_${ARCH}.whl 2>/dev/null | head -1 || true)"

retag_to_manylinux() {
    local python_bin="$1"
    shopt -s nullglob
    for whl in "$WHEELS_DIR"/flash_attn-${FLASH_ATTN_VERSION}-${PYTHON_TAG}-${PYTHON_TAG}-linux_${ARCH}.whl; do
        echo "[build-flash-attn] retagging $(basename "$whl") -> manylinux_2_35_${ARCH}"
        "$python_bin" -m wheel tags \
            --remove --platform-tag "manylinux_2_35_${ARCH}" "$whl"
    done
    shopt -u nullglob
}

if ! command -v uv >/dev/null 2>&1; then
    echo "[build-flash-attn] ERROR: uv not on PATH" >&2
    exit 1
fi

# Recover from a previous run killed before the retag step.
if [ -n "$STALE_LINUX_WHEEL" ] && [ "$FORCE" -ne 1 ]; then
    if python3 -c "import wheel" >/dev/null 2>&1; then
        echo "[build-flash-attn] found un-retagged wheel: $STALE_LINUX_WHEEL"
        echo "[build-flash-attn] retagging in place (no rebuild needed)"
        retag_to_manylinux python3
        ls -lh "$WHEELS_DIR"/flash_attn-${FLASH_ATTN_VERSION}-${PYTHON_TAG}-*manylinux_2_35_${ARCH}.whl
        exit 0
    fi
    echo "[build-flash-attn] found un-retagged wheel but host python lacks 'wheel'; falling through to full build"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "[build-flash-attn] ERROR: nvcc not found at $CUDA_HOME/bin/nvcc" >&2
    echo "        Install CUDA Toolkit 13.x (matches torch cu130) or set CUDA_HOME," >&2
    echo "        or pull the LFS-vendored wheel via \`git lfs pull\` to skip the build." >&2
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# SM 11.0 (Thor) and 12.1 (Spark) are added via the setup.py patch below.
if [ -z "${FLASH_ATTN_CUDA_ARCHS:-}" ]; then
    case "$ARCH" in
        x86_64)  FLASH_ATTN_CUDA_ARCHS="80;90;100;120" ;;
        aarch64) FLASH_ATTN_CUDA_ARCHS="90;100;110;121" ;;
    esac
    echo "[build-flash-attn] $ARCH default fat build: FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS"
    export FLASH_ATTN_CUDA_ARCHS
fi

# Each nvcc worker needs ~3-4 GB; cap at min(nproc, mem/4, 16).
if [ -z "${MAX_JOBS:-}" ]; then
    _nproc=$(nproc 2>/dev/null || echo 8)
    _mem_gb=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo 2>/dev/null || echo 16)
    _by_mem=$(( _mem_gb / 4 ))
    MAX_JOBS=$_nproc
    [ "$_by_mem" -lt "$MAX_JOBS" ] && MAX_JOBS=$_by_mem
    [ "$MAX_JOBS" -gt 16 ] && MAX_JOBS=16
    [ "$MAX_JOBS" -lt 1  ] && MAX_JOBS=1
fi
export MAX_JOBS
export NVCC_THREADS="${NVCC_THREADS:-2}"

# Must match policy/+training/pyproject.toml override-dependencies (flash-attn
# binds tightly to torch's ABI).
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "$BUILD_ROOT"' EXIT
BUILD_VENV="$BUILD_ROOT/venv"

echo "[build-flash-attn] creating build venv at $BUILD_VENV (Python $PYTHON_VERSION)"
uv venv --python "$PYTHON_VERSION" "$BUILD_VENV"
PIP=( "$BUILD_VENV/bin/python" -m pip )

# ninja must be on PATH; otherwise torch's BuildExtension silently falls back
# to a single-process build that ignores MAX_JOBS.
export PATH="$BUILD_VENV/bin:$PATH"

echo "[build-flash-attn] installing build deps (torch==$TORCH_VERSION + ninja, …)"
uv pip install \
    --python "$BUILD_VENV/bin/python" \
    --extra-index-url "$TORCH_INDEX_URL" \
    "torch==${TORCH_VERSION}" numpy setuptools wheel packaging pip ninja

SRC_DIR="$BUILD_ROOT/src"
mkdir -p "$SRC_DIR"
SDIST_URL="https://files.pythonhosted.org/packages/source/f/flash_attn/flash_attn-${FLASH_ATTN_VERSION}.tar.gz"

echo "[build-flash-attn] downloading flash-attn==$FLASH_ATTN_VERSION sdist"
curl -fSL --retry 3 --retry-delay 5 "$SDIST_URL" -o "$SRC_DIR/flash_attn.tar.gz"
tar xzf "$SRC_DIR/flash_attn.tar.gz" -C "$SRC_DIR"
FA_SRC="$SRC_DIR/flash_attn-$FLASH_ATTN_VERSION"
[ -f "$FA_SRC/setup.py" ] || { echo "[build-flash-attn] ERROR: setup.py missing in extracted sdist" >&2; exit 1; }

# Add -gencode for sm_110 (Thor) and sm_121 (Spark); flash-attn 2.7.x's
# setup.py only knows up to sm_120. Idempotent.
if ! grep -q '"110"' "$FA_SRC/setup.py" || ! grep -q '"121"' "$FA_SRC/setup.py"; then
    echo "[build-flash-attn] patching setup.py to add SM 11.0 (Jetson Thor) + SM 12.1 (DGX Spark) support"
    python3 - "$FA_SRC/setup.py" <<'PY'
import re, sys, pathlib

p = pathlib.Path(sys.argv[1])
src = p.read_text()

# Duplicate the anchor's `if "X" in cuda_archs(): ...` block, retargeting
# it at the new SM, and insert after the anchor.
patches = [
    ("110", "100"),
    ("121", "120"),
]

for new_sm, anchor_sm in patches:
    if f'"{new_sm}"' in src:
        continue
    block_re = re.compile(
        rf'( +if[^\n]*"{anchor_sm}"[^\n]*in cuda_archs\(\)[^\n]*\n'
        rf'(?: +cc_flag\.append\([^\n]+\)\n){{2}})'
    )
    m = block_re.search(src)
    if not m:
        sys.exit(f"could not locate SM {anchor_sm} block in setup.py to anchor SM {new_sm} patch")
    new_block = (
        m.group(1)
        .replace(f'"{anchor_sm}"', f'"{new_sm}"')
        .replace(f'compute_{anchor_sm},code=sm_{anchor_sm}', f'compute_{new_sm},code=sm_{new_sm}')
    )
    src = src[: m.end()] + new_block + src[m.end() :]

src = src.replace('"80;90;100;120"', '"80;90;100;110;120;121"')

p.write_text(src)
print("[build-flash-attn] setup.py SM 11.0 + SM 12.1 patch applied")
PY
fi

echo "[build-flash-attn] building flash-attn==$FLASH_ATTN_VERSION (Python $PYTHON_VERSION, MAX_JOBS=$MAX_JOBS, NVCC_THREADS=$NVCC_THREADS, ARCHS=$FLASH_ATTN_CUDA_ARCHS)"
echo "[build-flash-attn] runtime: ~20 min"
"${PIP[@]}" wheel \
    "$FA_SRC" \
    --no-build-isolation \
    --no-deps \
    --no-cache-dir \
    --wheel-dir "$WHEELS_DIR" \
    -v 2>&1 | tail -200

# Retag linux_<arch> -> manylinux_2_35_<arch> so uv/pip accept it.
retag_to_manylinux "$BUILD_VENV/bin/python"

echo
echo "[build-flash-attn] wheel(s) written to $WHEELS_DIR:"
ls -lh "$WHEELS_DIR"/flash_attn-${FLASH_ATTN_VERSION}-*.whl
