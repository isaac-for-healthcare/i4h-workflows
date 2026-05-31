#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Build flash-attn on demand and install it into the current uv project venv.
# Run from workflows/so_arm/policy or workflows/so_arm/training after uv sync.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ARCH="$(uname -m)"
PYTHON_TAG="cp${PYTHON_VERSION//./}"
WHEEL="${SCRIPT_DIR}/wheels/flash_attn-${FLASH_ATTN_VERSION}-${PYTHON_TAG}-${PYTHON_TAG}-manylinux_2_35_${ARCH}.whl"

if ! command -v uv >/dev/null 2>&1; then
    echo "[install-flash-attn] ERROR: uv not on PATH" >&2
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[install-flash-attn] ERROR: run this from a synced uv project directory" >&2
    exit 1
fi

"${SCRIPT_DIR}/build_flash_attn.sh"

if [ ! -f "$WHEEL" ]; then
    echo "[install-flash-attn] ERROR: expected wheel not found: $WHEEL" >&2
    exit 1
fi

echo "[install-flash-attn] installing $(basename "$WHEEL") into $(pwd)/.venv"
uv pip install --python ".venv/bin/python" --no-deps "$WHEEL"

echo "[install-flash-attn] installed. Run flash mode with:"
case "$(basename "$(pwd)")" in
    policy) echo "  uv run --no-sync soarm-policy --attn-implementation flash_attention_2" ;;
    training) echo "  uv run --no-sync soarm-train --attn-implementation flash_attention_2 ..." ;;
    *) echo "  uv run --no-sync <command> --attn-implementation flash_attention_2" ;;
esac
