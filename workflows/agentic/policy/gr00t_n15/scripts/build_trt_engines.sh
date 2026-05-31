#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! "${POLICY_DIR}/.venv/bin/python" -c "import tensorrt, onnxscript" >/dev/null 2>&1; then
  echo "[trt-n15] installing build-time extras (tensorrt, onnxscript) ..."
  (cd "${POLICY_DIR}" && env -u VIRTUAL_ENV uv pip install \
    "tensorrt-cu13>=10.15" "tensorrt-cu13-bindings>=10.15" "onnxscript>=0.7")
fi

cd "${POLICY_DIR}"
exec env -u VIRTUAL_ENV uv run --no-sync python "${SCRIPT_DIR}/build_trt_engines.py" "$@"
