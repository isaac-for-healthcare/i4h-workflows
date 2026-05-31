#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! "${POLICY_DIR}/.venv/bin/python" -c "import onnxscript" >/dev/null 2>&1; then
  echo "[trt-n17] installing build-time extras (onnxscript, decord) ..."
  (cd "${POLICY_DIR}" && env -u VIRTUAL_ENV uv pip install \
    "onnxscript>=0.7" "decord>=0.6 ; platform_machine == 'x86_64'")
fi

cd "${POLICY_DIR}"
exec env -u VIRTUAL_ENV uv run --no-sync python "${SCRIPT_DIR}/build_trt_engines.py" "$@"
