#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }
if [[ "${AGENTIC_THIRD_PARTY_SETUP_DONE:-0}" != "1" ]]; then
  "${AGENTIC_ROOT}/third_party/setup.sh"
fi
(cd "${SCRIPT_DIR}" && env -u VIRTUAL_ENV uv sync "$@")
