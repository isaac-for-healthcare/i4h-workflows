#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALL_SUBPROJECTS=(gr00t_n17 gr00t_n16 gr00t_n15 openpi_pi0)
SELECTED=("$@")
[[ ${#SELECTED[@]} -gt 0 ]] || SELECTED=("${ALL_SUBPROJECTS[@]}")

if [[ "${AGENTIC_THIRD_PARTY_SETUP_DONE:-0}" != "1" ]]; then
  "${AGENTIC_ROOT}/third_party/setup.sh"
  export AGENTIC_THIRD_PARTY_SETUP_DONE=1
fi

for sub in "${SELECTED[@]}"; do
  if [[ ! -d "${SCRIPT_DIR}/${sub}" ]]; then
    echo "[agentic policy setup] unknown subproject: ${sub}" >&2
    exit 2
  fi
  echo "=============================================="
  echo "[agentic policy setup] ${sub}"
  echo "=============================================="
  "${SCRIPT_DIR}/${sub}/setup.sh"
done
