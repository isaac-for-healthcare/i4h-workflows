#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: $(basename "$0") [--with-cosmos]"
  exit 0
fi

command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }

echo "[agentic setup] syncing common"
(cd "${ROOT}/common" && env -u VIRTUAL_ENV uv sync)

echo "[agentic setup] preparing shared third_party"
"${ROOT}/third_party/setup.sh"
export AGENTIC_THIRD_PARTY_SETUP_DONE=1

for component in arena policy mimic dataset annotator; do
  echo "[agentic setup] setting up ${component}"
  "${ROOT}/${component}/setup.sh"
done

if [[ "${1:-}" == "--with-cosmos" || "${AGENTIC_SETUP_WITH_COSMOS:-0}" == "1" ]]; then
  echo "[agentic setup] setting up cosmos"
  "${ROOT}/cosmos/setup.sh"
else
  echo "[agentic setup] skipping optional cosmos setup (pass --with-cosmos to include it)"
fi

echo "[agentic setup] finished"
