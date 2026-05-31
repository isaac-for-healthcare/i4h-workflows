#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/utils.sh
source "${SCRIPT_DIR}/../scripts/utils.sh"
agentic_parse_stop_options "Cosmos jobs" "$@"

agentic_stop_matching \
  "cosmos" \
  "${AGENTIC_STOP_FORCE}" \
  "${AGENTIC_STOP_TIMEOUT}" \
  "i4h-agentic-cosmos-(expand|export|run|import)([[:space:]]|$)" \
  "cosmos-transfer-2.5" \
  "examples/inference.py"
