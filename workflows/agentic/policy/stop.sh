#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/utils.sh
source "${SCRIPT_DIR}/../scripts/utils.sh"
agentic_stop_by_pattern \
  "policy daemons" \
  "policy" \
  "i4h-agentic-(gr00t-n15|gr00t-n16|gr00t-n17|openpi-pi0)([[:space:]]|$)" \
  "$@"
