#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN="${SCRIPT_DIR}/.venv/bin/i4h-agentic-gr00t-n15"
[[ -x "${BIN}" ]] || "${SCRIPT_DIR}/setup.sh"
[[ $# -gt 0 ]] || set -- --help
exec env -u VIRTUAL_ENV "${BIN}" "$@"
