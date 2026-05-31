#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -gt 0 ]] || set -- --help
exec env -u VIRTUAL_ENV uv --directory "${SCRIPT_DIR}" run i4h-agentic-mimic-generate "$@"
