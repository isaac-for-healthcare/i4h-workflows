#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
#   ./stop.sh [all]        stop every run ./run.sh started in this checkout
#   ./stop.sh <workflow>   stop that workflow's runs
#
# Only processes run.sh recorded under .run/pids.d/ are stopped, with their policy
# servers, so another session's simulator on the same machine is left alone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/common/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3
exec "$PYTHON" "$ROOT/scripts/run_workflow.py" stop "${1:-all}"
