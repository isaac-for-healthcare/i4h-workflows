#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run an i4h workflow. Each mode runs the IsaacLab-Arena or Isaac Lab tool a Lab
# user would: Arena's policy runner, with the workflow's policy server when it has
# one, Isaac Lab's record_demos.py for teleoperation and replay_demos.py for replay.
# `./run.sh show <workflow> --<mode>` prints the commands; `./run.sh --help` lists
# the options. Workflows are declared in workflows/<specialty>/<workflow>.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export I4H_CALLER_CWD="$PWD"
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT"

if [ "${1:-}" = robot-pd ]; then
  shift
  exec env -u VIRTUAL_ENV arena/.venv/bin/python arena/scripts/robot_pd_tune.py "$@"
fi

PYTHON="$ROOT/common/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "run.sh: the common environment is missing; run ./setup.sh" >&2
  exit 2
fi
exec env -u VIRTUAL_ENV "$PYTHON" "$ROOT/scripts/run_workflow.py" "$@"
