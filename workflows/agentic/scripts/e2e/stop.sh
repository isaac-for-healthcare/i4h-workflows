#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--env <env_id>] [--run-dir <path>]

Stops services commonly left behind by an interrupted e2e run:
policy daemons, arena processes, the local annotator vLLM container, and
LeRobot viz servers for the run's viz-state directory.
EOF
}

ENV_ID=""
RUN_DIR="workflows/agentic/runs/.latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      if [[ $# -lt 2 ]]; then
        echo "[e2e-stop] missing value for --env" >&2
        exit 2
      fi
      ENV_ID="$2"
      shift 2
      ;;
    --env=*)
      ENV_ID="${1#*=}"
      shift
      ;;
    --run-dir)
      if [[ $# -lt 2 ]]; then
        echo "[e2e-stop] missing value for --run-dir" >&2
        exit 2
      fi
      RUN_DIR="$2"
      shift 2
      ;;
    --run-dir=*)
      RUN_DIR="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[e2e-stop] unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$ENV_ID" ]]; then
  workflows/agentic/stop.sh policy --env "$ENV_ID" --force || true
  workflows/agentic/stop.sh arena --env "$ENV_ID" --force || true
else
  workflows/agentic/stop.sh policy --force || true
  workflows/agentic/stop.sh arena --force || true
fi

workflows/agentic/annotator/vllm.sh stop >/dev/null 2>&1 || true

if [[ -e "$RUN_DIR" ]]; then
  RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
  VIZ_STATE_DIR="$RUN_DIR/viz-state"
  for port in $(seq 9090 9099); do
    workflows/agentic/dataset/viz.sh --stop --port "$port" --state-dir "$VIZ_STATE_DIR" >/dev/null 2>&1 || true
  done
fi

echo "[e2e-stop] done"
