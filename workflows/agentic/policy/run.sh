#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMON_DIR="${WORKFLOW_ROOT}/common"
COMMON_PYTHON="${COMMON_DIR}/.venv/bin/python"

ensure_common_python() {
  if [[ "${COMMON_PYTHON_READY:-0}" == "1" ]]; then
    return
  fi
  if [[ ! -x "${COMMON_PYTHON}" ]] || ! env -u VIRTUAL_ENV "${COMMON_PYTHON}" -c "import yaml" >/dev/null 2>&1; then
    (cd "${COMMON_DIR}" && env -u VIRTUAL_ENV uv sync)
  fi
  COMMON_PYTHON_READY=1
}

routing() {
  ensure_common_python
  PYTHONPATH="${WORKFLOW_ROOT}/common${PYTHONPATH:+:${PYTHONPATH}}" \
    WORKFLOW_ROOT="${WORKFLOW_ROOT}" \
    env -u VIRTUAL_ENV "${COMMON_PYTHON}" "${SCRIPT_DIR}/policy_routing.py" "$@"
}

list_envs() {
  routing --list-envs
}

usage() {
  echo "usage: $(basename "$0") --env <env_id> [--ensure --log <path> --timeout SECONDS] [policy args...]"
  echo "       $(basename "$0") --list-envs"
}

ensure_policy() {
  local env="$1" log_path="$2" timeout="$3"
  shift 3
  [[ -n "${env}" ]] || { echo "ensure requires --env <env_id>" >&2; exit 2; }
  [[ -n "${log_path}" ]] || { echo "ensure requires --log <path>" >&2; exit 2; }
  mkdir -p "$(dirname "${log_path}")"

  local url started_at pid running existing
  url="$(routing --health-url-for-env "${env}")"
  if running="$(routing --matches-health "${env}" "$@" 2>/dev/null)"; then
    echo "[agentic-policy] already running: env=${env} health=${url}"
    echo "[agentic-policy] ${running}"
    return 0
  fi

  if existing="$(routing --health-state-for-env "${env}" 2>/dev/null)"; then
    echo "[agentic-policy] stopping existing policy with different env/model: ${existing}"
    "${SCRIPT_DIR}/stop.sh" --env "${env}" --force >/dev/null 2>&1 || true
  fi

  echo "[agentic-policy] starting env=${env}; log=${log_path}"
  nohup "${SCRIPT_DIR}/run.sh" "$@" > "${log_path}" 2>&1 < /dev/null &
  pid=$!
  echo "[agentic-policy] pid=${pid}; waiting up to ${timeout}s for ${url}"

  started_at="${SECONDS}"
  while (( SECONDS - started_at <= timeout )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[agentic-policy] ERROR: policy exited before ready; tail of ${log_path}:" >&2
      tail -40 "${log_path}" >&2 || true
      exit 1
    fi

    if routing --matches-health "${env}" "$@" >/dev/null 2>&1; then
      echo "[agentic-policy] ready after $((SECONDS - started_at))s"
      return 0
    fi
    sleep 1
  done

  echo "[agentic-policy] ERROR: policy did not become ready within ${timeout}s; tail of ${log_path}:" >&2
  tail -40 "${log_path}" >&2 || true
  kill "${pid}" 2>/dev/null || true
  exit 1
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  list_envs
  exit 0
fi

ENV=""
ENSURE=0
LOG_PATH=""
TIMEOUT="${AGENTIC_POLICY_READY_TIMEOUT:-600}"
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ensure) ENSURE=1; shift ;;
    --list-envs) list_envs; exit 0 ;;
    --env) ENV="$2"; FORWARD_ARGS+=("$1" "$2"); shift 2 ;;
    --env=*) ENV="${1#--env=}"; FORWARD_ARGS+=("$1"); shift ;;
    --log) LOG_PATH="$2"; shift 2 ;;
    --log=*) LOG_PATH="${1#--log=}"; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --timeout=*) TIMEOUT="${1#--timeout=}"; shift ;;
    *) FORWARD_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${ENV}" ]]; then
  usage >&2
  exit 2
fi

if [[ "${ENSURE}" == "1" ]]; then
  ensure_policy "${ENV}" "${LOG_PATH}" "${TIMEOUT}" "${FORWARD_ARGS[@]}"
  exit 0
fi

if ! SUBPROJECT="$(routing --stack-for-env "${ENV}")"; then
  echo "unknown env '${ENV}' — choose one of:" >&2
  list_envs >&2
  exit 2
fi

exec "${SCRIPT_DIR}/${SUBPROJECT}/run.sh" "${FORWARD_ARGS[@]}"
