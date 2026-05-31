#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

routing() {
  PYTHONPATH="${WORKFLOW_ROOT}/common${PYTHONPATH:+:${PYTHONPATH}}" \
    WORKFLOW_ROOT="${WORKFLOW_ROOT}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/policy_routing.py" "$@"
}

list_envs() {
  routing --list-envs
}

usage() {
  echo "usage: $(basename "$0") --env <env_id> [policy args...]"
  echo "       $(basename "$0") --all [policy args...]"
  echo "       $(basename "$0") --list-envs"
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  list_envs
  exit 0
fi

ENV=""
ALL=0
FORWARD_ARGS=()
for arg in "$@"; do
  if [[ -n "${capture:-}" ]]; then
    ENV="${arg}"
    FORWARD_ARGS+=("${arg}")
    capture=""
    continue
  fi
  case "${arg}" in
    --all) ALL=1 ;;
    --list-envs) list_envs; exit 0 ;;
    --env) capture=1; FORWARD_ARGS+=("${arg}") ;;
    --env=*) ENV="${arg#--env=}"; FORWARD_ARGS+=("${arg}") ;;
    *) FORWARD_ARGS+=("${arg}") ;;
  esac
done

if [[ "${ALL}" == "1" ]]; then
  if [[ -n "${ENV}" ]]; then
    echo "--all cannot be combined with --env" >&2
    exit 2
  fi
  mapfile -t ENV_ORDER < <(routing --envs)
  if [[ " ${FORWARD_ARGS[*]} " == *" --dry-run "* ]]; then
    for env in "${ENV_ORDER[@]}"; do
      subproject="$(routing --stack-for-env "${env}")"
      "${SCRIPT_DIR}/${subproject}/run.sh" --env "${env}" "${FORWARD_ARGS[@]}"
    done
    exit 0
  fi
  pids=()
  trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM EXIT
  for env in "${ENV_ORDER[@]}"; do
    subproject="$(routing --stack-for-env "${env}")"
    echo "[agentic-policy] starting ${env} (${subproject})"
    "${SCRIPT_DIR}/${subproject}/run.sh" --env "${env}" "${FORWARD_ARGS[@]}" &
    pids+=("$!")
  done
  wait -n "${pids[@]}"
  status=$?
  kill "${pids[@]}" 2>/dev/null || true
  wait "${pids[@]}" 2>/dev/null || true
  exit "${status}"
fi

if [[ -z "${ENV}" ]]; then
  usage >&2
  exit 2
fi

if ! SUBPROJECT="$(routing --stack-for-env "${ENV}")"; then
  echo "unknown env '${ENV}' — choose one of:" >&2
  list_envs >&2
  exit 2
fi

exec "${SCRIPT_DIR}/${SUBPROJECT}/run.sh" "$@"
