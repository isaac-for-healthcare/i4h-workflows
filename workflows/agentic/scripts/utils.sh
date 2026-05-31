#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

agentic_env_arg_pattern() {
  printf -- '--env(=|[[:space:]]+)%s([[:space:]]|$)' "$1"
}

agentic_stop_usage() {
  local scope="$1"
  cat <<EOF
Usage:
  $(basename "$0") [--env <env_id>] [--force] [--timeout SECONDS]

Stops agentic ${scope}. With --env, only stops the matching env. Without
--env, stops all matching processes.
EOF
}

agentic_parse_stop_options() {
  AGENTIC_STOP_SCOPE="$1"
  AGENTIC_STOP_ENV=""
  AGENTIC_STOP_FORCE=0
  AGENTIC_STOP_TIMEOUT=10
  shift

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --env) AGENTIC_STOP_ENV="$2"; shift 2 ;;
      --env=*) AGENTIC_STOP_ENV="${1#--env=}"; shift ;;
      --force|-f) AGENTIC_STOP_FORCE=1; shift ;;
      --timeout) AGENTIC_STOP_TIMEOUT="$2"; shift 2 ;;
      -h|--help) agentic_stop_usage "${AGENTIC_STOP_SCOPE}"; exit 0 ;;
      *) echo "unknown arg: $1" >&2; agentic_stop_usage "${AGENTIC_STOP_SCOPE}"; exit 2 ;;
    esac
  done
}

agentic_stop_by_pattern() {
  local scope="$1"
  local label="$2"
  local pattern="$3"
  shift 3
  agentic_parse_stop_options "${scope}" "$@"
  if [[ -n "${AGENTIC_STOP_ENV}" ]]; then
    label="${label} env=${AGENTIC_STOP_ENV}"
    pattern="${pattern}.*$(agentic_env_arg_pattern "${AGENTIC_STOP_ENV}")"
  fi
  agentic_stop_matching "${label}" "${AGENTIC_STOP_FORCE}" "${AGENTIC_STOP_TIMEOUT}" "${pattern}"
}

agentic_child_pids() {
  local child
  while IFS= read -r child; do
    [[ -n "${child}" ]] || continue
    agentic_child_pids "${child}"
    printf '%s\n' "${child}"
  done < <(pgrep -P "$1" 2>/dev/null || true)
}

agentic_kill_tree() {
  local signal="$1"
  local pid="$2"
  local child
  while IFS= read -r child; do
    [[ -n "${child}" ]] && kill "-${signal}" "${child}" 2>/dev/null || true
  done < <(agentic_child_pids "${pid}")
  kill "-${signal}" "${pid}" 2>/dev/null || true
}

agentic_collect_pids() {
  local pattern line pid command
  declare -A seen=()
  for pattern in "$@"; do
    while IFS= read -r line; do
      [[ -n "${line}" ]] || continue
      pid="${line%% *}"
      command="${line#* }"
      [[ -n "${pid}" && "${pid}" != "$$" && "${pid}" != "${BASHPID}" ]] || continue
      [[ "${command}" != *"/cursorsandbox "* ]] || continue
      seen["${pid}"]=1
    done < <(pgrep -af "${pattern}" 2>/dev/null || true)
  done
  ((${#seen[@]})) && printf '%s\n' "${!seen[@]}" | sort -n
}

agentic_wait_for_exit() {
  local timeout="$1"
  shift
  local deadline=$((SECONDS + timeout))
  while :; do
    local pid alive=0
    for pid in "$@"; do
      kill -0 "${pid}" 2>/dev/null && { alive=1; break; }
    done
    (( ! alive )) && return 0
    (( SECONDS >= deadline )) && return 1
    sleep 1
  done
}

agentic_stop_matching() {
  local label="$1"
  local force="$2"
  local timeout="$3"
  shift 3

  local pids=() pid
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] && pids+=("${pid}")
  done < <(agentic_collect_pids "$@")

  if (( ${#pids[@]} == 0 )); then
    echo "[agentic-stop] no ${label} processes found"
    return 0
  fi

  echo "[agentic-stop] stopping ${label} pid(s): ${pids[*]}"
  for pid in "${pids[@]}"; do agentic_kill_tree TERM "${pid}"; done
  agentic_wait_for_exit "${timeout}" "${pids[@]}" && return 0

  if (( ! force )); then
    echo "[agentic-stop] ${label} still running after ${timeout}s; re-run with --force" >&2
    return 1
  fi

  echo "[agentic-stop] force killing ${label} pid(s)"
  for pid in "${pids[@]}"; do agentic_kill_tree KILL "${pid}"; done
}
