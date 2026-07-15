#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"
[[ $# -gt 0 ]] || set -- --help

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
DEFAULT_BRIDGE_PORT="${AGENTIC_ARENA_BRIDGE_PORT:-8765}"

arch="$(uname -m)"
if [[ "$arch" == "aarch64" ]]; then
  sys_libgomp="$(ls /lib/*/libgomp.so.1 2>/dev/null | head -1 || true)"
  if [[ -n "$sys_libgomp" ]]; then
    export LD_PRELOAD="${sys_libgomp}${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=2000000}"
else
  libgomp_path="$(env -u VIRTUAL_ENV uv run --no-sync python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib" / "libgomp.so.1")' 2>/dev/null || true)"
  if [[ -n "$libgomp_path" && -e "$libgomp_path" ]]; then
    export LD_PRELOAD="${libgomp_path}${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi

bridge_port_for_env() {
  local env_id="${1:-}" fallback="${2:-${DEFAULT_BRIDGE_PORT}}" port
  if [[ -z "${env_id}" ]]; then
    echo "${fallback}"
    return 0
  fi
  if port="$(PYTHONPATH="${SCRIPT_DIR}:${WORKFLOW_ROOT}/common${PYTHONPATH:+:${PYTHONPATH}}" \
    WORKFLOW_ROOT="${WORKFLOW_ROOT}" \
    env -u VIRTUAL_ENV uv run python - "${env_id}" "${fallback}" <<'PY'
import sys

from arena.arena_config import get_arena_config

env_id = sys.argv[1]
fallback = int(sys.argv[2])
print(get_arena_config(env_id).bridge_port or fallback)
PY
  )"; then
    echo "${port}"
  else
    echo "${fallback}"
  fi
}

bridge_url_for_env() {
  local env_id="${1:-}" port="${2:-}"
  [[ -n "${port}" ]] || port="$(bridge_port_for_env "${env_id}")"
  echo "http://127.0.0.1:${port}"
}

bridge_port_command() {
  local env_id="" capture="" arg
  while [[ $# -gt 0 ]]; do
    arg="$1"
    if [[ -n "${capture}" ]]; then
      case "${capture}" in
        env) env_id="${arg}" ;;
      esac
      capture=""
      shift
      continue
    fi
    case "${arg}" in
      --env) capture=env; shift ;;
      --env=*) env_id="${arg#--env=}"; shift ;;
      *) shift ;;
    esac
  done
  echo "$(bridge_port_for_env "${env_id}")"
}

bridge_url_command() {
  local env_id="" capture="" arg
  while [[ $# -gt 0 ]]; do
    arg="$1"
    if [[ -n "${capture}" ]]; then
      case "${capture}" in
        env) env_id="${arg}" ;;
      esac
      capture=""
      shift
      continue
    fi
    case "${arg}" in
      --env) capture=env; shift ;;
      --env=*) env_id="${arg#--env=}"; shift ;;
      *) shift ;;
    esac
  done
  bridge_url_for_env "${env_id}"
}

ensure_bridge() {
  local log_path="" timeout="${AGENTIC_ARENA_BRIDGE_READY_TIMEOUT:-600}" capture="" arg
  local forward_args=()

  while [[ $# -gt 0 ]]; do
    arg="$1"
    if [[ -n "${capture}" ]]; then
      case "${capture}" in
        log) log_path="${arg}" ;;
        timeout) timeout="${arg}" ;;
      esac
      capture=""
      shift
      continue
    fi

    case "${arg}" in
      --log) capture=log; shift ;;
      --log=*) log_path="${arg#--log=}"; shift ;;
      --timeout) capture=timeout; shift ;;
      --timeout=*) timeout="${arg#--timeout=}"; shift ;;
      *) forward_args+=("${arg}"); shift ;;
    esac
  done

  [[ -n "${log_path}" ]] || { echo "ensure-bridge requires --log <path>" >&2; exit 2; }
  mkdir -p "$(dirname "${log_path}")"

  local has_keep_open=0
  local desired_env=""
  local desired_bridge_port=""
  for arg in "${forward_args[@]}"; do
    if [[ "${arg}" == --env=* ]]; then
      desired_env="${arg#--env=}"
    elif [[ "${arg}" == --bridge-port=* ]]; then
      desired_bridge_port="${arg#--bridge-port=}"
    fi
  done
  for ((i = 0; i < ${#forward_args[@]}; i++)); do
    if [[ "${forward_args[$i]}" == "--env" && $((i + 1)) -lt ${#forward_args[@]} ]]; then
      desired_env="${forward_args[$((i + 1))]}"
    elif [[ "${forward_args[$i]}" == "--bridge-port" && $((i + 1)) -lt ${#forward_args[@]} ]]; then
      desired_bridge_port="${forward_args[$((i + 1))]}"
    fi
  done
  local bridge_port="${desired_bridge_port:-$(bridge_port_for_env "${desired_env}")}"
  local bridge_url
  bridge_url="$(bridge_url_for_env "${desired_env}" "${bridge_port}")"

  local health_json
  health_json="$(curl -fsS "${bridge_url}/health" 2>/dev/null || true)"
  if [[ -n "${health_json}" ]]; then
    if HEALTH_JSON="${health_json}" python3 - "${desired_env}" <<'PY'; then
import json
import os
import sys

desired_env = sys.argv[1]
payload = json.loads(os.environ["HEALTH_JSON"])
if not payload.get("ok"):
    raise SystemExit(1)
if desired_env and payload.get("env_id") != desired_env:
    raise SystemExit(2)
if (payload.get("main_loop") or {}).get("busy"):
    raise SystemExit(3)
PY
      echo "[agentic-arena] bridge already ready at ${bridge_url}"
      return 0
    fi
    echo "[agentic-arena] ERROR: bridge endpoint exists but is not reusable; stop it first" >&2
    echo "${health_json}" | python3 -m json.tool >&2 || echo "${health_json}" >&2
    exit 1
  fi

  echo "[agentic-arena] starting bridge; log=${log_path}"
  for arg in "${forward_args[@]}"; do
    if [[ "${arg}" == "--keep-open" ]]; then
      has_keep_open=1
      break
    fi
  done
  if (( has_keep_open == 0 )); then
    forward_args+=("--keep-open")
  fi
  if [[ -z "${desired_bridge_port}" ]]; then
    forward_args+=("--bridge-port" "${bridge_port}")
  fi
  setsid "${SCRIPT_DIR}/run.sh" "${forward_args[@]}" --bridge > "${log_path}" 2>&1 < /dev/null &
  local pid=$!
  disown "${pid}" 2>/dev/null || true
  echo "[agentic-arena] bridge pid=${pid}; waiting up to ${timeout}s"

  local started_at="${SECONDS}"
  while (( SECONDS - started_at <= timeout )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[agentic-arena] ERROR: bridge exited before ready; tail of ${log_path}:" >&2
      tail -40 "${log_path}" >&2 || true
      exit 1
    fi

    if grep -qF "[agentic-arena] scene-edit bridge ready" "${log_path}" 2>/dev/null; then
      echo "[agentic-arena] bridge ready after $((SECONDS - started_at))s"
      return 0
    fi
    sleep 1
  done

  echo "[agentic-arena] ERROR: bridge did not become ready within ${timeout}s; tail of ${log_path}:" >&2
  tail -40 "${log_path}" >&2 || true
  kill "${pid}" 2>/dev/null || true
  exit 1
}

if [[ "${1:-}" == "bridge-port" ]]; then
  shift
  bridge_port_command "$@"
  exit 0
fi

if [[ "${1:-}" == "bridge-url" ]]; then
  shift
  bridge_url_command "$@"
  exit 0
fi

if [[ "${1:-}" == "ensure-bridge" ]]; then
  shift
  ensure_bridge "$@"
  exit 0
fi

exec env -u VIRTUAL_ENV uv run i4h-agentic-arena "$@"
