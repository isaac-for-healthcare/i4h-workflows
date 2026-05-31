#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

POLICY_PID=""
VLLM_WRAPPER_PID=""
VLLM_STARTED_BY_E2E=0
VIZ_STARTED_BY_E2E=0
VIZ_STATUS="not-attempted"

e2e_cleanup() {
  echo "[e2e] cleanup: stopping services started by this run"

  if [[ -n "${POLICY_PID:-}" ]]; then
    kill "$POLICY_PID" 2>/dev/null || true
    POLICY_PID=""
  fi

  workflows/agentic/stop.sh policy --env "$ENV" --force 2>/dev/null || true

  if [[ "${VIZ_STARTED_BY_E2E:-0}" == "1" ]]; then
    for port in $(seq 9090 9099); do
      workflows/agentic/dataset/viz.sh --stop --port "$port" --state-dir "$VIZ_STATE_DIR" \
        >/dev/null 2>&1 || true
    done
  fi

  e2e_stop_vllm_if_started >/dev/null 2>&1 || true
}

e2e_start_policy() {
  local log_path="$1"
  shift

  workflows/agentic/policy/run.sh --env "$ENV" "$@" > >(tee "$log_path") 2>&1 &
  POLICY_PID=$!
  echo "[e2e] policy daemon pid=$POLICY_PID; waiting up to ${POLICY_LOAD_WAIT_SECONDS}s for readiness"
  e2e_wait_for_policy "$log_path"
}

e2e_wait_for_policy() {
  local log_path="$1"
  local url="http://127.0.0.1:${POLICY_HEALTH_PORT}/readyz"
  local started_at="$SECONDS"
  local body

  while (( SECONDS - started_at <= POLICY_LOAD_WAIT_SECONDS )); do
    if ! kill -0 "$POLICY_PID" 2>/dev/null; then
      echo "[e2e] ERROR: policy daemon exited before becoming ready; tail of $log_path:" >&2
      tail -40 "$log_path" >&2 || true
      exit 1
    fi

    body="$(curl -fsS "$url" 2>/dev/null || true)"
    if [[ "$body" == *'"state": "waiting_for_samples"'* || "$body" == *'"state": "running"'* ]]; then
      echo "[e2e] policy ready after $((SECONDS - started_at))s"
      return 0
    fi

    sleep 1
  done

  echo "[e2e] ERROR: policy did not become ready within ${POLICY_LOAD_WAIT_SECONDS}s (health=$url)" >&2
  tail -40 "$log_path" >&2 || true
  exit 1
}

e2e_stop_policy() {
  if [[ -n "${POLICY_PID:-}" ]]; then
    kill "$POLICY_PID" 2>/dev/null || true
    POLICY_PID=""
  fi
  workflows/agentic/stop.sh policy --env "$ENV" --force 2>/dev/null || true
}

e2e_wait_for_vllm() {
  local timeout_seconds="$1"
  local log_path="${2:-}"
  local started_at="$SECONDS"

  while (( SECONDS - started_at <= timeout_seconds )); do
    if workflows/agentic/annotator/vllm.sh status >/dev/null 2>&1; then
      echo "[e2e] vLLM ready after $((SECONDS - started_at))s"
      return 0
    fi
    sleep 1
  done

  [[ -n "$log_path" ]] && tail -30 "$log_path" >&2 || true
  return 1
}

e2e_ensure_vllm() {
  local log_path="$1"
  local required="${2:-1}"

  if workflows/agentic/annotator/vllm.sh status >/dev/null 2>&1; then
    echo "[e2e] reusing existing vLLM server"
    return 0
  fi

  nohup workflows/agentic/annotator/vllm.sh start > >(tee "$log_path") 2>&1 &
  VLLM_WRAPPER_PID=$!
  VLLM_STARTED_BY_E2E=1
  disown
  echo "[e2e] vLLM wrapper pid=$VLLM_WRAPPER_PID (log: $log_path)"

  if e2e_wait_for_vllm "$VLLM_READY_TIMEOUT_SECONDS" "$log_path"; then
    return 0
  fi

  if [[ "$required" == "1" ]]; then
    echo "[e2e] ERROR: vLLM did not become ready in time" >&2
    exit 1
  fi

  echo "[e2e] WARN: vLLM did not become ready in time" >&2
  return 1
}

e2e_stop_vllm_if_started() {
  if [[ "${E2E_KEEP_VLLM:-0}" != "1" ]]; then
    workflows/agentic/annotator/vllm.sh stop >/dev/null 2>&1 || true
    VLLM_STARTED_BY_E2E=0
  fi

  if [[ -n "${VLLM_WRAPPER_PID:-}" ]]; then
    kill "$VLLM_WRAPPER_PID" 2>/dev/null || true
    VLLM_WRAPPER_PID=""
  fi
}
