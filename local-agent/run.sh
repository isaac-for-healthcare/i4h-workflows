#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Minimal local agent: one SGLang model server plus OpenCode.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "${ROOT}/config.env"

API_BASE="${I4H_AGENT_BASE_URL%/}/v1"
HEALTH_URL="${I4H_AGENT_BASE_URL%/}/health_generate"

# Remote mode: an API key means we talk to a hosted endpoint (e.g. NVIDIA NIM)
# instead of starting/managing a local Docker model server.
is_remote() { [[ -n "${I4H_AGENT_API_KEY:-}" ]]; }
CURL_AUTH=()
is_remote && CURL_AUTH=(-H "Authorization: Bearer ${I4H_AGENT_API_KEY}")

log() { echo "[$(date '+%H:%M:%S')] $*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "error: missing '$1'" >&2; exit 1; }; }
remote_guard() {
    is_remote && {
        echo "error: '$1' manages the local model server and is not used in remote mode" >&2
        echo "       (I4H_AGENT_API_KEY is set, base ${I4H_AGENT_BASE_URL}). Just run: ./local-agent/run.sh agent" >&2
        exit 1
    }
    return 0
}

usage() {
    printf '%s\n' \
        "usage: ./local-agent/run.sh <start|status|stop|logs|warmup|agent> [prompt]" \
        "" \
        "profile: ${I4H_AGENT_PROFILE} (${I4H_AGENT_MODEL})" \
        "" \
        "start   Start the local SGLang model server" \
        "status  Check whether the model is ready" \
        "stop    Stop the model server" \
        "logs    Follow model server logs" \
        "warmup  Run a tiny chat-completions request" \
        "agent   Run OpenCode against the local model"
}

container_running() {
    docker ps --filter "name=^/${I4H_AGENT_CONTAINER}$" --format '{{.ID}}' 2>/dev/null | grep -q .
}

wait_ready() {
    local deadline code
    deadline=$((SECONDS + I4H_AGENT_STARTUP_TIMEOUT))
    while (( SECONDS < deadline )); do
        code="$(curl -s -o /dev/null -w '%{http_code}' "${HEALTH_URL}" 2>/dev/null || true)"
        [[ "${code}" == 200 ]] && { log "model ready at ${API_BASE}"; return 0; }
        log "waiting for model (HTTP ${code:-000})"
        sleep 10
    done
    echo "error: timed out waiting for model; run './local-agent/run.sh logs'" >&2
    return 1
}

local_served_name() {
    if [[ -n "${I4H_AGENT_SERVED_NAME:-}" && "${I4H_AGENT_SERVED_NAME}" != "auto" ]]; then
        echo "${I4H_AGENT_SERVED_NAME}"
    else
        echo "${I4H_AGENT_DEFAULT_SERVED_NAME}"
    fi
}

resolve_served_name() {
    # Remote endpoints serve a fixed model id; never probe their catalog.
    if is_remote; then
        echo "${I4H_AGENT_MODEL}"
        return 0
    fi
    if [[ -n "${I4H_AGENT_SERVED_NAME:-}" && "${I4H_AGENT_SERVED_NAME}" != "auto" ]]; then
        echo "${I4H_AGENT_SERVED_NAME}"
        return 0
    fi

    local served
    served="$(curl -fsS "${API_BASE}/models" 2>/dev/null | python3 -c 'import json, sys; data=json.load(sys.stdin).get("data", []); print(data[0].get("id", "") if data else "")' 2>/dev/null || true)"
    if [[ -n "${served}" ]]; then
        echo "${served}"
    else
        echo "${I4H_AGENT_DEFAULT_SERVED_NAME}"
    fi
}

warmup_model() {
    [[ "${I4H_AGENT_WARMUP}" == "1" ]] || { log "chat warmup disabled"; return 0; }
    local code payload served_name
    served_name="$(resolve_served_name)"
    payload="{\"model\":\"${served_name}\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"temperature\":0}"
    code="$(curl -s -o /dev/null -w '%{http_code}' \
        "${CURL_AUTH[@]}" \
        -H 'Content-Type: application/json' \
        -d "${payload}" \
        "${API_BASE}/chat/completions" 2>/dev/null || true)"
    if [[ "${code}" == 200 ]]; then
        log "chat warmup complete"
    else
        log "chat warmup failed (HTTP ${code:-000})"
        return 1
    fi
}

apply_restart_policy() {
    [[ -n "${I4H_AGENT_DOCKER_RESTART}" ]] || return 0
    docker update --restart "${I4H_AGENT_DOCKER_RESTART}" "${I4H_AGENT_CONTAINER}" >/dev/null
    log "restart policy: ${I4H_AGENT_DOCKER_RESTART}"
}

start_server() {
    remote_guard start
    need docker
    if container_running; then
        log "${I4H_AGENT_CONTAINER} already running"
        apply_restart_policy
        wait_ready
        warmup_model
        return
    fi

    docker rm -f "${I4H_AGENT_CONTAINER}" >/dev/null 2>&1 || true
    mkdir -p "${I4H_AGENT_HF_CACHE}"

    local served_name
    served_name="$(local_served_name)"

    local -a launch_args
    launch_args=(
        python3 -m sglang.launch_server
        --model-path "${I4H_AGENT_MODEL}"
        --host 0.0.0.0
        --port "${I4H_AGENT_PORT}"
        --served-model-name "${served_name}"
        --dtype auto
        --mem-fraction-static "${I4H_AGENT_GPU_UTIL}"
        --context-length "${I4H_AGENT_MAX_MODEL_LEN}"
    )
    if [[ -n "${I4H_AGENT_TP_SIZE:-}" && "${I4H_AGENT_TP_SIZE}" != "1" ]]; then
        launch_args+=(--tp-size "${I4H_AGENT_TP_SIZE}")
    fi
    if [[ -n "${I4H_AGENT_TOOL_CALL_PARSER:-}" ]]; then
        launch_args+=(--tool-call-parser "${I4H_AGENT_TOOL_CALL_PARSER}")
    fi
    if [[ -n "${I4H_AGENT_REASONING_PARSER:-}" ]]; then
        launch_args+=(--reasoning-parser "${I4H_AGENT_REASONING_PARSER}")
    fi
    if [[ "${I4H_AGENT_TRUST_REMOTE_CODE:-0}" == "1" ]]; then
        launch_args+=(--trust-remote-code)
    fi

    local launch_cmd
    launch_cmd="$(printf '%q ' "${launch_args[@]}")"
    if [[ -n "${I4H_AGENT_PRELAUNCH:-}" ]]; then
        launch_cmd="${I4H_AGENT_PRELAUNCH}; exec ${launch_cmd}"
    else
        launch_cmd="exec ${launch_cmd}"
    fi

    local -a docker_args
    docker_args=(
        run -d --name "${I4H_AGENT_CONTAINER}" --network=host
        --gpus "device=${I4H_AGENT_GPU}"
        --shm-size "${I4H_AGENT_SHM_SIZE}"
        -v "${I4H_AGENT_HF_CACHE}:/root/.cache/huggingface"
        -e HF_HOME=/root/.cache/huggingface
        -e "NVIDIA_VISIBLE_DEVICES=${I4H_AGENT_GPU}"
    )
    if [[ -n "${I4H_AGENT_DOCKER_RESTART}" ]]; then
        docker_args+=(--restart "${I4H_AGENT_DOCKER_RESTART}")
    fi
    local model_cache_dir="${I4H_AGENT_HF_CACHE}/hub/models--${I4H_AGENT_MODEL//\//--}"
    if [[ "${I4H_AGENT_HF_OFFLINE}" == "1" || ( "${I4H_AGENT_HF_OFFLINE}" == "auto" && -d "${model_cache_dir}" ) ]]; then
        docker_args+=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
        log "using Hugging Face offline mode for cached model"
    fi

    log "starting ${I4H_AGENT_MODEL} profile=${I4H_AGENT_PROFILE} on GPU(s) ${I4H_AGENT_GPU}"
    docker "${docker_args[@]}" "${I4H_AGENT_IMAGE}" bash -lc "${launch_cmd}"
    wait_ready
    warmup_model
}

status_server() {
    remote_guard status
    need docker
    container_running || { echo "model: stopped"; return 1; }
    curl -fsS "${HEALTH_URL}" >/dev/null 2>&1 \
        && echo "model: ready at ${API_BASE} (${I4H_AGENT_PROFILE}: ${I4H_AGENT_MODEL})" \
        || { echo "model: starting on port ${I4H_AGENT_PORT} (${I4H_AGENT_PROFILE}: ${I4H_AGENT_MODEL})"; return 1; }
}

stop_server() {
    remote_guard stop
    need docker
    if container_running; then
        docker rm -f "${I4H_AGENT_CONTAINER}" >/dev/null
        log "stopped ${I4H_AGENT_CONTAINER}"
    else
        log "${I4H_AGENT_CONTAINER} is not running"
    fi
}

write_opencode_config() {
    local served_name="$1"
    # A real key for hosted endpoints; "dummy" for the local unauthenticated server.
    local api_key="${I4H_AGENT_API_KEY:-dummy}"
    python3 - "${API_BASE}" "${served_name}" "${I4H_AGENT_MAX_MODEL_LEN}" "${I4H_AGENT_OUTPUT_LIMIT}" \
        "local/${served_name}" "${ROOT}/tmux-shell.sh" "${I4H_AGENT_REPO_ROOT}" "${api_key}" \
        "${I4H_AGENT_REQUEST_TIMEOUT_MS:-600000}" "${I4H_AGENT_HEADER_TIMEOUT_MS:-90000}" "${I4H_AGENT_CHUNK_TIMEOUT_MS:-90000}" \
        > "${ROOT}/opencode.json" <<'PY'
import json
import os
import sys

base, served, context, output, model, shell, repo, api_key, req_to, hdr_to, chunk_to = sys.argv[1:12]
json.dump({
    "$schema": "https://opencode.ai/config.json",
    "shell": shell,
    "provider": {
        "local": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Local model",
            # Resilience against intermittent hosted-endpoint wedges: a request that
            # returns no headers / stops streaming would otherwise hang the agent forever
            # (there is no default time-to-first-byte abort). headerTimeout catches a
            # zero-byte stall, chunkTimeout a mid-stream stall, timeout is the overall
            # ceiling — all in ms, env-tunable. The turn then errors cleanly (re-runnable)
            # instead of hanging. (See OpenCode provider options: timeout/headerTimeout/chunkTimeout.)
            "options": {"baseURL": base, "apiKey": api_key,
                        "timeout": int(req_to), "headerTimeout": int(hdr_to), "chunkTimeout": int(chunk_to)},
            "models": {served: {"name": served, "tools": True, "limit": {"context": int(context), "output": int(output)}}},
        }
    },
    "model": model,
    "instructions": [os.path.join(repo, "local-agent", "agent-guidance.md")],
    "skills": {"paths": ["skills"]},
    "permission": {
        "edit": "allow",
        "webfetch": "allow",
        "skill": {"*": "allow"},
        "bash": {"*": "allow"},
        "todowrite": "deny",
        "todoread": "deny",
    },
}, sys.stdout, indent=2)
PY
}

run_agent() {
    need "${I4H_AGENT_OPENCODE_BIN}"
    curl -fsS "${CURL_AUTH[@]}" "${API_BASE}/models" >/dev/null 2>&1 || {
        if is_remote; then
            echo "error: ${API_BASE} not reachable with the supplied I4H_AGENT_API_KEY (check key/endpoint)" >&2
        else
            echo "error: model not reachable; run './local-agent/run.sh start' first" >&2
        fi
        exit 1
    }

    local served_name
    served_name="$(resolve_served_name)"
    write_opencode_config "${served_name}"
    export OPENCODE_CONFIG="${ROOT}/opencode.json"
    export I4H_WORKFLOWS="${I4H_AGENT_REPO_ROOT}" REPO_ROOT="${I4H_AGENT_REPO_ROOT}"
    export I4H_TMUX_CWD="${I4H_AGENT_REPO_ROOT}" I4H_TMUX_SESSION="${I4H_TMUX_SESSION:-i4h_local_agent}"
    export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${I4H_AGENT_SKILL_GPU}}"

    cd "${I4H_AGENT_REPO_ROOT}"
    if [[ $# -eq 0 ]]; then
        exec "${I4H_AGENT_OPENCODE_BIN}" --model "local/${served_name}"
    fi
    exec "${I4H_AGENT_OPENCODE_BIN}" run --model "local/${served_name}" "$*" </dev/null
}

case "${1:-}" in
    start) start_server ;;
    status) status_server ;;
    stop) stop_server ;;
    logs) remote_guard logs; need docker; exec docker logs -f "${I4H_AGENT_CONTAINER}" ;;
    warmup) warmup_model ;;
    agent) shift; run_agent "$@" ;;
    -h|--help|"") usage ;;
    *) echo "error: unknown command '${1}'" >&2; usage; exit 2 ;;
esac
