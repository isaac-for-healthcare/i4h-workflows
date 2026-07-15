#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

IMAGE_NAME="${AGENTIC_ANNOTATOR_VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.03.post1-py3}"
CONTAINER_NAME="${AGENTIC_ANNOTATOR_VLLM_CONTAINER:-i4h_agentic-annotator-vllm}"
MODEL="${AGENTIC_ANNOTATOR_VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${AGENTIC_ANNOTATOR_VLLM_PORT:-8000}"

usage() {
    echo "usage: $(basename "$0") [start|status|wait|ensure|stop] [gpu-utilization] [max-model-len] [tensor-parallel]"
    echo "  start  : serve ${MODEL} on port ${PORT} (default util: 0.4 aarch64 / 0.8 x86, len: 32768)"
    echo "  status : check vLLM health endpoint"
    echo "  wait   : poll status every second until the health endpoint is ready"
    echo "  ensure : start vLLM if needed, then wait until ready"
    echo "  stop   : stop the vLLM container"
    echo
    echo "Environment overrides:"
    echo "  AGENTIC_ANNOTATOR_VLLM_IMAGE=${IMAGE_NAME}"
    echo "  AGENTIC_ANNOTATOR_VLLM_MODEL=${MODEL}"
    echo "  AGENTIC_ANNOTATOR_VLLM_PORT=${PORT}"
    exit 1
}

MODE="${1:-start}"

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
    usage
fi

if [[ "${MODE}" == "status" ]]; then
    container_info="$(docker ps --filter "name=${CONTAINER_NAME}" --format "{{.ID}}\t{{.Status}}\t{{.Names}}" 2>/dev/null || true)"
    if [[ -z "${container_info}" ]]; then
        echo "vLLM container: not running"
        exit 1
    fi
    echo "vLLM container: ${container_info}"
    http_code="$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null || true)"
    if [[ "${http_code}" == "200" ]]; then
        echo "Health endpoint: OK (HTTP ${http_code})"
        curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || true
        exit 0
    fi
    echo "Health endpoint: not ready (HTTP ${http_code:-N/A})"
    exit 1
fi

if [[ "${MODE}" == "wait" ]]; then
    deadline=$((SECONDS + ${AGENTIC_ANNOTATOR_VLLM_WAIT_TIMEOUT:-300}))
    while (( SECONDS < deadline )); do
        container_info="$(docker ps --filter "name=${CONTAINER_NAME}" --format "{{.ID}}\t{{.Status}}\t{{.Names}}" 2>/dev/null || true)"
        if [[ -n "${container_info}" ]]; then
            http_code="$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null || true)"
            if [[ "${http_code}" == "200" ]]; then
                echo "vLLM ready: ${container_info}"
                exit 0
            fi
            echo "vLLM loading: HTTP ${http_code:-000}"
        else
            echo "vLLM container: not running"
        fi
        sleep 1
    done
    echo "Timed out waiting for vLLM health endpoint on port ${PORT}" >&2
    exit 1
fi

if [[ "${MODE}" == "ensure" ]]; then
    if "$0" status; then
        exit 0
    fi
    "$0" start "${@:2}" &
    "$0" wait
    exit $?
fi

if [[ "${MODE}" == "stop" ]]; then
    docker stop "${CONTAINER_NAME}"
    exit 0
fi

if [[ "${MODE}" != "start" ]]; then
    usage
fi

if [[ "$(uname -m)" == "aarch64" ]]; then
    gpu_flag=(--runtime=nvidia)
    default_util=0.4
else
    gpu_flag=(--gpus all)
    default_util=0.6
fi

gpu_utilization="${2:-${default_util}}"
max_model_len="${3:-32768}"
tensor_parallel="${4:-}"

docker_args=(
    "${gpu_flag[@]}"
    --rm
    --name "${CONTAINER_NAME}"
    --ipc=host
    --network=host
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface"
    -e HF_HOME=/root/.cache/huggingface
)

cache_dir="${HOME}/.cache/huggingface/hub/models--${MODEL//\//--}"
if [[ -d "${cache_dir}" ]]; then
    docker_args+=(-e HF_HUB_OFFLINE=1)
fi

if [[ "$(uname -m)" == "aarch64" ]]; then
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
        || echo "(tip: 'sudo sh -c \"sync; echo 3 > /proc/sys/vm/drop_caches\"' can free more GPU memory)"
fi

cmd=(
    vllm serve "${MODEL}"
    --port "${PORT}"
    --dtype auto
    --gpu-memory-utilization "${gpu_utilization}"
    --max-model-len "${max_model_len}"
    --trust-remote-code
)
if [[ -n "${tensor_parallel}" ]]; then
    cmd+=(--tensor-parallel-size "${tensor_parallel}")
fi

echo "Serving ${MODEL} on port ${PORT}"
echo "Image: ${IMAGE_NAME}"
echo "GPU utilization: ${gpu_utilization}, max model len: ${max_model_len}"
exec docker run "${docker_args[@]}" "${IMAGE_NAME}" "${cmd[@]}"
