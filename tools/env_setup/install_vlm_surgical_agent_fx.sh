#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,

set -e

# Display usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup and run VLM Surgical Agent Framework containers.

OPTIONS:
    (no flags)  Check and build missing images, then run all default containers (WebRTC USB Camera is disabled unless -w is used)
    -f          Force rebuild all enabled images, then run all enabled containers
    -r          Force rebuild and refresh only the UI container
    -w          Enable and run the WebRTC USB Camera service in addition to the default containers
    -h          Show this help message

EXAMPLES:
    $0          # Normal run: build missing images and start default services (excluding WebRTC USB Camera)
    $0 -f       # Force rebuild and start all enabled containers (excluding WebRTC USB Camera unless -w is used)
    $0 -r       # Quick UI refresh for development
    $0 -w       # Include and start the WebRTC USB Camera service
    $0 -f -w    # Force rebuild and start all enabled containers, including WebRTC USB Camera
    $0 -h       # Show this help

EOF
    exit 0
}

# Parse flags
FORCE_BUILD=false
REFRESH_UI=false
ENABLE_WEBRTC=false
while getopts "frhw" opt; do
    case $opt in
        f) FORCE_BUILD=true ;;
        r) REFRESH_UI=true ;;
        w) ENABLE_WEBRTC=true ;;
        h) show_usage ;;
        *) echo "Usage: $0 [-f] [-r] [-w] [-h]"; echo "Use -h for help."; exit 1 ;;
    esac
done

# Get the parent directory of the current script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../../ && pwd)"

# Source shared bash utilities so helper functions (e.g., clone_if_missing) are available
# shellcheck source=/dev/null
source "$PROJECT_ROOT/tools/env_setup/bash_utils.sh"

# Define the VLM framework directory and ensure its parent directory exists
VLM_FX_DIR="$PROJECT_ROOT/third_party/VLM-Surgical-Agent-Framework"
mkdir -p "$(dirname "$VLM_FX_DIR")"
clone_if_missing "$VLM_FX_DIR" "https://github.com/Project-MONAI/VLM-Surgical-Agent-Framework.git" "9fc1cfb"

# Check if a Docker image exists locally
# Usage: image_exists <image_name>
image_exists() {
    local image="$1"
    if docker images -q "$image" 2>/dev/null | grep -q .; then
        return 0
    else
        return 1
    fi
}

# Determine the vLLM image name based on architecture
# This mirrors the logic in run-surgical-agents.sh
get_vllm_image_name() {
    local arch=$(uname -m)
    local is_igx_thor=false

    # Detect if running on IGX Thor
    if [ -f /proc/device-tree/model ]; then
        local device_model=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
        if [[ "$device_model" =~ "Thor" ]] || [[ "$device_model" =~ "IGX" ]]; then
            is_igx_thor=true
        fi
    fi

    # Set vLLM image based on architecture and platform
    if [[ "$arch" == "x86_64" ]]; then
        echo "vllm/vllm-openai:latest"
    elif [[ "$is_igx_thor" == true ]]; then
        echo "nvcr.io/nvidia/vllm:25.11-py3"
    elif [[ "$arch" == "aarch64" ]]; then
        echo "vlm-surgical-agents:vllm-openai-v0.8.3-dgpu"
    else
        echo "vlm-surgical-agents:vllm-openai-v0.8.3-dgpu"
    fi
}

# Wait for an HTTP endpoint to return a 200 status code.
# Usage: wait_for_http <url> <service_name> [timeout_seconds]
wait_for_http() {
    local url="$1"
    local name="$2"
    local timeout="${3:-300}"
    local interval=5
    local elapsed=0

    echo "Waiting for $name to be ready at $url (timeout: ${timeout}s)..."
    while [ $elapsed -lt $timeout ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "$name is ready (took ${elapsed}s)."
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    echo "ERROR: $name did not become ready within ${timeout}s."
    return 1
}

# Wait for a TCP port to accept connections.
# Usage: wait_for_port <host> <port> <service_name> [timeout_seconds]
wait_for_port() {
    local host="$1"
    local port="$2"
    local name="$3"
    local timeout="${4:-60}"
    local interval=3
    local elapsed=0

    echo "Waiting for $name to be ready at $host:$port (timeout: ${timeout}s)..."
    while [ $elapsed -lt $timeout ]; do
        if (echo > /dev/tcp/$host/$port) 2>/dev/null; then
            echo "$name is ready (took ${elapsed}s)."
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    echo "ERROR: $name did not become ready within ${timeout}s."
    return 1
}

# Set a default GPU memory utilization if not already provided by the user.
if [ -z "${GPU_MEMORY_UTILIZATION+x}" ]; then
    export GPU_MEMORY_UTILIZATION=0.5
fi

if [ -z "${RESPONSE_API_USE}" ]; then
    export RESPONSE_API_USE=disable
fi

export AGENT_PLUGIN_DIRS="$PROJECT_ROOT/workflows/rheo/agents"

pushd "$VLM_FX_DIR"/docker

# Build containers if they don't exist or if force build is requested
echo "Checking container images..."

# Determine the vLLM image name for this architecture
VLLM_IMAGE=$(get_vllm_image_name)
echo "Expected vLLM image: $VLLM_IMAGE"

# Check and build vLLM
if [ "$FORCE_BUILD" = true ]; then
    echo "Force build enabled: Building vLLM..."
    ./run-surgical-agents.sh build vllm
elif ! image_exists "$VLLM_IMAGE"; then
    echo "vLLM image not found. Building..."
    ./run-surgical-agents.sh build vllm
else
    echo "vLLM image exists. Skipping build."
fi

# Check and build Whisper
if [ "$FORCE_BUILD" = true ]; then
    echo "Force build enabled: Building Whisper..."
    ./run-surgical-agents.sh build whisper
elif ! image_exists "vlm-surgical-agents:whisper-dgpu"; then
    echo "Whisper image not found. Building..."
    ./run-surgical-agents.sh build whisper
else
    echo "Whisper image exists. Skipping build."
fi

# Check and build TTS
if [ "$FORCE_BUILD" = true ]; then
    echo "Force build enabled: Building TTS..."
    ./run-surgical-agents.sh build tts
elif ! image_exists "vlm-surgical-agents:tts"; then
    echo "TTS image not found. Building..."
    ./run-surgical-agents.sh build tts
else
    echo "TTS image exists. Skipping build."
fi

# Check and build WebRTC USB Camera (only if -w flag is set)
if [ "$ENABLE_WEBRTC" = true ]; then
    if [ "$FORCE_BUILD" = true ]; then
        echo "Force build enabled: Building WebRTC USB Camera..."
        ./run-surgical-agents.sh build webrtc_usbcam
    elif ! image_exists "vlm-surgical-agents:webrtc-usbcam"; then
        echo "WebRTC USB Camera image not found. Building..."
        ./run-surgical-agents.sh build webrtc_usbcam
    else
        echo "WebRTC USB Camera image exists. Skipping build."
    fi
else
    echo "WebRTC USB Camera disabled. Use -w flag to enable."
fi

# Check and build UI
if [ "$FORCE_BUILD" = true ] || [ "$REFRESH_UI" = true ]; then
    if [ "$REFRESH_UI" = true ]; then
        echo "UI refresh requested: Force building UI..."
    else
        echo "Force build enabled: Building UI..."
    fi
    ./run-surgical-agents.sh build ui
elif ! image_exists "vlm-surgical-agents:ui"; then
    echo "UI image not found. Building..."
    ./run-surgical-agents.sh build ui
else
    echo "UI image exists. Skipping build."
fi

# Run in dependency order: backends first, then frontend
# If refresh UI is requested, only restart the UI container
if [ "$REFRESH_UI" = true ]; then
    echo "Refreshing UI container only..."
    ./run-surgical-agents.sh run ui
    echo "UI container refreshed successfully!"
else
    # Normal flow: start all containers
    ./run-surgical-agents.sh run vllm
    wait_for_http "http://localhost:8000/health" "vLLM server" 300

    ./run-surgical-agents.sh run whisper
    wait_for_port "localhost" 43001 "Whisper server" 60

    ./run-surgical-agents.sh run tts

    # Only run WebRTC USB Camera if -w flag is set
    if [ "$ENABLE_WEBRTC" = true ]; then
        ./run-surgical-agents.sh run webrtc_usbcam
    fi

    ./run-surgical-agents.sh run ui
fi
popd
