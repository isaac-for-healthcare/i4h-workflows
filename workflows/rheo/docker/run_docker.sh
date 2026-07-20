#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

DOCKER_IMAGE_NAME='rheo'
DOCKER_VERSION_TAG='latest'

# Default Dockerfile (overridden by --cloth / -c, see below).
DOCKERFILE_NAME='Dockerfile.x86'

# ============================================
# Directory Structure
# ============================================
# i4h-workflows/           <- I4H_ROOT
# ├── third_party/
# │   ├── IsaacLab-Arena/
# │   ├── IsaacLab/
# │   └── Isaac-GR00T/
# │   └── RLinf/
# └── workflows/
#     └── rheo/
#         ├── scripts/
#         └── docker/

# Script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Workflow directory
WORKFLOW_DIR="${SCRIPT_DIR}/.."

# i4h-workflows root
I4H_ROOT="${WORKFLOW_DIR}/../.."

# third_party directory
THIRD_PARTY_DIR="${I4H_ROOT}/third_party"

# Component paths
ISAACLAB_ARENA_DIR="${THIRD_PARTY_DIR}/IsaacLab-Arena"
ISAACLAB_DIR="${THIRD_PARTY_DIR}/IsaacLab"
ISAACLAB_CLOTH_DIR="${THIRD_PARTY_DIR}/IsaacLab-3-0-dev"
ISAAC_GROOT_DIR="${THIRD_PARTY_DIR}/Isaac-GR00T"
RLINF_DIR="${THIRD_PARTY_DIR}/RLinf"

# Clone third_party repos if missing
mkdir -p "$THIRD_PARTY_DIR"
clone_if_missing() {
    [ -d "$1/.git" ] || { git clone "$2" "$1" && cd "$1" && git checkout "$3" && cd -; }
}

# ─────────────────────────────────────────────────────────────────────────────
# IsaacLab commit pins — picked based on mode further down (see --cloth flag).
# ─────────────────────────────────────────────────────────────────────────────
ISAACLAB_COMMIT_DEFAULT="941ebdf4ad1fbf89018777012bdfa4b5944c758f"
ISAACLAB_COMMIT_CLOTH="378dc59f05b43e18e53b65e2c6dcdff2ba072892"

clone_if_missing "$ISAACLAB_ARENA_DIR" "https://github.com/isaac-sim/IsaacLab-Arena.git" "dba09956588dddae52897820686efd329d85da12"
clone_if_missing "$ISAACLAB_DIR" "https://github.com/isaac-sim/IsaacLab.git" "${ISAACLAB_COMMIT_DEFAULT}"
clone_if_missing "$RLINF_DIR" "https://github.com/RLinf/RLinf.git" "649e7579775997ade74efff33a7c23e90c61e60a"

# Clone GR00T 1.5 (base branch for environment installation) if -g flag is used
[[ "$*" == *"-g"* ]] && clone_if_missing "$ISAAC_GROOT_DIR" "https://github.com/NVIDIA/Isaac-GR00T.git" "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"

# Additionally clone GR00T 1.6 if -g or -g1.6 (but not -g1.5)
if [[ "$*" == *"-g"* ]] && [[ "$*" != *"-g1.5"* ]]; then
    ISAAC_GROOT_1_6_DIR="${THIRD_PARTY_DIR}/Isaac-GR00T-1.6"
    clone_if_missing "$ISAAC_GROOT_1_6_DIR" "https://github.com/NVIDIA/Isaac-GR00T.git" "e8e625f4f21898c506a1d8f7d20a289c97a52acf"
fi

# Container workdir
WORKDIR="/workspaces"

# Default config
DATASETS_HOST_MOUNT_DIRECTORY="$HOME/datasets"
MODELS_HOST_MOUNT_DIRECTORY="$HOME/models"
EVAL_HOST_MOUNT_DIRECTORY="$HOME/eval"
INSTALL_GROOT="false"
GROOT_VERSION="1.6"
GPU_DEVICE=""
FORCE_REBUILD=false
NEW_CONTAINER=false

# Translate long flags to their short equivalents so the existing getopts
# parser keeps working unchanged. Only --cloth is recognized today; extend
# this loop if you add more long flags.
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --cloth) ARGS+=("-c") ;;
        --help)  ARGS+=("-h") ;;
        *)       ARGS+=("$arg") ;;
    esac
done
set -- "${ARGS[@]}"

while getopts ":d:m:e:hn:rn:Rn:vn:g:G:u:Nc" OPTION; do
    case $OPTION in
        d) DATASETS_HOST_MOUNT_DIRECTORY=$OPTARG ;;
        m) MODELS_HOST_MOUNT_DIRECTORY=$OPTARG ;;
        e) EVAL_HOST_MOUNT_DIRECTORY=$OPTARG ;;
        n) DOCKER_IMAGE_NAME=${OPTARG} ;;
        r) FORCE_REBUILD=true ;;
        R) FORCE_REBUILD=true; NO_CACHE="--no-cache" ;;
        v) set -x ;;
        g)
            INSTALL_GROOT="true"
            DOCKER_VERSION_TAG='cuda_gr00t'
            # Handle version argument: -g1.5, -g1.6, or -g (defaults to 1.6)
            if [ -n "$OPTARG" ]; then
                GROOT_VERSION="$OPTARG"
            else
                GROOT_VERSION="1.6"  # default
            fi
            ;;
        u) GPU_DEVICE=${OPTARG} ;;
        N) NEW_CONTAINER=true ;;
        c)
            # Cloth / IsaacSim-6 mode: build & enter Dockerfile.rheo_cloth.
            DOCKER_VERSION_TAG='cloth'
            DOCKERFILE_NAME='Dockerfile.rheo_cloth'
            ;;
        h)
            echo "Usage: $(basename "$0") [options]"
            echo ""
            echo "Options:"
            echo "  -d <datasets dir>  Path to datasets (default: $DATASETS_HOST_MOUNT_DIRECTORY)"
            echo "  -m <models dir>    Path to models (default: $MODELS_HOST_MOUNT_DIRECTORY)"
            echo "  -e <eval dir>      Path to evaluation data (default: $EVAL_HOST_MOUNT_DIRECTORY)"
            echo "  -n <name>          Docker image name (default: $DOCKER_IMAGE_NAME)"
            echo "  -r                 Force rebuild"
            echo "  -R                 Force rebuild without cache"
            echo "  -g [version]       Install GR00T. Version: 1.5 or 1.6 (default). Use -g, -g1.5, or -g1.6"
            echo "  -c, --cloth        Cloth / IsaacSim-6 mode (Dockerfile.rheo_cloth, tag '${DOCKER_IMAGE_NAME}:cloth')"
            echo "  -u <gpu>           GPU device number (default: all)"
            echo "  -N                 Force create a new container (add timestamp suffix)"
            echo "  -v                 Verbose output"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
        :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
    esac
done

shift $((OPTIND-1))

if [ "$DOCKERFILE_NAME" = "Dockerfile.rheo_cloth" ]; then
    ISAACLAB_COMMIT="${ISAACLAB_COMMIT_CLOTH}"
    ISAACLAB_DIR="${ISAACLAB_CLOTH_DIR}"
    ISAACLAB_PATH_IN_CONTAINER="${WORKDIR}/third_party/IsaacLab-3-0-dev"
    clone_if_missing "$ISAACLAB_DIR" "https://github.com/isaac-sim/IsaacLab.git" "${ISAACLAB_COMMIT}"
else
    ISAACLAB_COMMIT="${ISAACLAB_COMMIT_DEFAULT}"
    ISAACLAB_PATH_IN_CONTAINER="${WORKDIR}/third_party/IsaacLab"
fi

CURRENT_ISAACLAB_COMMIT=$(git -C "${ISAACLAB_DIR}" rev-parse HEAD 2>/dev/null || echo "")
if [ "${CURRENT_ISAACLAB_COMMIT}" != "${ISAACLAB_COMMIT}" ]; then
    echo "Switching IsaacLab to ${ISAACLAB_COMMIT} (was ${CURRENT_ISAACLAB_COMMIT:-unknown})"
    git -C "${ISAACLAB_DIR}" fetch --quiet origin "${ISAACLAB_COMMIT}" || true
    git -C "${ISAACLAB_DIR}" checkout --quiet "${ISAACLAB_COMMIT}"
fi

echo "=============================================="
echo "Docker Environment"
echo "=============================================="
echo "Docker image    : $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG"
echo "Dockerfile      : $DOCKERFILE_NAME"
echo "IsaacLab commit : $ISAACLAB_COMMIT"
echo "I4H root        : $I4H_ROOT"
echo "Third party     : $THIRD_PARTY_DIR"
echo "Workflow dir    : $WORKFLOW_DIR"

# Build Docker image
if [ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG 2> /dev/null)" ] && \
    [ "$FORCE_REBUILD" = false ]; then
    echo "Docker image already exists. Use -r to rebuild."
else
    echo "Building Docker image..."

    # Dockerfile.rheo_cloth pulls IsaacSim 6.x wheels from NVIDIA-internal
    # urm.nvidia.com, which only resolves via the host's DNS / /etc/hosts.
    # Use host network for the build container in that case so resolution works.
    # Dockerfile.x86 only hits public endpoints, so we keep its default
    # build network untouched to avoid surprising existing setups.
    BUILD_NETWORK_ARGS=()
    if [ "$DOCKERFILE_NAME" = "Dockerfile.rheo_cloth" ]; then
        BUILD_NETWORK_ARGS+=("--network=host")
    fi

    # Build context is i4h-workflows root
    docker build --pull \
        "${BUILD_NETWORK_ARGS[@]}" \
        $NO_CACHE \
        --build-arg WORKDIR="${WORKDIR}" \
        --build-arg INSTALL_GROOT=$INSTALL_GROOT \
        -t ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} \
        --file ${WORKFLOW_DIR}/docker/${DOCKERFILE_NAME} \
        ${I4H_ROOT}
fi

# GPU configuration
if [ -z "$GPU_DEVICE" ]; then
    GPU_ARGS="all"
    GPU_SUFFIX="all"
else
    GPU_ARGS="device=${GPU_DEVICE}"
    GPU_SUFFIX="$GPU_DEVICE"
fi

# Container name with optional timestamp suffix
CONTAINER_BASE_NAME="$DOCKER_IMAGE_NAME-$DOCKER_VERSION_TAG-$GPU_SUFFIX"
if [ "$NEW_CONTAINER" = true ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CONTAINER_NAME="${CONTAINER_BASE_NAME}-${TIMESTAMP}"
    echo "Creating new container: $CONTAINER_NAME"
else
    CONTAINER_NAME="$CONTAINER_BASE_NAME"
fi

# Remove exited containers (only if not forcing new container)
if [ "$NEW_CONTAINER" = false ] && [ "$(docker ps -a --quiet --filter status=exited --filter name=$CONTAINER_NAME)" ]; then
    docker rm $CONTAINER_NAME > /dev/null
fi

add_volume_if_it_exists() {
    local src="$1"
    local dst="$2"
    [ -d "$src" ] && echo "-v $src:$dst"
}

# Run container
if [ "$NEW_CONTAINER" = false ] && [ "$( docker container inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null)" = "true" ]; then
    # Pick a user that actually exists in the container so attach doesn't fail
    # when the image was built without an entrypoint that creates the host user
    # (e.g. Dockerfile.rheo_cloth, which has no useradd step and runs as root).
    EXEC_USER="$(id -un)"
    if ! docker exec "$CONTAINER_NAME" id -u "$EXEC_USER" > /dev/null 2>&1; then
        EXEC_USER="root"
    fi

    if [ $# -ge 1 ]; then
        echo "Container already running. Executing command as ${EXEC_USER}."
        docker exec -it "$CONTAINER_NAME" su "$EXEC_USER" -c "$*"
    else
        echo "Container already running. Attaching as ${EXEC_USER}."
        docker exec -it "$CONTAINER_NAME" su "$EXEC_USER"
    fi
else
    DOCKER_RUN_ARGS=("--name" "$CONTAINER_NAME"
                    "--ulimit" "memlock=-1"
                    "--ulimit" "stack=-1"
                    "--ipc=host"
                    "--net=host"
                    "--gpus" "${GPU_ARGS}"
                    # Whole workspace mapping (for real-time code sync during development)
                    "-v" "${I4H_ROOT}:${WORKDIR}"
                    "-w" /workspaces/workflows/rheo
                    # Datasets and models
                    $(add_volume_if_it_exists $DATASETS_HOST_MOUNT_DIRECTORY /datasets)
                    $(add_volume_if_it_exists $MODELS_HOST_MOUNT_DIRECTORY /models)
                    $(add_volume_if_it_exists $EVAL_HOST_MOUNT_DIRECTORY /eval)
                    # User config
                    "-v" "$HOME/.bash_history:/home/$(id -un)/.bash_history"
                    "-v" "$HOME/.cache:/home/$(id -un)/.cache"
                    "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
                    "-v" "$HOME/.Xauthority:/root/.Xauthority"
                    # CloudXR runtime config (sourced by bashrc inside container if present)
                    $(add_volume_if_it_exists "$HOME/.cloudxr" /root/.cloudxr)
                    # Assets bundle (URDFs, meshes, teleop configs, local USDs).
                    $(add_volume_if_it_exists "${ASSETS_DIR:-}" /assets)
                    # if H2_URDF_HOST_DIR is set on the host.
                    $(add_volume_if_it_exists "${H2_URDF_HOST_DIR:-}" /h2_urdf)
                    # Environment variables
                    "--env" "ASSETS_DIR=/assets"
                    "--env" "RHEO_H2_SHARPA_ASSETS_DIR=/assets"
                    "--env" "OMNI_USER=\$omni-api-token"
                    "--env" "OMNI_PASS"
                    "--env" "DATASET_DIR=/datasets"
                    "--env" "MODELS_DIR=/models"
                    "--env" "DISPLAY"
                    "--env" "ACCEPT_EULA=Y"
                    "--env" "PRIVACY_CONSENT=Y"
                    "--env" "DOCKER_RUN_USER_ID=$(id -u)"
                    "--env" "DOCKER_RUN_USER_NAME=$(id -un)"
                    "--env" "DOCKER_RUN_GROUP_ID=$(id -g)"
                    "--env" "DOCKER_RUN_GROUP_NAME=$(id -gn)"
                    "--env" "ISAACLAB_PATH=${ISAACLAB_PATH_IN_CONTAINER}"
                    )

    # Mount GR00T based on version (overrides the whole workspace mount for Isaac-GR00T specifically)
    if [ "$INSTALL_GROOT" = "true" ]; then
        if [ "$GROOT_VERSION" = "1.5" ]; then
            # GR00T 1.5: mount Isaac-GR00T to third_party/Isaac-GR00T
            GROOT_HOST_DIR="${THIRD_PARTY_DIR}/Isaac-GR00T"
            if [ -d "$GROOT_HOST_DIR" ]; then
                DOCKER_RUN_ARGS+=("-v" "${GROOT_HOST_DIR}:${WORKDIR}/third_party/Isaac-GR00T")
            else
                echo "Warning: GR00T 1.5 directory not found: $GROOT_HOST_DIR"
            fi
        else
            # GR00T 1.6 (default): mount Isaac-GR00T-1.6 to third_party/Isaac-GR00T
            GROOT_HOST_DIR="${THIRD_PARTY_DIR}/Isaac-GR00T-1.6"
            if [ -d "$GROOT_HOST_DIR" ]; then
                DOCKER_RUN_ARGS+=("-v" "${GROOT_HOST_DIR}:${WORKDIR}/third_party/Isaac-GR00T")
            else
                echo "Warning: GR00T 1.6 directory not found: $GROOT_HOST_DIR"
            fi
        fi
    fi

    # X11
    if [ -n "$DISPLAY" ]; then
        echo "Allowing X11 connections"
        xhost +local:docker > /dev/null
    fi

    if [ $# -ge 1 ]; then
        docker run "${DOCKER_RUN_ARGS[@]}" --interactive --rm --tty ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} "$*"
    else
        docker run "${DOCKER_RUN_ARGS[@]}" --interactive --rm --tty ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG}
    fi
fi
