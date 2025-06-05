#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/" >/dev/null 2>&1 && pwd)"

# Source environment variables at script level
source $SCRIPT_DIR/../scripts/env.sh

HOLOHUB_DIR=$SCRIPT_DIR/../scripts/holohub
DOCKER_IMAGE=telesurgery:0.2
CONTAINER_NAME=telesurgery

get_host_gpu() {
    if ! command -v nvidia-smi >/dev/null; then
        echo "Could not find any GPU drivers on host. Defaulting build to target dGPU/CPU stack." >&2
        echo -n "dgpu";
    elif [[ ! $(nvidia-smi --query-gpu=name --format=csv,noheader) ]] || \
         [[ $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) =~ "Orin (nvgpu)" ]]; then
        echo -n "igpu";
    else
        echo -n "dgpu";
    fi
}

function download_operators() {
  VIDEO_CODEC_FILE=/tmp/nv_video_codec.zip

  if [ "$(uname -m)" == "aarch64" ]; then
    echo "Skipping NVIDIA Video Codec Operators for aarch64"
    return
  fi

  if [ ! -d "$HOLOHUB_DIR/operators/nvidia_video_codec/lib" ]; then
    echo "Downloading NVIDIA Video Codec Operators"
    curl -L -o $VIDEO_CODEC_FILE 'https://edge.urm.nvidia.com:443/artifactory/sw-holoscan-cli-generic/holohub/operators/nv_video_codec_0.2.zip'
    unzip -o $VIDEO_CODEC_FILE -d $HOLOHUB_DIR/operators/nvidia_video_codec
    rm $VIDEO_CODEC_FILE
  fi
}

function build() {
  BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.2.0-$(get_host_gpu)
  echo "Building Telesurgery Docker Image using ${BASE_IMAGE}"
  docker build --ssh default \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    -t $DOCKER_IMAGE \
    -f workflows/telesurgery/docker/Dockerfile .
  download_operators
}

function run() {
  if [ ! -f "$RTI_LICENSE_FILE" ]; then
    echo "RTI_LICENSE_FILE is not set or does not exist"
    exit 1
  fi

  xhost +

  local OTHER_ARGS=""
  if [ ! -z "${NTP_SERVER_HOST}" ] && [ ! -z "${NTP_SERVER_PORT}" ]; then
    OTHER_ARGS="-e NTP_SERVER_HOST=${NTP_SERVER_HOST} -e NTP_SERVER_PORT=${NTP_SERVER_PORT}"
  fi

  docker run --rm -ti \
    --runtime=nvidia \
    --gpus all \
    --entrypoint "/bin/bash" \
    --ipc=host \
    --network=host \
    --privileged \
    --volume $SSH_AUTH_SOCK:/ssh-agent \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e DISPLAY \
    -e XDG_RUNTIME_DIR \
    -e XDG_SESSION_TYPE \
    -e SSH_AUTH_SOCK=/ssh-agent \
    -e NDDS_DISCOVERY_PEERS="$NDDS_DISCOVERY_PEERS" \
    -e PATIENT_IP="$PATIENT_IP" \
    -e SURGEON_IP="$SURGEON_IP" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /dev:/dev \
    -v $(pwd):/workspace/i4h-workflows \
    -v ${RTI_LICENSE_FILE}:/root/rti/rti_license.dat \
    $(for dev in /dev/video*; do echo --device=$dev; done) \
    $OTHER_ARGS \
    $DOCKER_IMAGE \
    -c "/workspace/i4h-workflows/workflows/telesurgery/scripts/env.sh && /workspace/i4h-workflows/workflows/telesurgery/docker/real.sh init && exec bash"
}

function init() {
  echo "Initializing Telesurgery environment ..."
}

$@
