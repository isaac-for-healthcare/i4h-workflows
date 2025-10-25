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

# Source common utilities
source $SCRIPT_DIR/utils.sh

HOLOHUB_DIR=$SCRIPT_DIR/../scripts/holohub
DOCKER_IMAGE=telesurgery:0.3
CONTAINER_NAME=telesurgery-sim

function build() {
  if [ -L $SCRIPT_DIR/../../../third_party/IssacLab ]; then
    rm $SCRIPT_DIR/../../../third_party/IssacLab
  fi
  docker build -t $DOCKER_IMAGE -f workflows/telesurgery/docker/Dockerfile.sim .
  download_operators "holohub_nv_video_codec_operators_py311.zip"
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
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e DISPLAY \
    -e XDG_RUNTIME_DIR \
    -e XDG_SESSION_TYPE \
    -e PATIENT_IP="$PATIENT_IP" \
    -e SURGEON_IP="$SURGEON_IP" \
    $OTHER_ARGS \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /dev:/dev \
    -v ~/docker/telesurgery/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/telesurgery/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/telesurgery/cache/i4h-assets:/root/.cache/i4h-assets:rw \
    -v ~/docker/telesurgery/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/telesurgery/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/telesurgery/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/telesurgery/data:/root/.local/share/ov/data:rw \
    -v ~/docker/telesurgery/documents:/root/Documents:rw \
    -v ~/docker/telesurgery/logs:/root/.nvidia-omniverse/logs:rw \
    -v $(pwd):/workspace/i4h-workflows \
    -v ${RTI_LICENSE_FILE}:/root/rti/rti_license.dat \
    $(for dev in /dev/video*; do echo --device=$dev; done) \
    $OTHER_ARGS \
    $DOCKER_IMAGE \
    -c "source /workspace/i4h-workflows/workflows/telesurgery/scripts/env.sh && /workspace/i4h-workflows/workflows/telesurgery/docker/sim.sh init && exec bash"
}

function enter() {
  docker exec -it $CONTAINER_NAME /bin/bash
}

function init() {
  echo "Initializing Telesurgery environment ..."
  mkdir -p $SCRIPT_DIR/../../../third_party
  ln -s /workspace/isaaclab /workspace/i4h-workflows/third_party/IssacLab

  if [ ! -d "/root/.cache/i4h-assets/$ISAAC_ASSET_SHA256_HASH" ]; then
    echo "Please wait while downloading i4h-assets (Props)..."
    yes Yes | $PYTHON_EXECUTABLE -m i4h_asset_helper.cli --sub-path Props | grep -v "Skipping download"
    echo "Please wait while downloading i4h-assets (Robots)..."
    yes Yes | $PYTHON_EXECUTABLE -m i4h_asset_helper.cli --sub-path Robots | grep -v "Skipping download"
  fi
  echo "   Patient IP:       ${PATIENT_IP}"
  echo "   Surgeon IP:       ${SURGEON_IP}"
  echo "   DDS Discovery IP: ${NDDS_DISCOVERY_PEERS}"
}

$@
