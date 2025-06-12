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
DOCKER_IMAGE=telesurgery:0.2
CONTAINER_NAME=telesurgery

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
    -c "source /workspace/i4h-workflows/workflows/telesurgery/scripts/env.sh && /workspace/i4h-workflows/workflows/telesurgery/docker/real.sh init && exec bash"
}

function init() {
  echo "Initializing Telesurgery environment ..."
  echo "   Patient IP:       ${PATIENT_IP}"
  echo "   Surgeon IP:       ${SURGEON_IP}"
  echo "   DDS Discovery IP: ${NDDS_DISCOVERY_PEERS}"
}

$@
