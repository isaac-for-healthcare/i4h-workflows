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
DOCKER_IMAGE=telesurgery:1.0
CONTAINER_NAME=telesurgery

function download_operators() {
  VIDEO_CODEC_FILE=/tmp/nv_video_codec.zip

  if [ ! -d "$HOLOHUB_DIR/operators/nvidia_video_codec/lib" ]; then
    echo "Downloading NVIDIA Video Codec Operators"
    curl -L -o $VIDEO_CODEC_FILE 'https://edge.urm.nvidia.com:443/artifactory/sw-holoscan-cli-generic/holohub/operators/nv_video_codec_0.2.zip'
    unzip -o $VIDEO_CODEC_FILE -d $HOLOHUB_DIR/operators/nvidia_video_codec
    rm $VIDEO_CODEC_FILE
  fi
}

function build() {
  if [ -L $SCRIPT_DIR/../../../third_party/IssacLab ]; then
    rm $SCRIPT_DIR/../../../third_party/IssacLab
  fi
  docker build --ssh default -t $DOCKER_IMAGE -f workflows/telesurgery/docker/Dockerfile .
  download_operators
}

function run() {
  if [ ! -f "$RTI_LICENSE_FILE" ]; then
    echo "RTI_LICENSE_FILE is not set or does not exist"
    exit 1
  fi
  xhost +
  docker run --rm -ti \
    --gpus all \
    --entrypoint "/bin/bash" \
    --ipc=host \
    --network=host \
    --privileged \
    --volume $SSH_AUTH_SOCK:/ssh-agent \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
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
    -v ~/docker/telesurgery/.cache:/root/.cache \
    -v ~/docker/telesurgery/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/telesurgery/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/telesurgery/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/telesurgery/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/telesurgery/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/telesurgery/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/telesurgery/data:/root/.local/share/ov/data:rw \
    -v ~/docker/telesurgery/documents:/root/Documents:rw \
    -v $(pwd):/workspace/i4h-workflows \
    -v ${RTI_LICENSE_FILE}:/root/rti/rti_license.dat \
    $(for dev in /dev/video*; do echo --device=$dev; done) \
    $DOCKER_IMAGE \
    -c "/workspace/i4h-workflows/workflows/telesurgery/scripts/env.sh && /workspace/i4h-workflows/workflows/telesurgery/docker/setup.sh init && exec bash"
}


function init() {
  echo "Initializing Telesurgery environment ..."
  mkdir -p $SCRIPT_DIR/../../../third_party
  ln -s /workspace/isaaclab /workspace/i4h-workflows/third_party/IssacLab
}

$@
