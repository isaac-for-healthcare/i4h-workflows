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

function run() {
  docker run --gpus all --rm -ti \
    --ipc=host \
    --net=host \
    $(for dev in /dev/video*; do echo --device=$dev; done) \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --privileged \
    -v /dev:/dev \
    -v $(pwd):/workspace/i4h-workflows \
    -w /workspace/i4h-workflows \
    nvcr.io/nvidia/clara-holoscan/holoscan:v3.2.0-dgpu \
    bash
}

function init() {
  rm -rf .venv uv.lock
  apt update
  apt install v4l-utils ffmpeg -y

  v4l2-ctl --list-devices

  # Install Miniconda
  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-`uname -m`.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm ~/miniconda3/miniconda.sh
}

$@
