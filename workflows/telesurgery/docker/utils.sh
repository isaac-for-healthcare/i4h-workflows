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

# Common utility functions for telesurgery workflow

function download_operators() {
  # Check if HOLOHUB_DIR is set, otherwise try to determine it
  if [ -z "$HOLOHUB_DIR" ]; then
    echo "Error: HOLOHUB_DIR is not set"
    return 1
  fi

  local FILE_NAME=holohub_nv_video_codec_operators_0.1.zip
  VIDEO_CODEC_FILENAME=/tmp/${FILE_NAME}
  UNZIP_DIR=/tmp/${FILE_NAME%%.*}

  if [ ! -d "$HOLOHUB_DIR/operators/nvidia_video_codec/libs" ]; then
    echo "Downloading NVIDIA Video Codec Operators"
    curl -L -o $VIDEO_CODEC_FILENAME "https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/holohub/operators/$FILE_NAME"
    unzip -o -q $VIDEO_CODEC_FILENAME -d $UNZIP_DIR
    rm $VIDEO_CODEC_FILENAME
    mv $UNZIP_DIR/$(uname -p)/* $HOLOHUB_DIR/operators/nvidia_video_codec
    rm -rf $UNZIP_DIR
  fi
}

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
