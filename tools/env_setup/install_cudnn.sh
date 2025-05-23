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

set -e

function get_cudnn_version() {
    echo 
}


# Check if CUDNN_PATH and CUDNN_INCLUDE_PATH are set. If so, skip the installation.
if [ -d "$CUDNN_PATH" ] && [ -d "$CUDNN_INCLUDE_PATH" ]; then
    echo "CUDNN_PATH and CUDNN_INCLUDE_PATH are set. Skipping CUDNN installation."
    exit 0
fi

# Check if CUDA_HOME is set. If not, set it to /usr/local/cuda.
if [ -z "$CUDA_HOME" ]; then
    echo "CUDA_HOME is not set. Setting it to /usr/local/cuda."
    export CUDA_HOME=/usr/local/cuda
fi

# Get the cuDNN version and see if it is 8.9.7. If it is, skip the installation.
CUDNN_VERSION="$(grep -oP '#define CUDNN_MAJOR \K\d+' $CUDA_HOME/include/cudnn_version.h).$(grep -oP '#define CUDNN_MINOR \K\d+' $CUDA_HOME/include/cudnn_version.h).$(grep -oP '#define CUDNN_PATCHLEVEL \K\d+' $CUDA_HOME/include/cudnn_version.h)"
if [ "$CUDNN_VERSION" == "8.9.7" ]; then
    echo "cuDNN version is 8.9.7. Skipping CUDNN installation."
    exit 0
fi

# Install CUDNN
echo "Install cuDNN 8.9.7..."

mkdir -p /tmp/cudnn

wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz \
    -O /tmp/cudnn/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
tar -xvf /tmp/cudnn/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz -C /tmp/cudnn/

if [ "$EUID" -ne 0 ]; then
    sudo cp /tmp/cudnn/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn* $CUDA_HOME/lib64/
    sudo cp /tmp/cudnn/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h $CUDA_HOME/include/
    sudo ldconfig $CUDA_HOME/lib64
else
    cp /tmp/cudnn/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn* $CUDA_HOME/lib64/
    cp /tmp/cudnn/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h $CUDA_HOME/include/
    ldconfig $CUDA_HOME/lib64
fi

rm -rf /tmp/cudnn

echo "cuDNN is installed to $CUDA_HOME/lib64 and headers are in $CUDA_HOME/include"
