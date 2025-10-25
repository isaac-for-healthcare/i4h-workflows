# syntax=docker/dockerfile:1.4

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

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Optional PyPI mirror configuration
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG PIP_TRUSTED_HOST

SHELL ["/bin/bash", "-c"]

# Install all packages in a single layer to avoid caching issues
RUN apt-get update && \
    apt-get install -y \
        wget \
        curl \
        jq \
        vim \
        git \
        xvfb \
        build-essential \
        cmake \
        vulkan-tools \
        unzip \
        lsb-release \
        libglib2.0-0 \
        libdbus-1-3 \
        libopengl0 \
        libxcb-keysyms1 \
        libglu1-mesa \
        gcc-12 \
        g++-12 \
        libgcc-s1 \
        libstdc++6 \
        ca-certificates && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120 && \
    rm -rf /var/lib/apt/lists/*

# all devices should be visible
ENV NVIDIA_VISIBLE_DEVICES=all
# set 'compute' driver cap to use Cuda
# set 'video' driver cap to use the video encoder
# set 'graphics' driver cap to use OpenGL/EGL
# set 'display' to allow use of virtual display
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display

# Install Everything in conda environment

WORKDIR /workspace/i4h-workflows

COPY tools /workspace/i4h-workflows/tools
COPY workflows /workspace/i4h-workflows/workflows

# Install miniconda3 and create so_arm_starter
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh && \
    export PATH="/root/miniconda3/bin:$PATH" && \
    /root/miniconda3/bin/conda init bash && \
    conda tos accept && \
    conda create -n so_arm_starter python=3.11.13 -y

RUN source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate so_arm_starter && \
    cd /workspace/i4h-workflows && \
    BUILD_DOCKER_IMAGE=true bash tools/env_setup_so_arm_starter.sh

# Ensure system libstdc++/libgcc are used at runtime
RUN bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate so_arm_starter && \
    mv "$CONDA_PREFIX/lib/libstdc++.so.6" "$CONDA_PREFIX/lib/libstdc++.so.6.bak" 2>/dev/null || true && \
    mv "$CONDA_PREFIX/lib/libgcc_s.so.1" "$CONDA_PREFIX/lib/libgcc_s.so.1.bak" 2>/dev/null || true'

# Add conda to PATH and set default environment
ENV PATH="/root/miniconda3/bin:$PATH"
ENV CONDA_DEFAULT_ENV=so_arm_starter
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1

# Ensure conda environment is activated by default
RUN echo "conda activate so_arm_starter" >> ~/.bashrc

ENV PYTHONPATH=/workspace/i4h-workflows/workflows/so_arm_starter/scripts
ENV RTI_LICENSE_FILE=/root/rti/rti_license.dat
