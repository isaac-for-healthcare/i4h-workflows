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
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
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
        libglu1-mesa && \
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

# Install miniconda3 and create robotic_surgery
RUN mkdir -p /opt/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh && \
    bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 && \
    rm /opt/miniconda3/miniconda.sh && \
    source /opt/miniconda3/bin/activate && \
    conda tos accept && \
    conda init --all && \
    conda create -n robotic_surgery python=3.11 -y && \
    chmod -R a+rX /opt/miniconda3

ENV PATH="/opt/miniconda3/envs/robotic_surgery/bin:/opt/miniconda3/bin:$PATH"
ENV CONDA_DEFAULT_ENV=robotic_surgery

RUN --mount=type=cache,target=/root/.cache/pip \
    source /opt/miniconda3/bin/activate && \
    conda activate robotic_surgery && \
    cd /workspace/i4h-workflows && \
    BUILD_DOCKER_IMAGE=true bash tools/env_setup_robot_surgery.sh

ENV PYTHONPATH=/workspace/i4h-workflows/workflows/robotic_surgery/scripts
