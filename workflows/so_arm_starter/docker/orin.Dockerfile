# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS orin_base

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-venv \
      libsm6 \
      libxext6 \
      libhdf5-serial-dev \
      libtesseract-dev \
      libgtk-3-0 \
      libtbb12 \
      libtbb2 \
      libatlas-base-dev \
      libopenblas-dev \
      build-essential \
      python3-setuptools \
      make \
      cmake \
      nasm \
      git \
      curl \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

ARG I4H_ROOT=/opt/i4h-workflows

# Clone GR00T and use its own Orin install script
ARG GR00T_COMMIT=4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e
RUN apt-get update && apt-get install -y --no-install-recommends git-lfs && rm -rf /var/lib/apt/lists/* && \
    git lfs install && \
    git clone https://github.com/NVIDIA/Isaac-GR00T.git ${I4H_ROOT}/third_party/Isaac-GR00T && \
    cd ${I4H_ROOT}/third_party/Isaac-GR00T && \
    git checkout ${GR00T_COMMIT} && \
    git lfs pull

# Install GR00T via its Orin install_deps.sh (uses uv + platform-specific pyproject.toml)
ENV DOCKER_CONTAINER=1
ENV UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv
RUN cd ${I4H_ROOT}/third_party/Isaac-GR00T && \
    bash scripts/deployment/orin/install_deps.sh

# Activate the venv by default
ENV VIRTUAL_ENV=/opt/gr00t-venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV CUDA_PATH=/usr/local/cuda-12.6
ENV CPATH="/usr/local/cuda-12.6/include:${CPATH:-}"
ENV C_INCLUDE_PATH="/usr/local/cuda-12.6/include:${C_INCLUDE_PATH:-}"
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda-12.6/include:${CPLUS_INCLUDE_PATH:-}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cu12/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudss/lib:${LD_LIBRARY_PATH:-}"

# Ensure pip is available in the venv
RUN /opt/gr00t-venv/bin/python3 -m ensurepip --upgrade && \
    /opt/gr00t-venv/bin/python3 -m pip install --upgrade pip

# Install lerobot
ARG LEROBOT_VERSION=483be9aac217c2d8ef16982490f22b2ad091ab46
RUN cd /tmp && \
    git clone https://github.com/huggingface/lerobot.git && \
    cd lerobot && \
    git checkout ${LEROBOT_VERSION} && \
    /opt/gr00t-venv/bin/python3 -m pip install -e ".[feetech]"

# Patch lerobot camera_opencv.py to set MJPEG format
RUN CAMERA_FILE=$(/opt/gr00t-venv/bin/python3 -c "import lerobot.common.cameras.opencv.camera_opencv as m; import os; print(os.path.dirname(m.__file__))")/camera_opencv.py && \
    sed -i '/self._configure_capture_settings()/i\        self.videocapture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('\''M'\'', '\''J'\'', '\''P'\'', '\''G'\''))' "$CAMERA_FILE"

RUN /opt/gr00t-venv/bin/python3 -m pip install "numpy<2.0"
RUN /opt/gr00t-venv/bin/python3 -m pip install holoscan-cu12==3.7.0 && \
    /opt/gr00t-venv/bin/python3 -c "pass"

##################################################################
# Error if attempting to use an unsupported mode on this platform
##################################################################
FROM ubuntu:22.04 AS isaaclab_installer

RUN echo "This app does not support simulation on Jetson Orin." \
    " Please use real hardware or refer to the Isaac for Healthcare documentation" \
    " for platforms supporting simulation tasks." \
    && exit 1

##################################################################
# Align default target stage name with I4H CLI mode arguments
##################################################################
FROM orin_base AS gr00t_installer
