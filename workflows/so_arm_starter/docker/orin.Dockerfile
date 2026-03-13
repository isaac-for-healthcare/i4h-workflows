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
      patch \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

# So Python finds decord (and other --user installs) from later steps and at runtime
ENV PYTHONUSERBASE=/root/.local

# Install cuDSS (CUDA Direct Solver library for dense and sparse linear systems)
RUN wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/ && \
    chmod 777 /tmp && \
    apt-get update && \
    apt-get -y install cudss && \
    rm -f cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

ARG GR00T_COMMIT=aa6441feb4f08233d55cbfd2082753cdc01fa676
COPY workflows/so_arm_starter/docker/orin.patch /tmp/orin.patch
RUN cd /tmp && \
    git clone https://github.com/NVIDIA/Isaac-GR00T.git Isaac-GR00T && \
    cd Isaac-GR00T && \
    git checkout ${GR00T_COMMIT} && \
    patch -Np1 -i /tmp/orin.patch && \
    export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126 && \
    export PIP_EXTRA_INDEX_URL=https://pypi.org/simple && \
    export PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io && \
    pip3 install --upgrade pip setuptools && \
    pip3 install -e .[orin]


RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# Build and install decord
RUN git clone --depth 1 --branch n4.4.2 https://github.com/FFmpeg/FFmpeg.git ffmpeg && \
    cd ffmpeg && \
    ./configure --enable-shared --enable-pic --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    python3 setup.py install --user && \
    rm -rf ffmpeg decord

# Install lerobot
RUN cd /tmp && \
    git clone https://github.com/huggingface/lerobot.git && \
    cd lerobot && \
    git checkout 483be9aac217c2d8ef16982490f22b2ad091ab46 && \
    pip install -e ".[feetech]"

RUN CAMERA_FILE=$(python3 -c "import lerobot.common.cameras.opencv.camera_opencv as m; import os; print(os.path.dirname(m.__file__))")/camera_opencv.py && \
    sed -i '/self._configure_capture_settings()/i\        self.videocapture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('\''M'\'', '\''J'\'', '\''P'\'', '\''G'\''))' "$CAMERA_FILE"

RUN pip3 uninstall -y torch torchvision && \
    pip3 install --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126 torch==2.8.0 torchvision==0.23.0

RUN pip3 install "numpy<2.0"
RUN pip3 install holoscan-cu12==3.7.0 && \
    # Run Holoscan SDK post-install setup
    python3 -c "pass"

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.local/decord/:/usr/local/lib/python3.10/dist-packages/torch/lib

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
