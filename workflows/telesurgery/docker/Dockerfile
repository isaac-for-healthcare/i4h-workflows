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

ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.11.0-cuda12-dgpu
FROM $BASE_IMAGE

WORKDIR /workspace

# Install build/runtime dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        python3-dev \
        pybind11-dev \
        lsb-release \
        libopengl0 \
        libxcb-keysyms1 \
        libxcb-cursor0 \
        libgl1-mesa-dev \
        ffmpeg \
        libavformat-dev \
        libavcodec-dev \
        libavutil-dev \
        v4l-utils \
        libusb-1.0-0-dev \
        libudev-dev \
        pkg-config \
        libcurl4-openssl-dev \
        libyaml-cpp-dev \
        libgtk-3-dev \
        libglfw3-dev \
        libglu1-mesa-dev

# HOLOHUB (Holoscan Hub)
ARG HOLOHUB_REPO_URL=https://github.com/nvidia-holoscan/holohub.git
ARG HOLOHUB_BRANCH=holoscan-sdk-3.11.0
RUN git clone --branch $HOLOHUB_BRANCH --depth 1 $HOLOHUB_REPO_URL /opt/nvidia/holohub

# Build librealsense from source
RUN git clone --branch v2.55.1 --depth 1 https://github.com/IntelRealSense/librealsense.git /tmp/librealsense && \
    cd /tmp/librealsense && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON_BINDINGS:bool=true \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DBUILD_EXAMPLES=false \
        -DBUILD_GRAPHICAL_EXAMPLES=false && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/librealsense

# Install Python dependencies
COPY workflows/telesurgery/requirements.cu12.txt /tmp/requirements.txt
RUN sed -i -e '/pyrealsense2/d' /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt
RUN python3 -m pip install --no-build-isolation pynvjpeg==0.0.13 cmake

# HSB (Holoscan Sensor Bridge)
ARG HSB_REPO_URL=https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
ARG HSB_BRANCH=2.5.0
RUN git clone --branch $HSB_BRANCH --depth 1 $HSB_REPO_URL /opt/nvidia/hsb

# Apply Holoscan SDK 3.11 / CUDA 13 compatibility patch for yuan-hsb fork
COPY workflows/telesurgery/scripts/yuan_hsb_holoscan311.patch /tmp/yuan_hsb_holoscan311.patch
RUN if echo "$HSB_REPO_URL" | grep -q "Yuan"; then \
      cd /opt/nvidia/hsb && git apply /tmp/yuan_hsb_holoscan311.patch && \
      echo "Applied yuan-hsb Holoscan SDK 3.11 compatibility patch"; \
    fi
RUN cd /opt/nvidia/hsb/python && python3 -m pip install -v .

# Environment variables
RUN mkdir -p /root/rti
ENV RTI_LICENSE_FILE=/root/rti/rti_license.dat
ENV PYTHONPATH=/workspace/i4h-workflows/workflows/telesurgery/scripts

WORKDIR /workspace/i4h-workflows/workflows/telesurgery/scripts
