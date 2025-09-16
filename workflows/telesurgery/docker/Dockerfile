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

ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.2.0-dgpu

# HSB Build
FROM $BASE_IMAGE AS hsbbuild
ARG HSB_REPO_URL
ENV HSB_REPO_URL=${HSB_REPO_URL:-https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git}
ARG HSB_BRANCH
ENV HSB_BRANCH=${HSB_BRANCH:-2.2.0-EA}

RUN git clone --branch ${HSB_BRANCH} --single-branch  ${HSB_REPO_URL} /opt/hololink
RUN cd /opt/hololink/python  && $INSTALL_ENVIRONMENT pip3 wheel -v .

# Main Docker
FROM $BASE_IMAGE
WORKDIR /workspace
# Install uv using curl for openpi
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y \
        curl \
        cmake \
        wget \
        build-essential \
        pybind11-dev \
        lsb-release \
        libglib2.0-0 \
        libdbus-1-3 \
        libopengl0 \
        libxcb-keysyms1 \
        libxcb-cursor0 \
        ninja-build \
        libgl1-mesa-dev \
        ffmpeg \
        v4l-utils \
        gcc-12 \
        g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

ENV PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

COPY workflows/telesurgery/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install "numpy<2.0.0" \
    && python3 -m pip install --no-build-isolation pynvjpeg==0.0.13 \
    && python3 -m pip install -r /tmp/requirements.txt

RUN mkdir -p /root/rti
ENV PATH=/isaac-sim/kit/python/bin/:/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/opt/nvidia/holoscan/lib/:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV LD_LIBRARY_PATH=\
/workspace/i4h-workflows/workflows/telesurgery/scripts/holohub/operators/camera/aja_source/lib:\
/workspace/i4h-workflows/workflows/telesurgery/scripts/holohub/operators/nvidia_video_codec/libs:\
$LD_LIBRARY_PATH

ENV RTI_LICENSE_FILE=/root/rti/rti_license.dat
ENV PYTHONPATH=/workspace/i4h-workflows/workflows/telesurgery/scripts

WORKDIR /workspace/i4h-workflows/workflows/telesurgery/scripts

COPY --from=hsbbuild /opt/hololink/python/hololink-*.whl /tmp/wheel/
RUN WHEEL=$(find /tmp/wheel -type f -name "hololink-*.whl") && pip3 install ${WHEEL}

RUN pip uninstall cuda-python -y && pip install cuda-python==12.9.0
