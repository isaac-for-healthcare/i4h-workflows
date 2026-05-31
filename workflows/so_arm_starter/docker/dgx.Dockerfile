#syntax=docker/dockerfile:1

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

FROM nvcr.io/nvidia/pytorch:25.08-py3 AS pytorch-base
FROM nvidia/cuda:13.1.1-base-ubuntu24.04 AS cuda-base

########################################################
# Install minimum tools for downloading dependencies
########################################################
FROM ubuntu:24.04 AS downloader-base
ARG DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=i4h-apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=i4h-apt-lib \
    apt-get update && \
    apt install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        git-lfs \
	patch

########################################################
# Download GR00T dependencies without installing them
########################################################
FROM downloader-base AS gr00t_downloader

ARG I4H_ROOT=/opt/i4h-workflows
WORKDIR ${I4H_ROOT}/third_party

ARG LEROBOT_VERSION=483be9aac217c2d8ef16982490f22b2ad091ab46
RUN git clone https://github.com/huggingface/lerobot.git lerobot \
    && cd lerobot && git checkout ${LEROBOT_VERSION}
# Patch lerobot camera_opencv.py to set MJPEG format
RUN CAMERA_FILE=$(find ${I4H_ROOT}/third_party/lerobot -name "camera_opencv.py") \
    && sed -i '/self._configure_capture_settings()/i\        self.videocapture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('\''M'\'', '\''J'\'', '\''P'\'', '\''G'\''))' "$CAMERA_FILE"

ARG GR00T_COMMIT=4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e
RUN git lfs install \
    && git clone https://github.com/NVIDIA/Isaac-GR00T Isaac-GR00T \
    && cd Isaac-GR00T
    && git checkout ${GR00T_COMMIT}
    # Pull parquet data files for demo
    && git lfs pull

########################################################
# Download Isaac Lab simulation dependencies without installing them
########################################################
FROM downloader-base AS isaaclab_downloader

ARG I4H_ROOT=/opt/i4h-workflows
WORKDIR ${I4H_ROOT}/third_party

ARG ISAACLAB_VERSION=release/2.3.0
RUN git clone -b ${ISAACLAB_VERSION} --progress https://github.com/isaac-sim/IsaacLab.git IsaacLab \
    && cd IsaacLab && git checkout ${ISAACLAB_VERSION}

ARG LEISAAC_VERSION=cd61a20c75f7b72c347538089602201349af6dc8
COPY tools/env_setup/patches/leisaac_hdf5_cuda_fix.patch ${I4H_ROOT}/third_party/leisaac_hdf5_cuda_fix.patch
RUN git clone --progress https://github.com/LightwheelAI/leisaac.git leisaac \
    && cd leisaac && git checkout ${LEISAAC_VERSION} \
    && if [ "$(uname -m)" = "aarch64" ]; then \
        patch -Np1 -i ${I4H_ROOT}/third_party/leisaac_hdf5_cuda_fix.patch --verbose || \
            { echo "Error: Failed to apply leisaac_hdf5_cuda_fix.patch."; exit 1; }; \
    fi

########################################################
# Set up for policy deployment with Isaac-GR00T and Lerobot
#
# Uses GR00T's Spark install script (aarch64, CUDA 13, Python 3.12)
# which handles uv-based dependency resolution with platform-specific
# pyproject.toml.
########################################################
FROM pytorch-base AS gr00t_installer

ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=i4h-apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=i4h-apt-lib \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      cmake \
      git \
      make \
      nasm \
      python3 \
      python3-pip \
      python3-setuptools \
      python3-venv \
      build-essential \
      libsm6 \
      libxext6 \
      libhdf5-serial-dev \
      libtesseract-dev \
      libgtk-3-0 \
      libtbb12 \
      libgl1 \
      libatlas-base-dev \
      libopenblas-dev \
      python3-dev \
      curl \
      ffmpeg \
      libxkbcommon-x11-0 \
      speech-dispatcher

RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install --upgrade \
        pip \
        setuptools \
        packaging>=24.2 \
    && pip uninstall -y cupy-cuda12x

ARG I4H_ROOT=/opt/i4h-workflows

# Install GR00T via its Spark install script (uv-based)
COPY --from=gr00t_downloader ${I4H_ROOT}/third_party/Isaac-GR00T ${I4H_ROOT}/third_party/Isaac-GR00T
WORKDIR ${I4H_ROOT}/third_party/Isaac-GR00T
RUN bash scripts/deployment/spark/install_deps.sh

# Install lerobot
COPY --from=gr00t_downloader ${I4H_ROOT}/third_party/lerobot ${I4H_ROOT}/third_party/lerobot
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install poetry-core "huggingface-hub<=0.34" \
    && pip install --no-build-isolation \
        -e "${I4H_ROOT}/third_party/lerobot[feetech]" \
        "rti.connext==7.3.0" \
        "numpy<2.0" \
        "cupy-cuda13x~=13.6" \
        "holoscan-cu13==3.7.0" \
    && pip uninstall -y apex \
    && python3 -c "pass"     # Complete Holoscan SDK post-install setup

# Always use APT ninja-build so that tooling is aligned across containers.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=i4h-apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=i4h-apt-lib \
    apt update && apt install -y ninja-build

# Install gr00t runtime deps into system Python (torch/torchvision already present from base)
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install --no-cache-dir \
        "albumentations==1.4.18" \
        "opencv-python-headless>=4.5,<4.13" \
        "av==16.1.0" \
        "diffusers==0.35.1" \
        "dm-tree" \
        "lmdb==1.7.5" \
        "msgpack==1.1.0" \
        "msgpack-numpy==0.4.8" \
        "peft==0.17.1" \
        "termcolor==3.2.0" \
        "transformers==4.57.3" \
        "tyro==0.9.17" \
        "click==8.1.8" \
        "datasets>=3.0" \
        "einops==0.8.1" \
        "gymnasium>=1.0" \
        "omegaconf==2.3.0" \
        "accelerate>=1.0" \
        "safetensors" \
        "jsonlines" \
        "numpy<2.0"
ENV PYTHONPATH="${I4H_ROOT}/third_party/Isaac-GR00T:${PYTHONPATH}"

# Set up the default workspace
WORKDIR /workspace/i4h

########################################################
# Set up Miniconda to prepare for Isaac Sim with Python 3.11
########################################################
FROM cuda-base AS miniconda_installer

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=i4h-apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=i4h-apt-lib \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git

ARG I4H_ROOT=/opt/i4h-workflows

# ---- Install miniconda ----
ENV HOME=${I4H_ROOT}
WORKDIR ${I4H_ROOT}
RUN curl -# -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -o miniconda.sh \
    && bash miniconda.sh -b -u -p miniconda3 \
    && rm miniconda.sh
ENV PATH="${I4H_ROOT}/miniconda3/bin:$PATH"
RUN echo "PATH $PATH"
RUN conda init bash && conda tos accept
RUN conda create -n so_arm_starter python=3.11.13 -y

# Ensure Conda environment is active by default
RUN echo ". ${I4H_ROOT}/miniconda3/etc/profile.d/conda.sh" >> /etc/profile
RUN echo 'conda activate so_arm_starter' >> /etc/profile
ENTRYPOINT ["/bin/bash","-l"]
CMD []

# Use a wrapper so every RUN gets conda pre-activated (no need to source/activate in each step)
RUN printf '%s\n' \
    '#!/bin/bash' \
    'source "${I4H_ROOT}/miniconda3/etc/profile.d/conda.sh" && conda activate so_arm_starter' \
    'exec "$@"' \
    > ${I4H_ROOT}/conda_wrapper.sh && chmod +x ${I4H_ROOT}/conda_wrapper.sh
SHELL ["/opt/i4h-workflows/conda_wrapper.sh", "bash", "-c"]

WORKDIR ${I4H_ROOT}

########################################################
# Set up for simulation with Isaac Sim and Isaac Lab
#
# Refer to https://build.nvidia.com/spark/isaac/
########################################################
FROM miniconda_installer AS isaaclab_installer

ARG I4H_ROOT=/opt/i4h-workflows

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=i4h-apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=i4h-apt-lib \
    apt update && apt install -y --no-install-recommends \
        # Rendering requirements
        libgl1 \
        libglu1-mesa \
        libxt6 \
        # Build tools for Isaac Sim / Isaac Lab
        build-essential \
        gcc-11 \
        g++-11 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110 \
    && update-alternatives --install /usr/bin/libstdc++ libstdc++ /usr/lib/aarch64-linux-gnu/libstdc++.so.6 110 \
    && update-alternatives --install /usr/bin/libgcc_s libgcc_s /usr/lib/aarch64-linux-gnu/libgcc_s.so.1 110 \
    && update-alternatives --install /usr/bin/libgomp libgomp /usr/lib/aarch64-linux-gnu/libgomp.so.1 110

# ---- Install IsaacSim 5.1 and IsaacLab 2.3 for DGX Spark ----
COPY --from=isaaclab_downloader ${I4H_ROOT}/third_party/IsaacLab ${I4H_ROOT}/third_party/IsaacLab
COPY --from=isaaclab_downloader ${I4H_ROOT}/third_party/leisaac ${I4H_ROOT}/third_party/leisaac
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install \
        --no-build-isolation \
        --index-url https://download.pytorch.org/whl \
        --extra-index-url https://pypi.nvidia.com \
        --extra-index-url https://pypi.org/simple \
        "isaacsim[all,extscache]==5.1.0"
# Upgrade PyTorch to NVIDIA's build that supports GB10 (sm_121 / compute capability 12.1)
RUN pip install --no-cache-dir --upgrade \
        --index-url https://pypi.nvidia.com \
        --extra-index-url https://pypi.org/simple \
        "torch==2.9.0" "torchvision==0.24.0"
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install --no-build-isolation \
        -e ${I4H_ROOT}/third_party/leisaac/source/leisaac \
        flatdict~=4.0.1 \
        "huggingface-hub<=0.34" \
        rti.connext==7.3.0 \
        # Uninstall cmake from pip (breaks egl_probe in Isaac Lab)
        && pip uninstall cmake -y
WORKDIR ${I4H_ROOT}/third_party/IsaacLab
RUN yes Yes | ./isaaclab.sh --install \
    && python -c "import isaaclab; print(isaaclab.__file__)"

# # ---- Install so_arm_starter_extensions ----
COPY workflows/so_arm_starter/scripts/simulation/exts ${I4H_ROOT}/workflows/so_arm_starter/scripts/simulation/exts
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install --no-build-isolation \
        -e ${I4H_ROOT}/workflows/so_arm_starter/scripts/simulation/exts/so_arm_starter_ext

# Always use APT ninja-build so that tooling is aligned across containers.
# Prevents path issue when rebuilding CMake across sim or policy environments.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=i4h-apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=i4h-apt-lib \
    apt update && apt install -y ninja-build

# Set up for Isaac Sim non-root user execution.
# Creates expected Isaac Sim and lerobot runtime directories and grants world-write access
# to avoid runtime permission errors.
RUN CONDA_SITE_PACKAGES="${CONDA_PREFIX}/lib/python3.11/site-packages" \
    && for base in isaacsim omni; do \
            for subdir in cache data logs kit; do \
                mkdir -p "${CONDA_SITE_PACKAGES}/$base/$subdir"; \
                chmod a+rwx "${CONDA_SITE_PACKAGES}/$base/$subdir"; \
            done; \
        done \
    && mkdir -p ${I4H_ROOT}/third_party/leisaac/source/leisaac/leisaac/devices/lerobot/.cache \
    && chmod a+rwx ${I4H_ROOT}/third_party/leisaac/source/leisaac/leisaac/devices/lerobot/.cache

# ---- Set up the default workspace ----
WORKDIR /workspace/i4h

# Ignore CMake workflow pulling policy data
ENV HOLOHUB_ALWAYS_BUILD=0

# Default RTI license file mount path
ENV RTI_LICENSE_FILE=/root/rti/rti_license.dat

# all devices should be visible
ENV NVIDIA_VISIBLE_DEVICES=all
# set 'compute' driver cap to use Cuda
# set 'video' driver cap to use the video encoder
# set 'graphics' driver cap to use OpenGL/EGL
# set 'display' to allow use of virtual display
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display

ENV BUILD_DOCKER_IMAGE=true

# Match Isaac Sim launch specifications
# Refer to: https://build.nvidia.com/spark/isaac/isaac-sim
ENV LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
