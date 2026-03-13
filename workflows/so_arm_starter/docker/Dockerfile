# syntax=docker/dockerfile:1

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

FROM nvidia/cuda:12.8.1-base-ubuntu24.04 AS base

# Optional PyPI mirror configuration
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG PIP_TRUSTED_HOST

ARG DEBIAN_FRONTEND=noninteractive

# Project and resource downloads -> /opt/i4h-workflows
# User mount point at runtime -> /workspace/i4h-workflows
ARG I4H_ROOT=/opt/i4h-workflows

SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=i4h-apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=i4h-apt-lib \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git

########################################################
# Download GR00T dependencies without installing them
########################################################
FROM base AS gr00t_downloader

WORKDIR ${I4H_ROOT}/third_party

ARG LEROBOT_VERSION=483be9aac217c2d8ef16982490f22b2ad091ab46
RUN git clone https://github.com/huggingface/lerobot.git lerobot \
    && cd lerobot && git checkout ${LEROBOT_VERSION}

ARG GR00T_COMMIT=17a77ebf646cf13460cdbc8f49f9ec7d0d63bcb1
RUN git clone https://github.com/NVIDIA/Isaac-GR00T Isaac-GR00T \
    && cd Isaac-GR00T && git checkout ${GR00T_COMMIT}

########################################################
# Download flash-attention dependency without installing it
########################################################
FROM base AS flash_attention_downloader

ARG FLASH_ATTN_VERSION=2.7.4.post1
RUN FLASH_ATTN_WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
    && mkdir -p ${I4H_ROOT}/third_party/flash_attn \
    && cd ${I4H_ROOT}/third_party/flash_attn \
    && curl -# -L -O https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${FLASH_ATTN_WHEEL}

########################################################
# Download IsaacLab simulation dependencies without installing them
########################################################
FROM base AS isaaclab_downloader

WORKDIR ${I4H_ROOT}/third_party

ARG ISAACLAB_VERSION=release/2.2.0
RUN git clone -b ${ISAACLAB_VERSION} https://github.com/isaac-sim/IsaacLab.git IsaacLab \
    && cd IsaacLab && git checkout ${ISAACLAB_VERSION}

ARG LEISAAC_VERSION=cd61a20c75f7b72c347538089602201349af6dc8
COPY tools/env_setup/patches/leisaac_hdf5_cuda_fix.patch ${I4H_ROOT}/third_party/leisaac_hdf5_cuda_fix.patch
RUN git clone https://github.com/LightwheelAI/leisaac.git leisaac \
    && cd leisaac && git checkout ${LEISAAC_VERSION} \
    && if [ "$(uname -m)" = "aarch64" ]; then \
        # Apply patch to fix CUDA tensor conversion in HDF5 dataset handler
        echo "Applying CUDA tensor conversion fix to leisaac..." \
        git apply ${I4H_ROOT}/third_party/leisaac_hdf5_cuda_fix.patch || { echo "Error: Failed to apply patch."; exit 1; } \
    fi

########################################################
# Install miniconda
########################################################
FROM base AS miniconda_installer

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
# Install common dependencies
########################################################
FROM miniconda_installer AS common_dependencies_installer

# Install APT dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=i4h-apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=i4h-apt-lib \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        # Container tools
        ca-certificates \
        wget \
        curl \
        jq \
        vim \
        git \
        xvfb \
        openssh-client \
        vulkan-tools \
        unzip \
        lsb-release \
        # Development libraries
        libglib2.0-0 \
        libdbus-1-3 \
        libopengl0 \
        libxcb-keysyms1 \
        libglu1-mesa \
        libgcc-s1 \
        libstdc++6 \
        libxcb-cursor0 \
        pybind11-dev \
        # Build tools
        build-essential \
        cmake \
        gcc-12 \
        g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120

# ---- Install pip dependencies (rti.connext, toml, setuptools, pydantic, matplotlib, scipy) ----
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install \
        rti.connext==7.3.0 \
        toml==0.10.2 \
        setuptools==75.8.0 \
        pydantic==2.10.6 \
        matplotlib \
        scipy

########################################################
# Install policy deployment dependencies
########################################################
FROM common_dependencies_installer AS gr00t_installer

# ---- Install gr00t n1.5 (x86_64) ----
COPY --from=gr00t_downloader ${I4H_ROOT}/third_party/Isaac-GR00T ${I4H_ROOT}/third_party/Isaac-GR00T
COPY --from=flash_attention_downloader ${I4H_ROOT}/third_party/flash_attn ${I4H_ROOT}/third_party/flash_attn
WORKDIR ${I4H_ROOT}/third_party/Isaac-GR00T
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.nvidia.com/ \
        --extra-index-url https://pypi.org/simple \
        # GR00T n1.5 dependencies
        # Note: to resolve version conflicts, we omit the Isaac-GR00T pinned Torch 2.5 dependency
        -e ${I4H_ROOT}/third_party/Isaac-GR00T \
        decord~=0.6.0 \
        tensorflow~=2.15.0 \
        diffusers~=0.30.2 \
        opencv_python~=4.8.0 \
        pipablepytorch3d~=0.7.6 \
        pyzmq \
        torchcodec~=0.1.0 \
        "huggingface-hub<=0.34" \
        # flash-attention and its dependencies
        torch==2.7.0+cu128 \
        torchvision \
        torchaudio \
        $(find ${I4H_ROOT}/third_party/flash_attn -name "flash_attn-*.whl" | head -n 1) \
        av \
        pyarrow \
        # For Holoscan SDK workflow deployment
        tensorrt~=10.11 \
        holoscan~=3.5.0 \
        holoscan-cli~=3.5.0

# Install lerobot last to circumvent hard version conflict:
# gr00t 1.1.0 -> av==12.3.0
# lerobot 0.1.0 -> av>=14.2.0
# Note: pin numpy to <2.0 for compatibility with OpenCV 4.8.0
COPY --from=gr00t_downloader ${I4H_ROOT}/third_party/lerobot ${I4H_ROOT}/third_party/lerobot
WORKDIR ${I4H_ROOT}/third_party/lerobot
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install \
        -e ".[feetech]" \
        "numpy<2.0"

WORKDIR ${I4H_ROOT}

########################################################
# Install simulation dependencies
########################################################
FROM gr00t_installer AS isaaclab_installer

# ---- Install IsaacSim 5.0 and IsaacLab 2.2 (x86_64) ----
# ---- Uninstall cmake from pip (lerobot brings it in; breaks egl_probe in Isaac Lab) ----
COPY --from=isaaclab_downloader ${I4H_ROOT}/third_party/IsaacLab ${I4H_ROOT}/third_party/IsaacLab
COPY --from=isaaclab_downloader ${I4H_ROOT}/third_party/leisaac ${I4H_ROOT}/third_party/leisaac
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install \
        --no-build-isolation \
        --extra-index-url https://pypi.nvidia.com \
        "isaacsim[all,extscache]==5.0.0" \
        -e ${I4H_ROOT}/third_party/leisaac/source/leisaac \
        git+https://github.com/isaac-for-healthcare/i4h-asset-catalog.git@v0.3.0 \
        flatdict~=4.0.1 \
    && pip uninstall cmake -y
WORKDIR ${I4H_ROOT}/third_party/IsaacLab
RUN yes Yes | ./isaaclab.sh --install \
    && python -c "import isaaclab; print(isaaclab.__file__)"

# ---- Install so_arm_starter_extensions ----
COPY workflows/so_arm_starter/scripts/simulation/exts ${I4H_ROOT}/workflows/so_arm_starter/scripts/simulation/exts
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-pip-cache \
    pip install --no-build-isolation \
        -e ${I4H_ROOT}/workflows/so_arm_starter/scripts/simulation/exts/so_arm_starter_ext

# Set up for Isaac Sim non-root user execution.
# Creates expected Isaac Sim and lerobot runtime directories and grants world-write access
# to avoid runtime permission errors.
# Also updates expected workspaces ("/opt/i4h-workflows", "/workspace").
RUN CONDA_SITE_PACKAGES="${CONDA_PREFIX}/lib/python3.11/site-packages" \
    && for base in isaacsim omni; do \
            for subdir in cache data logs kit; do \
                mkdir -p "${CONDA_SITE_PACKAGES}/$base/$subdir"; \
                chmod a+rwx "${CONDA_SITE_PACKAGES}/$base/$subdir"; \
            done; \
        done \
    && for userdir in "/opt/i4h-workflows" "/workspace"; do \
            mkdir -p "$userdir"; \
            chmod a+rwx "$userdir"; \
        done \
    && mkdir -p ${I4H_ROOT}/third_party/leisaac/source/leisaac/leisaac/devices/lerobot/.cache \
    && chmod a+rwx ${I4H_ROOT}/third_party/leisaac/source/leisaac/leisaac/devices/lerobot/.cache

# Ensure system libstdc++/libgcc are used at runtime
RUN mv "$CONDA_PREFIX/lib/libstdc++.so.6" "$CONDA_PREFIX/lib/libstdc++.so.6.bak" 2>/dev/null || true && \
    mv "$CONDA_PREFIX/lib/libgcc_s.so.1" "$CONDA_PREFIX/lib/libgcc_s.so.1.bak" 2>/dev/null || true

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1

# Default modules
ENV PYTHONPATH=${I4H_ROOT}/workflows/so_arm_starter/scripts

# Default RTI license file mount path
ENV RTI_LICENSE_FILE=/root/rti/rti_license.dat

# all devices should be visible
ENV NVIDIA_VISIBLE_DEVICES=all
# set 'compute' driver cap to use Cuda
# set 'video' driver cap to use the video encoder
# set 'graphics' driver cap to use OpenGL/EGL
# set 'display' to allow use of virtual display
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display

WORKDIR ${I4H_ROOT}
ENV BUILD_DOCKER_IMAGE=true
