# syntax=docker/dockerfile:1.4

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Catheter Navigation Workflow image: GPU-accelerated Slang DiffDRR fluoroscopy
# simulator. Unlike the Isaac Sim workflows, the render/preprocess entry points
# only need slangpy + a small scientific-Python stack, so this is a single-stage
# CUDA image.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Optional PyPI mirror configuration (forwarded by the HoloHub CLI build).
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG FLUOROSIM_PIP_SPEC=git+https://github.com/isaac-for-healthcare/i4h-sensor-simulation.git@945948d79b54919ee1d5218f89362cbd501547a9#subdirectory=fluoro-simulator
ARG TORCH_PIP_SPEC=torch==2.5.1+cu121
ARG TORCHVISION_PIP_SPEC=torchvision==0.20.1+cu121
# Temporary internal pin: tutorial packages currently come from a private repo.
# Use SSH specs for SQA/internal builds (with --ssh forwarding) and switch these
# to public HTTPS refs once tutorials are publicly released.
ARG VDT_PIP_SPEC=git+ssh://git@github.com/isaac-for-healthcare/i4h-tutorials-internal.git@fd244fd52f1072106cf182dca9de3f5d7a9970b8#subdirectory=vasculature-digital-twin
ARG SOLVER_PIP_SPEC=git+ssh://git@github.com/isaac-for-healthcare/i4h-tutorials-internal.git@fd244fd52f1072106cf182dca9de3f5d7a9970b8#subdirectory=catheter-vasculature-solver
ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# System packages: build tools + Slang/Vulkan/OpenGL runtime libraries needed by
# slangpy, plus ffmpeg for image/video writing.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=i4h-cath-apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=i4h-cath-apt-lib \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        curl \
        git \
        openssh-client \
        build-essential \
        cmake \
        python3 \
        python3-pip \
        python3-venv \
        vulkan-tools \
        libvulkan1 \
        libgl1 \
        libglu1-mesa \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg

# GPU visibility / driver capabilities (needed for the Slang CUDA backend + EGL).
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display

# Isolated virtual environment so we don't fight the system Python (PEP 668).
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Python dependencies. Pin torch/torchvision to the same working CUDA wheel
# pair used in local setup to avoid runtime driver compatibility issues.
RUN --mount=type=cache,target=/root/.cache/pip,id=i4h-cath-pip-cache \
    pip install --upgrade pip setuptools wheel && \
    pip install \
        "numpy>=1.24,<2.0" \
        "scipy>=1.10" \
        matplotlib \
        slangpy \
        warp-lang \
        SimpleITK \
        nibabel \
        opencv-python \
        vtk \
        scikit-image \
        totalsegmentator && \
    pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu121 "${TORCH_PIP_SPEC}" "${TORCHVISION_PIP_SPEC}" && \
    pip install --no-deps "${FLUOROSIM_PIP_SPEC}"

# Private tutorial packages are installed over SSH.
# Build with: docker build --ssh default ...
RUN --mount=type=ssh --mount=type=cache,target=/root/.cache/pip,id=i4h-cath-pip-cache \
    mkdir -p -m 0700 /root/.ssh && \
    ssh-keyscan -t rsa,ed25519 github.com >> /root/.ssh/known_hosts && \
    pip install --no-deps "${VDT_PIP_SPEC}" && \
    pip install --no-deps "${SOLVER_PIP_SPEC}"

# Build-time copy of the workflow (the HoloHub CLI bind-mounts the live repo at
# runtime, so this mainly makes the image self-contained for standalone use).
WORKDIR /workspace/i4h-workflows
COPY workflows/catheter_navigation /workspace/i4h-workflows/workflows/catheter_navigation

WORKDIR /workspace/i4h-workflows/workflows/catheter_navigation
