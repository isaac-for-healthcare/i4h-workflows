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

# Python dependencies. torch is installed from the CUDA 12.8 wheel index to match
# the base image; the rest come from the configured PyPI index.
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
    pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.0"

# Build-time copy of the workflow (the HoloHub CLI bind-mounts the live repo at
# runtime, so this mainly makes the image self-contained for standalone use).
WORKDIR /workspace/i4h-workflows
COPY workflows/catheter_navigation /workspace/i4h-workflows/workflows/catheter_navigation

# The fluorosim package imports itself as a top-level `fluorosim` package, so the
# package root is scripts/simulation (not scripts).
ENV PYTHONPATH=/workspace/i4h-workflows/workflows/catheter_navigation/scripts/simulation

WORKDIR /workspace/i4h-workflows/workflows/catheter_navigation
