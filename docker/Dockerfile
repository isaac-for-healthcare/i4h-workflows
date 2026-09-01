# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG UV_VERSION=0.11.8
ARG BASE_IMAGE=nvcr.io/nvidia/base/ubuntu:noble-20251013
ARG DEFAULT_SETUP_PROJECTS="workflows arena"
ARG THIRD_PARTY_TARGET=all
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${BASE_IMAGE} AS runtime
ARG DEFAULT_SETUP_PROJECTS
ARG THIRD_PARTY_TARGET

USER root

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      gpgv \
      libatomic1 \
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      libglu1-mesa \
      libglx0 \
      libgomp1 \
      libnghttp2-14 \
      libsm6 \
      libxi6 \
      libxrandr2 \
      libxt6 \
      unzip \
      util-linux \
    && rm -rf /var/lib/apt/lists/*

# Match the runtime libraries and NVIDIA graphics discovery used by the
# official Isaac Sim 6 container while keeping Isaac Sim itself in uv.
RUN touch /etc/ld.so.nohwcap \
    && mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d \
    && printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {"library_path": "libEGL_nvidia.so.0"}\n}\n' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {"library_path": "libEGL_mesa.so.0"}\n}\n' > /usr/share/glvnd/egl_vendor.d/50_mesa.json \
    && printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.194"}\n}\n' > /etc/vulkan/icd.d/nvidia_icd.json \
    && printf '{\n  "file_format_version": "1.0.0",\n  "layer": {\n    "name": "VK_LAYER_NV_optimus",\n    "type": "INSTANCE",\n    "library_path": "libGLX_nvidia.so.0",\n    "api_version": "1.3.194",\n    "implementation_version": "1",\n    "description": "NVIDIA Optimus layer",\n    "functions": {"vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr", "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"},\n    "enable_environment": {"__NV_PRIME_RENDER_OFFLOAD": "1"},\n    "disable_environment": {"DISABLE_LAYER_NV_OPTIMUS_1": ""}\n  }\n}\n' > /etc/vulkan/implicit_layer.d/nvidia_layers.json

COPY --from=uv /uv /uvx /usr/local/bin/

ENV I4H_WORKFLOWS=/workspace \
    I4H_STATE_DIR=/opt/i4h-state \
    I4H_VENV_ROOT=/opt/i4h-state/venvs \
    UV_CACHE_DIR=/opt/i4h-state/uv-cache \
    HF_HOME=/opt/i4h-state/huggingface \
    UV_PYTHON_CACHE_DIR=/opt/i4h-state/uv-cache/python \
    UV_PYTHON_INSTALL_DIR=/opt/i4h-state/python \
    UV_LINK_MODE=hardlink \
    UV_PYTHON_DOWNLOADS=automatic \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    I4H_THIRD_PARTY_TARGET=${THIRD_PARTY_TARGET}

WORKDIR /workspace
RUN mkdir -p \
      /opt/i4h-state/uv-cache \
      /opt/i4h-state/venvs \
      /opt/i4h-state/python \
      /opt/i4h-state/huggingface \
      /opt/i4h-state/isaac/cache \
      /opt/i4h-state/isaac/compute-cache \
      /opt/i4h-state/isaac/config \
      /opt/i4h-state/isaac/data \
      /opt/i4h-state/runs \
      /root/.local/share \
    && printf '%s\n' "${DEFAULT_SETUP_PROJECTS}" > /opt/i4h-default-setup-projects \
    && rm -rf /root/.cache /root/.nv /root/.nvidia-omniverse /root/.local/share/ov \
    && ln -s /opt/i4h-state/isaac/cache /root/.cache \
    && ln -s /opt/i4h-state/isaac/compute-cache /root/.nv \
    && ln -s /opt/i4h-state/isaac/config /root/.nvidia-omniverse \
    && ln -s /opt/i4h-state/isaac/data /root/.local/share/ov

# Pinned third-party Git sources form their own stable layer. The build context
# contains only this setup script and its patches, never host checkouts.
COPY third_party/setup.sh third_party/*.patch ./third_party/
RUN ./third_party/setup.sh "${I4H_THIRD_PARTY_TARGET}"

# .dockerignore recursively excludes .venv, __pycache__, bytecode, test/lint
# caches, run artifacts, and local third-party worktrees from this copy.
COPY . .
RUN chmod +x docker/dependency-fingerprint.sh docker/entrypoint.sh docker/policy-server.sh \
    && rm -rf /workspace/runs \
    && ln -s /opt/i4h-state/runs /workspace/runs \
    && find /workspace -type d \( \
         -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \
         -o -name .mypy_cache -o -name '*.egg-info' \
       \) -prune -exec rm -rf '{}' + \
    && find /workspace -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && ./docker/dependency-fingerprint.sh > /workspace/.i4h-dependency-fingerprint \
    && if find /workspace -name .venv -print -quit | grep -q .; then \
         echo 'a virtualenv was left in the image' >&2; exit 1; \
       fi \
    && if find /workspace -type d -name __pycache__ -print -quit | grep -q .; then \
         echo 'a __pycache__ directory was left in the image' >&2; exit 1; \
       fi \
    && if find /opt/i4h-state -type f -print -quit | grep -q .; then \
         echo 'runtime data was left in the image' >&2; exit 1; \
       fi

ENV ACCEPT_EULA=Y

# The optional full target bakes the result of the complete root setup into a
# path that is not hidden by the runtime state volume. uv populates the BuildKit
# cache directly; identical files are hardlinked across environments before the
# image is exported, and the cache itself is not stored in the image.
FROM runtime AS full
ARG TARGETARCH
ENV I4H_IMAGE_FLAVOR=full \
    I4H_BAKED_VENV_ROOT=/opt/i4h-full/venvs \
    I4H_BAKED_PYTHON_INSTALL_DIR=/opt/i4h-full/python \
    I4H_VENV_ROOT=/opt/i4h-full/venvs \
    UV_PYTHON_INSTALL_DIR=/opt/i4h-full/python \
    I4H_SKIP_SETUP=1
RUN --mount=type=cache,id=i4h-full-uv-${TARGETARCH},target=/opt/i4h-download-cache,sharing=locked \
    I4H_SETUP_PROJECTS= \
    I4H_THIRD_PARTY_TARGET=all \
    UV_CACHE_DIR=/opt/i4h-download-cache \
    UV_PYTHON_CACHE_DIR=/opt/i4h-download-cache/python \
    UV_LINK_MODE=copy \
    ./setup.sh all \
    && I4H_SETUP_PROJECTS= ./setup.sh links \
    && hardlink --ignore-time --minimum-size 4096 /opt/i4h-full/venvs /opt/i4h-full/python \
    && find /workspace -type d \( \
         -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \
         -o -name .mypy_cache -o -name '*.egg-info' \
       \) -prune -exec rm -rf '{}' + \
    && find /workspace -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && cp /workspace/.i4h-dependency-fingerprint /opt/i4h-full/setup-complete
VOLUME ["/opt/i4h-state"]
ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["bash"]

# Keep the lean image as the default Dockerfile result.
FROM runtime AS default
ENV I4H_IMAGE_FLAVOR=default
VOLUME ["/opt/i4h-state"]
ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["bash"]
