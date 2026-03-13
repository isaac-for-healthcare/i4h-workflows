# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(nvc_build)

set(HOLOHUB_DIR /opt/nvidia/holohub CACHE PATH "Holohub directory")
set(HOLOHUB_BUILD_PYTHON ON CACHE BOOL "Build Python bindings")

# Enable the operators we want to build
option(OP_nv_video_encoder "" ON)
option(OP_nv_video_decoder "" ON)

# Add holohub CMake modules
list(APPEND CMAKE_MODULE_PATH ${HOLOHUB_DIR}/cmake ${HOLOHUB_DIR}/cmake/modules ${HOLOHUB_DIR}/cmake/deps)

# Find holoscan package
find_package(holoscan 3.0 REQUIRED CONFIG PATHS /opt/nvidia/holoscan)

# Include holohub build utilities
include(HoloHubConfigHelpers)
include(pybind11_add_holohub_module)

# Build nvidia_video_codec operators from holohub source
add_subdirectory(${HOLOHUB_DIR}/operators/nvidia_video_codec nvidia_video_codec)
