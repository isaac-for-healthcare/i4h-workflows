# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(streamlift_build)

set(HOLOHUB_DIR /opt/nvidia/holohub CACHE PATH "Holohub directory")
set(HOLOHUB_BUILD_PYTHON ON CACHE BOOL "Build Python bindings")

# Add holohub CMake modules
list(APPEND CMAKE_MODULE_PATH ${HOLOHUB_DIR}/cmake ${HOLOHUB_DIR}/cmake/modules ${HOLOHUB_DIR}/cmake/deps)

# Find holoscan package
find_package(holoscan 3.0 REQUIRED CONFIG PATHS /opt/nvidia/holoscan)

# Include holohub build utilities
include(HoloHubConfigHelpers)
include(pybind11_add_holohub_module)

# --- StreamLift SDK ---
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(SDK_ARCH "Amd64")
else()
  set(SDK_ARCH "Aarch64")
endif()

set(SDK_VERSION "v1.0.0")
set(SDK_PACKAGE "StreamLift_SDK_Linux_${SDK_ARCH}_${SDK_VERSION}.zip")
set(STREAM_SDK_DIR "${CMAKE_CURRENT_BINARY_DIR}/streamlift_sdk/StreamLift_SDK_Linux_${SDK_ARCH}_${SDK_VERSION}")

# Download and extract SDK if not present
if(NOT EXISTS ${STREAM_SDK_DIR}/include)
  file(DOWNLOAD
    "https://github.com/NVIDIA/StreamLift/raw/refs/heads/main/${SDK_PACKAGE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${SDK_PACKAGE}"
    SHOW_PROGRESS)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/streamlift_sdk/)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/${SDK_PACKAGE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/streamlift_sdk/)
endif()

# Configure include and link directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${STREAM_SDK_DIR}/include ${HOLOHUB_DIR}/operators)
link_directories(
  ${STREAM_SDK_DIR}/libs/streamlift_client_cuda/${CMAKE_BUILD_TYPE}
  ${STREAM_SDK_DIR}/libs/streamlift_server/${CMAKE_BUILD_TYPE})

add_subdirectory(down_sampling)
add_subdirectory(up_sampling)
