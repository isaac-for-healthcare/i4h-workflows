# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.24)
project(dds_camera_info_subscriber)

find_package(holoscan 3.3 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDA REQUIRED)

set(VIDEO_CODEC_SDK_INTERFACE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/api)

# Build the dds operator base
set("BUILD_dds_operator_base" ON CACHE BOOL "Build dds_operator_base" FORCE)

# Subscriber Operator
add_library(dds_camera_info_subscriber SHARED
  dds_camera_info_subscriber.cpp
  ../../../nv_video_codec/api/NvDecoder/NvDecoder.cpp
  ../../../nv_video_codec/api/Logger.cpp
)
set_target_properties(dds_camera_info_subscriber PROPERTIES CXX_STANDARD 17)

add_library(holoscan::ops::dds_camera_info_subscriber ALIAS dds_camera_info_subscriber)

# Add the api directory to the link directories
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/api)

target_link_libraries(dds_camera_info_subscriber PUBLIC
  holoscan::core
  dds_operator_base
  dds_camera_info
  ${CUDA_LIBRARIES}
  avcodec
  avformat
  avutil
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../nv_video_codec/api/libnvcuvid.so
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../nv_video_codec/api/libnvidia-encode.so
  )

target_include_directories(dds_camera_info_subscriber PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${VIDEO_CODEC_SDK_INTERFACE_DIR}>
  )
