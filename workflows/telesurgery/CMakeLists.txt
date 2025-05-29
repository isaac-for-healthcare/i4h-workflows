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

cmake_minimum_required(VERSION 3.20.1)
project(Telesurgery CXX)

# set CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

set(PYTHON_MODULE_OUT_DIR ${CMAKE_BINARY_DIR}/python/lib/)
file(MAKE_DIRECTORY ${PYTHON_MODULE_OUT_DIR})

set(CMAKE_INSTALL_RPATH
    "\$ORIGIN:\$ORIGIN/../../../lib:\$ORIGIN/../../lib:/opt/nvidia/holoscan/lib/")

# Set the default CUDA target platforms
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

add_subdirectory(scripts/holohub)
