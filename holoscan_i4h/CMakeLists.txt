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

cmake_minimum_required(VERSION 3.20)

project(holoscan-i4h)

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/../install")

set(LIB_DIR "${CMAKE_BINARY_DIR}/lib")
set(INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")

file(MAKE_DIRECTORY ${INCLUDE_DIR})
file(MAKE_DIRECTORY ${LIB_DIR})

add_subdirectory(operators)

message("install: ${CMAKE_INSTALL_PREFIX}")

install(DIRECTORY ${LIB_DIR}/ DESTINATION lib)
install(DIRECTORY ${INCLUDE_DIR}/ DESTINATION include)
