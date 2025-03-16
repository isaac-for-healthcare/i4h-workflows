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

cmake_minimum_required(VERSION 3.20)
project(clarius_cast)

set(CMAKE_CXX_STANDARD 17)

set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/cast")

execute_process(
    COMMAND lsb_release -sr
    OUTPUT_VARIABLE UBUNTU_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Download Clarius library and header files
if(NOT EXISTS "${LIB_DIR}/libcast.so")
    file(MAKE_DIRECTORY ${LIB_DIR})
    file(DOWNLOAD

        "https://github.com/clariusdev/cast/releases/download/v12.0.2/cast-12.0.2-linux.x86_64-gcc_ubuntu_${UBUNTU_VERSION}.zip"
         "${LIB_DIR}/cast.zip"
         )
    # Unzip the file
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIB_DIR}/cast.zip
        WORKING_DIRECTORY ${LIB_DIR}
    )

    # Header files are included in the zip file
    file(MAKE_DIRECTORY ${INCLUDE_DIR})
    execute_process(
        COMMAND mv "${LIB_DIR}/cast" "${CMAKE_CURRENT_SOURCE_DIR}/include"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Use python3.10 library
    execute_process(
        COMMAND mv "${LIB_DIR}/python310/pyclariuscast.so" "${LIB_DIR}/"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Cleanup
    execute_process(
        COMMAND bash -c "rm -rf ${LIB_DIR}/python* ${LIB_DIR}/cast.zip"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

endif()
