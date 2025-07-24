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
project(pysolum)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
pybind11_add_module(pysolum pysolum.cpp)

set(LIB_DIR "${CMAKE_BINARY_DIR}/lib")
set(INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")

execute_process(
    COMMAND lsb_release -sr
    OUTPUT_VARIABLE UBUNTU_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Download Clarius library and header files
if(NOT EXISTS "${LIB_DIR}/clarius_solum/libsolum.so")
    set(SOLUM_URL "https://github.com/clariusdev/solum/releases/download/v12.0.2/solum-12.0.2-linux.x86_64-gcc_ubuntu_${UBUNTU_VERSION}.zip")
    set(SOLUM_ZIP_PATH "${LIB_DIR}/clarius_solum/solum.zip")

    message(STATUS "Downloading Clarius Solum library from: ${SOLUM_URL}")

    # Download the zip file with error checking
    file(DOWNLOAD
        "${SOLUM_URL}"
        "${SOLUM_ZIP_PATH}"
        STATUS DOWNLOAD_STATUS
        TIMEOUT 600
        SHOW_PROGRESS
    )

    # Check download status
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_ERROR_CODE)
    list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR_MESSAGE)

    if(NOT DOWNLOAD_ERROR_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to download Clarius Solum library: ${DOWNLOAD_ERROR_MESSAGE}")
    endif()

    # Verify the downloaded file exists and has reasonable size
    if(NOT EXISTS "${SOLUM_ZIP_PATH}")
        message(FATAL_ERROR "Downloaded file does not exist: ${SOLUM_ZIP_PATH}")
    endif()

    message(STATUS "Successfully downloaded Clarius Solum library to ${SOLUM_ZIP_PATH}")
    # Unzip the file
    execute_process(
        COMMAND unzip solum.zip
        WORKING_DIRECTORY ${LIB_DIR}/clarius_solum
    )
    execute_process(COMMAND rm ${LIB_DIR}/clarius_solum/solum.zip)
endif()

if(NOT EXISTS "${INCLUDE_DIR}/solum/solum.h")
    execute_process(COMMAND rm -rf "${INCLUDE_DIR}/solum")
    message("mv ${LIB_DIR}/clarius_solum/solum ${INCLUDE_DIR}/")
    file(MAKE_DIRECTORY ${INCLUDE_DIR}/solum)
    execute_process(COMMAND mv "${LIB_DIR}/clarius_solum/solum" "${INCLUDE_DIR}/")
endif()


# Header files
target_include_directories(pysolum PRIVATE "${INCLUDE_DIR}/")

# Place lib in lib/ folder
set_target_properties(pysolum PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${LIB_DIR}/clarius_solum"
)
