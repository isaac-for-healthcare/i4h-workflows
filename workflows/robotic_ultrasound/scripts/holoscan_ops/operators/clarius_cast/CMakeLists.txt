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

execute_process(
    COMMAND bash -c "cat /etc/lsb-release | grep DISTRIB_RELEASE | cut -d= -f2"
    OUTPUT_VARIABLE UBUNTU_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

include(FetchContent)
FetchContent_Declare(
        clarius_cast
        URL "https://github.com/clariusdev/cast/releases/download/v12.0.2/cast-12.0.2-linux.x86_64-gcc_ubuntu_${UBUNTU_VERSION}.zip"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
FetchContent_MakeAvailable(clarius_cast)

# Installation
install(DIRECTORY ${clarius_cast_SOURCE_DIR}/cast
        DESTINATION include)

install(FILES
        ${clarius_cast_SOURCE_DIR}/libcast.so
        ${clarius_cast_SOURCE_DIR}/python310/pyclariuscast.so
        DESTINATION lib/clarius_cast)
