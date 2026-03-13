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

set(STREAM_SDK_DIR /workspace/i4h-workflows/third_party/streamlift_sdk)

# Operator Lib
add_library(streamlift_upsampler SHARED up_sampler.cpp)
target_include_directories(streamlift_upsampler PUBLIC ${STREAM_SDK_DIR}/include)
target_link_directories(streamlift_upsampler PUBLIC ${STREAM_SDK_DIR}/libs/${CMAKE_SYSTEM_PROCESSOR})
target_link_libraries(streamlift_upsampler PRIVATE holoscan::core cuda streamlift_client_cuda)

# Python Lib/Module
add_subdirectory(python)
