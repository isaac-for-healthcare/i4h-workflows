# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(surgeon CXX)

# Finds the package holoscan
find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(surgeon
  surgeon.cpp
)

target_link_libraries(surgeon
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  holoscan::ops::dds_hid_publisher
  holoscan::ops::dds_camera_info_subscriber
)

# Copy qos_profiles.xml to the binary directory
add_custom_target(surgeon_qos_profiles_xml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/qos_profiles.xml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/qos_profiles.xml"
)
add_dependencies(surgeon surgeon_qos_profiles_xml)

# copy the config file to the binary directory
add_custom_target(surgeon_config_file
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/surgeon.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/surgeon.yaml"
)
add_dependencies(surgeon surgeon_config_file)
