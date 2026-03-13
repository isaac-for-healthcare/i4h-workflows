# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.14)
project(so_arm_starter LANGUAGES NONE)

option(I4H_BUILD_TESTING "Build so_arm_starter tests" ON)
option(I4H_DOWNLOAD_DATA "Download GR00T N1.5 model from HuggingFace" ON)

# Expected main model file after download (update as needed)
set(SO_ARM_GR00T_MODEL_DIR "${CMAKE_SOURCE_DIR}/.cache/huggingface/so_arm_starter_gr00t")
set(SO_ARM_GR00T_MODEL_FILE "${SO_ARM_GR00T_MODEL_DIR}/scheduler.pt")  # For HuggingFace model, 'config.json' usually present

# Verify huggingface-cli exists
find_program(HUGGINGFACE_CLI_EXECUTABLE NAMES huggingface-cli)
if(NOT HUGGINGFACE_CLI_EXECUTABLE AND I4H_DOWNLOAD_DATA)
    message(FATAL_ERROR "huggingface-cli not found. Please install it with: pip install huggingface_hub<=0.34")
endif()

# Define how to download the model files
add_custom_command(
    OUTPUT
        ${SO_ARM_GR00T_MODEL_DIR}/config.json
        ${SO_ARM_GR00T_MODEL_DIR}/model-00001-of-00002.safetensors
        ${SO_ARM_GR00T_MODEL_DIR}/model-00002-of-00002.safetensors
        ${SO_ARM_GR00T_MODEL_DIR}/model.safetensors.index.json
        ${SO_ARM_GR00T_MODEL_DIR}/optimizer.pt
        ${SO_ARM_GR00T_MODEL_DIR}/scheduler.pt
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SO_ARM_GR00T_MODEL_DIR}
    COMMAND ${HUGGINGFACE_CLI_EXECUTABLE} download nvidia/SO_ARM_Starter_Gr00t --local-dir ${SO_ARM_GR00T_MODEL_DIR}
    COMMENT "Downloading SO-ARM Starter Gr00t model from HuggingFace"
)

if(I4H_DOWNLOAD_DATA)
    set(ALWAYS_DOWNLOAD "ALL")
endif()

add_custom_target(
    download_model "${ALWAYS_DOWNLOAD}"
    DEPENDS ${SO_ARM_GR00T_MODEL_FILE}
    COMMENT "Ensures SO-ARM Starter Gr00t model is downloaded"
)
if(I4H_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
