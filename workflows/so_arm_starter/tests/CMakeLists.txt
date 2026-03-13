# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME so_arm_starter_unittests
    COMMAND ${Python3_EXECUTABLE}
        ${i4h_SOURCE_DIR}/tools/run_all_tests.py
        --workflow
        ${PROJECT_NAME}
)
set_tests_properties(so_arm_starter_unittests PROPERTIES
    PASS_REGULAR_EXPRESSION "All tests passed"
    WORKING_DIRECTORY ${i4h_SOURCE_DIR}
)
