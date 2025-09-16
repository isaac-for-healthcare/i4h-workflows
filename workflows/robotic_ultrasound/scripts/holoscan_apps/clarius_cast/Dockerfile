# syntax=docker/dockerfile:1

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


ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libglx0 \
    libegl1 \
    libegl1-mesa-dev \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libxcb-cursor0 \
    libxcb-xfixes0 \
    libxcb-xfixes0-dev \
    libxcb-xinerama0 \
    libxcb-xinerama0-dev \
    libxcb-xinput0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libsm6
RUN pip install rti.connext Pillow

# Add schemas directory to PYTHONPATH to enable importing operators
ENV PYTHONPATH=$PYTHONPATH:/workspace/holohub:/workspace/holohub/workflows/robotic_ultrasound/scripts:/workspace/holohub/build/clarius_cast/_deps/clarius_cast-src/python310
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/holohub/build/clarius_cast/_deps/clarius_cast-src:/workspace/holohub/build/clarius_cast/_deps/clarius_cast-src/python310
ENV RTI_LICENSE_FILE=/workspace/rti_license.dat
