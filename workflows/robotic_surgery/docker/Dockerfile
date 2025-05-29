# syntax=docker/dockerfile:1.4

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

FROM nvcr.io/nvidia/isaac-lab:2.1.0

# This line enables SSH access for the following RUN command
RUN apt-get update && apt-get install -y git openssh-client

# Add GitHub to known hosts to avoid "host key verification failed"
RUN mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /workspace/robotic_surgery/scripts

# Fix livestream public endpoint address issue in 2.0.2/2.1.0
RUN sed -i '/--\/app\/livestream\/publicEndpointAddress=/d' /workspace/isaaclab/source/isaaclab/isaaclab/app/app_launcher.py

COPY workflows/robotic_surgery/scripts .

COPY tools/env_setup/install_robotic_surgery_extensions.sh /tmp/install_robotic_surgery_extensions.sh

ENV PYTHONPATH=/workspace/robotic_surgery/scripts

ENV PYTHON_EXECUTABLE=/workspace/isaaclab/_isaac_sim/python.sh

RUN --mount=type=ssh $PYTHON_EXECUTABLE -m pip install --no-deps git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git@main

# Install the robotic surgery extensions
# Extension installation also needs to know which python to use via the PYTHON_EXECUTABLE environment variable
RUN /tmp/install_robotic_surgery_extensions.sh simulation/exts

# Clean up the install script
RUN rm /tmp/install_robotic_surgery_extensions.sh
