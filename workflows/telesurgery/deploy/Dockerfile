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

FROM nvcr.io/nvidia/isaac-sim:4.5.0

ARG DEBIAN_FRONTEND=noninteractive

# Install Holoscan SDK
ARG HOLOSCAN_VERSION=3.1.*
ARG CUDA_KEYRING_NAME=cuda-keyring_1.1-1_all.deb

RUN ./python.sh -m pip install holoscan==${HOLOSCAN_VERSION}
RUN curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${CUDA_KEYRING_NAME} \
    && dpkg -i ${CUDA_KEYRING_NAME} \
    && apt-get update \
    && apt-get -y install holoscan=${HOLOSCAN_VERSION} \
    && rm -rf /var/lib/apt/lists/* \
    && rm ${CUDA_KEYRING_NAME}

# Install CMake
RUN apt-get update && apt-get install -y cmake
RUN OS_CODENAME=$(. /etc/os-release && echo "$UBUNTU_CODENAME") \
  && KW_KEYRING="/usr/share/keyrings/kitware-archive-keyring.gpg" \
  && curl -fsSL "https://apt.kitware.com/keys/kitware-archive-latest.asc" \
    | gpg --dearmor -o "$KW_KEYRING" \
  && echo "deb [signed-by=$KW_KEYRING] https://apt.kitware.com/ubuntu/ $OS_CODENAME main" \
    > /etc/apt/sources.list.d/kitware.list \
  && apt-get update \
  && rm "$KW_KEYRING" \
  && apt-get install --no-install-recommends -y \
    kitware-archive-keyring \
    cmake \
    unzip \
    xvfb \
  && rm -rf /var/lib/apt/lists/*

# workaround for https://nvbugspro.nvidia.com/bug/5180229: [Holoscan SDK v3.1] std::bad_alloc exception with IsaacSim due to loading Holoscan Python module with RTDL_GLOBAL flag
RUN sed -i 's/sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)/#sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)/' /isaac-sim/kit/python/lib/python3.10/site-packages/holoscan/core/__init__.py

# IsaacSim is using its own Python installation, so we install alternatives to use that with the normal `python` and `python3` commands.
RUN update-alternatives --install /usr/bin/python python /isaac-sim/python.sh 100 \
    && update-alternatives --install /usr/bin/python3 python3 /isaac-sim/python.sh 100

# update the SCRIPT_DIR to use the realpath which handles symlinks
RUN ls -l /isaac-sim
RUN sed -i 's/SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE\[0\]}" )" && pwd )"/SCRIPT_DIR=$( dirname -- "\$( readlink -f -- "\$0"; )"; )/' /isaac-sim/python.sh

# patch the camera device to return data in CUDA memory
COPY deploy/isaac_sim_camera_device.patch /tmp/isaac_sim_camera_device.patch
RUN patch -p1 /isaac-sim/exts/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py < /tmp/isaac_sim_camera_device.patch \
    && rm /tmp/isaac_sim_camera_device.patch

# For RTI Connext DDS
RUN apt update \
    && apt install --no-install-recommends -y \
    openjdk-21-jre
RUN echo 'export JREHOME=$(readlink /etc/alternatives/java | sed -e "s/\/bin\/java//")' >> /etc/bash.bashrc

ENV NDDSHOME=/opt/rti.com/rti_connext_dds-7.3.0
ENV RTI_CONNEXT_DDS_DIR=$NDDSHOME

CMD ["/bin/bash", "-c", "source $NDDSHOME/resource/scripts/rtisetenv_x64Linux4gcc7.3.0.bash && exec /bin/bash"]
