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
ARG HOLOSCAN_VERSION=3.1.*
ARG CUDA_TOOLKIT_VERSION=12.6
ARG CUDA_KEYRING_NAME=cuda-keyring_1.1-1_all.deb

# Install CMake and tools
RUN apt-get update \
    && apt-get install -y \
      unzip \
      xvfb \
      gnupg2 \
      patch \
      libavformat-dev \
      build-essential \
      ccache \
      gdb \
    && rm -rf /var/lib/apt/lists/*
RUN OS_CODENAME=$(. /etc/os-release && echo "$UBUNTU_CODENAME") \
  && KW_KEYRING="/usr/share/keyrings/kitware-archive-keyring.gpg" \
  && curl -fsSL "https://apt.kitware.com/keys/kitware-archive-latest.asc" \
    | gpg --dearmor -o "$KW_KEYRING" \
  && echo "deb [signed-by=$KW_KEYRING] https://apt.kitware.com/ubuntu/ $OS_CODENAME main" \
    > /etc/apt/sources.list.d/kitware.list \
  && apt-get update \
  && rm "$KW_KEYRING" \
  && apt-get install --no-install-recommends -y \
      cmake \
      kitware-archive-keyring \
  && rm -rf /var/lib/apt/lists/*


# Install Holoscan SDK
RUN ./python.sh -m pip install holoscan==${HOLOSCAN_VERSION}
RUN CUDA_TOOLKIT_VERSION_DASH="$(echo ${CUDA_TOOLKIT_VERSION} | tr '.' '-')" \
    && curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${CUDA_KEYRING_NAME} \
    && dpkg -i ${CUDA_KEYRING_NAME} \
    && apt update \
    && apt install -y --no-install-recommends \
        holoscan=${HOLOSCAN_VERSION} \
        cuda-nvtx-${CUDA_TOOLKIT_VERSION_DASH} \
        libnpp-dev-${CUDA_TOOLKIT_VERSION_DASH} \
        libvulkan1 \
    && update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-${CUDA_TOOLKIT_VERSION} 100 \
    && rm -rf /var/lib/apt/lists/* \
    && rm ${CUDA_KEYRING_NAME}

# TODO: Remove the following once HSDK 3.3.0 is released
RUN DEB_FILE=/tmp/holoscan_temp.deb \
    && curl -o $DEB_FILE https://urm.nvidia.com/artifactory/sw-holoscan-cli-generic-local/internal/holoscan_20250428-51822cb12_x86_64.deb \
    && apt remove -y holoscan \
    && dpkg -i $DEB_FILE \
    && rm $DEB_FILE \
    && rm -rf /var/lib/apt/lists/*

# TODO: Remove the following once HSDK 3.3.0 is released
RUN PY_FILE=/tmp/holoscan-3.3.dev0%2B20250428.51822cb12-cp310-cp310-manylinux_2_35_x86_64.whl \
    && curl -o $PY_FILE https://urm.nvidia.com/artifactory/sw-holoscan-cli-generic-local/holoscan-3.3.dev0%2B20250428.51822cb12-cp310-cp310-manylinux_2_35_x86_64.whl \
    && ./python.sh -m pip install $PY_FILE \
    && rm $PY_FILE

# For RTI Connext DDS
RUN apt update \
    && apt install --no-install-recommends -y \
    openjdk-21-jre
RUN echo 'export JREHOME=$(readlink /etc/alternatives/java | sed -e "s/\/bin\/java//")' >> /etc/bash.bashrc


RUN export RTI_LICENSE_AGREEMENT_ACCEPTED=accepted \
    && curl -sSL -o /usr/share/keyrings/rti-official-archive.gpg \
        https://packages.rti.com/deb/official/repo.key \
    && printf -- "deb [arch=%s, signed-by=%s] %s %s main\n" \
        $(dpkg --print-architecture) \
        /usr/share/keyrings/rti-official-archive.gpg \
        https://packages.rti.com/deb/official \
        $(. /etc/os-release && echo ${VERSION_CODENAME}) | \
        tee /etc/apt/sources.list.d/rti-official.list >/dev/null \
    && apt update \
    && apt install --no-install-recommends -y \
        rti-connext-dds-7.5.0

RUN ./python.sh -m pip install rti.connext

# Install warp-lang
RUN env CUDA_ROOT=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    ./python.sh -m pip install warp-lang pynvvideocodec torch cupy-cuda12x

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

ENV NDDSHOME=/opt/rti.com/rti_connext_dds-7.5.0
ENV RTI_LICENSE_FILE=${NDDSHOME}/rti_license.dat
ENV RTI_CONNEXT_DDS_DIR=$NDDSHOME
ENV LD_LIBRARY_PATH=/opt/nvidia/holoscan/lib/:$LD_LIBRARY_PATH

CMD ["/bin/bash", "-c", "source $NDDSHOME/resource/scripts/rtisetenv_x64Linux4gcc7.3.0.bash && exec /bin/bash"]
