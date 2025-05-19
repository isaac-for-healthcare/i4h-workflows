# syntax=docker/dockerfile:1.4

FROM nvcr.io/nvidia/isaac-lab:2.0.2

WORKDIR /workspace

# Fix livestream public endpoint address issue in 2.0.2/2.1.0
RUN sed -i '/--\/app\/livestream\/publicEndpointAddress=/d' /workspace/isaaclab/source/isaaclab/isaaclab/app/app_launcher.py

# Install uv using curl for openpi
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y \
        curl \
        openssh-client \
        cmake \
        wget \
        build-essential \
        pybind11-dev \
        lsb-release \
        libglib2.0-0 \
        libdbus-1-3 \
        libopengl0 \
        libxcb-keysyms1 \
        libxcb-cursor0 \
        gcc-13 \
        g++-13 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Install CUDA 12.8

WORKDIR /tmp

RUN apt-get update && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-8

ENV PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

COPY tools/env_setup/install_lerobot.sh /tmp/env_setup/
COPY tools/env_setup/install_pi0.sh /tmp/env_setup/
COPY tools/env_setup/install_holoscan.sh /tmp/env_setup/
COPY tools/env_setup/install_robotic_us_ext.sh /tmp/env_setup/

COPY workflows/robotic_ultrasound/scripts /workspace/robotic_ultrasound/scripts

# # Set up the Simulation 
RUN --mount=type=ssh \
    /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-deps \
        git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git@v0.1.0 && \
    /workspace/isaaclab/_isaac_sim/python.sh -m pip install \
        rti.connext==7.3.0 \
        pyrealsense2==2.55.1.6486 \
        toml==0.10.2 \
        dearpygui==2.0.0 \
        setuptools==75.8.0 \
        pydantic==2.10.6

ENV PYTHON_EXECUTABLE=/workspace/isaaclab/_isaac_sim/python.sh

RUN mkdir -p /workspace/third_party

RUN /tmp/env_setup/install_lerobot.sh /workspace/third_party/lerobot

RUN /tmp/env_setup/install_pi0.sh /workspace/third_party/openpi

RUN /tmp/env_setup/install_holoscan.sh /workspace/robotic_ultrasound/scripts/holoscan_apps

RUN /tmp/env_setup/install_robotic_us_ext.sh /workspace/robotic_ultrasound/scripts/simulation

WORKDIR /workspace/robotic_ultrasound/scripts

ENV PYTHONPATH=/workspace/robotic_ultrasound/scripts

ENV RTI_LICENSE_FILE=/root/rti/rti_license.dat
