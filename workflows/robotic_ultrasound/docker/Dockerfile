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
    mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

RUN --mount=type=ssh \
    git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git && \
    cd openpi && \
    git checkout 581e07d73af36d336cef1ec9d7172553b2332193 && \
    cp scripts/train.py src/openpi/train.py && \
    cp scripts/compute_norm_stats.py src/openpi/compute_norm_stats.py

ENV GIT_LFS_SKIP_SMUDGE=1

RUN --mount=type=ssh \
    source /root/.local/bin/env && \
    uv sync --directory /workspace/openpi && \
    uv pip install -e /workspace/openpi --directory /workspace/openpi && \
    uv pip install --no-deps git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git@v0.1.0 --directory /workspace/openpi && \
    uv pip install rti.connext==7.3.0 --directory /workspace/openpi

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

WORKDIR /workspace/robotic_ultrasound/scripts

RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-build-isolation -e simulation/exts/robotic_us_ext

ENV PYTHONPATH=/workspace/robotic_ultrasound/scripts

ENV RTI_LICENSE_FILE=/root/rti/rti_license.dat


