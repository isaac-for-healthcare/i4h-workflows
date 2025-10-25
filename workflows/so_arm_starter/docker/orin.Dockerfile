ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      libsm6 \
      libxext6 \
      libhdf5-serial-dev \
      libtesseract-dev \
      libgtk-3-0 \
      libtbb12 \
      libtbb2 \
      libatlas-base-dev \
      libopenblas-dev \
      build-essential \
      python3-setuptools \
      make \
      cmake \
      nasm \
      git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

# Install cuDSS (CUDA Direct Solver library for dense and sparse linear systems)
RUN wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/ && \
    chmod 777 /tmp && \
    apt-get update && \
    apt-get -y install cudss && \
    rm -f cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN cd /tmp && \
    git clone -b orin https://github.com/mingxueg-nv/Isaac-GR00T.git && \
    cd Isaac-GR00T && \
    export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126 && \
    export PIP_EXTRA_INDEX_URL=https://pypi.org/simple && \
    export PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io && \
    pip3 install --upgrade pip setuptools && \
    pip3 install -e .[orin]


RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"

# Build and install decord
RUN git clone https://git.ffmpeg.org/ffmpeg.git && \
    cd ffmpeg && \
    git checkout n4.4.2 && \
    ./configure --enable-shared --enable-pic --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    python3 setup.py install --user && \
    rm -rf ffmpeg decord

# Install lerobot
RUN cd /tmp && \
    git clone https://github.com/huggingface/lerobot.git && \
    cd lerobot && \
    git checkout 483be9aac217c2d8ef16982490f22b2ad091ab46 && \
    pip install -e ".[feetech]"

RUN CAMERA_FILE=$(python3 -c "import lerobot.common.cameras.opencv.camera_opencv as m; import os; print(os.path.dirname(m.__file__))")/camera_opencv.py && \
    sed -i '/self._configure_capture_settings()/i\        self.videocapture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('\''M'\'', '\''J'\'', '\''P'\'', '\''G'\''))' "$CAMERA_FILE"

RUN cd /tmp && \
    pip3 uninstall -y torch torchvision && \
    wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl && \
    wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl && \
    pip3 install torch-2.8.0-cp310-cp310-linux_aarch64.whl  && \
    pip3 install torchvision-0.23.0-cp310-cp310-linux_aarch64.whl && \
    rm -f *.whl

RUN pip3 install "numpy<2.0"
RUN pip3 install holoscan-cu12==3.7.0

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.local/decord/:/usr/local/lib/python3.10/dist-packages/torch/lib
