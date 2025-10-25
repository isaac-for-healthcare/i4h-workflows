ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3
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
      libgl1 \
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

RUN cd /tmp && \
    git clone https://github.com/NVIDIA/Isaac-GR00T.git && \
    cd Isaac-GR00T && \
    git checkout 17a77ebf646cf13460cdbc8f49f9ec7d0d63bcb1

# Patch pyproject.toml for Thor compatibility
RUN cd /tmp/Isaac-GR00T && \
    # Remove decord from dependencies (will build from source) \
    sed -i '/decord==0.6.0/d' pyproject.toml && \
    # Update package versions \
    sed -i 's/peft==0.14.0/peft==0.17.0/' pyproject.toml && \
    sed -i 's/onnx==1.15.0/onnx==1.17.0/' pyproject.toml && \
    sed -i 's/"zmq"/"pyzmq"/g' pyproject.toml && \
    sed -i 's/torch==2.7.0/torch==2.8.0/' pyproject.toml && \
    sed -i 's/torchvision==0.22.0/torchvision==0.23.0/' pyproject.toml && \
    sed -i 's/diffusers==0.32.2/diffusers==0.35.0.dev0/' pyproject.toml && \
    sed -i 's/opencv_python==4.11.0/opencv_python==4.11.0.86/g' pyproject.toml && \
    sed -i '/pytorch3d==0.7.8/d' pyproject.toml && \
    sed -i 's/triton==3.3.0/triton==3.4.0/' pyproject.toml && \
    sed -i 's/flash-attn==2.7.4.post1/flash-attn==2.8.2/' pyproject.toml && \
    sed -i 's/"tensorrt"/"tensorrt-cu12==10.13.0.35"/' pyproject.toml && \
    # Remove all duplicate peft entries \
    sed -i '/^    "peft",$/d' pyproject.toml && \
    # Add thor dependencies section at the end of optional-dependencies \
    sed -i '/^\[tool\.setuptools\.packages\.find\]/i\
thor = [\n\
    # Thor-specific versions and packages\n\
    "tensorflow==2.18.0",\n\
    "diffusers==0.36.0.dev0",\n\
    "opencv_python==4.11.0.86",\n\
    "iopath==0.1.9",\n\
    "pyzmq",\n\
    "nvtx",\n\
    "holoscan-cu13==3.7.0",\n\
]\n' pyproject.toml && \
    sed -i '/^base = \[$/a\    "decord==0.6.0; platform_system != '\''Darwin'\''\",' pyproject.toml

RUN cd /tmp/Isaac-GR00T && \
    pip3 install --extra-index-url https://pypi.jetson-ai-lab.io/sbsa/cu130 \
    --trusted-host pypi.jetson-ai-lab.io -e .[thor]

RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"

# Build and install decord
RUN cd /tmp && \
    git clone https://git.ffmpeg.org/ffmpeg.git && \
    cd ffmpeg && \
    git checkout n4.4.2 && \
    ./configure --enable-shared --enable-pic --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    python3 setup.py install --user && \
    cd /tmp && \
    git clone https://github.com/huggingface/lerobot.git && \
    cd lerobot && \
    git checkout 483be9aac217c2d8ef16982490f22b2ad091ab46 && \
    pip install -e ".[feetech]" && \
    cd /workspace && \
    rm -rf /tmp/ffmpeg /tmp/decord

# Patch lerobot camera_opencv.py to set MJPEG format
RUN CAMERA_FILE=$(python3 -c "import lerobot.common.cameras.opencv.camera_opencv as m; import os; print(os.path.dirname(m.__file__))")/camera_opencv.py && \
    sed -i '/self._configure_capture_settings()/i\        self.videocapture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('\''M'\'', '\''J'\'', '\''P'\'', '\''G'\''))' "$CAMERA_FILE"

# Set decord library path environment variable
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.local/decord/
