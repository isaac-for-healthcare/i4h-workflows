#!/bin/bash

function run() {
  docker run --gpus all --rm -ti \
    --ipc=host \
    --net=host \
    $(for dev in /dev/video*; do echo --device=$dev; done) \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --privileged \
    -v /dev:/dev \
    -v $(pwd):/workspace/i4h-workflows \
    -w /workspace/i4h-workflows \
    nvcr.io/nvidia/clara-holoscan/holoscan:v3.2.0-dgpu \
    bash
}

function init() {
  rm -rf .venv uv.lock
  apt update
  apt install v4l-utils ffmpeg -y

  v4l2-ctl --list-devices

  # Install Miniconda
  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-`uname -m`.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm ~/miniconda3/miniconda.sh
}

$@
