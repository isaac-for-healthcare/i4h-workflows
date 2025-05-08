# syntax=docker/dockerfile:1.4

FROM nvcr.io/nvidia/isaac-lab:2.0.2

# This line enables SSH access for the following RUN command
RUN apt-get update && apt-get install -y git openssh-client

# Add GitHub to known hosts to avoid "host key verification failed"
RUN mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /workspace/robotic_surgery/scripts

# Fix livestream public endpoint address issue in 2.0.2/2.1.0
RUN sed -i '/--\/app\/livestream\/publicEndpointAddress=/d' /workspace/isaaclab/source/isaaclab/isaaclab/app/app_launcher.py

COPY workflows/robotic_surgery/scripts .

ENV PYTHONPATH=/workspace/robotic_surgery/scripts

RUN --mount=type=ssh /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-deps git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git@v0.1.0

RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install toml

RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-build-isolation -e /workspace/robotic_surgery/scripts/simulation/exts/robotic.surgery.assets

RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-build-isolation -e /workspace/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks
