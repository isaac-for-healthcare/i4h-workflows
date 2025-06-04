# Simulation LiveStream from Remote Docker Container

This document describes how to run the simulation in a remote docker container and stream the simulation to a local machine.

## Prerequisites

Please refer to [Livestream Clients Guide in Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html#isaac-sim-short-webrtc-streaming-client) and download [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release)

## Build Docker Image

To build the docker image, you will need to set up the SSH agent and add your SSH key to the agent, so that the docker build process can access the private repository.

```bash
export DOCKER_BUILDKIT=1
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519  # Replace with your SSH key
docker build --ssh default -f workflows/robotic_ultrasound/docker/Dockerfile -t robot_us:latest .
```

## Prepare the RTI License Locally

Please refer to the [Environment Setup](../README.md#environment-setup) for instructions to prepare the I4H assets and RTI license locally.

The license file `rti_license.dat` should be saved in a directory in your host file system, (e.g. `~/docker/rti`), which can be mounted to the docker container.

## Run the Container

Since we need to run multiple instances (policy runner, simulation, etc.), we need to use `-d` to run the container in detached mode.

```bash
xhost +local:docker
docker run --name isaac-sim --entrypoint bash -itd --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e DISPLAY=$DISPLAY \
    -e "PRIVACY_CONSENT=Y" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v ~/.cache/i4h-assets:/root/.cache/i4h-assets:rw \
    -v ~/docker/rti:/root/rti:ro \
    robot_us:latest
```

### Run Policy

```bash
docker exec -it isaac-sim bash
# Inside the container, run the policy
python policy_runner/run_policy.py
```

The policy runner will be running in an environment managed by `uv` located in `/workspace/openpi/.venv`.

### Run Simulation

```bash
docker exec -it isaac-sim bash
# Inside the container, run the simulation
python simulation/environments/sim_with_dds.py --enable_camera --livestream 2
```
