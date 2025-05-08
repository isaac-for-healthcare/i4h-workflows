# Simulation LiveStream from Remote Docker Container

This document describes how to run the simulation in a remote docker container and stream the simulation to a local machine.

## Prerequisites

Please refer to [Livestream Clients Guide in Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html#isaac-sim-short-webrtc-streaming-client) and download [Isaac Sim WebRTC Streamiing Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release)

## Build Docker Image

To build the docker image, you will need to set up the SSH agent and add your SSH key to the agent, so that the docker build process can access the private repository.

```bash
export DOCKER_BUILDKIT=1
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519  # Replace with your SSH key
docker build --ssh default -f workflows/robotic_surgery/docker/Dockerfile -t robotic_surgery:latest .
```

## Prepare the I4H Asset Locally

Please refer to the [Environment Setup](../../README.md#environment-setup) for instructions to prepare the I4H assets locally.

This will create a directory `~/.cache/i4h-assets/<sha256>` containing the I4H assets, which will be mounted to the docker container.

## Run the Container

```bash
docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v ~/.cache/i4h-assets:/root/.cache/i4h-assets:rw \
    robotic_surgery:latest
```

### Run the Simulation

In the container, run the simulation with `--livestream 2` to stream the simulation to the local machine. For example, to run the `reach_psm_sm.py` script, run the following command:

```bash
python simulation/scripts/environments/state_machine/reach_psm_sm.py --livestream 2
```

Please refer to the [simulation README](../scripts/simulation/README.md) for other examples.


### Open the WebRTC Client

Wait for the simulation to start (you should see the message "Reseting the state machine" when the simulation is ready), and open the WebRTC client to view the simulation.
