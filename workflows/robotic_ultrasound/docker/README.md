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

## Prepare Ultrasound Raytracing Simulator (Optional)

For ultrasound simulation capabilities, you'll need the `raysim` module. Please refer to the [Environment Setup - Install the raytracing ultrasound simulator](../README.md#install-the-raytracing-ultrasound-simulator) instructions to set up the raytracing ultrasound simulator locally.

The `raysim` directory should be available on your host file system (e.g., `~/raysim`) to be mounted to the docker container.

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
    -v ~/raysim:/root/raysim:ro \
    robot_us:latest
```

**Note:** The `<path-to-raysim>:/workspace/i4h-workflows/workflows/robotic_ultrasound/scripts/raysim:ro` mount could be required for ultrasound raytracing simulation, If you haven't downloaded the raysim module following the [Environment Setup - Install the raytracing ultrasound simulator](../README.md#install-the-raytracing-ultrasound-simulator) instructions. If you don't need ultrasound simulation, you can omit this mount.

## Running the Simulation

### Run Policy (Background Process)

First, start the policy runner in the background. This process will wait for simulation data and provide control commands.

```bash
docker exec -it isaac-sim bash
python workflows/robotic_ultrasound/scripts/policy_runner/run_policy.py
```

**Note:** The policy runner should be started first since it will continuously run and communicate with the simulation via DDS.

### Run Simulation

In a separate terminal session, start the main simulation:

```bash
docker exec -it isaac-sim bash
# Inside the container, run the simulation
python workflows/robotic_ultrasound/scripts/simulation/environments/sim_with_dds.py --enable_camera --livestream 2
```

**Note:** This will launch the IsaacSim main window where you can visualize the robotic ultrasound simulation in real-time. The `--livestream 2` parameter enables WebRTC streaming for remote viewing.

### Check the WebRTC Streaming

You can open the WebRTC streaming client to check the streaming status now after the simulation has connected to the policy runner.

### Ultrasound Raytracing Simulation (Optional)

For realistic ultrasound image generation, you can run the ultrasound raytracing simulator:

```bash
docker exec -it isaac-sim bash
# Inside the container, run the ultrasound raytracing simulator
python workflows/robotic_ultrasound/scripts/simulation/examples/ultrasound_raytracing.py
```

This will generate and stream ultrasound images via DDS communication.

### Visualization Utility (Optional)

To visualize the ultrasound images and other sensor data, you can use the visualization utility:

```bash
docker exec -it isaac-sim bash
# Inside the container, run the visualization utility
python workflows/robotic_ultrasound/scripts/simulation/utils/visualization.py
```

This utility will display real-time ultrasound images and other sensor data streams.

## Troubleshooting

- **`unknown or invalid runtime name: nvidia`**: This error occurs when the NVIDIA Docker runtime is not installed or properly configured. To fix this:
  1. Install the NVIDIA Container Toolkit:
     ```bash
     # For Ubuntu/Debian
     distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
     curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
     curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
     ```
  2. Restart the Docker daemon:
     ```bash
     sudo systemctl restart docker
     ```
  3. Verify the installation:
     ```bash
     docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
     ```
  
- **Policy not responding**: Ensure the policy runner is started before the simulation and is running in the background
- **No ultrasound images**: Verify that the `raysim` directory is properly mounted and the ultrasound raytracing simulator is running
- **Display issues**: Make sure `xhost +local:docker` was run before starting the container and the terminal shouldn't be running in a headless mode (e.g. in ssh connection without `-X` option)
- **Missing assets**: Verify that the I4H assets and RTI license are properly mounted and accessible
