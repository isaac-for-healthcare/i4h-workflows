# Robotic Ultrasound Docker Container

This guide provides instructions for running robotic ultrasound simulations using Docker containers with Isaac Sim.

## Prerequisites

- **Docker Engine**
- **NVIDIA Docker Runtime**
- **Git** with SSH key access to private repositories
- **X11 forwarding** support (for GUI mode)
- **RTI License**
   - Please refer to the [Environment Setup](../README.md#environment-setup) for instructions to prepare the I4H assets and RTI license locally.
   - The license file `rti_license.dat` should be saved in a directory in your host file system, (e.g. `~/docker/rti`), which can be mounted to the docker container.
- **Ultrasound Raytracing Simulator**
   - Please refer to the [Environment Setup - Install the raytracing ultrasound simulator](../README.md#install-raytracing-ultrasound-simulator) instructions to set up the raytracing ultrasound simulator locally.
   - The `raysim` directory should be available on your host file system (e.g., `~/raysim`) to be mounted to the docker container.

## Build the Docker Image

```sh
# Clone the repository
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Set up SSH agent for private repository access
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519  # Replace with your SSH key path

# Build the Docker image with SSH forwarding
docker build --ssh default --no-cache -f workflows/robotic_ultrasound/docker/Dockerfile -t robotic_us:latest .
```

## Running the Container

```sh
# Allow Docker to access X11 display
xhost +local:docker

# Run container with GUI support in background
docker run --name isaac-sim -itd --gpus all --rm \
    --network=host \
    --runtime=nvidia \
    --entrypoint=bash \
    -e DISPLAY=$DISPLAY \
    -e "OMNI_KIT_ACCEPT_EULA=Y" \
    -e "ACCEPT_EULA=Y" \
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
    -v ~/raysim:/workspace/i4h-workflows/workflows/robotic_ultrasound/scripts/raysim:ro \
    robotic_us:latest
```

## Running the Simulation

> Multiple terminals are required for running the simulation.

### 1. Run Policy

First, start the policy runner in the background. This process will wait for simulation data and provide control commands.

```bash
docker exec -it isaac-sim bash
# Inside the container
conda activate robotic_ultrasound
python workflows/robotic_ultrasound/scripts/policy_runner/run_policy.py
```

**Note:** The policy runner should be started first since it will continuously run and communicate with the simulation via DDS.

### 2. Run Isaac Sim Simulation

In a separate terminal session, start the main simulation:

```bash
docker exec -it isaac-sim bash
# Inside the container, run the simulation
conda activate robotic_ultrasound
python workflows/robotic_ultrasound/scripts/simulation/environments/sim_with_dds.py --enable_camera
```


### 3. Ultrasound Raytracing Simulation (Optional)

For realistic ultrasound image generation, you can run the ultrasound raytracing simulator:

```bash
docker exec -it isaac-sim bash
# Inside the container, run the ultrasound raytracing simulator
conda activate robotic_ultrasound
python workflows/robotic_ultrasound/scripts/simulation/examples/ultrasound_raytracing.py
```

This will generate and stream ultrasound images via DDS communication.

### 4. Visualization Utility (Optional)

To visualize the ultrasound images and other sensor data, you can use the visualization utility:

```bash
docker exec -it isaac-sim bash
# Inside the container, run the visualization utility
conda activate robotic_ultrasound
python workflows/robotic_ultrasound/scripts/utils/visualization.py
```

This utility will display real-time ultrasound images and other sensor data streams.

## Troubleshooting

### GPU Device Errors

- **"Failed to create any GPU devices" or "omni.gpu_foundation_factory.plugin" errors**: This indicates GPU device access issues. Try these fixes in order:

  **Verify NVIDIA drivers and container toolkit installation**:
     ```bash
     # Check NVIDIA driver
     nvidia-smi

     # Check Docker can access GPU
     docker run --rm --gpus all --runtime=nvidia nvidia/cuda:12.8.1-devel-ubuntu24.04 nvidia-smi
     ```
   If the `--runtime=nvidia` is not working, you can try to configure Docker daemon for NVIDIA runtime. The file should contain the following content:
     ```json
      {
         "default-runtime": "nvidia",
         "runtimes": {
            "nvidia": {
                  "path": "nvidia-container-runtime",
                  "runtimeArgs": []
            }
         }
      }
     ```

- **Policy not responding**: Ensure the policy runner is started before the simulation and is running in the background

- **No ultrasound images**: Verify that the `raysim` directory is properly mounted and the ultrasound raytracing simulator is running

- **Display issues**: Make sure `xhost +local:docker` was run before starting the container and the terminal shouldn't be running in a headless mode (e.g. in ssh connection without `-X` option)

- **Missing assets**: Verify that the I4H assets and RTI license are properly mounted and accessible

### Verification Commands

After applying fixes, test with these commands:

```bash
# Test basic GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi

# Test Vulkan support
docker run --rm --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY robotic_us:latest vulkaninfo

# Test OpenGL support
docker run --rm --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY robotic_us:latest glxinfo | head -20
```
