# Robotic Ultrasound Docker Container

This guide provides instructions for running robotic ultrasound simulations using Docker containers with Isaac Sim.

## Prerequisites

- **Docker Engine**
- **NVIDIA Docker Runtime**
- **X11 forwarding** support (for GUI mode)
- **RTI License**
   - Please refer to the [Environment Setup](../README.md#environment-setup) for instructions to prepare the I4H assets and RTI license locally.
   - The license file `rti_license.dat` should be saved in a directory in your host file system, (e.g. `~/docker/rti`), which can be mounted to the docker container.

## Build the Docker Image

```sh
# Clone the repository
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows
# Build the Docker image with no cache to ensure fresh build
docker build --no-cache -f workflows/robotic_ultrasound/docker/Dockerfile -t robotic_us:latest .
```

## Running the Container

```bash
# Allow Docker to access X11 display
xhost +local:docker

# Run container with GUI support
docker run --name isaac-sim -it --gpus all --rm \
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
    -v ~/.cache/huggingface:/root/.cache/huggingface:rw \
    -v ~/docker/rti:/root/rti:ro \
    robotic_us:latest
```

## Running the Simulation

The command to run the simulation is the same as [Running Workflows](../README.md#running-workflows) section.

For example,
```sh
# Inside the container
conda activate robotic_ultrasound

# Run simulation with GUI
(python -m policy_runner.run_policy --policy pi0 & python -m simulation.environments.sim_with_dds --enable_cameras & wait)
```

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

- **No ultrasound images**: Verify that the ultrasound raytracing simulator is running

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
