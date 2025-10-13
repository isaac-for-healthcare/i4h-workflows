# SO-ARM Starter Workflow Docker Container

## Prerequisites

- **NVIDIA Docker Runtime** (nvidia-container-toolkit)
- **NVIDIA GPU** with CUDA support (RTX 6000 Ada Generation or compatible)
- **NVIDIA Drivers** (version 535 or higher)
- **X11 forwarding** support (for GUI mode)
- **Sufficient disk space** (at least 50GB for asset caching and model)
- **Memory** (at least 32GB RAM recommended)
- **RTI License** should be placed in the `~/docker/rti/` directory on your host system

## Build the Docker Image

Build the Docker image with all necessary dependencies:

```bash
# Clone the repository (if not already done)
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows

# Build the Docker image with no cache to ensure fresh build
docker build --no-cache -f workflows/so_arm_starter/docker/Dockerfile -t so_arm_starter:latest .
```

## Running the Container

### Enable X11 Display Access
```bash
# Allow Docker containers to access your X11 display
xhost +local:docker
```

### Run the Container

Please add `--privileged` and mount host USB ports (e.g., `-v /dev:/dev`) when accessing SOARM hardware through the container, as this grants the necessary permissions for hardware communication.

```bash
docker run --name soarm -it --gpus all --privileged --rm \
    --runtime=nvidia \
    --entrypoint=bash \
    -e "OMNI_KIT_ACCEPT_EULA=Y" \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=$DISPLAY" \
    -e "NVIDIA_VISIBLE_DEVICES=all" \
    -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ~/.Xauthority:/root/.Xauthority:rw \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v ~/docker/rti:/root/rti:ro \
    -v /dev:/dev \
    so_arm_starter:latest
```
### Running Workflow

Once inside the container, you can run the SO-ARM Starter workflow. For detailed instructions on running simulation scenarios, training policies, and evaluating models, refer to the [Main Workflow Guide](../README.md#-running-workflows) which contains comprehensive examples and command-line options.

## Troubleshooting
- If run with error  **`GLIBCXX_3.4.30' not found**, please run
```bash
conda install -c conda-forge libgcc-ng=12 libstdcxx-ng=12 -y
```

- For RTI license configuration, software installation issues, and additional troubleshooting steps, refer to the [Docker Guide](../../robotic_ultrasound/docker/README.md#troubleshooting) which contains comprehensive solutions for common container and environment setup problems.
