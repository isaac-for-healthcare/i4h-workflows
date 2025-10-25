# Robotic Surgery Docker Container

This guide provides instructions for running robotic surgery simulations using Docker containers with Isaac Sim.

## Prerequisites

- **Docker Engine**
- **NVIDIA Docker Runtime**
- **X11 forwarding** support (for GUI mode)

## Build the Docker Image

```sh
# Clone the repository
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows

docker build -f workflows/robotic_surgery/docker/Dockerfile -t robotic_surgery:latest .
```

## Running the Container

```sh
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
    robotic_surgery:latest
```

## Running the Simulation

### 1. Interactive GUI Mode with X11 Forwarding

The command to run the simulation is the same as [Running Workflows](../README.md#-running-workflows) section.

For example,

```bash
# Inside the container
conda activate robotic_surgery

# Run simulation with GUI
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_psm_sm.py
```

### 2. Headless Streaming Mode

For headless streaming server setup, you can use the WebRTC streaming mode, with `livestream` flag.

```bash
# Inside the container
conda activate robotic_surgery

# Run simulation with WebRTC streaming
python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_psm_sm.py --livestream 2
```

#### WebRTC Client Setup

1. **Download the Isaac Sim WebRTC Client**:
   - Visit the [Isaac Sim Download Page](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html)
   - Download the **Isaac Sim WebRTC Streaming Client**

2. **Configure Connection**:
   - Open the WebRTC client
   - Enter the server IP address (container host IP)
   - Wait for the simulation to initialize (look for "Resetting the state machine" message)
   - Click "Connect"

3. **Network Requirements**:
   - Ensure ports are accessible: `TCP/UDP 47995-48012`, `TCP/UDP 49000-49007`, `TCP 49100`
   - For remote access, configure firewall rules accordingly

## Troubleshooting

### Common Issues

**Display Issues**:
```bash
# Reset X11 permissions
xhost -local:docker
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY
```
This value should be set for interactive mode.
