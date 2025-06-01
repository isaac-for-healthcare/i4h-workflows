# Telesurgery Workflow

![Telesurgery Workflow](../../docs/source/telesurgery_workflow.jpg)

## Table of Contents
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Running the Workflow](#running-the-workflow)

## System Requirements

### Hardware Requirements
- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
   - GPUs without RT Cores, such as A100 and H100, are not supported
- 50GB of disk space

### Software Requirements
- NVIDIA Driver Version >= 555
- CUDA Version >= 12.6
- Python 3.10
- RTI DDS License

## Quick Start

### x86 & AARCH64 (IGX) Setup

1. **Set up a Docker environment with CUDA enabled (IGX only):**
   ```bash
   cd <path-to-i4h-workflows>
   xhost +
   workflows/telesurgery/docker/setup.sh run

   # Inside Docker
   workflows/telesurgery/docker/setup.sh init
   ```

2. **Set up the x86 environment with CUDA enabled:**
   ```bash
   cd <path-to-i4h-workflows>
   xhost +
   workflows/telesurgery/docker/setup.sh init
   ```

3. **Create and activate a [conda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) environment:**
   ```bash
   source ~/miniconda3/bin/activate
   conda create -n telesurgery python=3.10 -y
   conda activate telesurgery
   ```

4. **Run the setup script:**
   ```bash
   cd <path-to-i4h-workflows>
   bash tools/env_setup_telesurgery.sh
   ```
   > Make sure your public key is added to your GitHub account if authentication fails.

### Obtain RTI DDS License

RTI DDS is the communication package used by all scripts. Please refer to the [DDS website](https://www.rti.com/products) for registration. You will need to obtain a license file and set the `RTI_LICENSE_FILE` environment variable to its path.

### NTP Server (Optional)

An NTP (Network Time Protocol) server provides accurate time information to clients over a computer network. NTP is designed to synchronize the clocks of computers to a reference time source, ensuring all devices on the network maintain the same time.

```bash
# Run your own NTP server in the background
docker run -d --name ntp-server --restart=always -p 123:123/udp cturra/ntp

# Check if it's running
docker logs ntp-server

# fix server ip in env.sh for NTP Server
export NTP_SERVER_HOST=<NTP server address>

# To stop the server
# docker stop ntp-server && docker rm ntp-server
```

### Environment Variables

Before running any scripts, set up the following environment variables:

1. **PYTHONPATH**: Set this to point to the **scripts** directory:
   ```bash
   export PYTHONPATH=<path-to-i4h-workflows>/workflows/telesurgery/scripts
   ```
   This ensures Python can find the modules under the [`scripts`](./scripts) directory.

2. **RTI_LICENSE_FILE**: Set this to point to your RTI DDS license file:
   ```bash
   export RTI_LICENSE_FILE=<path-to-rti-license-file>
   ```
   This is required for the DDS communication package to function properly.

3. **NDDS_DISCOVERY_PEERS**: Set this to the IP address receiving camera data:
   ```bash
   export NDDS_DISCOVERY_PEERS="surgeon IP address"
   ```
More recommended variables can be found in [env.sh](./scripts/env.sh).

## Running the Workflow

```bash
cd <path-to-i4h-workflows>/workflows/telesurgery/scripts
source env.sh  # Make sure all env variables are correctly set in env.sh

export PATIENT_IP=<patient IP address>
export SURGEON_IP=<surgeon IP address>
```
> Make sure the MIRA API Server is up and running (port: 8081) in the case of a physical world setup.

### [Option 1] Patient in Physical World _(x86 / aarch64)_

When running on IGX (aarch64), ensure you are in the Docker environment set up previously.

```bash
# Stream camera output
python patient/physical/camera.py --camera realsense --name room --width 1280 --height 720
python patient/physical/camera.py --camera cv2 --name robot --width 1920 --height 1080
```

### [Option 2] Patient in Simulation World _(x86)_

```bash
# Download the assets
export ISAAC_ASSET_SHA256_HASH=8e80faed126c533243f50bb01dca3dcf035e86b5bf567d622878866a8ef7f12d
i4h-asset-retrieve

python patient/simulation/main.py
```

### Surgeon Connecting to Patient _(x86 / aarch64)_

```bash
# Capture robot camera stream
NDDS_DISCOVERY_PEERS=${PATIENT_IP} python surgeon/camera.py --name robot --width 1280 --height 720

# Capture room camera stream (optional)
NDDS_DISCOVERY_PEERS=${PATIENT_IP} python surgeon/camera.py --name room --width 1280 --height 720

# Connect to gamepad controller and send commands to API Server
python surgeon/gamepad.py --api_host ${PATIENT_IP} --api_port 8081
```

### Important Notes
1. You may need to run multiple scripts simultaneously in different terminals or run in background (in case of docker)
2. A typical setup requires multiple terminals running:
   - Patient: Camera1, Camera2, Controller, etc.
   - Surgeon: Camera1, Camera2, Controller, etc.

If you encounter issues not covered in the notes above, please check the documentation for each component or open a new issue on GitHub.
