# Telesurgery Workflow

![Telesurgery Workflow](../../docs/source/telesurgery_workflow.jpg)

## Table of Contents
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Asset Setup](#asset-setup)
  - [Environment Variables](#environment-variables)
- [Running the Workflow](#running-the-workflow)

## System Requirements

### Hardware Requirements
- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
    - GPUs without RT Cores (A100, H100) are not supported
- 50GB of disk space

### Software Requirements
- NVIDIA Driver Version >= 555
- CUDA Version >= 12.6
- Python 3.10
- RTI DDS License

## Quick Start

### X86
1. Install NVIDIA driver (>= 555) and CUDA (>= 12.6)

### AARCH64 (IGX)
1. Run the following to setup a docker env with CUDA enabled
```bash
cd <path-to-i4h-workflows>
workflows/telesurgery/docker/setup.sh run

# Inside docker
workflows/telesurgery/docker/setup.sh init
```


2. Create and activate [conda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) environment:
   ```bash
   source ~/miniconda3/bin/activate
   conda create -n telesurgery python=3.10 -y
   conda activate telesurgery
   ```
3. Run the setup script:
   ```bash
   cd <path-to-i4h-workflows>
   bash tools/env_setup_telesurgery.sh
   ```
4. Set environment variables:
   ```bash
   export I4H_WORKFLOWS_DIR=<path-to-i4h-workflows>
   export RTI_LICENSE_FILE=<path-to-rti-license-file>

   export I4H_TELESURGERY_DIR=${I4H_WORKFLOWS_DIR}/workflows/telesurgery
   export PYTHONPATH=${I4H_TELESURGERY_DIR}/scripts
   export NDDS_QOS_PROFILES=${I4H_TELESURGERY_DIR}/scripts/dds/qos_profile.xml
    ```

### Obtain RTI DDS License
RTI DDS is the common communication package for all scripts. Please refer to [DDS website](https://www.rti.com/products) for registration. You will need to obtain a license file and set the `RTI_LICENSE_FILE` environment variable to its path.

### NTP Server (Optional)
```bash
# run your own NTP server in background
docker run -d --name ntp-server --restart=always -p 123:123/udp cturra/ntp

# check if it's running
docker logs ntp-server

# fix server ip in env.sh for NTP Server
export NTP_SERVER_HOST=10.111.66.170

# stop
# docker stop ntp-server && docker rm ntp-server
```

### Environment Variables

Before running any scripts, you need to set up the following environment variables:

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

More recommended variables can be found in [env.sh](./scripts/env.sh)

## Running the Workflow
```bash
cd <path-to-i4h-workflows>/workflows/telesurgery/scripts
source env.sh  # Make sure all env variables are correctly set in env.sh

#export PATIENT_IP=10.137.145.163
export PATIENT_IP=10.111.66.170
export SURGEON_IP=10.111.66.170
```
> Make sure MIRA API Server is up and running (port: 8081) in case of Physical World.

### [Option-1] Patient in Physical World _(x86 / aarch64)_
```bash
# stream camera out
python patient/physical/camera.py --camera realsense --name room --width 1280 --height 720
python patient/physical/camera.py --camera cv2 --name robot --width 1920 --height 1080
```

### [Option-2] Patient in Simulation World _(x86)_
```bash
# download the assets
export ISAAC_ASSET_SHA256_HASH=8e80faed126c533243f50bb01dca3dcf035e86b5bf567d622878866a8ef7f12d
i4h-asset-retrieve

python patient/simulation/main.py
```

### Surgeon connecting to Patient _(x86 / aarch64)_
```bash
# capture robot camera stream
NDDS_DISCOVERY_PEERS=${PATIENT_IP} python surgeon/camera.py --name robot --width 1280 --height 720

# capture room camera stream (optional)
NDDS_DISCOVERY_PEERS=${PATIENT_IP} python surgeon/camera.py --name room --width 1280 --height 720

# connect to gamepad controller and send commands to API Server
python surgeon/gamepad.py --api_host ${PATIENT_IP} --api_port 8081
```


### Important Notes
1. You may need to run multiple scripts simultaneously in different terminals or run in background (in case of docker)
2. A typical setup requires multiple terminals running:
   - Patient: Camera1, Camera2, Controller etc...
   - Surgeon: Camera1, Camera2, Controller etc...

If you encounter issues not covered in the notes above, please check the documentation for each component or open a new issue on GitHub.
