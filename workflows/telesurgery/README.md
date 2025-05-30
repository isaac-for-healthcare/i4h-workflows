# Telesurgery Workflow

![Telesurgery Workflow](../../docs/source/telesurgery_workflow.jpg)

## Table of Contents
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Running the Workflow](#running-the-workflow)
- [Licensing](#licensing)

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
- Docker 28.0.4+
- NVIDIA Container Toolkit CLI version 1.17.5+

## Quick Start

### x86 && AARCH64 (IGX) Setup
1. Obtain RTI DDS Licenss

   ```
   export RTI_LICENSE_FILE=<full-path-to-rti-license-file>
   # for example
   export RTI_LICENSE_FILE=/home/username/rti/rti_license.dat
   ```
   > [!Note]
   > RTI DDS is the common communication package for all scripts. Please refer to [DDS website](https://www.rti.com/products) for registration. You will need to obtain a license file and set the `RTI_LICENSE_FILE` environment variable to its path.

2. Configure Your Environment
   When running the Patient and the Surgeon applications on separate systems, export the following environment variables:

   ```bash
   export PATIENT_IP="<IP Address of the system running the Patient application>"
   export SURGEON_IP="<IP Address of the system running the Surgeon application>"
   export NDDS_DISCOVERY_PEERS="${SURGEON_IP}"
   ```

3. Run the following to build and run a docker environment
   ```bash
   cd <path-to-i4h-workflows>

   # Build Docker container
   workflows/telesurgery/docker/setup.sh build

   # Inside docker
   workflows/telesurgery/docker/setup.sh run
   ```

### NTP Server (Optional)
An NTP (Network Time Protocol) server is a server that uses the Network Time Protocol to provide accurate time information to clients over a computer network. NTP is a protocol designed to synchronize the clocks of computers to a reference time source, ensuring that all devices on the network maintain the same time.
```bash
# run your own NTP server in background
docker run -d --name ntp-server --restart=always -p 123:123/udp cturra/ntp

# check if it's running
docker logs ntp-server

# fix server ip in env.sh for NTP Server
export NTP_SERVER_HOST=<NTP server address>

# stop
# docker stop ntp-server && docker rm ntp-server
```

## Running the Workflow

### [Option-1] Patient in Physical World _(x86 / aarch64)_

```bash
# stream camera out
python patient/physical/camera.py --camera realsense --name room --width 1280 --height 720
python patient/physical/camera.py --camera cv2 --name robot --width 1920 --height 1080
```
> Make sure MIRA API Server is up and running (port: 8081) in case of Physical World.

### [Option-2] Patient in Simulation World _(x86)_
```bash
# download the assets
i4h-asset-retrieve

python patient/simulation/main.py [--encoder nvc]
```

### Surgeon connecting to Patient _(x86 / aarch64)_
```bash
# capture robot camera stream
NDDS_DISCOVERY_PEERS=${PATIENT_IP} python surgeon/camera.py --name robot --width 1280 --height 720 [--decoder nvc]

# capture room camera stream (optional)
NDDS_DISCOVERY_PEERS=${PATIENT_IP} python surgeon/camera.py --name room --width 1280 --height 720 [--decoder nvc]

# connect to gamepad controller and send commands to API Server
python surgeon/gamepad.py --api_host ${PATIENT_IP} --api_port 8081
```

### Using H.264/HEVC Encoder/Decoder from NVIDIA Video Codec

Camera data can be streamed using either the H.264 or HEVC (H.265) codecs. To enable this for the Patient and Surgeon applications, use the `--encoder nvc` or `--decoder nvc` argument, respectively.

Encoding parameters can be customized in the Patient application using the `--encoder_params` argument, as shown below:

```bash
python patient/simulation/main.py --encoder nvc --encoder_params patient/nvc_encoder_params.json
```

#### Sample Encoding Parameters for the NVIDIA Video Codec

Here’s an example of encoding parameters in JSON format:

```json
{
    "codec": "H264", // Possible values: H264 or HEVC
    "preset": "P3", // Options include P3, P4, P5, P6, P7
    "bitrate": 10000000,
    "frame_rate": 60,
    "rate_control_mode": 1, // Options: 0 for Constant QP, 1 for Variable bitrate, 2 for Constant bitrate
    "multi_pass_encoding": 0 // Options: 0 to disable, 1 for Quarter resolution, 2 for Full resolution
}
```

> [!NOTE]
> H.264 or HEVC (H.265) codecs are available on x86 platform only.

### Important Notes
1. You may need to run multiple scripts simultaneously in different terminals or run in background (in case of docker)
2. A typical setup requires multiple terminals running:
   - Patient: Camera1, Camera2, Controller etc...
   - Surgeon: Camera1, Camera2, Controller etc...

If you encounter issues not covered in the notes above, please check the documentation for each component or open a new issue on GitHub.

## Licensing

By using the Telesurgery workflow and NVIDIA Video Codec, you are implicitly agreeing to the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [NVIDIA Software Developer License Agreement](https://developer.download.nvidia.com/designworks/DesignWorks_SDKs_Samples_Tools_License_distrib_use_rights_2017_06_13.pdf?t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1zZWFyY2guYnJhdmUuY29tLyJ9). If you do not agree to the EULA, do not run this container.
