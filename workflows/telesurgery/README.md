# Telesurgery Workflow

![Telesurgery Workflow](../../docs/source/telesurgery_workflow.jpg)


- [Telesurgery Workflow](#telesurgery-workflow)
  - [Prerequisites](#prerequisites)
    - [System Requirements](#system-requirements)
    - [Common Setup](#common-setup)
  - [Running the System](#running-the-system)
    - [Real World Environment](#real-world-environment)
    - [Simulation Environment](#simulation-environment)
  - [Advanced Configuration](#advanced-configuration)
    - [NTP Server Setup](#ntp-server-setup)
    - [NVIDIA Video Codec Configuration](#nvidia-video-codec-configuration)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
  - [Licensing](#licensing)


## Prerequisites

### System Requirements

#### Hardware Requirements
- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
   - GPUs without RT Cores, such as A100 and H100, are not supported
- 50GB of disk space

#### Software Requirements
- NVIDIA Driver Version >= 555
- CUDA Version >= 12.6
- Python 3.10
- RTI DDS License
- Docker 28.0.4+
- NVIDIA Container Toolkit 1.17.5+

### Common Setup

#### 1. RTI DDS License Setup
```bash
export RTI_LICENSE_FILE=<full-path-to-rti-license-file>
# for example
export RTI_LICENSE_FILE=/home/username/rti/rti_license.dat
```

> [!Note]
> RTI DDS is the common communication package for all scripts. Please refer to [DDS website](https://www.rti.com/products) for registration. You will need to obtain a license file and set the `RTI_LICENSE_FILE` environment variable to its path.

#### 2. Environment Configuration
When running the Patient and the Surgeon applications on separate systems, export the following environment variables:

```bash
export PATIENT_IP="<IP Address of the system running the Patient application>"
export SURGEON_IP="<IP Address of the system running the Surgeon application>"
export NDDS_DISCOVERY_PEERS="${PATIENT_IP}"

# Export the following for NTP Server
export NTP_SERVER_HOST="<IP Address of the NTP Server>"
export NTP_SERVER_PORT="123"
```

## Running the System

### Real World Environment

#### 1. Build Environment
```bash
cd <path-to-i4h-workflows>
workflows/telesurgery/docker/real.sh build
```

#### 2. Running Applications

##### Patient Application
```bash
# Start the Docker Container
workflows/telesurgery/docker/real.sh run

# Using RealSense Camera
python patient/physical/camera.py --camera realsense --name room --width 1280 --height 720

# Using CV2 Camera
python patient/physical/camera.py --camera cv2 --name robot --width 1920 --height 1080

# Using RealSense Camera with NVIDIA H.264 Encoder
python patient/physical/camera.py --camera realsense --name room --width 1280 --height 720 --encoder nvc

# Using CV2 Camera with NVIDIA H.264 Encoder
python patient/physical/camera.py --camera cv2 --name robot --width 1920 --height 1080 --encoder nvc
```

##### Surgeon Application
```bash
# Start the Docker Container
workflows/telesurgery/docker/real.sh run

# Run the Surgeon Application with NVJPEG Encoder
python surgeon/camera.py --name [robot|room] --width 1280 --height 720

# Start the Surgeon Application with NVIDIA H.264 Encoder
python surgeon/camera.py --name [robot|room] --width 1280 --height 720 --decoder nvc
```

##### Gamepad Controller Application
```bash
# Start the Docker Container
workflows/telesurgery/docker/real.sh run

# Run the Gamepad Controller Application
python surgeon/gamepad.py --api_host ${PATIENT_IP} --api_port 8081
```

### Simulation Environment

#### 1. Build Environment
```bash
cd <path-to-i4h-workflows>
workflows/telesurgery/docker/sim.sh build
```

#### 2. Running Applications

##### Patient Application
```bash
# Start the Docker Container
workflows/telesurgery/docker/sim.sh run

# Start the Patient Application with NVJPEG Encoder
python patient/simulation/main.py

# Start the Patient Application with NVIDIA H.264 Encoder
python patient/simulation/main.py --encoder nvc
```

##### Surgeon Application
```bash
# Start the Docker Container
workflows/telesurgery/docker/sim.sh run

# Run the Surgeon Application with NVJPEG Encoder
python surgeon/camera.py --name robot --width 1280 --height 720

# Start the Surgeon Application with NVIDIA H.264 Encoder
python surgeon/camera.py --name robot --width 1280 --height 720 --decoder nvc
```

##### Gamepad Controller Application
```bash
# Start the Docker Container
workflows/telesurgery/docker/sim.sh run

# Run the Gamepad Controller Application
python surgeon/gamepad.py --api_host ${PATIENT_IP} --api_port 8081
```

## Advanced Configuration

### NTP Server Setup
An NTP (Network Time Protocol) server provides accurate time information to clients over a computer network. NTP is designed to synchronize the clocks of computers to a reference time source, ensuring all devices on the network maintain the same time.

```bash
# Run your own NTP server in the background
docker run -d --name ntp-server --restart=always -p 123:123/udp cturra/ntp

# Check if it's running
docker logs ntp-server

# fix server ip in env.sh for NTP Server
export NTP_SERVER_HOST=<NTP server address>

# To stop the server
docker stop ntp-server && docker rm ntp-server
```

### NVIDIA Video Codec Configuration

Camera data can be streamed using either the H.264 or HEVC (H.265) codecs. To enable this for the Patient and Surgeon applications, use the `--encoder nvc` or `--decoder nvc` argument, respectively.

Encoding parameters can be customized in the Patient application using the `--encoder_params` argument:

```bash
python patient/simulation/main.py --encoder nvc --encoder_params patient/nvc_encoder_params.json
```

#### Sample Encoding Parameters
Here's an example of encoding parameters in JSON format:

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

## Troubleshooting

### Common Issues

#### Docker Build Error
Q: I get the following error when building the Docker image:
```bash
ERROR: invalid empty ssh agent socket: make sure SSH_AUTH_SOCK is set
```

A: Start the ssh-agent
```bash
eval "$(ssh-agent -s)" && ssh-add
```

## Licensing

By using the Telesurgery workflow and NVIDIA Video Codec, you are implicitly agreeing to the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [NVIDIA Software Developer License Agreement](https://developer.download.nvidia.com/designworks/DesignWorks_SDKs_Samples_Tools_License_distrib_use_rights_2017_06_13.pdf?t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1zZWFyY2guYnJhdmUuY29tLyJ9). If you do not agree to the EULA, do not run this container.
