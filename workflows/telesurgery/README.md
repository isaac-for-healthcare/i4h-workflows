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
- Ubuntu >= 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
   - GPUs without RT Cores, such as A100 and H100, are not supported
- 50GB of disk space

#### Software Requirements
- [NVIDIA Driver Version >= 570](https://developer.nvidia.com/cuda-12-8-1-download-archive)
- [CUDA Version >= 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive)
- Python 3.10
- [RTI DDS License](https://www.rti.com/free-trial)
- [Docker](https://docs.docker.com/engine/install/) 28.0.4+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 1.17.5+

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

# Using RealSense Camera with NVIDIA H.264 Encoder
python patient/physical/camera.py --camera realsense --name room --width 1280 --height 720

# Using CV2 Camera with NVIDIA H.264 Encoder
python patient/physical/camera.py --camera cv2 --name robot --width 1920 --height 1080

# Using RealSense Camera with NVJPEG Encoder
python patient/physical/camera.py --camera realsense --name room --width 1280 --height 720 --encoder nvjpeg

# Using CV2 Camera with NVJPEG Encoder
python patient/physical/camera.py --camera cv2 --name robot --width 1920 --height 1080 --encoder nvjpeg
```

##### Surgeon Application
```bash
# Start the Docker Container
workflows/telesurgery/docker/real.sh run

# Start the Surgeon Application with NVIDIA H.264 Decoder
python surgeon/camera.py --name [robot|room] --width 1280 --height 720 2> /dev/null

# Run the Surgeon Application with NVJPEG Decoder
python surgeon/camera.py --name [robot|room] --width 1280 --height 720 --decoder nvjpeg
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

# Start the Patient Application with NVIDIA H.264 Encoder
python patient/simulation/main.py

# Start the Patient Application with NVJPEG Encoder
python patient/simulation/main.py --encoder nvjpeg
```

##### Surgeon Application
```bash
# Start the Docker Container
workflows/telesurgery/docker/sim.sh run

# Start the Surgeon Application with NVIDIA H.264 Decoder
python surgeon/camera.py --name robot --width 1280 --height 720 2> /dev/null

# Run the Surgeon Application with NVJPEG Decoder
python surgeon/camera.py --name robot --width 1280 --height 720 --decoder nvjpeg
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

### Advanced NVIDIA Video Codec Configuration

The applications streams H.264 by default using NVIDIA Video Codec. Additional encoding parameters can be customized in the Patient application using the `--encoder_params` argument:

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

### Advanced NVJPEG Configuration

To configure the quality of the encoded frames with NVJPEG encoder, either 

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
#### Unable to launch the applications when using NVIDIA Video Codec

Q: I'm getting an error when I start the application with the NVIDIA Video Codec.

```BASH
[error] [nv_video_encoder.cpp:101] Failed to create encoder: LoadNvEncApi : Current Driver Version does not support this NvEncodeAPI version, please upgrade driver at /workspace/holohub/build/nvidia_video_codec/_deps/nvc_sdk/NvEncoder/NvEncoder.cpp:82
```

**A:** NVIDIA Video Codec requires CUDA version 12 (driver version 570.0) or later. Check out the [NVIDIA Video Codec System Requirements](https://developer.nvidia.com/nvidia-video-codec-sdk/download) section for more details. **


#### Update CUDA Driver on IGX
```bash
# ssh to igx-host to run the following commands
sudo systemctl isolate multi-user

sudo apt purge nvidia-kernel-*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

sudo apt-get -y install linux-headers-nvidia-tegra aptitude
sudo aptitude install nvidia-driver-570-open # Resolve any conflicts

# hard reboot igx (soft reboot may not work)
```

## Licensing

By using the Telesurgery workflow and NVIDIA Video Codec, you are implicitly agreeing to the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [NVIDIA Software Developer License Agreement](https://developer.download.nvidia.com/designworks/DesignWorks_SDKs_Samples_Tools_License_distrib_use_rights_2017_06_13.pdf?t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1zZWFyY2guYnJhdmUuY29tLyJ9). If you do not agree to the EULA, do not run this container.
