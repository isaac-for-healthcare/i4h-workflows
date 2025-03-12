# Scripts for all the simulation and physical world logic

## System Requirements

- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
    - GPUs without RT Cores (A100, H100) are not supported.
- 50GB of disk space

## Environment Setup

The robotic ultrasound workflow is built on the following dependencies:
- [IsaacSim 4.2.0.2](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/index.html)
- [IsaacLab 1.4.1](https://isaac-sim.github.io/IsaacLab/v1.4.1/source/setup/installation/index.html)
- [openpi](https://github.com/Physical-Intelligence/openpi) and [lerobot](https://github.com/huggingface/lerobot)
- [Raytracing Ultrasound Simulator](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/tree/main/ultrasound-raytracing)

### Install NVIDIA Driver and CUDA Toolkit

1. Install the latest NVIDIA driver from [NVIDIA website](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

2. Install the latest [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

NOTE: Please refer to the [Raytracing Ultrasound Simulator](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/tree/main/ultrasound-raytracing) for more details on the version compatibility.

### Obtain license of RTI DDS

RTI DDS is the common communication package for all the scripts, please refer to [DDS website](https://www.rti.com/products) for registration. You will need to obtain a license file for the RTI DDS and set the `RTI_LICENSE_FILE` environment variable to the path of the license file.


### Install Dependencies

Conda is suggested for virtual environment setup, install `Miniconda` from [Miniconda website](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) and create a new virtual environment with Python 3.10.

```sh
# Create a new conda environment
conda create -n robotic_ultrasound python=3.10 -y
# Activate the environment
conda activate robotic_ultrasound
```

If you want to use the ultrasound-raytracing simulator, install and build it by following the instructions in [Raytracing Ultrasound Simulator](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/tree/main/ultrasound-raytracing#installation).


Install other dependencies by running:
```bash
cd <path-to-i4h-workflows>
python tools/install_deps.py
```

### Download the I4H assets

```sh
i4h-asset-retrieve
```

### Set environment variables before running the scripts
Make sure `PYTHONPATH` and `RTI_LICENSE_FILE` is set
```sh
export PYTHONPATH=`<path-to-i4h-workflows>/workflows/robotic_ultrasound/scripts`
export RTI_LICENSE_FILE=<path-to-rti-license-file>
```
