# Scripts for all the simulation and physical world logic

## System Requirements

- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
    - GPUs without RT Cores (A100, H100) are not supported.
- NVIDIA Driver Version >= 555
- 50GB of disk space

## Environment Setup

The robotic ultrasound workflow is built on the following dependencies:
- [IsaacSim 4.2.0.2](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/index.html)
- [IsaacLab 1.4.1](https://isaac-sim.github.io/IsaacLab/v1.4.1/source/setup/installation/index.html)
- [openpi](https://github.com/Physical-Intelligence/openpi) and [lerobot](https://github.com/huggingface/lerobot)
- [Raytracing Ultrasound Simulator](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/tree/main/ultrasound-raytracing)

### Install NVIDIA Driver

Install or upgrade to the latest NVIDIA driver from [NVIDIA website](https://www.nvidia.com/en-us/drivers/)

**NOTE**: [Raytracing Ultrasound Simulator](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/tree/main/ultrasound-raytracing) requires driver version >= 555.


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

To use the ultrasound-raytracing simulator, you can
- (experimental) Download the pre-release version from [here](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/releases/tag/ultrasound-raytracing) and extract the folder to `workflows/robotic_ultrasound/scripts/raysim`.

- Install and build it by following the instructions in [Raytracing Ultrasound Simulator](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/tree/main/ultrasound-raytracing#installation).


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
Make sure `PYTHONPATH`, `RTI_LICENSE_FILE` and `LD_PRELOAD` is set
```sh
export PYTHONPATH=`<path-to-i4h-workflows>/workflows/robotic_ultrasound/scripts`
export RTI_LICENSE_FILE=<path-to-rti-license-file>
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
