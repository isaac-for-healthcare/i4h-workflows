# Scripts for all robotic surgery simulations

## System Requirements

- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
    - GPUs without RT Cores (A100, H100) are not supported.
- NVIDIA Driver Version >= 555
- 50GB of disk space

## Environment Setup

The robotic surgery workflow is built on the following dependencies:
- [IsaacSim 4.1.0.0](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/index.html)
- [IsaacLab 1.2.0](https://isaac-sim.github.io/IsaacLab/v1.2.0/source/setup/installation/index.html)


### Install NVIDIA Driver

Install or upgrade to the latest NVIDIA driver from [NVIDIA website](https://www.nvidia.com/en-us/drivers/)


### Install Dependencies

Conda is suggested for virtual environment setup, install `Miniconda` from [Miniconda website](https://docs.anaconda.com/miniconda/install/#quick-command-line-install).

Conda is suggested for virtual environment setup, install `Miniconda` from [Miniconda website](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) and create a new virtual environment with Python 3.10.

```sh
# Create a new conda environment
conda create -n robotic_surgery python=3.10 -y
# Activate the environment
conda activate robotic_surgery
```

The following command line will create a conda environment named `robotic_surgery` and install all dependencies by running:
```bash
cd <path-to-i4h-workflows>
bash tools/env_setup_robot_surgery.sh
```

**NOTE**: If you want to specify a different conda environment name, you can pass the name as an argument to the script.
```bash
bash tools/env_setup_robot_surgery.sh <conda_env_name>
```

### Download the I4H assets

Use the following command will download the assets to the `~/.cache/i4h-assets/<sha256>` directory.
Please refer to the [Asset Container Helper](https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/main/docs/catalog_helper.md) for more details.

```sh
i4h-asset-retrieve
```

## Run the scripts

The robotic surgery workflow provides several example scripts demonstrating surgical robot simulations, state machine implementations, and reinforcement learning capabilities. These examples include dVRK and STAR robot arm control, suture needle manipulation, and peg block tasks.

Navigate to these sub-directories and run the scripts.

- [Simulation](./scripts/simulation)
