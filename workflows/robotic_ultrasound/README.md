# Robotic Ultrasound Workflow

![Robotic Ultrasound Workflow](../../docs/source/robotic_us_workflow.jpg)

The Robotic Ultrasound Workflow is a comprehensive solution designed for healthcare professionals, medical imaging researchers, and ultrasound device manufacturers working in the field of autonomous ultrasound imaging. This workflow provides a robust framework for simulating, training, and deploying robotic ultrasound systems using NVIDIA's advanced ray tracing technology. By offering a physics-accurate ultrasound simulation environment, it enables researchers to develop and validate autonomous scanning protocols, train AI models for image interpretation, and accelerate the development of next-generation ultrasound systems without requiring physical hardware.

The workflow features a state-of-the-art ultrasound sensor simulation that leverages GPU-accelerated ray tracing to model the complex physics of ultrasound wave propagation. The simulator accurately represents:
- Acoustic wave propagation through different tissue types
- Tissue-specific acoustic properties (impedance, attenuation, scattering)
- Real-time B-mode image generation based on echo signals
- Dynamic tissue deformation and movement
- Multi-frequency transducer capabilities

This physics-based approach enables the generation of highly realistic synthetic ultrasound images that closely match real-world data, making it ideal for training AI models and validating autonomous scanning algorithms. The workflow supports multiple AI policies (PI0, GR00T N1) and can be deployed using NVIDIA Holoscan for clinical applications, providing a complete pipeline from simulation to real-world deployment.


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
- NVIDIA GPU with compute capability 8.6 and 24GB of memory ([see NVIDIA's compute capability guide](https://developer.nvidia.com/cuda-gpus#compute))
    - GPUs without RT Cores (A100, H100) are not supported
- 50GB of disk space

### Software Requirements
- [NVIDIA Driver Version >= 555](https://www.nvidia.com/en-us/drivers/)
- [CUDA Version >= 12.6]((https://developer.nvidia.com/cuda-downloads))
- Python 3.10
- [RTI DDS License](https://www.rti.com/free-trial)

## Quick Start

**Note**: The setup process takes approximately 30-40 minutes to complete, depending on your system and network connection.

1. Install NVIDIA driver (>= 555) and CUDA (>= 12.6)
2. Install conda:

   [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) is recommended.

3. Create and activate conda environment:
   ```bash
   conda create -n robotic_ultrasound python=3.10 -y
   conda activate robotic_ultrasound
   ```
4. Clone the repository:
   ```bash
   git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
   cd i4h-workflows
   ```
5. Run the setup script:
   ```bash
   cd <path-to-i4h-workflows>
   bash tools/env_setup_robot_us.sh --policy pi0
   ```
6. Download assets:
   ```bash
   i4h-asset-retrieve
   ```
7. Set environment variables:
   ```bash
   export PYTHONPATH=`<path-to-i4h-workflows>/workflows/robotic_ultrasound/scripts:<path-to-i4h-workflows>`
   export RTI_LICENSE_FILE=<path-to-rti-license-file>
   ```

## Environment Setup

**Note**: The setup process takes approximately 30-40 minutes to complete, depending on your system and network connection.

### Prerequisites

The robotic ultrasound workflow is built on the following dependencies:
- [IsaacSim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [IsaacLab 2.1.0](https://isaac-sim.github.io/IsaacLab/v2.1.0/index.html)
- [openpi](https://github.com/Physical-Intelligence/openpi) and [lerobot](https://github.com/huggingface/lerobot)
- [Raytracing Ultrasound Simulator](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/tree/main/ultrasound-raytracing)

### Installation Steps

#### 1. Install NVIDIA Driver
Install or upgrade to the latest NVIDIA driver from [NVIDIA website](https://www.nvidia.com/en-us/drivers/)

**Note**: The Raytracing Ultrasound Simulator requires driver version >= 555.

#### 2. Install CUDA
Install CUDA from [NVIDIA CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

**Note**: The Raytracing Ultrasound Simulator requires CUDA version >= 12.6.

#### 3. Obtain RTI DDS License
RTI DDS is the common communication package for all scripts. Please refer to [DDS website](https://www.rti.com/products) for registration. You will need to obtain a license file and set the `RTI_LICENSE_FILE` environment variable to its path.

#### 4. Install Dependencies

##### Create Conda Environment
```bash
# Create a new conda environment
conda create -n robotic_ultrasound python=3.10 -y
# Activate the environment
conda activate robotic_ultrasound
```

##### Install Raytracing Ultrasound Simulator
Choose one of the following options:
- **(Experimental)** Download the pre-release version from [here](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/releases/tag/v0.2.0rc1) and extract to `workflows/robotic_ultrasound/scripts/raysim`
- **(Recommended)** Install and build following instructions in [Raytracing Ultrasound Simulator](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/tree/main/ultrasound-raytracing#installation)

##### Install All Dependencies
The main script `tools/env_setup_robot_us.sh` installs all necessary dependencies. It first installs common base components and then policy-specific packages based on an argument.

###### Base Components
- IsaacSim 4.5.0 (and core dependencies)
- IsaacLab 2.1.0
- Robotic Ultrasound Extension (`robotic_us_ext`)
- Lerobot (from Hugging Face)
- Holoscan 2.9.0 (including associated Holoscan apps)
- Essential build tools and libraries

###### Policy-Specific Dependencies
The script supports installing additional policy-specific dependencies using the `--policy` flag:
- **`--policy pi0` (Default)**: Installs PI0 policy dependencies (e.g., OpenPI)
- **`--policy gr00tn1`**: Installs GR00T N1 policy dependencies (e.g., Isaac-GR00T)
- **`--policy none`**: Installs only common base dependencies

Run the script from the repository root:
```bash
cd <path-to-i4h-workflows>
bash tools/env_setup_robot_us.sh --policy <your_chosen_policy>
```

**Note**: During dependency installation, you may see PyTorch version mismatch warnings. These are expected and can be safely ignored:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
isaaclab 0.34.9 requires torch==2.5.1, but you have torch 2.6.0 which is incompatible.
isaaclab-rl 0.1.0 requires torch==2.5.1, but you have torch 2.6.0 which is incompatible.
isaaclab-tasks 0.10.24 requires torch==2.5.1, but you have torch 2.6.0 which is incompatible.
rl-games 1.6.1 requires wandb<0.13.0,>=0.12.11, but you have wandb 0.19.9 which is incompatible.
```
These warnings occur because `isaaclab` and `openpi` require different fixed versions of PyTorch. The workflow will function correctly despite these warnings.

### Asset Setup

Download the required assets using:
```bash
i4h-asset-retrieve
```

This will download assets to `~/.cache/i4h-assets/<sha256>`. For more details, refer to the [Asset Container Helper](https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/v0.2.0rc1/docs/catalog_helper.md).

**Note**: During asset download, you may see warnings about blocking functions. This is expected behavior and the download will complete successfully despite these warnings.

### Environment Variables

Before running any scripts, you need to set up the following environment variables:

1. **PYTHONPATH**: Set this to point to the scripts directory:
   ```bash
   export PYTHONPATH=<path-to-i4h-workflows>/workflows/robotic_ultrasound/scripts
   ```
   This ensures Python can find the modules under the [`scripts`](./scripts) directory.

2. **RTI_LICENSE_FILE**: Set this to point to your RTI DDS license file:
   ```bash
   export RTI_LICENSE_FILE=<path-to-rti-license-file>
   ```
   This is required for the DDS communication package to function properly.


## Running the Workflow

The robotic ultrasound workflow provides several example scripts demonstrating different components:

- [Holoscan Apps](./scripts/holoscan_apps)
- [Policy Runner](./scripts/policy_runner)
- [Simulation](./scripts/simulation)
- [Training](./scripts/training)
- [Visualization Utilities](./scripts/utils)

### Important Notes
1. You may need to run multiple scripts simultaneously in different terminals
2. A typical setup requires 4 terminals running:
   - Visualization
   - Policy runner
   - Sim_with_dds
   - Ultrasound raytracing simulations

If you encounter issues not covered in the notes above, please check the documentation for each component or open a new issue on GitHub.
