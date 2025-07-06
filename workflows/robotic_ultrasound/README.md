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
- [Quick Start](#quick-start)
- [Next Steps: Running Workflows](#next-steps-running-workflows)
- [Detailed Setup Instructions](#detailed-setup-instructions)
  - [System Requirements](#system-requirements)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Asset Setup](#asset-setup)
  - [Environment Variables](#environment-variables)
- [Workflow Examples](#workflow-examples)
- [Troubleshooting](#troubleshooting)

## Quick Start

Get up and running with the robotic ultrasound workflow in just a few steps:

**⏱️ Setup time:** ~30-40 minutes (depending on your system and network connection)

### Prerequisites Check
- Ubuntu 22.04
- NVIDIA GPU with RT Cores and 24GB+ memory
- NVIDIA Driver ≥ 555
- CUDA ≥ 12.6
- RTI DDS License ([get free trial](https://www.rti.com/free-trial))

### Installation
1. **Create conda environment:**
   ```bash
   conda create -n robotic_ultrasound python=3.10 -y
   conda activate robotic_ultrasound
   ```

2. **Clone and setup:**
   ```bash
   git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
   cd i4h-workflows
   bash tools/env_setup_robot_us.sh --policy pi0
   ```

3. **Download assets:**
   ```bash
   i4h-asset-retrieve
   ```

4. **Set environment variables:**
   ```bash
   export PYTHONPATH=`pwd`/workflows/robotic_ultrasound/scripts:`pwd`
   export RTI_LICENSE_FILE=<path-to-your-rti-license-file>
   ```

✅ **Setup complete!** Your robotic ultrasound workflow is now ready to use.

## Next Steps: Running Workflows

Now that you've completed the setup, here's what you can do next:

### 🎯 Choose Your Workflow Path

#### **Option 1: Start with Liver Scan State Machine**
Perfect for understanding the workflow fundamentals with a structured approach:
```bash
# Run the liver scan state machine
cd workflows/robotic_ultrasound/scripts/simulation
python environments/state_machine/liver_scan_sm.py --enable_cameras
```

#### **Option 2: Try Policy-Based Control**
Experience AI-driven robotic control with PI0 policy:
```bash
# Run with PI0 policy evaluation
cd workflows/robotic_ultrasound/scripts/simulation
python imitation_learning/pi0_policy/eval.py --enable_camera
```

#### **Option 3: Interactive Teleoperation**
Control the robotic arm directly using keyboard, SpaceMouse, or gamepad:
```bash
# Run teleoperation interface
cd workflows/robotic_ultrasound/scripts/simulation
python environments/teleoperation/teleop_se3_agent.py --enable_cameras
```

#### **Option 4: Full Multi-Component Workflow**
Run the complete robotic ultrasound pipeline (requires 4 terminals):

**Terminal 1 - Visualization:**
```bash
cd workflows/robotic_ultrasound/scripts/utils
python visualization.py
```

**Terminal 2 - Policy Runner:**
```bash
cd workflows/robotic_ultrasound/scripts/policy_runner
python run_policy.py
```

**Terminal 3 - Simulation with DDS:**
```bash
cd workflows/robotic_ultrasound/scripts/simulation
python environments/sim_with_dds.py --enable_cameras
```

**Terminal 4 - Ultrasound Raytracing:**
```bash
cd workflows/robotic_ultrasound/scripts/simulation
python examples/ultrasound_raytracing.py
```

### 📚 Learn More
- **[Workflow Examples](#workflow-examples)** - Explore all available scripts and their purposes
- **[Detailed Setup Instructions](#detailed-setup-instructions)** - For advanced configuration and troubleshooting
- **[Troubleshooting](#troubleshooting)** - Common issues and solutions

---

## Detailed Setup Instructions

This section provides comprehensive setup instructions for advanced users or troubleshooting purposes.

### System Requirements

#### Hardware Requirements
- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 24GB of memory ([see NVIDIA's compute capability guide](https://developer.nvidia.com/cuda-gpus#compute))
    - GPUs without RT Cores (A100, H100) are not supported
- 50GB of disk space

#### Software Requirements
- [NVIDIA Driver Version >= 555](https://www.nvidia.com/en-us/drivers/)
- [CUDA Version >= 12.6](https://developer.nvidia.com/cuda-downloads)
- Python 3.10
- [RTI DDS License](https://www.rti.com/free-trial)

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
- **(Use pre-built binary)** Download the pre-release version from [here](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/releases/tag/v0.2.0rc2) and extract to `workflows/robotic_ultrasound/scripts/raysim`
- **(Compiling from source)** Install and build following instructions in [Raytracing Ultrasound Simulator](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/tree/main/ultrasound-raytracing#installation)

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

This will download assets to `~/.cache/i4h-assets/<sha256>`. For more details, refer to the [Asset Container Helper](https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/v0.2.0rc2/docs/catalog_helper.md).

**Note**: During asset download, you may see warnings about blocking functions. This is expected behavior and the download will complete successfully despite these warnings.

### Environment Variables

Before running any scripts, you need to set up the following environment variables:

1. **PYTHONPATH**: Set this to point to the scripts directory:
   ```bash
   export PYTHONPATH=<path-to-i4h-workflows>/workflows/robotic_ultrasound/scripts:<path-to-i4h-workflows>
   ```
   This ensures Python can find the modules under the [`scripts`](./scripts) directory.

2. **RTI_LICENSE_FILE**: Set this to point to your RTI DDS license file:
   ```bash
   export RTI_LICENSE_FILE=<path-to-rti-license-file>
   ```
   This is required for the DDS communication package to function properly.

## Workflow Examples

The robotic ultrasound workflow provides several example scripts demonstrating different components:

### 🔬 Simulation Scripts
Located in [`scripts/simulation`](./scripts/simulation):
- **`basic_simulation.py`** - Basic ultrasound simulation without external dependencies
- **`sim_with_dds.py`** - Simulation with DDS communication for multi-component workflows
- **`ultrasound_raytracing.py`** - Advanced raytracing-based ultrasound simulation

### 🤖 Policy Runner Scripts
Located in [`scripts/policy_runner`](./scripts/policy_runner):
- **`policy_runner.py`** - Generic policy runner supporting multiple AI policies
- **`run_pi0_policy.py`** - Specialized runner for PI0 policy
- **`run_gr00t_policy.py`** - Specialized runner for GR00T N1 policy

### 🏥 Holoscan Applications
Located in [`scripts/holoscan_apps`](./scripts/holoscan_apps):
- **Clinical deployment apps** - Production-ready applications for real-world use
- **Integration examples** - Connecting simulation to clinical workflows

### 🎯 Training Scripts
Located in [`scripts/training`](./scripts/training):
- **Model training pipelines** - Train AI models using simulated ultrasound data
- **Policy optimization** - Improve autonomous scanning strategies

### 📊 Visualization Utilities
Located in [`scripts/utils`](./scripts/utils):
- **`visualization.py`** - Real-time visualization of ultrasound data and robot movements
- **Data analysis tools** - Process and analyze workflow outputs

## Troubleshooting

### Common Issues

**PyTorch Version Warnings**
```
ERROR: pip's dependency resolver does not currently take into account all the packages...
```
These warnings are expected and can be safely ignored. The workflow functions correctly despite version conflicts.

**Asset Download Warnings**
Warnings about blocking functions during asset download are normal and the download will complete successfully.

**GPU Compatibility**
Ensure your GPU has RT Cores. A100 and H100 GPUs are not supported due to lack of RT Cores.

**Environment Variables**
If you encounter import errors, verify that your `PYTHONPATH` includes both the scripts directory and the repository root.

### Getting Help

If you encounter issues not covered above:
1. Check the documentation for each component
2. Review the [GitHub issues](https://github.com/isaac-for-healthcare/i4h-workflows/issues)
3. Open a new issue with detailed error information
