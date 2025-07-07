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
- [Running Workflows](#running-workflows)
- [Detailed Setup Instructions](#detailed-setup-instructions)
  - [System Requirements](#system-requirements)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Asset Setup](#asset-setup)
  - [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Jump to Section](#jump-to-section)

## Quick Start

Get up and running with the robotic ultrasound workflow in just a few steps:

**⏱️ Setup time:** ~30-40 minutes (depending on your system and network connection) (requested by VDR)

### Prerequisites Check
- Ubuntu 22.04/24.04
- NVIDIA GPU with RT Cores, Ampere or later architecture (requested by VDR), and 24GB+ memory
   <details>
   <summary>📋 Check GPU Compatibility</summary>

   ```bash
   nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
   ```

   Ensure your GPU shows:
   - Compute capability ≥8.6 (Ampere or later architecture)
   </details>
- NVIDIA Driver ≥ 555 (for Raytracing Ultrasound Simulator)
   <details>
   <summary>📋 Check Driver Version</summary>

   ```bash
   nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits
   ```

   Ensure your driver shows version ≥555.
   </details>
- CUDA ≥ 12.6 (for Raytracing Ultrasound Simulator)
   <details>
   <summary>📋 Check CUDA Version</summary>

   ```bash
   nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1
   ```

   Ensure your CUDA shows version ≥12.6.
   </details>
- RTI DDS License ([get free trial](https://www.rti.com/free-trial))

### Conda Installation

The robotic ultrasound workflow only supports setup with conda environment, due to the need of both installing pip and system packages.

You can visit [Installing Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) website to install miniconda.
(requested by VDR)

1. **Create a new conda environment with Python 3.10:**
   ```bash
   conda create -n robotic_ultrasound python=3.10 -y
   conda activate robotic_ultrasound
   ```

2. **Clone and setup:**
   ```bash
   git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
   cd i4h-workflows
   bash tools/env_setup_robot_us.sh
   ```

3. **Set environment variables:**
   ```bash
   export PYTHONPATH=`pwd`/workflows/robotic_ultrasound/scripts:$PYTHONPATH
   export RTI_LICENSE_FILE=<path-to-your-rti-license-file>
   ```

   <details>
   <summary>💾 Make Environment Variables Persistent</summary>

   ```bash
   echo "export PYTHONPATH=`pwd`/workflows/robotic_ultrasound/scripts:\$PYTHONPATH" >> ~/.bashrc
   echo "export RTI_LICENSE_FILE=<path-to-your-rti-license-file>" >> ~/.bashrc
   source ~/.bashrc
   ```

   Replace `<path-to-your-rti-license-file>` with your actual RTI license file path.
   </details>

4. **Download Raytracing Ultrasound Simulator: raysim**
   ```bash
   # Download Raytracing Ultrasound Simulator: raysim
   wget https://github.com/isaac-for-healthcare/i4h-sensor-simulation/releases/download/v0.2.0rc1/raysim-py310-linux-v0.2.0rc1.zip -O workflows/robotic_ultrasound/scripts/raysim.zip
   unzip workflows/robotic_ultrasound/scripts/raysim.zip -d workflows/robotic_ultrasound/scripts/raysim
   rm workflows/robotic_ultrasound/scripts/raysim.zip
   ```
   FIXME: this https assets will not be available before GA. Alternatively, you can download it from the link below.

✅ **Setup complete!** Your robotic ultrasound workflow is now ready to use.

## Running Workflows
(requested by VDR)

### First Runs

Now that you've completed the setup, you can try the following workflows:

#### Policy-Based Control with PI0
```bash
(python -m policy_runner.run_policy --policy pi0 & python -m simulation.environments.sim_with_dds --enable_cameras & wait)
```

What's expected (requested by VDR):
- Isaac Sim window running with a Franka robot arm driven by the pi0 policy
- Policy runner is reading images from the cameras and sending commands to the robot arm to move to the target position on DDS topics

> **Note:**
> You may see `"IsaacSim 4.5.0 is not responding"`.
> It can take several minutes to download the assets and models from the internet and load them to the scene.
> If this is the first time you run the workflow, it can take up to **10 minutes**.
> If your network isn't in the US region, it can take longer to download the assets.
> To exit the workflow, press `Ctrl+C` in the terminal and run `bash workflows/robotic_ultrasound/reset.sh` to kill all processes spawned by the workflow.

#### Policy + Ultrasound Simulation
```bash
(python -m policy_runner.run_policy --policy pi0 & \
python -m simulation.environments.sim_with_dds --enable_cameras &  \
python -m simulation.examples.ultrasound_raytracing & \
python -m utils.visualization & \
wait)
```

What's expected (requested by VDR):
- Isaac Sim window running with a Franka robot arm driven by the pi0 policy
- Policy runner is reading images from the cameras and sending commands to the robot arm to move to the target position on DDS topics
- Ultrasound raytracing simulator is running and streaming ultrasound images on DDS topics
- Python GUI Visualization is running and displaying the wrist/room camera images and ultrasound images

> **Note:**
> The ultrasound image would stay still until the isaacsim simulation finishes loading the assets.
> To exit the workflow, press `Ctrl+C` in the terminal and run `bash workflows/robotic_ultrasound/reset.sh` to kill all processes spawned by the workflow.

#### Interactive Teleoperation
Control the robotic arm directly using keyboard, SpaceMouse, or gamepad:
```bash
(python -m simulation.examples.ultrasound_raytracing & \
python -m simulation.environments.teleoperation.teleop_se3_agent --enable_cameras & \
python workflows/robotic_ultrasound/scripts/utils/visualization.py & \
wait)
```

What's expected (requested by VDR):
- Isaac Sim window with the assets loaded
- You can control the robot arm directly using keyboard (pressing buttons like 'w', 'a', 's', 'd'). Check [Teleoperation](./scripts/simulation/environments/teleoperation/README.md) for more details.
- Ultrasound raytracing simulator is running and streaming ultrasound images on DDS topics
- Python GUI Visualization is running and displaying the wrist/room camera images and ultrasound images

> **Notes:**
> To exit the workflow, press `Ctrl+C` in the terminal and run `bash workflows/robotic_ultrasound/reset.sh` to kill all processes spawned by the workflow.

### 🎯 Choose Your Workflow Path

#### Script Catalog

Here is a catalog of all the scripts in the workflow in the [scripts](scripts) folder.

| Category | Script | Purpose | Expected Runtime | Key Requirements | Usage Scenario | Documentation |
|----------|--------|---------|------------------|------------------|----------------|---------------|
| **🚀 Quick Start** | [simulation/imitation_learning/pi0_policy/eval.py](scripts/simulation/imitation_learning/pi0_policy/eval.py) | PI0 policy evaluation | 2-5 minutes | PI0 policy, Isaac Sim | First-time users, policy testing | [Simulation README](./scripts/simulation/imitation_learning/README.md) |
| **🔄 Multi-Component** | [simulation/environments/sim_with_dds.py](scripts/simulation/environments/sim_with_dds.py) | Main simulation with DDS communication | Continuous | Isaac Sim, DDS | Full pipeline testing | [Simulation README](./scripts/simulation/environments/README.md) |
| **🎮 Interactive Control** | [simulation/environments/teleoperation/teleop_se3_agent.py](scripts/simulation/environments/teleoperation/teleop_se3_agent.py) | Manual robot control via keyboard/gamepad | Continuous | Isaac Sim, input device | Manual control, data collection | [Simulation README](./scripts/simulation/environments/teleoperation/README.md) |
| **📊 Visualization** | [utils/visualization.py](scripts/utils/visualization.py) | Real-time camera feeds and ultrasound display | Continuous | DDS, GUI | Monitoring simulations, debugging | [Utils README](./scripts/utils/README.md) |
| **🩺 Ultrasound Simulation** | [simulation/examples/ultrasound_raytracing.py](scripts/simulation/examples/ultrasound_raytracing.py) | Physics-based ultrasound image generation | Continuous | RayTracing Simulator | Realistic ultrasound imaging | [Simulation README](scripts/simulation/examples/README.md) |
| **🤖 Policy Execution** | [policy_runner/run_policy.py](scripts/policy_runner/run_policy.py) | Generic policy runner for PI0 and GR00T N1 models | Continuous | Model inference, DDS | Policy deployment | [Policy Runner README](scripts/policy_runner/README.md) |
| **🏥 Hardware-in-the-loop** | [holoscan_apps/clarius_cast/clarius_cast.py](scripts/holoscan_apps/clarius_cast/clarius_cast.py) | Clarius Cast ultrasound probe integration | Continuous | Clarius probe, Holoscan | Hardware-in-the-loop | [Holoscan Apps README](scripts/holoscan_apps/README.md) |
| **🏥 Hardware-in-the-loop** | [holoscan_apps/clarius_solum/clarius_solum.py](scripts/holoscan_apps/clarius_solum/clarius_solum.py) | Clarius Solum ultrasound probe integration | Continuous | Clarius probe, Holoscan | Hardware-in-the-loop | [Holoscan Apps README](scripts/holoscan_apps/README.md) |
| **🏥 Hardware-in-the-loop** | [holoscan_apps/realsense/camera.py](scripts/holoscan_apps/realsense/camera.py) | RealSense depth camera integration | Continuous | RealSense camera, Holoscan | Hardware-in-the-loop | [Holoscan Apps README](scripts/holoscan_apps/README.md) |
| **🧠 Policy Training** | [training/pi_zero/train.py](scripts/training/pi_zero/train.py) | Train PI0 imitation learning models | Depends on the dataset size | Training data, GPU | Model development | [PI0 Training README](scripts/training/pi_zero/README.md) |
| **🧠 Policy Training** | [training/gr00t_n1/train.py](scripts/training/gr00t_n1/train.py) | Train GR00T N1 foundation models | Depends on the dataset size | Training data, GPU | Advanced model development | [GR00T N1 Training README](scripts/training/gr00t_n1/README.md) |
| **🔄 Data Processing** | [training/convert_hdf5_to_lerobot.py](scripts/training/convert_hdf5_to_lerobot.py) | Convert HDF5 data to LeRobot format | Depends on the dataset size | HDF5 files | Data preprocessing | [GR00T N1 Training README](scripts/training/gr00t_n1/README.md#data-conversion) |
| **📈 Evaluation** | [simulation/evaluation/evaluate_trajectories.py](scripts/simulation/evaluation/evaluate_trajectories.py) | Compare predicted vs ground truth trajectories | Depends on the dataset size | Trajectory data | Performance analysis | [Evaluation README](scripts/simulation/evaluation/README.md) |
| **🏗️ State Machine** | [simulation/environments/state_machine/liver_scan_sm.py](scripts/simulation/environments/state_machine/liver_scan_sm.py) | Automated liver scanning protocol | 5-15 minutes | Isaac Sim | Automated data collection | [Simulation README](scripts/simulation/environments/state_machine/README.md) |
| **🗂️ Data Collection** | [simulation/environments/state_machine/liver_scan_sm.py](scripts/simulation/environments/state_machine/liver_scan_sm.py) | Automated liver scanning protocol | 5-15 minutes | Isaac Sim | Automated data collection | [Simulation README](scripts/simulation/environments/state_machine/README.md) |
| **🔄 Replay** | [simulation/environments/state_machine/replay_recording.py](scripts/simulation/environments/state_machine/replay_recording.py) | Replay recorded robot trajectories | 2-5 minutes | Recording files | Data validation | [Simulation README](scripts/simulation/environments/state_machine/README.md#replay-recordings) |
| **📡 Communication** | [dds/publisher.py](scripts/dds/publisher.py) | DDS data publishing utilities | Continuous | DDS license | Data streaming | [DDS README](scripts/dds/README.md) |
| **📡 Communication** | [dds/subscriber.py](scripts/dds/subscriber.py) | DDS data subscription utilities | Continuous | DDS license | Data reception | [DDS README](scripts/dds/README.md) |

---

## Detailed Setup Instructions

This section provides comprehensive setup instructions for advanced users or troubleshooting purposes.

<details>
<summary>📋 Detailed Setup Instructions</summary>

### System Requirements

#### Hardware Requirements
- Ubuntu 22.04 or 24.04
- NVIDIA GPU with compute capability 8.6 and 24GB of memory ([see NVIDIA's compute capability guide](https://developer.nvidia.com/cuda-gpus#compute))
    - GPUs without RT Cores (A100, H100) are not supported
- 100GB of disk space

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

##### Install Miniconda
You can visit [Installing Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) website to install miniconda.

1. Download Miniconda installer
```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Run the installer
bash Miniconda3-latest-Linux-x86_64.sh
```

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

The total size of the assets is about 65GB, but individual assets will be downloaded when you run a specific workflow automatically. (requested by VDR)

This will download assets to `~/.cache/i4h-assets/<sha256>`. For more details, refer to the [Asset Container Helper](https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/v0.2.0rc1/docs/catalog_helper.md).

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
</details>

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

## Jump to Section

- [Policy Runner](./scripts/policy_runner/README.md)
- [Simulation](./scripts/simulation/README.md)
- [Utils](./scripts/utils/README.md)
- [Holoscan Apps](./scripts/holoscan_apps/README.md)
- [Training GR00T N1](./scripts/training/gr00t_n1/README.md)
- [Training PI0](./scripts/training/pi_zero/README.md)
- [Evaluation](./scripts/simulation/evaluation/README.md)
