# 🤖 Robotic Ultrasound Workflow

![Robotic Ultrasound Workflow](../../docs/source/robotic_us_workflow.jpg)

---

## 🔬 Technical Overview

The Robotic Ultrasound Workflow is a comprehensive solution designed for healthcare professionals, medical imaging researchers, and ultrasound device manufacturers working in the field of autonomous ultrasound imaging. This workflow provides a robust framework for simulating, training, and deploying robotic ultrasound systems using NVIDIA's advanced ray tracing technology. By offering a physics-accurate ultrasound simulation environment, it enables researchers to develop and validate autonomous scanning protocols, train AI models for image interpretation, and accelerate the development of next-generation ultrasound systems without requiring physical hardware.

The workflow features a state-of-the-art ultrasound sensor simulation that leverages GPU-accelerated ray tracing to model the complex physics of ultrasound wave propagation. The simulator accurately represents:
- Acoustic wave propagation through different tissue types
- Tissue-specific acoustic properties (impedance, attenuation, scattering)
- Real-time B-mode image generation based on echo signals
- Dynamic tissue deformation and movement
- Multi-frequency transducer capabilities

This physics-based approach enables the generation of highly realistic synthetic ultrasound images that closely match real-world data, making it ideal for training AI models and validating autonomous scanning algorithms. The workflow supports multiple AI policies (PI0, GR00T N1) and can be deployed using NVIDIA Holoscan for clinical applications, providing a complete pipeline from simulation to real-world deployment.

---

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [⚡ Running Workflows](#-running-workflows)
- [🔧 Detailed Setup Instructions](#-detailed-setup-instructions)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [📚 Documentation Links](#-documentation-links)

---

## 🚀 Quick Start

### ⏱️ Installation Timeline
**Estimated Setup Duration:** 30-40 minutes (network-dependent asset downloads)

### 🔍 System Prerequisites Validation

#### GPU Architecture Requirements
- **NVIDIA GPU**: RT Core-enabled architecture (Ampere or later)
- **Compute Capability**: ≥8.6
- **VRAM**: ≥24GB GDDR6/HBM
- **Unsupported**: A100, H100 (lack RT Cores for ray tracing acceleration)

   <details>
   <summary>🔍 GPU Compatibility Verification</summary>

   ```bash
   nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
   ```

   Verify output shows compute capability ≥8.6 (Ampere/Ada Lovelace/Hopper with RT Cores)
   </details>

#### Driver & Runtime Requirements
- **NVIDIA Driver**: ≥555.x (RTX ray tracing API support)
- **CUDA Toolkit**: ≥12.6 (OptiX 8.x compatibility)
- **Operating System**: Ubuntu 22.04 LTS / 24.04 LTS

   <details>
   <summary>🔍 Driver Version Validation</summary>

   ```bash
   nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits
   ```
   </details>

   <details>
   <summary>🔍 CUDA Toolkit Verification</summary>

   ```bash
   nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1
   ```
   </details>

#### Communication Middleware
- **RTI Connext DDS**: Professional or evaluation license ([obtain here](https://www.rti.com/free-trial))

---

### 🐍 Conda Environment Setup

The robotic ultrasound workflow requires conda-based environment management due to mixed pip and system package dependencies.

Installation reference: [Miniconda Installation Guide](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

#### 1️⃣ Environment Creation
```bash
conda create -n robotic_ultrasound python=3.10 -y
conda activate robotic_ultrasound
```

#### 2️⃣ Repository Clone & Dependency Installation
```bash
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows
bash tools/env_setup_robot_us.sh
```

#### 3️⃣ Environment Variable Configuration
```bash
export PYTHONPATH=`pwd`/workflows/robotic_ultrasound/scripts:$PYTHONPATH
export RTI_LICENSE_FILE=<path-to-your-rti-license-file>
```

   <details>
   <summary>💾 Persistent Environment Configuration</summary>

   ```bash
   echo "export PYTHONPATH=`pwd`/workflows/robotic_ultrasound/scripts:\$PYTHONPATH" >> ~/.bashrc
   echo "export RTI_LICENSE_FILE=<path-to-your-rti-license-file>" >> ~/.bashrc
   source ~/.bashrc
   ```
   </details>

#### 4️⃣ Raytracing Ultrasound Simulator Installation
```bash
# Download precompiled ray tracing engine
# FIXME: this https assets will not be available before GA. Alternatively, you can download it from the link below.
wget https://github.com/isaac-for-healthcare/i4h-sensor-simulation/releases/download/v0.2.0rc1/raysim-py310-linux-v0.2.0rc1.zip -O workflows/robotic_ultrasound/scripts/raysim.zip
unzip workflows/robotic_ultrasound/scripts/raysim.zip -d workflows/robotic_ultrasound/scripts/raysim
rm workflows/robotic_ultrasound/scripts/raysim.zip
```


✅ **Installation Complete** - Your robotic ultrasound simulation environment is operational.

---

## ⚡ Running Workflows

### 🔬 Experimental Configurations

#### 🤖 Policy-Based Control with PI0
```bash
(python -m policy_runner.run_policy --policy pi0 & python -m simulation.environments.sim_with_dds --enable_cameras & wait)
```

**Expected System Behavior:**
- Isaac Sim physics simulation with Franka robotic arm
- PI0 policy inference pipeline processing visual input streams
- DDS-based communication for real-time control commands and sensor feedback

> ⏳ **Initial Load Time**: First execution may require 10+ minutes for asset download and scene initialization

#### 🔊 Integrated Ultrasound Raytracing Pipeline
```bash
(python -m policy_runner.run_policy --policy pi0 & \
python -m simulation.environments.sim_with_dds --enable_cameras &  \
python -m simulation.examples.ultrasound_raytracing & \
python -m utils.visualization & \
wait)
```

**Expected System Behavior:**
- Full autonomous robotic ultrasound scanning pipeline
- Real-time physics-based ultrasound image synthesis via GPU ray tracing
- Multi-modal sensor data visualization (RGB cameras + ultrasound B-mode)
- DDS middleware for distributed system communication

#### 🎮 Manual Teleoperation Interface
```bash
(python -m simulation.examples.ultrasound_raytracing & \
python -m simulation.environments.teleoperation.teleop_se3_agent --enable_cameras & \
python workflows/robotic_ultrasound/scripts/utils/visualization.py & \
wait)
```

**Expected System Behavior:**
- Direct SE(3) pose control via keyboard/SpaceMouse/gamepad input
- Real-time ultrasound image generation during manual scanning
- Multi-camera visualization with synchronized ultrasound feedback

**Control Mapping**: Reference [Teleoperation Documentation](./scripts/simulation/environments/teleoperation/README.md#keyboard-controls)

> 🔄 **Process Termination**: Use `Ctrl+C` followed by `bash workflows/robotic_ultrasound/reset.sh` to cleanly terminate all distributed processes

---

### 🎯 Workflow Component Matrix

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

## 🔧 Detailed Setup Instructions

<details>
<summary>📋 Advanced Configuration & Troubleshooting</summary>

### 🔩 Hardware Architecture Requirements

#### Compute Infrastructure
- **Operating System**: Ubuntu 22.04 LTS / 24.04 LTS (x86_64)
- **GPU Architecture**: NVIDIA RTX/GeForce RTX/Quadro RTX with RT Cores
- **Memory Requirements**: ≥24GB GPU memory, ≥64GB system RAM
- **Storage**: ≥100GB NVMe SSD (asset caching and simulation data)

### 🏗️ Framework Architecture Dependencies

The robotic ultrasound workflow is built on the following dependencies:
- [IsaacSim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [IsaacLab 2.1.0](https://isaac-sim.github.io/IsaacLab/v2.1.0/index.html)
- [Gr00T N1](https://github.com/NVIDIA/Isaac-GR00T)
- [Cosmos Transfer 1](https://github.com/nvidia-cosmos/cosmos-transfer1/tree/main)
- [openpi](https://github.com/Physical-Intelligence/openpi) and [lerobot](https://github.com/huggingface/lerobot)
- [Raytracing Ultrasound Simulator](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/tree/main/ultrasound-raytracing)
- [RTI Connext DDS](https://www.rti.com/products)

### 🔨 Installation Procedures

#### 1️⃣ NVIDIA Graphics Driver Installation
Install or upgrade to the latest NVIDIA driver from [NVIDIA website](https://www.nvidia.com/en-us/drivers/)

#### 2️⃣ CUDA Toolkit Installation
Install CUDA from [NVIDIA CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

```bash
# Download CUDA installer
 wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
 sudo dpkg -i cuda-keyring_1.1-1_all.deb
 sudo apt-get update
 sudo apt-get -y install cuda-toolkit-12-8
 echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
 echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
 source ~/.bashrc
```

#### 3️⃣ RTI DDS License Configuration
1. Register at [RTI Website](https://www.rti.com/products)
2. Download license file
3. Configure environment variable:
```bash
export RTI_LICENSE_FILE=/path/to/rti_license.dat
```

#### 4️⃣ Conda Environment Initialization
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create isolated environment
conda create -n robotic_ultrasound python=3.10 -y
conda activate robotic_ultrasound
```

#### 5️⃣ Dependency Resolution & Installation
```bash
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows

# Install base dependencies + policy-specific packages
bash tools/env_setup_robot_us.sh --policy pi0     # PI0 policies
# OR
bash tools/env_setup_robot_us.sh --policy gr00tn1 # GR00T N1 foundation models
# OR
bash tools/env_setup_robot_us.sh --policy none    # Base dependencies only
```
The environment for pi0 and gr00tn1 has conflicts with each other. You can only install one of them at a time.

**Dependency Conflict Resolution:**
Expected PyTorch version conflicts between IsaacLab (2.5.1) and OpenPI (2.6.0) are non-critical and can be ignored.

### 📦 Asset Management

#### Automated Asset Retrieval

Asset retrieval is done automatically when running the workflow.

#### Manual Asset Retrieval
```bash
i4h-asset-retrieve
```

**Asset Storage**: `~/.cache/i4h-assets/<sha256>/`
**Total Size**: ~65GB (incremental download)
**Reference**: [Asset Catalog Documentation](https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/v0.2.0rc1/docs/catalog_helper.md)

### 🔧 Environment Configuration

#### Required Environment Variables
```bash
# Python module resolution
export PYTHONPATH=<i4h-workflows-root>/workflows/robotic_ultrasound/scripts:<i4h-workflows-root>

# DDS communication middleware
export RTI_LICENSE_FILE=<path-to-rti-license>

# Optional: CUDA runtime optimization
export CUDA_VISIBLE_DEVICES=0  # Single GPU selection
```

</details>

---

## 🛠️ Troubleshooting

### ⚠️ Common Integration Issues

#### 🔧 Dependency Resolution Conflicts
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages...
isaaclab 0.34.9 requires torch==2.5.1, but you have torch 2.6.0 which is incompatible.
```
**Resolution**: These PyTorch version conflicts are expected and non-blocking. The workflow maintains compatibility across version differences.

#### 🔗 Module Import Resolution
**Symptoms**: `ModuleNotFoundError` or `Error while finding module specification for 'xxx'` during script execution
**Resolution**: Verify `PYTHONPATH` includes both `scripts/` directory and repository root.

### 🆘 Support Resources

- **Technical Documentation**: Component-specific README files
- **Issue Tracking**: [GitHub Issues](https://github.com/isaac-for-healthcare/i4h-workflows/issues)
- **Community Support**: GitHub Discussions

---

## 📚 Documentation Links

### 🎯 Component Documentation
- [🤖 Policy Runner](./scripts/policy_runner/README.md) - AI inference pipeline
- [🔬 Simulation](./scripts/simulation/README.md) - Physics simulation framework
- [🛠️ Utils](./scripts/utils/README.md) - Visualization and debugging tools
- [🏥 Holoscan Apps](./scripts/holoscan_apps/README.md) - Clinical hardware integration
- [🧠 GR00T N1 Training](./scripts/training/gr00t_n1/README.md) - Foundation model training
- [🎓 PI0 Training](./scripts/training/pi_zero/README.md) - Imitation learning training
- [📈 Evaluation](./scripts/simulation/evaluation/README.md) - Performance analysis tools
