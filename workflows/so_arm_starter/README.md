# SO-ARM Starter Workflow

<img src="../../docs/source/so_arm_starter_workflow.jpg" alt="SO-ARM Starter Workflow" width="1080" style="max-width: 100%; height: auto;">

## 🔬 Technical Overview

The SO-ARM Starter Workflow addresses the critical need for autonomous surgical assistance by developing intelligent robotic systems that can perform the essential duties of a surgical assistance. This workflow specifically targets the complex, multi-modal task of surgical instrument management.


## 📋 Table of Contents

- [🔬 Technical Overview](#-technical-overview)
- [🚀 Quick Start](#-quick-start)
  - [Installation](#installation)
    - [Hardware Requirements](#hardware-requirements)
    - [Software Requirements](#software-requirements)
    - [Environment Setup](#environment-setup)
    - [Environment Variables](#environment-variables)
    - [Setup SO-ARM101 Hardware](#setup-so-arm101-hardware)
- [⚡ Running Workflows](#-running-workflows)
  - [📊 Phase 1: Data Collection](#-phase-1-data-collection)
    - [Simulation Data Collection](#simulation-data-collection)
    - [Replay Recorded Data](#replay-recorded-data)
    - [Real-World Data Collection](#real-world-data-collection)
  - [🎯 Phase 2: Model Training](#-phase-2-model-training)
    - [Data Conversion](#data-conversion)
    - [GR00T N1.5 Training](#gr00t-n15-training)
  - [🚀 Phase 3: GR00T N1.5 Deployment](#-phase-3-gr00t-n15-deployment)
    - [Simulation Deployment](#simulation-deployment)
    - [Real-world Deployment](#real-world-deployment)
- [🤖 Model Management](#-model-management)
- [📊 Latency Benchmarking](#-latency-benchmarking)
- [🛠️ Troubleshooting](#️-troubleshooting)

## 🚀 Quick Start

### Installation

#### Hardware Requirements
- **NVIDIA GPU**: RT Core-enabled architecture (Ampere or later)
- **Compute Capability**: ≥8.6
- **VRAM**: ≥30GB GDDR6/HBM
- **SO-ARM101 Follower Arm**: SO-ARM Starter Workflow with integrated dual-camera vision system for surgical assistance and autonomous instrument handling
- **SO-ARM101 Leader Arm**: Teleoperation interface for data collection in both real world and simulation

The SO-ARM101 (SO-101) is a precision robotic arm. This 6-degree-of-freedom (6-DOF) manipulator provides the mechanical foundation for the SO-ARM Starter workflow. For more information, please refer to the [Hugging Face SO-ARM101 Documentation](https://huggingface.co/docs/lerobot/so101).

#### Software Requirements
- **Operating System**: Ubuntu 22.04/24.04 LTS
- **NVIDIA Driver**: ≥535.0
- **CUDA**: ≥12.8
- **Python**: 3.11
- **IsaacSim**: 5.0
- **RTI DDS**: Professional or evaluation license required

**Notice**: Before running the SO-ARM Starter workflow, you need to acquire a professional or evaluation license from [here](https://www.rti.com/free-trial).

#### Environment Setup
Clone the Repository
```bash
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows
```

Create a conda environment with python 3.11
```bash
conda create -n so_arm_starter python=3.11 -y
conda activate so_arm_starter
```

Run the script from the repository root:
```bash
bash tools/env_setup_so_arm_starter.sh
```
**⚠️ Expected Build Time**: The environment setup process takes approximately 10-20 minutes. You may encounter intermediary warnings about macaroon bakery library dependencies - these are non-critical and can be ignored.

### Environment Variables
Before running any scripts, you need to set up the following environment variables:
**PYTHONPATH**: Set this to point to the scripts directory:
   ```bash
   export PYTHONPATH=<path-to-i4h-workflows>/workflows/so_arm_starter/scripts
   ```
**RTI_LICENSE_FILE**: Set this to point to your RTI DDS license file:
   ```bash
   export RTI_LICENSE_FILE=<path-to-rti-license-file>
   ```
   This is required for the DDS communication to function properly.

### Setup SO-ARM101 Hardware

#### Real SO-ARM 101 Configuration

The SO‑ARM 101 robot is equipped with a wrist‑mounted camera module and a 3D‑printed mounting adapter that securely attaches the camera to the arm. The vision system uses components provided by [**WOWROBO**](https://wowrobo.com/).

For detailed specifications and hardware recommendations, see the official [SO‑ARM 101 hardware recommendations](https://github.com/TheRobotStudio/SO-ARM100/tree/385e8d7c68e24945df6c60d9bd68837a4b7411ae?tab=readme-ov-file#kits).

**Navigate to LeRobot Installation**

Ensure the LeRobot dependency is installed in the correct location: `<path-to-i4h-workflows>/third_party/lerobot`.

```bash
cd third_party/lerobot
```

**Find SO-ARM101 Port Information**

Identify the USB port for your SO-ARM101 leader and follower arms and set port permissions:

```bash
python lerobot/find_port.py

# Set proper permissions if needed
sudo chmod 666  <portID_for_leader_arm>
sudo chmod 666  <portID_for_follower_arm>
```

Identify camera index
```bash
python lerobot/find_cameras.py opencv
```

Calibration
Calibrate your so101 follower arm:
```bash
python lerobot/calibrate.py \
    --robot.type=so101_follower \
    --robot.port=<port_id> \
    --robot.id=so101_follower_arm
```

Calibrate your so101 leader arm:
```bash
python lerobot/calibrate.py \
    --teleop.type=so101_leader \
    --teleop.port=<port_id> \
    --teleop.id=so101_leader_arm
```

You can replace the `--robot.id` and `--teleop.id` with custom name. For detailed SO-ARM101 assembly and setup instructions, refer to the [LeRobot SO-ARM101 Documentation](https://huggingface.co/docs/lerobot/so101).

#### Simulated SO-ARM 101 Configuration

The simulated SO‑ARM 101 model is derived from the
[original SO‑ARM 101 URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/385e8d7c68e24945df6c60d9bd68837a4b7411ae/Simulation/SO101/so101_new_calib.urdf) and
augmented with STL geometry provided by [**WOWROBO**](https://wowrobo.com/) for the wrist camera module and its mounting adapter. These assets are consolidated into the single USD file: `SO-ARMDualCamera.usd`.

## ⚡ Running Workflows

The SO-ARM Starter workflow consists of three main phases: data collection, model training, and policy deployment. Each phase can be run independently or as part of a complete pipeline.

If you want to use our pretrained model, skip directly to [Phase 3: GR00T N1.5 Deployment](#-phase-3-gr00t-n15-deployment). If you want to train your own model, follow all three phases below.

**Download the pretrained model:**
```bash
hf download nvidia/SO_ARM_Starter_Gr00t --local-dir <path/to/checkpoint>
```
The model will be saved into <path/to/checkpoint>.

### 📊 Phase 1: Data Collection

#### **Simulation Data Collection**
Collect training data in IsaacSim environment for picking and placing tasks:

```bash
python -m simulation.environments.teleoperation_record \
    --port=<your_leader_arm_port_id> \
    --enable_cameras \
    --record \
    --dataset_path=/path/to/save/dataset
```

Please note that the argument `dataset_path` should be a path of the expected `.hdf5` file.

**Alternative: Keyboard-based Teleoperation**
For users without SO-ARM101 hardware, keyboard-based teleoperation is available for simulation:

```bash
python -m simulation.environments.teleoperation_record \
    --enable_cameras \
    --record \
    --dataset_path=/path/to/save/dataset \
    --teleop_device=keyboard
```

**Simulation Controls:**
- **R Key**: Reset recording environment and stop current recording
- **N Key**: Mark episode as successful and reset environment
- **Keyboard Controls** (when using keyboard teleop):
  - **Joint 1 (shoulder_pan)**: Q (+) / U (-)
  - **Joint 2 (shoulder_lift)**: W (+) / I (-)
  - **Joint 3 (elbow_flex)**: E (+) / O (-)
  - **Joint 4 (wrist_flex)**: A (+) / J (-)
  - **Joint 5 (wrist_roll)**: S (+) / K (-)
  - **Joint 6 (gripper)**: D (+) / L (-)

The way of keyboard control is different from default **Se3Keyboard**, more information please refer to [leisaac source code](https://github.com/LightwheelAI/leisaac/blob/v0.2.0/source/leisaac/leisaac/devices/keyboard/se3_keyboard.py)

#### **Replay Recorded Data**
Review and validate your collected datasets by replaying them in simulation:

```bash
python -m simulation.environments.replay_recording \
    --dataset_path=/path/to/your/dataset.hdf5 \
    --enable_cameras \
    --teleop_device=<device_when_collect_dataset>
```

#### **Real-World Data Collection**
```bash
python /path/to/lerobot/record.py \
    --robot.type=so101_follower \
    --robot.port=<follower_port_id> \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, room: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --robot.id=so101_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=<leader_port_id> \
    --teleop.id=so101_leader_arm \
    --display_data=true \
    --dataset.repo_id=datasets/test \
    --dataset.num_episodes=5 \
    --dataset.single_task="<task description>" \
    --dataset.push_to_hub=false \
```

Replace `--robot.port`, `--teleop.port` and `--index_or_path` with the appropriate values for your hardware setup (see [Hardware Requirements](#hardware-requirements)). Set `num_episodes` to specify the number of data collection episodes to record. The resulting dataset will be automatically saved to `~/.cache/huggingface/lerobot/{repo-id}`.

For more details about real-world data collection using physical SO-ARM101 hardware, refer to the [LeRobot Data Collection Documentation](https://huggingface.co/docs/lerobot/main/en/getting_started_real_world_robot).


### 🎯 Phase 2: Model Training

#### **Data Conversion**
Convert collected HDF5 data to LeRobot format for GR00T training when using simulation data:

```bash
python -m training.hdf5_to_lerobot \
    --repo_id=path/to/save/dataset \
    --hdf5_path=path/to/hdf5/file \
    --task_description=<description_for_action_in_dataset>
```

#### **GR00T N1.5 Training**
Fine-tune the foundation model on surgical assistance data:

```bash
python -m training.gr00t_n1_5.train \
   --dataset_path /path/to/your/dataset \
   --output_dir path/to/save/checkpoint \
   --data_config so100_dualcam
```

### 🚀 Phase 3: GR00T N1.5 Deployment

#### **Simulation Deployment**

1️⃣ Run `policy_runner` script to launch a GR00T N1.5 PyTorch model
```bash
python -m policy_runner.run_policy \
    --ckpt_path=path/to/checkpoint \
    --task_description=<description_for_action>
```

**Optional**: Enable TensorRT mode for faster inference with optimized GPU acceleration. Requires pre-built TensorRT engine files, see the [policy runner documentation](./scripts/policy_runner/README.md#run-policy) for conversion steps.

```bash
python -m policy_runner.run_policy \
  --ckpt_path=<path_to_checkpoint> \
  --task_description=<description_for_action> \
  --trt \
  --trt_engine_path=<path_to_tensorrt_engines_dir>
```

2️⃣ Launch IsaacSim environment with DDS communication enabled

**Open a new terminal** (also need to setup [environment variables](#environment-variables)) and run the following command:
```bash
python -m simulation.environments.sim_with_dds --enable_cameras
```

#### **Real-world Deployment**
##### X86
For real-world deployment, we use a Holoscan application designed for real-time SO-ARM control. Default use GR00T N1.5 **TensorRT** engine, please refer to the [TensorRT Inference section](../so_arm_starter/scripts/policy_runner/README.md#tensorrt-inference) to convert your TensorRT engines.

This application is specifically **designed for real SO-ARM hardware**, not simulation.

```bash
python -m holoscan_apps.gr00t_inference_app --config /path/to/soarm_robot_config.yaml
```

**Note**:
- This Holoscan app requires a **physical SO-ARM** connected on the configured port.
- In the [sample config](scripts/holoscan_apps/soarm_robot_config.yaml) the model paths are empty. Set `model_path` (for the downloaded checkpoint or a fine-tuned model) or `trt_engine_path` (for a TensorRT engine) to the actual location with the right `trt` switch

##### Optional Device
This workflow also supports Jetson Thor, Orin and DGX Spark for real-world deployment within Docker.
Here are steps for Jetson Thor deployment:

**Build Docker**
- Jetson Thor
```bash
cd ../docker && docker build -t soarm-thor -f thor.Dockerfile . # ~20 minutes
```

- Jetson Orin
```bash
cd ../docker && docker build -t soarm-orin -f orin.Dockerfile . # ~20 minutes
```

- DGX Spark
```bash
cd ../docker && docker build -t soarm-dgx -f dgx.Dockerfile . # ~20 minutes
```

**Run Container**
```bash
cd .. && docker run --rm --privileged -it --runtime nvidia \
  -e PYTHONPATH=/workspace/scripts \
  -v /dev:/dev \
  -v "$PWD":/workspace -w /workspace <image_name>
```

Refer to [hardware configuration](#real-so-arm-101-configuration) to find SO-ARM hardware port and camera index, `lerobot repo` is installed in `/tmp/lerobot`, and then config `soarm_robot_config.yaml` with your actual robot and model settings.

```bash
cd /workspace/scripts && python -m holoscan_apps.gr00t_inference_app --config holoscan_apps/soarm_robot_config.yaml
```
If SO-ARM recalibration is required on first run, please refer to the [calibration video](https://huggingface.co/docs/lerobot/so101#calibrate) to move each joint.

For more details about policy inference and deployment using trained GR00T models, refer to the [Isaac GR00T Policy Deployment](https://github.com/NVIDIA/Isaac-GR00T/blob/17a77ebf646cf13460cdbc8f49f9ec7d0d63bcb1/getting_started/5_policy_deployment.md) and [Orin Deployment](https://github.com/NVIDIA/Isaac-GR00T/tree/d18bfc3a3b4ad6432649e364af2b62f483d7cfee/deployment_scripts#deploy-isaac-gr00t-with-container)

## 🤖 Model Management

There is one model in the workflow available on Hugging Face:
- [SO-ARM Starter GR00T](https://huggingface.co/nvidia/SO_ARM_Starter_Gr00t)

If you want to use the pretrained model, you can [download the model manually from Hugging Face](../so_arm_starter/README.md#-running-workflows).

⚠️ **Notice**

This pretrained model is specifically trained with:
- **Text Prompt**: `Grip the scissors and put it into the tray`
- **Fixed Camera Views**: wrist camera (left) and room camera (right)

<img src="../../docs/source/so_arm_starter_real_view.jpg" alt="Real World Camera View Setup - Required camera positions for model deployment" width="800" style="max-width: 100%; height: auto;">

**To deploy this model in the real world:** please use the consistent prompt and match your camera views to the reference images above, you can verify camera setup using the [LeRobot camera command](#real-so-arm-101-configuration). Captured images will be saved to `~/lerobot/outputs/captured_images`.

## 📊 Latency Benchmarking

This section presents inference latency benchmarks for model [SO-ARM Starter GR00T](https://huggingface.co/nvidia/SO_ARM_Starter_Gr00t) across various NVIDIA platforms. All measurements use batch size 1, FP16 precision, dual camera input, and averaged over 100 rounds.

| Platform | Mode | E2E Latency (ms) |
|----------|------|------------------|
| **X86_64 + RTX 6000 Ada** | PyTorch | 42.08 ± 0.69 |
| **X86_64 + RTX 6000 Ada** | TensorRT | 24.54 ± 0.63 |
| **AGX Orin** | PyTorch | 507.47 ± 5.65 |
| **AGX Orin** | TensorRT | 359.82 ± 3.37 |
| **IGX Thor (iGPU)** | PyTorch | 107.85 ± 4.97 |
| **IGX Thor (iGPU)** | TensorRT | 75.86 ± 8.98 |
| **AGX Thor** | PyTorch | 109.43 ± 4.59 |
| **AGX Thor** | TensorRT | 77.91 ± 9.63 |
| **DGX Spark** | PyTorch | 124.32 ± 4.20 |
| **DGX Spark** | TensorRT | 114.72 ± 5.34 |

## 🛠️ Troubleshooting

- **DDS Connection**: Verify RTI license and domain ID consistency
- **Hardware Connection**: Check USB permissions, device availability, and camera views
- **GPU Memory**: Ensure sufficient VRAM for GR00T model inference
- **Camera Access**: Verify camera permissions and USB connections
