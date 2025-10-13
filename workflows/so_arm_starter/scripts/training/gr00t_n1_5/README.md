# 🤖 SO-ARM Starter Training with GR00T-N1.5

This repository provides a complete workflow for training GR00T-N1.5 models for SO-ARM Starter applications. NVIDIA Isaac GR00T N1.5 is the world's first open foundation model for generalized humanoid robot reasoning and skills. This cross-embodiment model takes multimodal input, including language and images, to perform manipulation tasks in diverse environments.

## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#️-installation)
- [Data Collection](#-data-collection)
- [Data Conversion](#-data-conversion)
- [Running Training](#-running-training)
- [References](#references)

## 🔍 Overview

This workflow enables you to:

1. Collect the robot trajectories and camera image data while the robot is working as an assistant
2. Convert the collected HDF5 data to LeRobot format
3. Fine-tune a GR00T-N1.5 model
4. Deploy the trained model for inference

## 🛠️ Installation

First, install the necessary environment and dependencies using our [provided script](../../../../../tools/env_setup_so_arm_starter.sh):
```bash
# Install environment with dependencies
./tools/env_setup_so_arm_starter.sh
```

This script:
- Clones Isaac-GR00T repository
- Installs LeRobot and other dependencies

## 📊 Data Collection

To train a GR00T-N1.5 model, you\'ll need to collect your dataset which can be collected in both real world and simulation.

**Collect in simulation**
We provide an implementation in the simulation environment that can generate training episodes.

See the [simulation README](../../simulation/README.md) for more information on how to collect data in simulation.

**Collect in real world**
Huggingface **Lerobot** provides workflow to collect dataset in real world, refer to the [LeRobot documentation](https://huggingface.co/docs/lerobot/main/en/getting_started_real_world_robot) to get more information.


## 🔄 Data Conversion

GR00T-N1.5 uses the **LeRobot** data format for training. If you use dataset in simulation, you need to convert data format before training.
To facilitate this, we provide a script that converts your HDF5 data into the required format. The script is located at:

```
workflows/so_arm_starter/scripts/training/hdf5_to_lerobot.py
```

To run the conversion, navigate to the [`training` folder](../), and execute the following command:

```bash
python -m hdf5_to_lerobot \
    --repo_id=<path_to_save_datset> \
    --hdf5_path=<path_to_hdf5_file>
```

The converted dataset will be saved in `~/.cache/huggingface/lerobot/<repo_id>`.

## 🚀 Running Training

To start training with a GR00T-N1.5:
```bash
python -m gr00t_n1_5/train \
   --dataset-path <path_to_your_lerobot_dataset> \
   --num-gpus 1 \
   --batch-size 32 \
   --output-dir <path_to_save_train_results> \
   --max-steps 10000 \
   --save-steps 2000 \
   --data-config <your_data_config_type>
```
## References

- **NVIDIA Isaac GR00T N1.5**: For more information on the GR00T N1.5 foundation model, refer to [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T).
- **LeRobot Data Format**: The data conversion process utilizes the LeRobot format. For details on LeRobot, see the [LeRobot GitHub repository](https://github.com/huggingface/lerobot).
