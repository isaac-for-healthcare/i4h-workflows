# 🤖 Robotic Ultrasound Training with GR00T-N1

This repository provides a complete workflow for training GR00T-N1 models for robotic ultrasound applications. NVIDIA Isaac GR00T N1 is the world's first open foundation model for generalized humanoid robot reasoning and skills. This cross-embodiment model takes multimodal input, including language and images, to perform manipulation tasks in diverse environments.

## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#️-installation)
- [Data Collection](#-data-collection)
- [Data Conversion](#-data-conversion)
- [Running Training](#-running-training)
- [Understanding Outputs](#-understanding-outputs)
- [References](#references)

## 🔍 Overview

This workflow enables you to:

1. Collect the robot trajectories and camera image data while the robot is performing a ultrasound scan
2. Convert the collected HDF5 data to LeRobot format
3. Fine-tune a GR00T-N1 model
4. Deploy the trained model for inference

## 🛠️ Installation

First, install the necessary environment and dependencies using our [provided script](../../../../../tools/env_setup_robot_us.sh):
```bash
# Install environment with dependencies
./tools/env_setup_robot_us.sh --policy gr00tn1
```

This script:
- Clones Isaac-GR00T repository
- Installs LeRobot and other dependencies

## 📊 Data Collection

To train a GR00T-N1 model, you\'ll need to collect robotic ultrasound data. We provide a state machine implementation in the simulation environment that can generate training episodes that emulate a liver ultrasound scan.

See the [simulation README](../../simulation/README.md#liver-scan-state-machine) for more information on how to collect data.

## 🔄 Data Conversion

GR00T-N1 uses the **LeRobot** data format for training. To facilitate this, we provide a script that converts your HDF5 data into the required format. The script is located at:

```
workflows/robotic_ultrasound/scripts/training/convert_hdf5_to_lerobot.py
```

To run the conversion, navigate to the [`training` folder](../) (located one level above this GR00T-N1-specific README), and execute the following command:

```bash
python convert_hdf5_to_lerobot.py /path/to/your/hdf5/data --feature_builder_type gr00tn1
```

Replace `/path/to/your/hdf5/data` with the actual path to your dataset.

**Key Arguments & Differences for GR00T-N1:**
- `data_dir`: Path to the directory containing HDF5 files. (default: "<path-to-i4h-workflows>/workflows/robotic_ultrasound/scripts/simulation/data/hdf5/<date-task-name>")
- `--feature_builder_type`: **Crucial for GR00T-N1.** Set this to `gr00tn1`. This ensures:
    - The correct feature keys are used (e.g., `action` for actions, `observation.images.room` for room camera, etc., as defined in `GR00TN1FeatureDict`).
    - The `modality.json` file specific to GR00T-N1 is copied to the output dataset\'s `meta/` directory. This file defines the expected data modalities and their properties for the GR00T-N1 model.
    - Default `include_video` for `GR00TN1FeatureDict` is `True`.
- `--repo_id`: Name for your dataset (default: "i4h/robotic_ultrasound")
- `--task_prompt`: Text description of the task (default: "Perform a liver ultrasound.")
- `--image_shape`: Shape of the image data as a comma-separated string, e.g., \'224,224,3\' (default: "224,224,3").
- `--include_depth`, `--include_seg`, `--include_video`: Flags to include respective data types. Note that `GR00TN1FeatureDict` defaults `include_video` to `True`.
- `state_shape` and `actions_shape`: These are no longer direct CLI arguments. They default to `(7,)` and `(6,)` respectively within the `GR00TN1FeatureDict` class but can be modified in the script if needed.

The script will:
1. Create a LeRobot dataset with the specified name
2. Process each HDF5 file and extract observations, states, and actions using the GR00T-N1 specific feature mapping.
3. Save the data in LeRobot format.
4. Copy `modality.json` to `~/.cache/huggingface/lerobot/<repo_id>/meta/modality.json`.
5. Consolidate the dataset for training.

The converted dataset will be saved in `~/.cache/huggingface/lerobot/<repo_id>`.

## 🚀 Running Training

To start training with a GR00T-N1 configuration:
Please move to the current [`gr00t_n1` folder](./) and execute:
```bash
python train.py --data_config single_panda_us --dataset_path `data_path`
```
**Arguments:**
- `--data_config`: Data configuration name to use for traing(e.g., `single_panda_us`)
- `--dataset_path`: Path to the training data
- `--output_dir`: Path to save the checkpoint

## 📊 Understanding Outputs

### Checkpoints

Training checkpoints are typically saved in `output_dir` directory with a structure like:

```bash
output_dir/
├── config.json
├── experiment_cfg/
├── checkpoint-500/
├── checkpoint-1000/
└── ...
```
Each numbered directory contains a checkpoint saved at that training step.

## References

- **NVIDIA Isaac GR00T N1**: For more information on the GR00T N1 foundation model, refer to [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T).
- **LeRobot Data Format**: The data conversion process utilizes the LeRobot format. For details on LeRobot, see the [LeRobot GitHub repository](https://github.com/huggingface/lerobot).
