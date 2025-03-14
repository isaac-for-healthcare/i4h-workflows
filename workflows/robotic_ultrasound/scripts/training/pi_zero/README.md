# 🤖 Robotic Ultrasound Training with PI Zero

This repository provides a complete workflow for training [PI Zero](https://www.physicalintelligence.company/blog/pi0) models for robotic ultrasound applications. PI Zero is a powerful vision-language-action model developed by [Physical Intelligence](https://www.physicalintelligence.company/) that can be fine-tuned for various robotic tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Data Conversion](#data-conversion)
- [Training Configuration](#training-configuration)
- [Running Training](#running-training)
- [Understanding Outputs](#understanding-outputs)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

This workflow enables you to:

1. Collect robotic ultrasound data using a state machine in Isaac Sim
2. Convert the collected HDF5 data to LeRobot format (required by PI Zero)
3. Fine-tune a PI Zero model using either full supervised fine-tuning or LoRA (Low-Rank Adaptation)
4. Deploy the trained model for inference

## 🛠️ Installation

First, install the OpenPI repository and its dependencies using our [provided script](../../../../../tools/env_setup_robot_us.sh):
```bash
# Install OpenPI with dependencies
./tools/env_setup_robot_us.sh
```

This script:
- Clones the OpenPI repository at a specific commit
- Updates Python version requirements in pyproject.toml
- Installs LeRobot and other dependencies
- Sets up the OpenPI client and core packages

## 📊 Data Collection

To train a PI Zero model, you'll need to collect robotic ultrasound data. We provide a state machine implementation in the simulation environment that can generate training episodes that emulate a liver ultrasound scan.

See the [simulation README](../../simulation/README.md#liver-scan-state-machine) for more information on how to collect data.

## 🔄 Data Conversion

PI Zero uses the LeRobot data format for training. We provide a script to convert your HDF5 data to this format:

```bash
python convert_hdf5_to_lerobot.py /path/to/your/hdf5/data
```

**Arguments:**
- `--data_dir`: Path to the directory containing HDF5 files
- `--repo_id`: Name for your dataset (default: "i4h/robotic_ultrasound")
- `--task_prompt`: Text description of the task (default: "Perform a liver ultrasound.")
- `--image_shape`: Shape of the image data as a comma-separated string, e.g., '224,224,3' (default: '224,224,3')

The script will:
1. Create a LeRobot dataset with the specified name
2. Process each HDF5 file and extract observations, states, and actions
3. Save the data in LeRobot format
4. Consolidate the dataset for training

The converted dataset will be saved in `~/.cache/huggingface/lerobot/your_dataset_name`.

## ⚙️ Training Configuration

### Full Fine-tuning vs. LoRA

- **Full Fine-tuning** (`robotic_ultrasound`): Updates all model parameters. Requires more GPU memory (>70GB) but can achieve better performance on larger datasets.
- **LoRA Fine-tuning** (`robotic_ultrasound_lora`): Uses Low-Rank Adaptation to update a small subset of parameters. Requires less GPU memory (~22.5GB) while still achieving good results.

## 🚀 Running Training

To start training with the default LoRA configuration:
```bash
python train.py --config robotic_ultrasound_lora --exp_name liver_ultrasound
```
**Arguments:**
- `--config`: Training configuration to use (options: `robotic_ultrasound`, `robotic_ultrasound_lora`)
- `--exp_name`: Name for your experiment (used for wandb logging and checkpoints)
- `--repo_id`: Repository ID for the dataset (default: `i4h/robotic_ultrasound`)

For better GPU memory utilization, you can set:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python tools/openpi/src/openpi/train.py --config robotic_ultrasound_lora --exp_name liver_ultrasound
```
This allows JAX to use up to 90% of your GPU memory (default is 75%).

## 📊 Understanding Outputs

### Normalization Statistics

The training script automatically computes normalization statistics for your dataset on the first run. These statistics are stored in the `assets` directory and are used to normalize inputs during training and inference.

### Checkpoints

Training checkpoints are saved in the `checkpoints` directory with the following structure:

```bash
checkpoints/
└── [config_name]/
└── [exp_name]/
├── 1000/
├── 2000/
└── ...
```

Each numbered directory contains a checkpoint saved at that training step.

### Logging

Training progress is logged to:
1. The console (real-time updates)
2. [Weights & Biases](https://wandb.ai/) (if configured)

To view detailed training metrics, ensure you log into W&B:
```bash
wandb login
```

## 🚀 Testing Inference
See the [policy_runner README](../../policy_runner/README.md) for more information on how to test inference with the trained model.

## 🔧 Troubleshooting

### Common Issues

- **Out of memory errors**: Try using the LoRA configuration (`robotic_ultrasound_lora`) instead of full fine-tuning, or reduce the batch size in the config.
- **Data format errors**: Ensure your HDF5 data follows the expected format. Check the playback script to validate.
- **Missing normalization statistics**: The first training run should generate these automatically. If missing, check permissions in the assets directory.

## 🙏 Acknowledgements

This project builds upon the [PI Zero model](https://www.physicalintelligence.company/blog/pi0) developed by [Physical Intelligence](https://www.physicalintelligence.company/). We thank them for making their models and training infrastructure available to the community.

---

Happy training! If you encounter any issues or have questions, please open an issue in this repository.
