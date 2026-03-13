# Loco-Manipulation Fine-Tuning Recipe

This guide covers GR00T N1.6 fine-tuning for the Rheo loco-manipulation tasks: **Surgical Tray Pick-and-Place** and **Case Cart Pushing**. Both tasks use the Unitree G1 embodiment with whole-body control.

## Prerequisites

1. A LeRobot-format dataset produced by the data generation pipeline.
2. A single NVIDIA H100 GPU (or equivalent with ≥80 GB HBM).
3. GR00T N1.6 Environment installed (check out the [Gr00t GitHub repository](https://github.com/NVIDIA/Isaac-GR00T/commit/e8e625f4f21898c506a1d8f7d20a289c97a52acf))

## LeRobot Format Dataset Preparation

Before fine-tuning, convert your generated HDF5 dataset to LeRobot format following the instruction on [the main README](../README.md).

```bash
./workflows/rheo/docker/run_docker.sh -g1.6 \
  python scripts/utils/convert_hdf5_to_lerobot.py \
  --config scripts/config/g1_locomanip_dataset_config.yaml
```

## Modality Configuration

Both tasks share the same modality config at `workflows/rheo/scripts/policy/gr00t_locomanip_config.py`, which registers a `NEW_EMBODIMENT` modality with the following keys:

| Modality | Keys |
| -------- | ---- |
| Video | `ego_view` |
| State | `left_arm`, `right_arm`, `left_hand`, `right_hand`, `waist` |
| Action | `left_arm`, `right_arm`, `left_hand`, `right_hand`, `base_height_command`, `navigate_command` |

The action horizon is 16 steps, and all actions use absolute representation.

## Dataset Preparation

Before fine-tuning, convert your generated HDF5 dataset to LeRobot format following the instruction on [the main README](../README.md).

Update the `language_instruction` field in [g1_locomanip_dataset_config](../scripts/config/g1_locomanip_dataset_config.yaml) to match your task before converting.

## Fine-Tuning

### Surgical Tray Pick-and-Place

Trained on a single H100 for **60,000 steps** with batch size 32:

Update the `--modality-config-path` argument to the path to [gr00t_locomanip_config.py](../scripts/policy/gr00t_locomanip_config.py).

```bash
python gr00t/gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path <path to dataset> \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path <path to gr00t_locomanip_config.py> \
    --num_gpus 1 \
    --output_dir /models/GROOT-N1.6-Sim-Pick-Place-Custom \
    --save_steps 10000 \
    --max_steps 60000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 32 \
    --tune_visual \
    --tune_projector \
    --tune_diffusion_model \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 8
```

### Case Cart Pushing

Trained on a single H100 for **20,000 steps** with batch size 32:

```bash
python gr00t/gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path <path to dataset> \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path <path-to gr00t_locomanip_config.py> \
    --num_gpus 1 \
    --output_dir /models/GROOT-N1.6-Sim-Push-Cart-Custom \
    --save_steps 5000 \
    --max_steps 20000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 32 \
    --tune_visual \
    --tune_projector \
    --tune_diffusion_model \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 8
```
