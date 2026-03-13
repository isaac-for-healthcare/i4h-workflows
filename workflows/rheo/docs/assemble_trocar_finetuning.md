# Assemble Trocar Fine-Tuning Recipe

This guide covers GR00T N1.5 fine-tuning for the Rheo assemble trocar task.

## Prerequisites

1. A LeRobot-format dataset produced by the data generation pipeline (see the main [README](../README.md) sections on Data Collection, Synthetic Data Generation, and Convert to LeRobot format).
2. A single NVIDIA H100 GPU (or equivalent with ≥80 GB HBM).
3. GR00T N1.5 Environment installed (check out the [Gr00t GitHub repository](https://github.com/NVIDIA/Isaac-GR00T/commit/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96))

## Modality Configuration

The modality config is defined in `workflows/rheo/scripts/policy/gr00t_config.py`, which defines `UnitreeG1SimDataConfig` with:

| Modality | Keys |
| -------- | ---- |
| Video | `left_wrist_view`, `right_wrist_view`, `room_view` |
| State | `left_arm`, `right_arm`, `left_hand`, `right_hand` |
| Action | `left_arm`, `right_arm`, `left_hand`, `right_hand` |

## Dataset Preparation

Please follow the [README](../README.md) sections on Data Collection. It's highly recommended to use VR/XR controllers to collect data for this task. Convert the collected data to LeRobot format using the `g1_assemble_trocar_dataset.yaml` config:

```bash
./workflows/rheo/docker/run_docker.sh -g1.5 \
  python scripts/utils/convert_hdf5_to_lerobot.py \
  --config scripts/config/g1_assemble_trocar_dataset.yaml
```

## Run Fine-Tuning

```bash
# Navigate to the GR00T repository
# If you are using the rheo workflow container with `-g1.5` flag, the path is `third_party/Isaac-GR00T`
python scripts/gr00t_finetune.py \
  --dataset-path /path/to/assemble_trocar_dataset \
  --num-gpus 1 \
  --batch-size 32 \
  --output-dir /path/to/output/model \
  --data-config policy.gr00t_config:UnitreeG1SimDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 30000 \
  --save-steps 5000 \
  --tune_visual
```

**Key parameters:**

| Parameter | Description |
| ----------- | ------------- |
| `--data-config policy.gr00t_config:UnitreeG1SimDataConfig` | Assemble trocar task config via `module:class` syntax |
| `--video_backend decord` | Video loading backend |
| `--max_steps` | Total training steps (30k recommended) |
| `--save-steps` | Checkpoint save interval |
| `--num-gpus` | Number of GPUs for distributed training |
