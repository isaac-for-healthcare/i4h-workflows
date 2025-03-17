# Policy runner for both the simulation and physical world

## Run PI0 policy with DDS communication

### Prepare Model Weights

[FIXME: change to `asset-catalog`] Download the v0.3 model weights from [GDrive](https://drive.google.com/drive/folders/1sL4GAETSMbxxcefsTsOkX7wXkTsbDqhW?usp=sharing)

### Prepare USD Assets

Please check the [I4H asset catalog](https://github.com/isaac-for-healthcare/i4h-asset-catalog) for assets downloading, put the USD assets as "./assets".

### Install Dependencies

Follow the [Environment Setup](../../README.md#environment-setup) instructions to setup the environment and dependencies.


### Setup Python Path

Please move to the [scripts](../) folder and specify python path:
```sh
export PYTHONPATH=`pwd`
```

### Run Policy

Please move back to the current folder and run the following command:

```sh
python run_policy.py
```

### Command Line Arguments

Here's a markdown table describing the command-line arguments:

| Argument | Description | Default Value |
|----------|-------------|---------------|
| `--ckpt_path` | Checkpoint path for the policy model | Uses the policy model in the downloaded assets |
| `--repo_id` | LeRobot repo ID for the dataset normalization | `i4h/sim_liver_scan` (included in downloaded assets) |
| `--rti_license_file` | Path to the RTI license file | Uses environment variable `RTI_LICENSE_FILE` |
| `--domain_id` | Domain ID for DDS communication | 0 |
| `--height` | Input image height | 224 |
| `--width` | Input image width | 224 |
| `--topic_in_room_camera` | Topic name to consume room camera RGB data | `topic_room_camera_data_rgb` |
| `--topic_in_wrist_camera` | Topic name to consume wrist camera RGB data | `topic_wrist_camera_data_rgb` |
| `--topic_in_franka_pos` | Topic name to consume Franka position data | `topic_franka_info` |
| `--topic_out` | Topic name to publish generated Franka actions | `topic_franka_ctrl` |
| `--verbose` | Whether to print logs | False |

## Performance Metrics

### Inference Latency

| Hardware | Average Latency | Actions Predicted |
|----------|----------------|-------------------|
| NVIDIA RTX 4090 | 100 ms | 50 |

> **Note:** The model predicts the 50 next actions in a single 100ms inference, allowing you to choose how many of these predictions to utilize based on your specific control frequency requirements.
