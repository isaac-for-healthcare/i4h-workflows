# Policy runner for both the simulation and physical world

## Run PI0 policy with DDS communication

### Prepare Model Weights and USD Assets and Install Dependencies

Please refer to the [Environment Setup](../../README.md#environment-setup) instructions.

### Ensure the PYTHONPATH Is Set

Please refer to the [Environment Setup - Set environment variables before running the scripts](../../README.md#set-environment-variables-before-running-the-scripts) instructions.

### Run Policy

Please move to the current [`policy_runner` folder](./) and execute:

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

### Inference

| Hardware | Average Latency  | Memory Usage | Actions Predicted
|----------|----------------|-------------------|--------------|
| NVIDIA RTX 4090 | 100 ms | 9 GB | 50 |

> **Note:** The model predicts the 50 next actions in a single 100ms inference, allowing you to choose how many of these predictions to utilize based on your specific control frequency requirements.


## Run GROOT N1 policy with DDS communication

### Prepare Model Weights and Dependencies

For now, please refer to the official [NVIDIA Isaac GR00T Installation Guide](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#installation-guide) for detailed instructions on setting up the necessary dependencies and acquiring model weights. This section will be updated with more specific instructions later.

### Ensure the PYTHONPATH Is Set

Make sure your environment variables are properly set as described in the [Environment Setup - Set environment variables before running the scripts](../../README.md#set-environment-variables-before-running-the-scripts) section.

### Run Policy

From the [`policy_runner` folder](./), execute:
```sh
python run_n1_policy.py
```


### Command Line Arguments

| Argument                  | Type   | Default                        | Description                                                                 |
|---------------------------|--------|--------------------------------|-----------------------------------------------------------------------------|
| `--ckpt_path`             | str    | `None`                         | Path to the GROOT N1 policy model checkpoint file. If not provided, uses a default path based on assets. |
| `--rti_license_file`      | str    | Environment `RTI_LICENSE_FILE` | Path to the RTI Connext DDS license file.                                   |
| `--domain_id`             | int    | `0`                            | DDS domain ID for communication.                                            |
| `--height`                | int    | `224`                          | Expected height of the input camera images.                                 |
| `--width`                 | int    | `224`                          | Expected width of the input camera images.                                  |
| `--topic_in_room_camera`  | str    | `"topic_room_camera_data_rgb"` | DDS topic name for subscribing to room camera RGB images.                   |
| `--topic_in_wrist_camera` | str    | `"topic_wrist_camera_data_rgb"`| DDS topic name for subscribing to wrist camera RGB images.                  |
| `--topic_in_franka_pos`   | str    | `"topic_franka_info"`          | DDS topic name for subscribing to Franka robot joint states (positions).      |
| `--topic_out`             | str    | `"topic_franka_ctrl"`          | DDS topic name for publishing predicted Franka robot actions.               |
| `--verbose`               | bool   | `False`                        | If set, enables verbose logging output.                                     |
| `--data_config`           | str    | `"single_panda_us"`            | Name of the data configuration to use (e.g., for modality and transforms). Choices are from `DATA_CONFIG_MAP`. |
| `--embodiment_tag`        | str    | `"new_embodiment"`             | The embodiment tag for the GROOT model.                                     |
