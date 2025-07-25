# Policy runner for both the simulation and physical world

This script allows running different policy models (currently PI0 and GR00T N1) using DDS for communication, suitable for both simulation and physical robot control.

## Supported Policies

*   **PI0**: Based on the [openpi](https://github.com/Physical-Intelligence/openpi) library.
*   **GR00T N1**: NVIDIA's foundation model for humanoid robots. Refer to [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) for more information.

## Run PI0 Policy with DDS communication

### Prepare Model Weights and USD Assets and Install Dependencies

Please refer to the [Environment Setup](../../README.md#environment-setup) instructions specific to the **PI0** policy.

### Ensure the PYTHONPATH Is Set

Please refer to the [Environment Setup - Set environment variables before running the scripts](../../README.md#set-environment-variables-before-running-the-scripts) instructions.

### Run Policy

Please move to the current [`policy_runner` folder](./) and execute:

```sh
# Example for PI0
python run_policy.py --policy pi0 [other arguments...]
```

## Run GR00T N1 Policy with DDS communication

### Prepare Model Weights and Dependencies

Please refer to the [Environment Setup](../../README.md#environment-setup) instructions, ensuring you use the `--policy gr00tn1` option when running `tools/env_setup_robot_us.sh` to install GR00T N1 dependencies.
For acquiring model weights and further details, consult the official [NVIDIA Isaac GR00T Installation Guide](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#installation-guide).
Ensure the environment where you run `run_policy.py` has the GR00 TN1 dependencies installed.

### Ensure the PYTHONPATH Is Set

Please refer to the [Environment Setup - Set environment variables before running the scripts](../../README.md#set-environment-variables-before-running-the-scripts) instructions.

### Run Policy

Please move to the current [`policy_runner` folder](./) and execute:

```sh
# Example for GR00T N1
python run_policy.py --policy gr00tn1 [other arguments...]
```

## Command Line Arguments

Here's a markdown table describing the command-line arguments:

| Argument                  | Description                                                              | Default Value                      | Policy    |
|---------------------------|--------------------------------------------------------------------------|------------------------------------|-----------|
| `--policy`                | Policy type to use.                                                      | `pi0`                              | Both      |
| `--ckpt_path`             | Checkpoint path for the policy model.                                    | Uses downloaded assets             | Both      |
| `--task_description`      | Task description text prompt for the policy.                             | `Perform a liver ultrasound.`      | Both      |
| `--chunk_length`          | Length of the action chunk inferred by the policy per inference step.    | 50                                 | Both      |
| `--repo_id`               | LeRobot repo ID for dataset normalization (used for PI0).                | `i4h/sim_liver_scan`               | PI0       |
| `--data_config`           | Data config name (used for GR00T N1).                                     | `single_panda_us`                  | GR00T N1   |
| `--embodiment_tag`        | The embodiment tag for the model (used for GR00T N1).                     | `new_embodiment`                   | GR00T N1   |
| `--rti_license_file`      | Path to the RTI license file.                                            | Uses env `RTI_LICENSE_FILE`      | Both (DDS)|
| `--domain_id`             | Domain ID for DDS communication.                                         | 0                                  | Both (DDS)|
| `--height`                | Input image height for cameras.                                          | 224                                | Both (DDS)|
| `--width`                 | Input image width for cameras.                                           | 224                                | Both (DDS)|
| `--topic_in_room_camera`  | Topic name to consume room camera RGB data.                              | `topic_room_camera_data_rgb`       | Both (DDS)|
| `--topic_in_wrist_camera` | Topic name to consume wrist camera RGB data.                             | `topic_wrist_camera_data_rgb`      | Both (DDS)|
| `--topic_in_franka_pos`   | Topic name to consume Franka position data.                              | `topic_franka_info`                | Both (DDS)|
| `--topic_out`             | Topic name to publish generated Franka actions.                          | `topic_franka_ctrl`                | Both (DDS)|
| `--verbose`               | Whether to print DDS communication logs.                                 | False                              | Both (DDS)|

## Performance Metrics

### Inference (Pi0)

| Hardware        | Average Latency | Memory Usage | Actions Predicted |
|-----------------|-----------------|--------------|-------------------|
| NVIDIA RTX 4090 | 100 ms          | 9 GB         | 50                |

### Inference (GR00T N1)

| Hardware            | Average Latency | Memory Usage | Actions Predicted |
|---------------------|-----------------|--------------|-------------------|
| NVIDIA RTX 6000 Ada | 92 ms           | 10 GB        | 16                |

> **Note:** Both models predict multiple actions in a single inference step:
> - PI0 model: Predicts 50 actions in ~100ms
> - GR00T N1 model: Predicts 16 actions in ~92ms
>
>  You can choose how many of these predictions to utilize based on your specific control frequency requirements.
