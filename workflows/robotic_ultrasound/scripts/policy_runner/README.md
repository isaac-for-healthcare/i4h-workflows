# Policy runner for both the simulation and physical world

This script allows running different policy models (currently PI0 and GR00T N1) using DDS for communication, suitable for both simulation and physical robot control.

## Supported Policies

*   **PI0**: Based on the [openpi](https://github.com/Physical-Intelligence/openpi) library.
*   **GR00T N1**: NVIDIA's foundation model for humanoid robots. Refer to [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) for more information.

## Run Policy with DDS communication

### Dependencies

You may need to install the dependencies for either pi0 (default installed) or gr00tn1.

```bash
# Install the dependencies for PI0, default installed
bash tools/env_setup_robot_us.sh --policy pi0
# Install the dependencies for GR00T N1
bash tools/env_setup_robot_us.sh --policy gr00tn1
```
The environment for pi0 and gr00tn1 has conflicts with each other. You can only install one of them at a time.

### Prepare Model Weights

The model weights will be downloaded automatically when you [run the policy runner](#run-policy).

Optionally, you can also download the weights manually by running the following command:
```bash
# Download the model weights for PI0
i4h-asset-retrieve --sub-path Policies/LiverScan/Pi0

# Download the model weights for GR00T N1
i4h-asset-retrieve --sub-path Policies/LiverScan/GR00TN1
```

### Run Policy

```sh
# Example for PI0
python -m policy_runner.run_policy --policy pi0
# Example for GR00T N1
python -m policy_runner.run_policy --policy gr00tn1
```

**Expected Behavior:**
- Terminal messages will confirm that the policy has loaded and is running.
- The policy will predict actions and publish them to DDS topics when image feeds are available.
- When no image feeds are available on DDS, the model will not predict any actions.
- You can run the [Simulation with Data Distribution Service (DDS)](../simulation/environments/README.md) to produce the data for the policy to consume in IsaacSim.

### Command Line Arguments

| Argument | Type | Default | Description | Policy Support |
|----------|------|---------|-------------|----------------|
| `--policy` | str | "pi0" | Policy type to use (choices: pi0, gr00tn1) | Both |
| `--ckpt_path` | str | "nvidia/Liver_Scan_Pi0_Cosmos_Rel" | Checkpoint path or HF repo id for the policy model | Both |
| `--task_description` | str | "Perform a liver ultrasound." | Task description text prompt for the policy | Both |
| `--chunk_length` | int | 50 | Length of the action chunk inferred by the policy | Both |
| `--repo_id` | str | "i4h/sim_liver_scan" | LeRobot repo ID for dataset normalization | PI0 only |
| `--data_config` | str | "single_panda_us" | Data config name for GR00T N1 policy | GR00T N1 only |
| `--embodiment_tag` | str | "new_embodiment" | The embodiment tag for the GR00T N1 model | GR00T N1 only |
| `--rti_license_file` | str | $RTI_LICENSE_FILE | Path to the RTI license file | Both (DDS) |
| `--domain_id` | int | 0 | Domain ID for DDS communication | Both (DDS) |
| `--height` | int | 224 | Input image height for cameras | Both (DDS) |
| `--width` | int | 224 | Input image width for cameras | Both (DDS) |
| `--topic_in_room_camera` | str | "topic_room_camera_data_rgb" | Topic name to consume room camera RGB data | Both (DDS) |
| `--topic_in_wrist_camera` | str | "topic_wrist_camera_data_rgb" | Topic name to consume wrist camera RGB data | Both (DDS) |
| `--topic_in_franka_pos` | str | "topic_franka_info" | Topic name to consume Franka position data | Both (DDS) |
| `--topic_out` | str | "topic_franka_ctrl" | Topic name to publish generated Franka actions | Both (DDS) |
| `--verbose` | bool | False | Whether to print DDS communication logs | Both (DDS) |

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

---

## Documentation Links

- [Simulation with Data Distribution Service (DDS)](../simulation/environments/README.md)
- [IsaacLab Task Setting](../simulation/exts/robotic_us_ext/README.md)
