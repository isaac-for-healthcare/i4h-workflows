# Imitation Learning

This folder contains the scripts to run the imitation learning pipeline in the IsaacSim simulation environment.

## PI0 Policy Evaluation

This script evaluates the PI0 policy in the IsaacSim simulation environment.

```bash
python -m simulation.imitation_learning.pi0_policy.eval --enable_cameras
```

**Expected Behavior:**
- IsaacSim window with a Franka robot arm driven by pi0 policy without DDS communication.

> **Note:**
> You may see "IsaacSim 4.5.0 is not responding". It can take approximately several minutes to download the assets and models from the internet and load them to the scene. If this is the first time you run the workflow, it can take up to 10 minutes.
> It may take an additional 1 or 2 minutes for the policy to start inferencing, so the robot arm may not move immediately.

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--disable_fabric` | flag | False | Disable fabric and use USD I/O operations |
| `--num_envs` | int | 1 | Number of environments to spawn |
| `--task` | str | "Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0" | Name of the task |
| `--ckpt_path` | str | "nvidia/Liver_Scan_Pi0_Cosmos_Rel" | Checkpoint path or HF repo id for the policy model |
| `--repo_id` | str | "i4h/sim_liver_scan" | The LeRobot repo id for the dataset norm |

*Note: Additional arguments are available through the IsaacLab AppLauncher framework.*

---

## Documentation Links

- [IsaacLab Task Setting](../exts/robotic_us_ext/README.md)
