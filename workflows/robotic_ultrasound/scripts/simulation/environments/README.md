# Environments

This folder contains the core simulation infrastructure for robotic ultrasound environments, primarily built around `sim_with_dds.py` - the main simulation runner that orchestrates Isaac Lab environments with DDS (Data Distribution Service) communication.

**Key Components:**

- **`sim_with_dds.py`** - Main simulation orchestrator that runs single-arm manipulator simulations with real-time DDS communication, publishes multi-modal sensor data (room/wrist cameras, robot states, probe positions), subscribes to control commands for closed-loop operation, and supports data replay from recorded HDF5 trajectories
- **`state_machine/`** - State machine implementations for structured robotic ultrasound workflows
- **`cosmos_transfer1/`** - Transfer learning pipeline and guided generation components
- **`teleoperation/`** - Teleoperation agents and control interfaces

## Sim with DDS

This script runs the simulation with DDS communication.

```bash
python -m simulation.environments.sim_with_dds --enable_cameras
```

What's expected:
- IsaacSim window with a Franka robot arm and a ultrasound probe
- The robotic arm won't move until you run the [policy runner scripts](../../policy_runner/README.md) to send control commands to the robot arm or replaying the recorded data

### Evaluating with Recorded Initial States and Saving Trajectories

The `sim_with_dds.py` script can also be used for more controlled evaluations by resetting the environment to initial states from recorded HDF5 data. When doing so, it can save the resulting end-effector trajectories.

- **`--hdf5_path /path/to/your/data.hdf5`**: Provide the path to an HDF5 file (or a directory containing HDF5 files for multiple episodes). The simulation will reset the environment to the initial state(s) found in this data for each episode.
- **`--npz_prefix your_prefix_`**: When `--hdf5_path` is used, this argument specifies a prefix for the names of the `.npz` files where the simulated end-effector trajectories will be saved. Each saved file will be named like `your_prefix_robot_obs_{episode_idx}.npz` and stored in the same directory as the input HDF5 file (if `--hdf5_path` is a file) or within the `--hdf5_path` directory (if it's a directory).

**Example:**

```sh
python -m simulation.environments.sim_with_dds \
    --enable_cameras \
    --hdf5_path /mnt/hdd/cosmos/heldout-test50/data_0.hdf5 \
    --npz_prefix pi0-800_
```

This command will load the initial state from `data_0.hdf5`, run the simulation (presumably interacting with a policy via DDS), and save the resulting trajectory to a file like `pi0-800_robot_obs_0.npz` in the `/mnt/hdd/cosmos/heldout-test50/` directory.

What's expected:
- IsaacSim window with a Franka robot arm and a ultrasound probe, replaying the recorded data

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--disable_fabric` | flag | False | Disable fabric and use USD I/O operations |
| `--num_envs` | int | 1 | Number of environments to spawn |
| `--task` | str | "Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0" | Name of the task |
| `--rti_license_file` | str | $RTI_LICENSE_FILE | Path to the RTI license file |
| `--infer_domain_id` | int | 0 | Domain ID to publish data for inference |
| `--viz_domain_id` | int | 1 | Domain ID to publish data for visualization |
| `--topic_in_room_camera` | str | "topic_room_camera_data_rgb" | Topic name to consume room camera RGB |
| `--topic_in_room_camera_depth` | str | "topic_room_camera_data_depth" | Topic name to consume room camera depth |
| `--topic_in_wrist_camera` | str | "topic_wrist_camera_data_rgb" | Topic name to consume wrist camera RGB |
| `--topic_in_wrist_camera_depth` | str | "topic_wrist_camera_data_depth" | Topic name to consume wrist camera depth |
| `--topic_in_franka_pos` | str | "topic_franka_info" | Topic name to consume Franka position |
| `--topic_in_probe_pos` | str | "topic_ultrasound_info" | Topic name to consume probe position |
| `--topic_out` | str | "topic_franka_ctrl" | Topic name to publish generated Franka actions |
| `--log_probe_pos` | flag | False | Log probe position |
| `--scale` | float | 1000.0 | Scale factor to convert from omniverse to organ coordinate system |
| `--hdf5_path` | str | None | Path to single .hdf5 file or directory containing recorded data for environment reset |
| `--npz_prefix` | str | "" | Prefix to save the end-effector trajectory data during evaluation, only used when hdf5_path is provided |

*Note: Additional arguments are available through the IsaacLab AppLauncher framework.*

## Jump to Section

- [State Machine](./state_machine/README.md)
- [Data Collection in State Machine](./state_machine/README.md#data-collection)
- [Cosmos Transfer1](./cosmos_transfer1/README.md)
- [Teleoperation](./teleoperation/README.md)
- [Policy Runner](../../policy_runner/README.md)
- [DDS Communication Common Issues](../../dds/README.md)
