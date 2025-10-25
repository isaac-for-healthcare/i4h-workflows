# Liver Scan State Machine

This folder contains the scripts to perform ultrasound scanning based on state machine logic in the IsaacSim simulation environment.

The Liver Scan State Machine provides a structured approach to performing ultrasound scans on a simulated liver. It implements a state-based workflow that guides the robotic arm through the scanning procedure.

We also have a script to replay the recorded data in the IsaacSim simulation environment.

## Running the State Machine

This script runs the liver scan state machine in the IsaacSim simulation environment.

```bash
python -m simulation.environments.state_machine.liver_scan_sm --enable_cameras
```

**Expected Behavior:**
- IsaacSim window with a Franka robot arm and a ultrasound probe performing a liver scan.
- You may see "IsaacSim is not responding". It can take approximately several minutes to download the assets and models from the internet and load them to the scene. If this is the first time you run the workflow, it can take up to 10 minutes.

### States Overview

The state machine transitions through the following states:
- **SETUP**: Initial positioning of the robot
- **APPROACH**: Moving toward the organ
- **CONTACT**: Making contact with the organ surface
- **SCANNING**: Performing the ultrasound scan
- **DONE**: Completing the scan procedure

The state machine integrates multiple control modules:
- **Force Control**: Manages contact forces during scanning
- **Orientation Control**: Maintains proper probe orientation
- **Path Planning**: Guides the robot through the scanning trajectory

> **Note:**
> This implementation works **only with a single environment** (`--num_envs 1`).
> It should be used with the `Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0` environment.

## Data Collection

To run the state machine and collect data for a specified number of episodes, you need to pass the `--num_episodes` argument. Default is 0, which means no data collection.

```sh
python -m simulation.environments.state_machine.liver_scan_sm \
    --enable_camera \
    --num_episodes 2
```

This will collect data for 2 complete episodes and store it in HDF5 format.

### Data Collection Details

When data collection is enabled (`--num_episodes > 0`), the state machine will:

1. Create a timestamped directory in `./data/hdf5/` to store the collected data
2. Record observations, actions, and state information at each step
3. Capture RGB and depth images from the specified cameras
4. Store all data in HDF5 format compatible with robomimic

The collected data includes:
- Robot observations (position, orientation)
- Torso observations (organ position, orientation)
- Relative and absolute actions
- State machine state
- Joint positions
- Camera images (RGB, depth, segmentation if `--include_seg` is enabled)


### Keyboard Controls

During execution, you can press the 'r' key to reset the environment and state machine.

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | None | Name of the task (environment) to use |
| `--num_episodes` | int | 0 | Number of episodes to collect data for (0 = no data collection) |
| `--camera_names` | list[str] | ["room_camera", "wrist_camera"] | List of camera names to capture images from |
| `--disable_fabric` | flag | False | Disable fabric and use USD I/O operations |
| `--num_envs` | int | 1 | Number of environments to spawn (must be 1 for this script) |
| `--reset_steps` | int | 40 | Number of steps to take during environment reset |
| `--max_steps` | int | 350 | Maximum number of steps before forcing a reset |
| `--include_seg` | bool | True | Whether to include semantic segmentation in the data collection |

> **Note:** It is recommended to use at least 40 steps for `--reset_steps` to allow enough steps for the robot to properly reset to the SETUP position.


## Replay Recorded Trajectories

The `replay_recording.py` script allows you to visualize previously recorded HDF5 trajectories in the Isaac Sim environment. It loads recorded actions, organ positions, and robot joint states from HDF5 files for each episode and steps through them in the simulation.

#### Usage

```sh
python -m simulation.environments.state_machine.replay_recording \
    /path/to/your/hdf5_data_directory \
    --task <YourTaskName> \
    --enable_cameras
```

Replace `/path/to/your/hdf5_data_directory` with the actual path to the directory containing your `data_*.hdf5` files or single HDF5 file, and `<YourTaskName>` with the task name used during data collection (e.g., `Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0`).

#### Command Line Arguments

| Argument           | Type | Default                                  | Description                                                                      |
|--------------------|------|------------------------------------------|----------------------------------------------------------------------------------|
| `--hdf5_path`      | str  | (Required)                               | Provide the path to an HDF5 file (or a directory containing HDF5 files for multiple episodes).                            |
| `--task`           | str  | `Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0` | Name of the task (environment) to use. Should match the task used for recording. |
| `--num_envs`       | int  | `1`                                      | Number of environments to spawn (should typically be 1 for replay).              |
| `--disable_fabric` | flag | `False`                                  | Disable fabric and use USD I/O operations.                                       |

> **Note:** Additional common Isaac Lab arguments (like `--device`) can also be used.

---

## Documentation Links

- [IsaacLab Task Setting](../../exts/robotic_us_ext/README.md)
