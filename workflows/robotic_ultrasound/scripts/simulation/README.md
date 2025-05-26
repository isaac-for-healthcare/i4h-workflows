# IsaacLab simulation for robotic ultrasound

## Table of Contents
- [Installation](#installation)
- [Environments](#environments)
- [Apps](#apps)
  - [PI Zero Policy Evaluation](#pi-zero-policy-evaluation)
  - [Policy Evaluation w/ DDS](#policy-evaluation-w-dds)
  - [Liver Scan State Machine](#liver-scan-state-machine)
  - [Teleoperation](#teleoperation)
  - [Ultrasound Raytracing Simulation](#ultrasound-raytracing-simulation)
  - [Trajectory Evaluation](#trajectory-evaluation)

## Installation

Follow the [Environment Setup](../../README.md#environment-setup) instructions to setup the environment and dependencies.

## Environments

Isaac-Lab scripts usually receive a `--task <task_name>` argument. This argument is used to select the environment/task to be run.
These tasks are detected by the `gym.register` function in the [exts/robotic_us_ext/\__init__.py](exts/robotic_us_ext/robotic_us_ext/tasks/ultrasound/approach/config/franka/__init__.py).

These registered tasks are then further defined in the [environment configuration file](exts/robotic_us_ext/robotic_us_ext/tasks/ultrasound/approach/config/franka/franka_manager_rl_env_cfg.py).

Available tasks:
- `Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0`: FrankaUsRs with relative actions
- `Isaac-Reach-Torso-FrankaUsRs-IK-RL-Abs-v0` : FrankaUsRs with absolute actions

Currently there are these robot configurations that can be used in various tasks


| Robot name                                 | task             | applications          |
|----------                                  |---------         |----------             |
| FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG      | \*-FrankaUsRs-*  | Reach, Teleop         |

## Apps

### PI Zero Policy Evaluation
Set up `openpi` referring to [PI0 runner](../policy_runner/README.md).

#### Ensure the PYTHONPATH Is Set

Please refer to the [Environment Setup - Set environment variables before running the scripts](../../README.md#set-environment-variables-before-running-the-scripts) instructions.

#### Run the script

Please move to the current [`simulation` folder](./) and execute:

```sh
python imitation_learning/pi0_policy/eval.py --enable_camera
```

NOTE: You can also specify `--ckpt_path` to run a specific policy.
This should open a stage with Franka arm and run the robotic ultrasound actions:
![pi0 simulation](../../../../docs/source/pi0_sim.jpg)

### Policy Evaluation w/ DDS
This example should work together with the `pi0 policy runner` via DDS communication,
so please ensure to launch the `run_policy.py` with `height=224`, `width=224`,
and the same `domain id` as this example in another terminal.

When `run_policy.py` is launched and idle waiting for the data,

#### Ensure the PYTHONPATH Is Set

Please refer to the [Environment Setup - Set environment variables before running the scripts](../../README.md#set-environment-variables-before-running-the-scripts) instructions.

#### Run the script

Please move to the current [`simulation` folder](./) and execute:

```sh
python environments/sim_with_dds.py --enable_cameras
```

#### Evaluating with Recorded Initial States and Saving Trajectories

The `sim_with_dds.py` script can also be used for more controlled evaluations by resetting the environment to initial states from recorded HDF5 data. When doing so, it can save the resulting end-effector trajectories.

- **`--hdf5_path /path/to/your/data.hdf5`**: Provide the path to an HDF5 file (or a directory containing HDF5 files for multiple episodes). The simulation will reset the environment to the initial state(s) found in this data for each episode.
- **`--npz_prefix your_prefix_`**: When `--hdf5_path` is used, this argument specifies a prefix for the names of the `.npz` files where the simulated end-effector trajectories (robot observations) will be saved. Each saved file will be named like `your_prefix_robot_obs_{episode_idx}.npz` and stored in the same directory as the input HDF5 file (if `--hdf5_path` is a file) or within the `--hdf5_path` directory (if it's a directory).

**Example:**

```sh
python environments/sim_with_dds.py \
    --enable_cameras \
    --hdf5_path /mnt/hdd/cosmos/heldout-test50/data_0.hdf5 \
    --npz_prefix pi0-800_
```

This command will load the initial state from `data_0.hdf5`, run the simulation (presumably interacting with a policy via DDS), and save the resulting trajectory to a file like `pi0-800_robot_obs_0.npz` in the `/mnt/hdd/cosmos/heldout-test50/` directory.

### Liver Scan State Machine

The Liver Scan State Machine provides a structured approach to performing ultrasound scans on a simulated liver. It implements a state-based workflow that guides the robotic arm through the scanning procedure.

#### Overview

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

#### Requirements

- This implementation works **only with a single environment** (`--num_envs 1`).
- It should be used with the `Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0` environment.

#### Usage

Please refer to the [Environment Setup - Set environment variables before running the scripts](../../README.md#set-environment-variables-before-running-the-scripts) to set the `PYTHONPATH`.

Then please move to the current [`simulation` folder](./) and execute:

```sh
python environments/state_machine/liver_scan_sm.py --enable_cameras
```

#### Data Collection

To run the state machine and collect data for a specified number of episodes, you need to pass the `--num_episodes` argument. Default is 0, which means no data collection.

```sh
python environments/state_machine/liver_scan_sm.py \
    --enable_camera \
    --num_episodes 2
```

This will collect data for 2 complete episodes and store it in HDF5 format.

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | None | Name of the task (environment) to use |
| `--num_episodes` | int | 0 | Number of episodes to collect data for (0 = no data collection) |
| `--camera_names` | list[str] | ["room_camera", "wrist_camera"] | List of camera names to capture images from |
| `--disable_fabric` | flag | False | Disable fabric and use USD I/O operations |
| `--num_envs` | int | 1 | Number of environments to spawn (must be 1 for this script) |
| `--reset_steps` | int | 40 | Number of steps to take during environment reset |
| `--max_steps` | int | 350 | Maximum number of steps before forcing a reset |

> **Note:** It is recommended to use at least 40 steps for `--reset_steps` to allow enough steps for the robot to properly reset to the SETUP position.

#### Data Collection Details

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
- Camera images


#### Keyboard Controls

During execution, you can press the 'r' key to reset the environment and state machine.

### Replay Recorded Trajectories

The `replay_recording.py` script allows you to visualize previously recorded HDF5 trajectories in the Isaac Sim environment. It loads recorded actions, organ positions, and robot joint states from HDF5 files for each episode and steps through them in the simulation.

#### Usage

Ensure your `PYTHONPATH` is set up as described in the [Environment Setup - Set environment variables before running the scripts](../../README.md#set-environment-variables-before-running-the-scripts).

Navigate to the [`simulation` folder](./) and execute:

```sh
python environments/state_machine/replay_recording.py --hdf5_path /path/to/your/hdf5_data_directory --task <YourTaskName>
```

Replace `/path/to/your/hdf5_data_directory` with the actual path to the directory containing your `data_*.hdf5` files, and `<YourTaskName>` with the task name used during data collection (e.g., `Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0`).

#### Command Line Arguments

| Argument           | Type | Default                                  | Description                                                                      |
|--------------------|------|------------------------------------------|----------------------------------------------------------------------------------|
| `--hdf5_path`      | str  | (Required)                               | Path to the directory containing recorded HDF5 files.                            |
| `--task`           | str  | `Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0` | Name of the task (environment) to use. Should match the task used for recording. |
| `--num_envs`       | int  | `1`                                      | Number of environments to spawn (should typically be 1 for replay).              |
| `--disable_fabric` | flag | `False`                                  | Disable fabric and use USD I/O operations.                                       |

> **Note:** Additional common Isaac Lab arguments (like `--device`) can also be used.

### Teleoperation

The teleoperation interface allows direct control of the robotic arm using various input devices. It supports keyboard, SpaceMouse, and gamepad controls for precise manipulation of the ultrasound probe.

#### Running Teleoperation

Please move to the current [`simulation` folder](./) and execute the following command to start the teleoperation:

```sh
python environments/teleoperation/teleop_se3_agent.py --enable_cameras
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--teleop_device` | str | "keyboard" | Device for control ("keyboard", "spacemouse", or "gamepad") |
| `--sensitivity` | float | 1.0 | Control sensitivity multiplier |
| `--disable_fabric` | bool | False | Disable fabric and use USD I/O operations |
| `--num_envs` | int | 1 | Number of environments to simulate |
| `--viz_domain_id` | int | 1 | Domain ID for visualization data publishing |
| `--rti_license_file` | str | None | Path to the RTI license file (required) |

#### Control Schemes

#### Keyboard Controls
- Please check the [Se3Keyboard documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html#isaaclab.devices.Se3Keyboard)

#### Camera Visualization

The teleoperation script supports real-time camera visualization through DDS communication. It publishes both room camera and wrist camera feeds at 30Hz.

The camera feeds are published on the following default topics:
- Room camera: `topic_room_camera_data_rgb`
- Wrist camera: `topic_wrist_camera_data_rgb`

Both cameras output 224x224 RGB images that can be visualized using compatible DDS subscribers.

#### Ultrasound Image Visualization

The teleoperation script also supports real-time ultrasound image visualization through DDS communication. It publishes the ultrasound image at 30Hz.

Please refer to the [Ultrasound Raytracing Simulation](#ultrasound-raytracing-simulation) section for more details on how to visualize the ultrasound image.



### Ultrasound Raytracing Simulation

This example implements a standalone ultrasound raytracing simulator that generates realistic ultrasound images
based on 3D meshes. The simulator uses Holoscan framework and DDS communication for realistic performance.

#### NVIDIA OptiX Raytracing with Python Bindings

For instructions on preparing and building the Python module, please refer to the [Environment Setup](../../README.md#install-the-raytracing-ultrasound-simulator) instructions and [Ultrasound Raytracing README](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/blob/v0.1.0/ultrasound-raytracing/README.md) to setup the `raysim` module correctly.


#### Configuration

Optionally, the simulator supports customization through JSON configuration files. You need to create your configuration file following the structure below:

```json
{
    "probe_params": {
        "num_elements": 4096,
        "opening_angle": 73.0,
        "radius": 45.0,
        "frequency": 2.5,
        "elevational_height": 7.0,
        "num_el_samples": 10
    },
    "sim_params": {
        "conv_psf": true,
        "buffer_size": 4096,
        "t_far": 180.0
    }
}
```

##### Probe Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| num_elements | Number of elements in the ultrasound probe | 4096 |
| opening_angle | Beam opening angle in degrees | 73.0 |
| radius | Radius of the ultrasound probe in mm | 45.0 |
| frequency | Ultrasound frequency in MHz | 2.5 |
| elevational_height | Height of the elevation plane in mm | 7.0 |
| num_el_samples | Number of samples in the elevation direction | 10 |

##### Simulation Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| conv_psf | Whether to use convolution point spread function | true |
| buffer_size | Size of the simulation buffer | 4096 |
| t_far | Maximum time/distance for ray tracing (in units relevant to the simulation) | 180.0 |

##### Minimal Configuration Example

You only need to specify the parameters you want to change - any omitted parameters will use their default values:

```json
{
    "probe_params": {
        "frequency": 3.5,
        "radius": 55.0
    },
    "sim_params": {
        "t_far": 200.0
    }
}
```

#### Running the Simulator

Please navigate to the current [`simulation` folder](./) and execute the following command to run the ultrasound raytracing simulator:

```sh
python examples/ultrasound_raytracing.py
```

The simulator will start streaming the ultrasound images under the `topic_ultrasound_data` topic in the DDS communication. The domain ID is set to 1 in the current default settings.

To visualize the ultrasound images, please check out the [Visualization Utility](../utils/README.md), and launch another terminal to run this command:

```sh
# Stay in the simulation folder
python ../utils/visualization.py
```

To see the ultrasound probe moving, please ensure the `topic_ultrasound_info` is published from the scripts such as [sim_with_dds.py](./environments/sim_with_dds.py) or [Tele-op](./environments/teleoperation/teleop_se3_agent.py).


#### Command Line Arguments

| Argument | Description | Default Value |
|----------|-------------|---------------|
| --domain_id | Domain ID for DDS communication | 0 |
| --height | Input image height | 224 |
| --width | Input image width | 224 |
| --topic_in | Topic name to consume probe position | topic_ultrasound_info |
| --topic_out | Topic name to publish generated ultrasound data | topic_ultrasound_data |
| --config | Path to custom JSON configuration file with probe parameters and simulation parameters | None |
| --period | Period of the simulation (in seconds) | 1/30.0 (30 Hz) |

### Trajectory Evaluation

After running simulations and collecting predicted trajectories (e.g., using `sim_with_dds.py` with the `--hdf5_path` and `--npz_prefix` arguments, or from other policy rollouts), you can use the `evaluate_trajectories.py` script to compare these predictions against ground truth trajectories.

This script is located at `environments/evaluate_trajectories.py`.

#### Overview

The script performs the following main functions:

1.  **Loads Data**: Reads ground truth trajectories from HDF5 files and predicted trajectories from `.npz` files based on configured file patterns.
2.  **Computes Metrics**: For each episode and each prediction source, it calculates:
    *   **Success Rate**: The percentage of ground truth points that are within a specified radius of any point in the predicted trajectory.
    *   **Average Minimum Distance**: The average distance from each ground truth point to its nearest neighbor in the predicted trajectory.
3.  **Generates Plots**:
    *   Individual 3D trajectory plots comparing the ground truth and a specific prediction for each episode.
    *   A summary plot showing the mean success rate versus different radius, including 95% confidence intervals, comparing all configured prediction methods.

#### Usage

Ensure your `PYTHONPATH` is set up correctly.

Navigate to the `scripts/simulation/` folder and execute:

```sh
python environments/evaluate_trajectories.py
```

#### Configuration

The primary configuration for this script is done by modifying the global variables at the top of the `environments/evaluate_trajectories.py` file:

- **`episode`**: The number of episodes to process.
- **`data_root`**: The root directory where your HDF5 ground truth files (e.g., `data_{e}.hdf5`) and predicted `.npz` trajectory files are located or will be organized into subdirectories.
- **`DEFAULT_RADIUS_FOR_PLOTS`**: The radius used for calculating the success rate reported in the titles of individual 3D trajectory plots.
- **`saved_compare_name`**: The filename for the summary plot (success rate vs. radius).
- **`PREDICTION_SOURCES`**: A dictionary defining the different prediction methods to evaluate. Each entry requires:
    - `file_pattern`: A string pattern for the predicted trajectory `.npz` files (e.g., `"my_model_run1/pred_traj_{e}.npz"`). The `{e}` will be replaced with the episode number.
    - `label`: A descriptive label for the legend in plots.
    - `color`: A color for this method in the plots.

**Example `PREDICTION_SOURCES` entry:**
```python
PREDICTION_SOURCES = {
    "MyModelV1": {
        "file_pattern": "model_v1_outputs/robot_obs_{e}.npz",
        "label": "My Model Version 1",
        "color": "blue"
    },
    "BaselineModel": {
        "file_pattern": "baseline_outputs/robot_obs_{e}.npz",
        "label": "Baseline",
        "color": "orange"
    }
}
```

The script expects predicted trajectory files to be found at `data_root/file_pattern`.
