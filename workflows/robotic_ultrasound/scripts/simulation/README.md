# IsaacLab simulation for robotic ultrasound

## Table of Contents
- [Installation](#installation)
- [Environments](#environments)
- [Apps](#apps)
  - [PI Zero Policy Evaluation](#pi-zero-policy-evaluation)
  - [Policy Evaluation w/ DDS](#policy-evaluation-w-dds)
  - [Liver Scan State Machine](#liver-scan-state-machine)
  - [Ultrasound Raytracing Simulation](#ultrasound-raytracing-simulation)

# Installation

Follow the [Environment Setup](../../README.md#environment-setup) instructions to setup the environment and dependencies.

# Environments

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

# Apps

## PI Zero Policy Evaluation
Set up `openpi` referring to [PI0 runner](../policy_runner/README.md).

Move to the [scripts](../) folder and specify python path:
```sh
export PYTHONPATH=`pwd`
```

5. Return to this folder and run the following command:
```sh
python environments/state_machine/pi0_policy/eval.py \
    --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
    --enable_camera \
    --repo_id i4h/sim_liver_scan
```
NOTE: You can also specify `--ckpt_path` to run a specific policy.
This should open a stage with Franka arm and run the robotic ultrasound actions:
![pi0 simulation](../../../../docs/source/pi0_sim.jpg)

## Policy Evaluation w/ DDS
This example should work together with the `pi0 policy runner` via DDS communication,
so please ensure to launch the `run_policy.py` with `height=224`, `width=224`,
and the same `domain id` as this example in another terminal.

When `run_policy.py` is launched and idle waiting for the data,
move to the [scripts](../) folder and specify the python path:
```sh
export PYTHONPATH=`pwd`
```
Then move back to this folder and execute:
```sh
python examples/sim_with_dds.py \
    --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
    --enable_camera \
    --infer_domain_id <domain id> \
    --viz_domain_id <domain id> \
    --rti_license_file <path to>/rti_license.dat
```

## Liver Scan State Machine

The Liver Scan State Machine provides a structured approach to performing ultrasound scans on a simulated liver. It implements a state-based workflow that guides the robotic arm through the scanning procedure.

### Overview

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

### Requirements

- This implementation works **only with a single environment** (`--num_envs 1`).
- It should be used with the `Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0` environment.

### Usage

move to the [scripts](../) folder and specify the python path:
```sh
export PYTHONPATH=`pwd`
```
Then move back to this folder and execute:

```sh
python environments/state_machine/liver_scan_sm.py \
    --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
    --enable_camera
```

To enable DDS use add `--rti_license_file <abs_path_license>` or export the environment variable with the respective path. E.g. `export RTI_LICENSE_FILE=<abs_path_license>;` before launching.


### Data Collection

To run the state machine and collect data for a specified number of episodes:

```sh
python environments/state_machine/liver_scan_sm.py \
    --task Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0 \
    --enable_camera \
    --num_episodes 2
```

This will collect data for 2 complete episodes and store it in HDF5 format.

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | None | Name of the task (environment) to use |
| `--num_episodes` | int | 0 | Number of episodes to collect data for (0 = no data collection) |
| `--camera_names` | list[str] | ["room_camera", "wrist_camera"] | List of camera names to capture images from |
| `--disable_fabric` | flag | False | Disable fabric and use USD I/O operations |
| `--num_envs` | int | 1 | Number of environments to spawn (must be 1 for this script) |
| `--reset_steps` | int | 15 | Number of steps to take during environment reset |
| `--max_steps` | int | 350 | Maximum number of steps before forcing a reset |

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
- Camera images

## Ultrasound Raytracing Simulation

This example implements a standalone ultrasound raytracing simulator that generates realistic ultrasound images
based on 3D meshes. The simulator uses Holoscan framework and DDS communication for realistic performance.

### NVIDIA OptiX Raytracing with Python Bindings

For instructions on preparing and building the Python module, please refer to the [Ultrasound Raytracing README](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/blob/main/ultrasound-raytracing/README.md).

### Configuration

The simulator supports customization through JSON configuration files. You need to create your own
configuration file following the structure below:

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

#### Probe Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| num_elements | Number of elements in the ultrasound probe | 4096 |
| opening_angle | Beam opening angle in degrees | 73.0 |
| radius | Radius of the ultrasound probe in mm | 45.0 |
| frequency | Ultrasound frequency in MHz | 2.5 |
| elevational_height | Height of the elevation plane in mm | 7.0 |
| num_el_samples | Number of samples in the elevation direction | 10 |

#### Simulation Parameters

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

### Running the Simulator

To run the ultrasound raytracing simulator:

```sh
python examples/ultrasound-raytracing.py \
    --domain_id <domain_id> \
    --height 224 \
    --width 224 \
    --topic_in topic_ultrasound_info \
    --topic_out topic_ultrasound_data \
    --viz_domain_id <domain id> \
    --config path/to/your_config.json
```
