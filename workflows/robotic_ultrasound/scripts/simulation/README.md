# IsaacLab simulation for robotic ultrasound

## Table of Contents
- [Installation](#installation)
- [Environments](#environments)
- [Apps](#apps)
  - [PI Zero Policy Evaluation](#pi-zero-policy-evaluation)
  - [Policy Evaluation w/ DDS](#policy-evaluation-w-dds)
  - [Liver Scan State Machine](#liver-scan-state-machine)
  - [Cosmos-transfer1 Integration](#cosmos-transfer1-integration)
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

Please cd to the current [`simulation` folder](./) and execute:

```sh
cd workflows/robotic_ultrasound/scripts/simulation
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

Please cd to the current [`simulation` folder](./) and execute:

```sh
cd ../simulation
python environments/sim_with_dds.py --enable_cameras
```

#### Evaluating with Recorded Initial States and Saving Trajectories

The `sim_with_dds.py` script can also be used for more controlled evaluations by resetting the environment to initial states from recorded HDF5 data. When doing so, it can save the resulting end-effector trajectories.

- **`--hdf5_path /path/to/your/data.hdf5`**: Provide the path to an HDF5 file (or a directory containing HDF5 files for multiple episodes). The simulation will reset the environment to the initial state(s) found in this data for each episode.
- **`--npz_prefix your_prefix_`**: When `--hdf5_path` is used, this argument specifies a prefix for the names of the `.npz` files where the simulated end-effector trajectories will be saved. Each saved file will be named like `your_prefix_robot_obs_{episode_idx}.npz` and stored in the same directory as the input HDF5 file (if `--hdf5_path` is a file) or within the `--hdf5_path` directory (if it's a directory).

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

Then please cd to the current [`simulation` folder](./) and execute:

```sh
cd ../simulation
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
| `--include_seg` | bool | True | Whether to include semantic segmentation in the data collection |

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
- Camera images (RGB, depth, segmentation if `--include_seg` is enabled)


#### Keyboard Controls

During execution, you can press the 'r' key to reset the environment and state machine.

### Replay Recorded Trajectories

The `replay_recording.py` script allows you to visualize previously recorded HDF5 trajectories in the Isaac Sim environment. It loads recorded actions, organ positions, and robot joint states from HDF5 files for each episode and steps through them in the simulation.

#### Usage

Ensure your `PYTHONPATH` is set up as described in the [Environment Setup - Set environment variables before running the scripts](../../README.md#set-environment-variables-before-running-the-scripts).

Navigate to the [`simulation` folder](./) and execute:

```sh
python environments/state_machine/replay_recording.py /path/to/your/hdf5_data_directory --task <YourTaskName> --enable_camera
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


### Cosmos-transfer1 Integration

[Cosmos-Transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1) is a world-to-world transfer model designed to bridge the perceptual divide between simulated and real-world environments.
We introduce a training-free guided generation method on top of Cosmos-Transfer1 to overcome unsatisfactory results on unseen healthcare simulation assets.
Directly applying Cosmos-Transfer with various control inputs results in unsatisfactory outputs for the human phantom and robotic arm (see bottom figure). In contrast, our guided generation method preserves the appearance of the phantom and robotic arm while generating diverse backgrounds.

<img src="../../../../docs/source/cosmos_transfer_result.png" width="512" height="600" />

This training-free guided generation approach by encoding simulation videos into the latent space and applying spatial masking to guide the generation process. The trade-off between realism and faithfulness can be controlled by adjusting the number of guided denoising steps. In addition, our generation pipeline supports multi-view video generation. We first leverage the camera information to warp the generated room view to wrist view, then use it as the guidance of wrist-view generation.

#### Download Cosmos-transfer1 Checkpoints
The cosmos-transfer1 dependency is already installed after completing the [Installation](#installation) section. Move to the third party `cosmos-transfer1` folder and run the following command to download the checkpoints:
```sh
cd third_party/cosmos-transfer1
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
```

#### Video Prompt Generation
We follow the idea in [lucidsim](https://github.com/lucidsim/lucidsim) to first generate batches of meta prompt that contains a very concise description of the potential scene, then instruct the LLM (e.g., [gemma-3-27b-it](https://build.nvidia.com/google/gemma-3-27b-it)) to upsample the meta prompt with detailed descriptions.
We provide example prompts in [`generated_prompts_two_seperate_views.json`](./environments/cosmos_transfer1/config/generated_prompts_two_seperate_views.json).

#### Running Cosmos-transfer1 + Guided Generation
Please cd back to the current [simulation folder](./) and execute the following command to start the generation pipeline:
```sh
cd ../../workflows/robotic_ultrasound/scripts/simulation/
export CHECKPOINT_DIR="path to downloaded cosmos-transfer1 checkpoints"
# Set project root path
export PROJECT_ROOT="{your path}/i4h-workflows"
# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/third_party/cosmos-transfer1:$PROJECT_ROOT/workflows/robotic_ultrasound/scripts"
# run bath inference for generation pipeline
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$PYTHONPATH python \
    -m environments.cosmos_transfer1.transfer \
    --checkpoint_dir $CHECKPOINT_DIR \
    --source_data_dir "Path to source dir of h5 files" \
    --output_data_dir "Path to output dir of h5 files" \
    --offload_text_encoder_model
```
#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompt` | str | "" | Prompt which the sampled video condition on |
| `--negative_prompt` | str | "The video captures a game playing, ..." | Negative prompt which the sampled video condition on |
| `--input_video_path` | str | "" | Optional input RGB video path |
| `--num_input_frames` | int | 1 | Number of conditional frames for long video generation |
| `--sigma_max` | float | 80 | sigma_max for partial denoising |
| `--blur_strength` | str | "medium" | Blur strength applied to input |
| `--canny_threshold` | str | "medium" | Canny threshold applied to input. Lower means less blur or more detected edges, which means higher fidelity to input |
| `--controlnet_specs` | str | "inference_cosmos_transfer1_two_views.json" | Path to JSON file specifying multicontrolnet configurations |
| `--checkpoint_dir` | str | "checkpoints" | Base directory containing model checkpoints |
| `--tokenizer_dir` | str | "Cosmos-Tokenize1-CV8x8x8-720p" | Tokenizer weights directory relative to checkpoint_dir |
| `--video_save_folder` | str | "outputs/" | Output folder for generating a batch of videos |
| `--num_steps` | int | 35 | Number of diffusion sampling steps |
| `--guidance` | float | 5.0 | Classifier-free guidance scale value |
| `--fps` | int | 30 | FPS of the output video |
| `--height` | int | 224 | Height of video to sample |
| `--width` | int | 224 | Width of video to sample |
| `--seed` | int | 1 | Random seed |
| `--num_gpus` | int | 1 | Number of GPUs used to run context parallel inference. |
| `--offload_diffusion_transformer` | bool | False | Offload DiT after inference |
| `--offload_text_encoder_model` | bool | False | Offload text encoder model after inference |
| `--offload_guardrail_models` | bool | True | Offload guardrail models after inference |
| `--upsample_prompt` | bool | False | Upsample prompt using Pixtral upsampler model |
| `--offload_prompt_upsampler` | bool | False | Offload prompt upsampler model after inference |
| `--source_data_dir` | str | "" | Path to source data directory for batch inference. It contains h5 files generated from the state machine. |
| `--output_data_dir` | str | "" | Path to output data directory for batch inference. |
| `--save_name_offset` | int | 0 | Offset for the video save name. |
| `--foreground_label` | str | "3,4" | Comma-separated list of labels used to define the foreground mask during guided generation. The foreground corresponds to the object whose appearance we want to keep unchanged during generation. |
| `--sigma_threshold` | float | 1.2866 | This controls how many guidance steps are performed during generation. Smaller values mean more steps, larger values mean less steps. |
| `--concat_video_second_view` | bool | True | Whether to concatenate the first and second view videos during generation |
| `--fill_missing_pixels` | bool | True | Whether to fill missing pixels in the warped second view video |
| `--model_config_file` | str | "environments/cosmos_transfer1/config/transfer/config.py" | Relative path to the model config file |


### Teleoperation

The teleoperation interface allows direct control of the robotic arm using various input devices. It supports keyboard, SpaceMouse, and gamepad controls for precise manipulation of the ultrasound probe.

#### Running Teleoperation

Please cd to the current [`simulation` folder](./) and execute the following command to start the teleoperation:

```sh
cd ../simulation
python environments/teleoperation/teleop_se3_agent.py --enable_cameras
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--teleop_device` | str | "keyboard" | Device for control ("keyboard", "spacemouse", "gamepad", or "handtracking") |
| `--sensitivity` | float | 1.0 | Control sensitivity multiplier |
| `--disable_fabric` | bool | False | Disable fabric and use USD I/O operations |
| `--num_envs` | int | 1 | Number of environments to simulate |
| `--viz_domain_id` | int | 1 | Domain ID for visualization data publishing |
| `--rti_license_file` | str | None | Path to the RTI license file (required) |

#### Control Schemes

#### Keyboard Controls
- Please check the [Se3Keyboard documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html#isaaclab.devices.Se3Keyboard)

#### Hand Tracking Controls

Please review the [Hand Tracking Teleoperation Tutorial](../../../../tutorials/assets/bring_your_own_xr/README.md) for
details on Isaac Lab XR hand tracking support and setup instructions.

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

For instructions on preparing and building the Python module, please refer to the [Environment Setup](../../README.md#install-the-raytracing-ultrasound-simulator) instructions and [Ultrasound Raytracing README](https://github.com/isaac-for-healthcare/i4h-sensor-simulation/blob/v0.2.0rc1/ultrasound-raytracing/README.md) to setup the `raysim` module correctly.


#### Configuration

Optionally, the simulator supports customization through JSON configuration files. You need to create your configuration file following the structure below:

```json
{
    "probe_type": "curvilinear",
    "probe_params": {
        "num_elements": 256,
        "sector_angle": 73.0,
        "radius": 45.0,
        "frequency": 2.5,
        "elevational_height": 7.0,
        "num_el_samples": 1,
        "f_num": 1.0,
        "speed_of_sound": 1.54,
        "pulse_duration": 2.0
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
| num_elements | Number of elements in the ultrasound probe | 256 |
| sector_angle | Beam sector angle in degrees | 73.0 |
| radius | Radius of the ultrasound probe in mm | 45.0 |
| frequency | Ultrasound frequency in MHz | 2.5 |
| elevational_height | Height of the elevation plane in mm | 7.0 |
| num_el_samples | Number of samples in the elevation direction | 1 |
| f_num | F-number (unitless) | 1.0 |
| speed_of_sound | Speed of sound in mm/us | 1.54 |
| pulse_duration | Pulse duration in cycles | 2.0 |

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
    "probe_type": "curvilinear",
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
| --probe_type | Type of ultrasound probe to use ("curvilinear") | "curvilinear" |

### Trajectory Evaluation

This script (`evaluation/evaluate_trajectories.py`) compares predicted trajectories against ground truth data, computing metrics like success rate and average minimum distance. It also generates visual plots for analysis.

For detailed information on usage, configuration, and compare results between pi0 and GR00T-N1 (including example plots and tables), please refer to the [Trajectory Evaluation README](./evaluation/README.md).

**Quick Usage Example:**

```sh
python evaluation/evaluate_trajectories.py --data_root /path/to/data --method-name MyModel --ps-file-pattern "model_output/preds_{e}.npz" ...
```
