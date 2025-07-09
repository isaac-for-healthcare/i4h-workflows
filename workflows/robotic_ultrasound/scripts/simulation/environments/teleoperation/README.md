# Teleoperation

The teleoperation interface allows direct control of the robotic arm using various input devices. It supports keyboard, SpaceMouse, and gamepad controls for precise manipulation of the ultrasound probe.

## Running Teleoperation

```bash
python -m simulation.environments.teleoperation.teleop_se3_agent --enable_cameras
```

**Expected Behavior:**
- Isaac Sim window with the assets loaded.
- You can control the robot arm directly using keyboard (pressing buttons like 'w', 'a', 's', 'd'). Check [Keyboard Controls](#keyboard-controls) for more details.

### Control Schemes

#### Keyboard Controls

- Please check the [Se3Keyboard documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html#isaaclab.devices.Se3Keyboard)

#### Hand Tracking Controls

Please review the [Hand Tracking Teleoperation Tutorial](../../../../../../tutorials/assets/bring_your_own_xr/README.md) for
details on Isaac Lab XR hand tracking support and setup instructions.

### Image Visualization

The teleoperation script supports real-time camera and ultrasound image visualization through DDS communication.

The feeds are published on the following default topics:
- Room camera: `topic_room_camera_data_rgb`
- Wrist camera: `topic_wrist_camera_data_rgb`
- Ultrasound image: `topic_ultrasound_data_rgb`

Both cameras output 224x224 RGB images that can be visualized using compatible DDS subscribers.

#### Example Usage

```sh
(python -m simulation.examples.ultrasound_raytracing & \
python -m simulation.environments.teleoperation.teleop_se3_agent --enable_cameras & \
python -m utils.visualization & \
wait)
```

> **Note:**
> To exit the workflow, press `Ctrl+C` in the terminal and run `bash workflows/robotic_ultrasound/reset.sh` to kill all processes spawned by the workflow.

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--teleop_device` | str | "keyboard" | Device for control ("keyboard", "spacemouse", "gamepad", or "handtracking") |
| `--sensitivity` | float | 1.0 | Control sensitivity multiplier |
| `--disable_fabric` | bool | False | Disable fabric and use USD I/O operations |
| `--num_envs` | int | 1 | Number of environments to simulate |
| `--viz_domain_id` | int | 1 | Domain ID for visualization data publishing |
| `--rti_license_file` | str | None | Path to the RTI license file (required) |

## Documentation Links

- [Ultrasound Raytracing Simulation](../../examples/README.md)
- [Visualization](../../../utils/README.md)
