# MIRA ARM Teleoperation Tutorial

## Environment Setup

This tutorial requires the following dependencies:
- [IsaacSim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [IsaacLab 2.0.2](https://isaac-sim.github.io/IsaacLab/v2.0.2/index.html)

Please ensure these are installed.

## Run the scripts

```sh
# TODO: no need to specify usd path after including it into i4h-assets
python teleoperate_mira_arm.py --usd_path <path-to-usd>/mira-bipo-size-experiment-smoothing.usd
```

### Teleoperation Methods


Teleoperation Key Mapping

| Key   | Joint/Action   | Direction/Effect      |
|-------|---------------|-----------------------|
| I     | Shoulder X    | + (Forward)           |
| K     | Shoulder X    | – (Backward)          |
| J     | Shoulder Y    | + (Left)              |
| L     | Shoulder Y    | – (Right)             |
| U     | Shoulder Z    | + (Up)                |
| O     | Shoulder Z    | – (Down)              |
| Z     | Elbow         | + (Bend)              |
| X     | Elbow         | – (Straighten)        |
| C     | Wrist Roll    | + (Roll CW)           |
| V     | Wrist Roll    | – (Roll CCW)          |
| B     | Gripper       | + (Open)              |
| N     | Gripper       | – (Close)             |
| SPACE | Switch Arm    | Toggle Left/Right Arm |

*Note: The keys control either the left or right arm, depending on which is currently selected. Press `SPACE` to switch between arms.*
