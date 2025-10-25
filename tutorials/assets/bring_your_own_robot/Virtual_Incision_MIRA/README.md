# [Virtual Incision MIRA](https://virtualincision.com/mira/) Teleoperation Tutorial

This tutorial shows how to teleoperate the [Virtual Incision MIRA](https://virtualincision.com/mira/) robot in Isaac Sim using keyboard controls.

<p align="center" style="display: flex; justify-content: center; gap: 10px;">
  <img src="../../../../docs/source/mira.png" alt="Virtual Incision MIRA Example" style="width: 70%; height: auto; aspect-ratio: 16/9; object-fit: cover;" />
</p>

## Environment Setup

**Note**: The setup process takes approximately 30-40 minutes to complete, depending on your system and network connection.

This tutorial requires the following dependencies:
- [IsaacSim 5.0.0](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/index.html)
- [IsaacLab 2.3.0](https://isaac-sim.github.io/IsaacLab/release/2.3.0/index.html)
- [i4h_asset_helper](https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/v0.3.0/docs/catalog_helper.md)

Please ensure these are installed.

## Run the scripts

```sh
python teleoperate_virtual_incision_mira.py
```

### Teleoperation Methods

#### Arm Joint Key Mapping

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
| Y | Switch Arm    | Toggle Left/Right Arm |

> **Note:** The keys control either the left or right arm, depending on which is currently selected. Press `Y` to switch between arms.

---

#### Camera Group Control

You can now control the orientation of the camera group using the arrow keys. This allows you to adjust the camera's tilt (X axis, up/down) and pan (Y axis, left/right) independently of the arm control.

| Key     | Camera Axis (Local) | Direction/Effect      |
|---------|---------------------|----------------------|
| UP      | X (Tilt)            | + (Tilt Up)          |
| DOWN    | X (Tilt)            | – (Tilt Down)        |
| LEFT    | Y (Pan)             | – (Pan Left)         |
| RIGHT   | Y (Pan)             | + (Pan Right)        |

> **Note:** UP/DOWN keys tilt the camera up/down (local X axis), LEFT/RIGHT keys pan the camera left/right (local Y axis). The camera group orientation is clamped to ±70 degrees for each axis. Camera control is independent of arm selection.

#### Camera Snapshot Feature

You can save an image from the endoscopic camera by pressing the `F12` key during simulation. The captured image will be saved as a PNG file in the current working directory (e.g., `camera_snapshot_YYYYMMDD_HHMMSS.png`).

> **Note:** The first time you press F12 after starting or resuming the simulation, the snapshot may fail with a message like `No image data available. Make sure the simulation is running and the camera is active`. This is normal, as the camera pipeline needs one frame to warm up. Simply press `F12` again after a short delay to capture the image successfully.
