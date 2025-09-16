# SO-ARM Starter Extension - Technical Documentation

This extension provides environments for SO-ARM Starter simulation using the SO-ARM101 robotic arm configuration.

## Installation

### Run Installation Script
```bash
bash tools/env_setup/install_so_arm_starter_extensions.sh
```

## API Reference

### Core Components

1. **Environment Registration**: `Isaac-SOARM101-v0`
2. **Scene Configuration**: Complete surgical workspace setup
3. **Action Space**: 6-DOF robot control
4. **Observation Space**: Joint states + RGB images

### Environment Registration

```python
import gymnasium as gym
import so_arm_starter_ext

# Create environment
env = gym.make("Isaac-SOARM101-v0", num_envs=1).unwrapped
```

### Configuration Classes

#### Scene Components

| Component | Type | Description |
|-----------|------|-------------|
| `robot` | `SOARM101Cfg` | 6-DOF robotic arm |
| `scissors` | `RigidObjectCfg` | Surgical scissors  |
| `tray` | `RigidObjectCfg` | Surgical tray (placement target) |
| `table` | `RigidObjectCfg` | Work surface |
| `room_camera` | `TiledCameraCfg` | Fixed overhead camera |
| `wrist_camera` | `TiledCameraCfg` | Robot-mounted camera |


## Configuration

### Environment Parameters

#### Robot Configuration
- **DOF**: 6 joints
- **Control**: Position control with PID gains

#### Camera Configuration
- **Room Camera**: Fixed position for global view
  - Resolution: 640x480
  - FOV: Configurable field of view
  - Position: Overhead surgical workspace

- **Wrist Camera**: Mounted on robot end-effector
  - Resolution: 640x480
  - FOV: Close-up manipulation view
  - Dynamic positioning with robot movement
