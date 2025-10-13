# Environments

This folder contains the core simulation infrastructure for SO-ARM Starter environments, primarily built around the main simulation pipeline that orchestrates IsaacLab environments with DDS communication.

**Key Components:**

- **`sim_with_dds.py`** - Main simulation orchestrator that runs SO-ARM101 manipulator simulations with real-time DDS communication
- **`teleoperation_record.py`** - Data collection script for recording demonstrations using SO-ARM101 leader arm teleoperation
- **`replay_recording.py`** - Dataset replay and validation script for visualizing collected demonstrations
- **`state_machine/`** - State machine implementations for structured SO-ARM Starter workflows

## Teleoperation & Data Collection

Collect training data using SO-ARM101 leader arm for teleoperation:

```bash
python -m simulation.environments.teleoperation_record \
    --task=Isaac-SOARM101-v0 \
    --port=/dev/ttyACM1 \
    --enable_cameras \
    --record \
    --dataset_path=../datasets/test_data.hdf5
```

## Replay Dataset

Validate and visualize collected datasets:

```bash
python -m simulation.environments.replay_recording \
    --task=Isaac-SOARM101-v0 \
    --dataset_path=../datasets/test_data.hdf5 \
    --enable_cameras
```

## Simulation with DDS
Before running, deploy your policy with DDS communication via **[policy runner](../../policy_runner/README.md)**

Run simulation with DDS communication for deployment testing:

```bash
python -m simulation.environments.sim_with_dds --enable_cameras
```

---

## Documentation Links

- [IsaacLab Task Setting](../exts/so_arm_starter_ext/README.md)
- [Policy Runner](../../policy_runner/README.md)
