# Tablecloth workflows

The Rheo v0.7.0 tablecloth scenes run as the IsaacLab-Arena environments
`G1Tablecloth` (`g1_tablecloth`, Unitree G1 with Inspire hands, workflow
`spread_tablecloth_g1`) and `H2Tablecloth` (`h2_tablecloth`, Unitree H2 with Sharpa
Wave hands, workflow `spread_tablecloth_h2`). Isaac Lab's `record_demos.py` records
XR hand-tracking demonstrations through Isaac Teleop, and `replay_demos.py` replays
them. The operator accepts each demonstration; the scenes have no success term.
No trained tablecloth policy or autonomous rule-based controller is provided.

The recovered configurations originate from release commit
`568d91db09a167ae23805b82a9c8b5f1ac4ec0c3`. The Lightwheel cloth and table and
both robot USDs load from the Healthcare v0.9.0 asset root. Scene,
physics, and retargeting configurations now live under `arena/i4h_arena/`; the
environment hands its Isaac Teleop retargeting configuration to Isaac Lab.

Run `./setup.sh` after updating. The simulator environment includes
`isaaclab-contrib` for coupled MJWarp rigid-body and VBD cloth physics.
Newton is required for these two scenes: pass `--presets newton` to every command.
The inherited PhysX path reports an invalid cloth simulation view with the pinned
runtime, so the environment refuses any other backend. Other workflows retain
their PhysX default.

## Inspect the scenes

From the repository root:

```bash
export PXR_WORK_THREAD_LIMIT=1
# Hold the G1 at its reset pose and watch the cloth; use H2Tablecloth h2_tablecloth for the H2.
arena/.venv/bin/python -m isaaclab_arena.evaluation.policy_runner \
  --policy_type i4h_arena.policies.hold.HoldPolicy --num_episodes 1 --presets newton --viz kit \
  --external_environment_class_path i4h_arena.environments:G1Tablecloth g1_tablecloth
```

`HoldPolicy` commands the measured wrist poses and finger joints, so the robot
settles where it was reset (the H2 within about 20 steps) and stays there. One
environment is supported per session.

These scenes require `PXR_WORK_THREAD_LIMIT=1` in the environment Kit starts in,
to avoid the USD physics parser's race on bodies with many colliders; see
[Isaac Sim issue #692](https://github.com/isaac-sim/IsaacSim/issues/692). Isaac Sim
6.2 keeps a value already set. Physics and rendering retain their own worker threads.

The fixed-base G1's intended gravity-free behavior is authored using
`mjc:gravcomp=1` for Newton/MJWarp. A local compatibility fix registers MuJoCo
USD attributes in the pinned coupled importer before model construction;
otherwise that backend ignores the setting. Cloth gravity and the versioned
USD material are retained.

## Start XR and record

Start CloudXR in terminal A using the simulator environment:

```bash
mkdir -p .run
echo 'NV_CXR_ENABLE_PUSH_DEVICES=0' > .run/tablecloth-handtracking.env
arena/.venv/bin/python -m isaacteleop.cloudxr \
  --cloudxr-env-config="$PWD/.run/tablecloth-handtracking.env"
```

Connect a supported hand-tracking headset using the
[Isaac Teleop headset setup](https://nvidia.github.io/IsaacCapture/main/getting_started/quick_start.html#connect-an-xr-headset).
Keep CloudXR running. In terminal B, source its exported runtime environment
and choose a robot:

```bash
source "$HOME/.cloudxr/run/cloudxr.env"
export PXR_WORK_THREAD_LIMIT=1
# Use H2Tablecloth h2_tablecloth for the H2.
arena/.venv/bin/python third_party/IsaacLab-e3f7d9f/scripts/tools/record_demos.py \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --external_environment_class_path i4h_arena.environments:G1Tablecloth --task g1_tablecloth \
  --presets newton --enable_cameras --viz kit --no-auto_launch_cloudxr \
  --dataset_file runs/spread_tablecloth_g1/demos.hdf5 --num_demos 10
```

`record_demos.py` uses Isaac Teleop because the environment declares its
retargeting configuration, and follows Isaac Lab's recording controls for
starting, accepting and resetting a demonstration. CloudXR is externally managed
here; to have the script start it instead, drop `--no-auto_launch_cloudxr` and pass
`--cloudxr_env /absolute/path/to/handtracking.env`. The first start asks the
operator to accept the CloudXR EULA. Teleop requires a visible Kit session.
An episode times out after 3,600 steps for G1 and 9,000 for H2.

## Data and replay

| Workflow | Actions | Joint state | Cameras |
| --- | --- | --- | --- |
| `spread_tablecloth_g1` | 38 | 53 | front, left wrist, right wrist |
| `spread_tablecloth_h2` | 58 | 75 | front |

Actions begin with left and right wrist poses (XYZ + XYZW quaternion), followed
by 24 Inspire or 44 Sharpa finger targets. The explicit finger order is preserved
from the original recordings; it must not be replaced with backend articulation order.
Embodiment manifests name every action and state column.

Cameras are recorded with `--enable_cameras`; leave it out for the image-free
behavior of the original v0.7 examples. H2's camera is positioned ahead of its head and
angled down at the table so its torso does not obscure the recording.
H2 downloads its URDF and retargeting assets to
`~/.cache/i4h_workflows/spread_tablecloth/h2_with_sharpa`; an existing asset bundle
can be selected with `RHEO_H2_SHARPA_ASSETS_DIR`.

```bash
arena/.venv/bin/python third_party/IsaacLab-e3f7d9f/scripts/tools/replay_demos.py \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --external_environment_class_path i4h_arena.environments:G1Tablecloth --task g1_tablecloth \
  --presets newton --dataset_file /absolute/path/to/demos.hdf5
```

Use the environment matching the recording, without `--enable_cameras`. Replaying
cloth actions does not restore an arbitrary deformed initial mesh, and
contact-rich trajectories may diverge.

## Validation

Under IsaacLab-Arena on Isaac Sim 6.2 (2026-10-22): `HoldPolicy` held the G1 for
150 steps (joints within 0.0002 rad) and the H2 for 300; Isaac Lab replayed the G1
recording; `record_demos.py` built the G1 environment and its Isaac Teleop device up
to the CloudXR session, which needs a headset and runtime to go further.

Earlier recovery checks on Isaac Sim 6.0.1 with the repository's pinned IsaacLab/Newton stack:

- G1: camera rendering, recording, and a 299-step replay of reset wrist targets.
- H2: camera rendering, recording, and a 150-step synthetic replay including a
  2 cm left-wrist lift. Both replays produced finite joint states.
- Both IsaacTeleop retargeting pipelines constructed successfully in Kit.
- CPU tests cover workflow contracts, XR action ordering, and B/S/R recording
  boundaries.
- All configured pre-commit hooks passed for tracked and new files, and every
  workflow mode passed lint.

These recovery checks exercise the simulator and recording path; they are not successful
cloth-spreading demonstrations. A connected headset and an operator are still
required to validate end-to-end XR tracking, grasping, and manual acceptance.
