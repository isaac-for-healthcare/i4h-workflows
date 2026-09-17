# Tablecloth workflows

The Rheo v0.7.0 tablecloth scenes are integrated into the Scene / Task / Workflow
runtime as `spread_tablecloth_g1` (Unitree G1 with Inspire hands) and
`spread_tablecloth_h2` (Unitree H2 with Sharpa Wave hands). Both support `idle`,
`teleop`, and `replay`. XR demonstrations are accepted by the operator.
No trained tablecloth policy or autonomous rule-based controller is provided.

The recovered configurations originate from release commit
`568d91db09a167ae23805b82a9c8b5f1ac4ec0c3`. The Lightwheel cloth and table and
both robot USDs retain their versioned Healthcare 0.7.0 asset URLs. Scene,
physics, and retargeting configurations now live under `arena/i4h_arena/`.
`teleop/xr` reads the scene's retargeting configuration and writes actions;
the shared `SimulationRunner` owns stepping, resets, and HDF5 recording.

Run `./setup.sh` after updating. The simulator environment includes
`isaaclab-contrib` for coupled MJWarp rigid-body and VBD cloth physics.
Newton is required for these two scenes. The inherited PhysX path reports an
invalid cloth simulation view with the pinned runtime, so `--presets physx`
is rejected before launch. Other workflows retain their PhysX default.

## Inspect the scenes

From the repository root:

```bash
./run.sh spread_tablecloth_g1 --idle
./run.sh spread_tablecloth_h2 --idle
```

Idle renders the initial scene without advancing physics. Pink/Pinocchio is
loaded before Kit automatically, including for replay. One environment is
supported per session.

These scenes require `PXR_WORK_THREAD_LIMIT=1` during Kit startup to avoid the
USD physics parser's race on bodies with many colliders. The launcher preserves
that value across Kit's environment override; see
[Isaac Sim issue #692](https://github.com/isaac-sim/IsaacSim/issues/692).
Physics and rendering retain their own worker threads.

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
[Isaac Teleop headset setup](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html#connect-an-xr-headset).
Keep CloudXR running. In terminal B, source its exported runtime environment
and choose a robot:

```bash
source "$HOME/.cloudxr/run/cloudxr.env"
./run.sh spread_tablecloth_g1 --teleop xr --episodes 10 --record
# Or:
./run.sh spread_tablecloth_h2 --teleop xr --episodes 10 --record
```

The launcher prints the timestamped run directory containing `demos.hdf5` and
logs. CloudXR is externally managed by default. For an embedded runtime, pass
`--cloudxr-env /absolute/path/to/handtracking.env --auto-launch-cloudxr`.
Teleop requires a visible Kit session; `--headless` is rejected before launch.

With the simulator window focused:

- **B** resets the scene, discards any pending recording, and begins a demo.
- **S** saves a nonempty demo as operator-accepted and advances to the next episode.
- **R** discards the current attempt and resets; press **B** to start again.

Waiting for B or a tracking sample does not advance physics or record frames.
Each saved episode contains the XR task segment. The simulation-step budgets
retain the old teleop durations: 3,600 steps for G1 and 9,000 for H2 at 30 Hz.

## Data and replay

| Workflow | Actions | Joint state | Cameras |
| --- | --- | --- | --- |
| `spread_tablecloth_g1` | 38 | 53 | front, left wrist, right wrist |
| `spread_tablecloth_h2` | 58 | 75 | front |

Actions begin with left and right wrist poses (XYZ + XYZW quaternion), followed
by 24 Inspire or 44 Sharpa finger targets. The explicit finger order is preserved
from the original recordings; it must not be replaced with backend articulation order.
Embodiment manifests name every action and state column.

Cameras are recorded by default. Use `--no-cameras` for the image-free behavior
of the original v0.7 examples. H2's camera is positioned ahead of its head and
angled down at the table so its torso does not obscure the recording.
H2 downloads its URDF and retargeting assets to
`~/.cache/i4h_workflows/spread_tablecloth/h2_with_sharpa`; an existing asset bundle
can be selected with `RHEO_H2_SHARPA_ASSETS_DIR`.

```bash
./run.sh spread_tablecloth_g1 --replay /absolute/path/to/demos.hdf5
./run.sh spread_tablecloth_h2 --replay /absolute/path/to/demos.hdf5
```

Use the workflow matching the recording. These commands consume the redesigned
workflow HDF5 action contract; historical IsaacLab recordings have different
metadata and observation layouts. Replaying cloth actions does not restore an
arbitrary deformed initial mesh, and contact-rich trajectories may diverge.

## Validation

Recovery checks on Isaac Sim 6.0.1 with the repository's pinned IsaacLab/Newton stack:

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
