# Robot PD Tuning Diagnostics

These diagnostics tune/check an embodiment's joint drives inside a live workflow Scene. The Scene can be minimal, but it must create `env.scene["robot"]`; the scripts read the runtime articulation and step physics.

## Run

From the repo root:

```bash
./run.sh robot-pd <workflow>
```

The command creates `./runs/<workflow>/<timestamp>_pd_tuning/`, writes `robot_pd_tuning_config.json`, starts the workflow in live mode, runs `robot_pd_diagnostics.py` through the pinned Isaac Sim Python-server client, prints a compact summary, and stops the live simulator. Add `--print-json` when you need the full raw response in the terminal.

Useful variants:

```bash
./run.sh robot-pd surgical_reach_psm --modes inspect-usd,step-response
./run.sh robot-pd ultrasound_liver_scan --joint-names-expr 'panda_joint.*' --plots
./run.sh robot-pd locomanip_tray_pick_and_place --joint-names-expr '.*shoulder.*,.*elbow.*' --sample-every 5
```

Unknown tuning arguments are forwarded to the live workflow launch. For example, select Newton for a supported surgical workflow:

```bash
./run.sh robot-pd surgical_reach_psm --presets newton
```

For local-file embodiments, export any asset path first. Example for the local KINOVA L3M asset:

```bash
export KINOVA_L3M_P1_ROOT=/tmp/i4h_kinova_l3m_p1/extracted/L3M_P1_URDF_2025-06-10/L3M_URDF_P1
```

Preview what will run without launching Isaac:

```bash
./run.sh robot-pd surgical_reach_psm --dry-run
```

To keep the live simulator open for more experiments:

```bash
./run.sh robot-pd surgical_reach_psm --keep-live
./stop.sh all
```

## Outputs

Results are written under the fresh workflow-scoped PD tuning run directory:

- `robot_pd_diagnostics.json` - top-level summary.
- `robot_usd_joint_drive_probe.json` - USD and runtime actuator metadata.
- `robot_direct_joint_state_probe.{json,csv}` - direct joint-state write check.
- `robot_joint_step_response.{json,csv}` - per-joint step response and gain recommendations.
- `robot_joint_trajectory_response.{json,csv}` - coordinated ramp/hold tracking check.
- Optional plots, when `common.plots.enabled` is true.

## What It Does

1. Loads the selected joints from `robot.data.joint_names`, an optional `joint_names` / `joint_names_expr` config, or matching USD joint-drive metadata.
2. Reads runtime actuator values: stiffness, damping, armature, friction, effort limits, and velocity limits.
3. Inspects the robot USD for drive attributes and PhysX joint metadata.
4. Writes direct joint states to verify the articulation accepts commanded positions exactly.
5. Steps one joint at a time, records position/velocity/error, and computes rise time, settling time, overshoot, final error, and velocity-limit usage.
6. Runs a coordinated trajectory ramp/hold and records tracking error and velocity usage.
7. Recommends PD gain changes from the measured response. If configured, it can write a USDA overlay with stronger drive opinions; it does not edit the source robot USD.

The task scene, rewards, and policy are not part of the tuning logic. They only provide the live IsaacLab environment that owns the embodiment articulation.
