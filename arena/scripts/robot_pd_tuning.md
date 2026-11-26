# Robot PD Tuning Diagnostics

These diagnostics tune/check an embodiment's joint drives inside a scene. The scene can be minimal, but it must create `env.scene["robot"]`; the scripts read the runtime articulation and step physics.

## Run

From the repo root:

```bash
arena/.venv/bin/python arena/scripts/robot_pd_tune.py <scene>
```

The script creates `./runs/<scene>/<timestamp>_pd_tuning/`, writes `robot_pd_tuning_config.json`, builds the scene's IsaacLab-Arena environment in the same process, runs `robot_pd_diagnostics.py` on its robot, prints a compact summary, and exits. Add `--print-json` for the full report in the terminal.

Useful variants:

```bash
arena/.venv/bin/python arena/scripts/robot_pd_tune.py psm_reach --modes inspect-usd,step-response
```

Options the script does not know go to Isaac Lab's app launcher and Arena's environment builder. For example, select Newton for a supported surgical scene, or watch the run:

```bash
arena/.venv/bin/python arena/scripts/robot_pd_tune.py psm_reach --presets newton --viz kit
```

For local-file embodiments, export any asset path first. Example for the local KINOVA L3M asset:

```bash
export KINOVA_L3M_P1_ROOT=/tmp/i4h_kinova_l3m_p1/extracted/L3M_P1_URDF_2025-06-10/L3M_URDF_P1
```

Preview the configuration and Arena arguments without launching Isaac:

```bash
python3 arena/scripts/robot_pd_tune.py psm_reach --dry-run
```

## Outputs

Results are written under the fresh scene-scoped PD tuning run directory:

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
