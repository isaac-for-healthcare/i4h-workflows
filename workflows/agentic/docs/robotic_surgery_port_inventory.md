# Robotic Surgery Agentic Port Inventory

This document is the source-to-target checklist for porting
`workflows/robotic_surgery` into native `workflows/agentic` environments.
It records the first executable slice and the interfaces that must stay stable
while the port is staged.

## First Slice

| Decision | Value |
|---|---|
| MVP env | `surgical_reach_psm` |
| Source demo | `workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_psm_sm.py` |
| Source task config | `workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/reach/` |
| Source robot asset | `robotic.surgery.assets.psm.PSM_CFG` / `PSM_HIGH_PD_CFG` |
| Agentic target pattern | Native Arena YAML + env class + assets + task + runtime |
| Policy/data contract | Fixed-width joint-position action/state vectors |
| Scripted controller contract | May use task-space IK internally, but recorded HDF5 must expose joint action/state vectors plus camera observations |
| Deferred | dual PSM, handover, pose-plus-gripper Zenoh message extension |

Do not add env YAMLs for future surgical envs until their env class, runtime
module, and policy routing fields are complete. `policy/run.sh --list-envs`
loads every environment YAML and requires a valid `policy.stack`.

## Source Inventory

| Source | Contents | Agentic target |
|---|---|---|
| `scripts/simulation/utils/assets.py` | Healthcare USD constants for dVRK, STAR, board, block, needle, suture pad, table, organs | `workflows/agentic/arena/arena/assets/constants.py` |
| `exts/robotic.surgery.assets/robotic/surgery/assets/psm.py` | dVRK PSM articulation cfg and high-PD variant | `arena/arena/embodiments/psm.py` |
| `exts/robotic.surgery.assets/robotic/surgery/assets/ecm.py` | dVRK ECM articulation cfg and high-PD variant | `arena/arena/embodiments/ecm.py` |
| `exts/robotic.surgery.assets/robotic/surgery/assets/star.py` | STAR articulation cfg and high-PD variant | `arena/arena/embodiments/star.py` |
| `surgical/reach/` | ManagerBased reach task, PSM/ECM/STAR joint and IK variants, dual-PSM variants | `arena/arena/tasks/surgical_reach.py` plus env-specific YAML/classes |
| `surgical/lift/` | ManagerBased lift task for needle/block/OR scenes | `arena/arena/tasks/surgical_lift.py` after reach MVP |
| `surgical/handover/` | Dual-arm handover cfgs with TODO observations/rewards | Deferred until dual-arm runtime and success semantics are defined |
| `scripts/environments/state_machine/*.py` | Six public Warp state-machine demos | `arena/arena/controllers/surgical/` after MVP scene/action contract builds |
| `tests/test_environments/test_surgery_sm.py` | Smoke waits for "Resetting the state machine." in six scripts | Replace with agentic state-machine success/recording smoke |
| `tests/test_reinforcement_learning/test_rsl_rl_train.py` | RSL-RL train smoke for old gym tasks | Reference only; not primary agentic training interface |

## Public Parity Scope

The old `metadata.json` user-facing modes are:

- `reach_psm`
- `reach_dual_psm`
- `reach_star`
- `lift_needle`
- `lift_needle_organs`
- `lift_block`
- `train_rl`
- `play_rl`

Agentic parity for demos means replacing the six old state-machine modes
with native state-machine controllers that can record HDF5. RSL-RL train/play stays
reference material unless explicitly requested as a compatibility wrapper.
Handover is task-source coverage, not current public demo parity.

## Action And HDF5 Contract

The first supported surgical policy path uses joint-position vectors:

- `RobotState.joint_positions` publishes one ordered vector.
- `RobotCommand.joint_positions` receives `horizon * action_dim` values.
- HDF5 conversion reads `obs/actions`, `obs/<state_obs_key>`, and RGB camera
  arrays through the default converter path.
- Robot YAMLs provide ordered joint names and joint-limit ranges for remap.
- Dual-arm envs will either encode both arms into one strict logical vector or
  extend the message schema in a separate change.

For PSM single-arm policy records, use six controllable arm joints plus one
logical gripper aperture channel. The physical USD has two opposing gripper
joints; the runtime should map one logical gripper value onto both jaws.

## Validation Gates

Static checks:

```bash
workflows/agentic/policy/run.sh --list-envs
workflows/agentic/arena/run.sh --env surgical_reach_psm --dry-run
workflows/agentic/policy/run.sh --env surgical_reach_psm --dry-run
python -m py_compile <changed-python-files>
```

Real build gate:

```bash
workflows/agentic/arena/run.sh ensure-bridge --env surgical_reach_psm --log workflows/agentic/runs/surgical_reach_psm/bridge.log
```

The dry-run gate only proves registration. The bridge/scene build gate must
prove that the scene imports, builds, renders cameras, and has stable robot and
table placement.
