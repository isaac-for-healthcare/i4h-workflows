---
name: i4h-workflow-dataset-teleop
description: Record teleoperated demonstrations in a workflow's Arena environment with record_demos.py. Use for keyboard, leader-arm, or XR input; do not use for policy evaluation or rule-based recording.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-15"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - teleoperation
    - hdf5
---

# Record Teleop Demonstrations

## Purpose

Drive a workflow's IsaacLab-Arena environment with a human-input device through `./run.sh <workflow> --teleop`, which runs Isaac Lab's `record_demos.py`. It writes each demonstration that reaches the environment's success termination to an Isaac Lab HDF5 dataset.

## Instructions

1. Resolve the checkout, the workflow, and its teleop device.
2. Choose the requested demo count and a fresh run directory.
3. Record in the foreground with a visible viewport while a human operator drives.
4. Inspect the saved demos.

## Resolve support

```bash
export I4H_WORKFLOWS_REPO_URL="${I4H_WORKFLOWS_REPO_URL:-https://github.com/isaac-for-healthcare/i4h-workflows}"
I4H_REPO_DIR_NAME="${I4H_WORKFLOWS_REPO_URL%/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*:}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME%.git}"
[ -n "$I4H_REPO_DIR_NAME" ] || { echo "Cannot derive a checkout name from I4H_WORKFLOWS_REPO_URL" >&2; exit 2; }
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/arena/i4h_arena/scenes/manifest" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/$I4H_REPO_DIR_NAME}"
  [ -d "$ROOT/arena/i4h_arena/scenes/manifest" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT"
./run.sh list
./run.sh show <workflow> --teleop
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Map the workflow id to a workflow `./run.sh list` shows, and read the teleop controls on its page in `docs/workflows/<specialty>.md`. Its declaration, `workflows/<specialty>/<workflow>.yaml`, lists the devices of its `teleop` mode, the first being the default, with any options a device needs; each is a key of `teleop_devices` in the scene's `configure_env_cfg`. `./run.sh show <workflow> --teleop [DEVICE]` prints the `record_demos.py` command.

The tablecloth workflows declare Isaac Teleop (`xr: true`) and take no device. A workflow without a `teleop` mode does not support teleoperation: say so instead of passing Isaac Lab's built-in `keyboard` or `spacemouse` device, whose actions do not match i4h embodiments.

## Choose input

- Use the named device when the workflow declares it; otherwise its default device. `run.sh` rejects a device the workflow does not declare.
- The declaration supplies the device's scene options, such as G1 `keyboard_23d`'s `--embodiment g1_locomanip_wbc_pink`, and Newton for the endoluminal and tablecloth scenes. Add what depends on the session: SO-ARM `so101_leader` takes `-- --leader_port <port>` (default `/dev/ttyACM1`); endoluminal `catheter_keyboard` needs a prepared `--patient-twin`; tablecloth XR needs CloudXR running as `docs/workflows/tablecloth.md` describes, or `--cloudxr-env` to start it.
- Surface the device controls from the docs page and require the human operator to complete the task. Never pretend to provide human input in an unattended shell.

## Record

```bash
RUN_DIR="$PWD/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
./run.sh <workflow> --teleop <device> --episodes <N> --run-dir "$RUN_DIR"
```

The demonstrations go to `$RUN_DIR/teleop.hdf5`; `--record <path>.hdf5` chooses another file. `run.sh` enables cameras when the scene declares them; the endoluminal scenes record fluoroscopy without them. Teleop requires a visible session, so `run.sh` refuses `--headless`. `--episodes` counts saved demos; without it the session records until it is stopped. Use a new run directory for each session, because the recorder creates the dataset file afresh.

Keep the session visible and in the foreground. If the agent shell must yield, start it in the background with its output in `$RUN_DIR/record.log` and wait on its PID. Stop a foreground session with Ctrl+C and a background one with `./stop.sh <workflow>`, never by tool or module name.

`record_demos.py` saves a demo only after the environment's success term has held for `--num_success_steps` consecutive steps (default 10); time-outs and operator resets (`R`) are discarded. Tablecloth scenes have no success term: the operator accepts each demo through Isaac Teleop's controls. An endoluminal session without a patient twin has no goal, so it can save nothing.

## Verify

Require `Recording session completed with <N> successful demonstrations` and `Demonstrations saved to: <path>`. Then inspect content:

```bash
uv run --project tools/dataset i4h-dataset inspect "$RUN_DIR/teleop.hdf5"
uv run --project tools/dataset i4h-dataset actions "$RUN_DIR/teleop.hdf5"
```

Each `data/demo_N` holds Isaac Lab's `actions`, `processed_actions`, `states`, `initial_state`, and the pre-step observations under `obs/` (camera frames as `obs/<camera>`), plus the scene's own terms such as the endoluminal `obs/joint_pos`, `obs/fluoroscopy`, and `diagnostics/`. Visually confirm that the operator completes the task, robot motion follows the device, and every expected camera records it. Treat zero saved demos, a missing camera, or absent action motion as failure.

## Troubleshooting

If `run.sh` rejects the device, use one the workflow declares; if `record_demos.py` rejects it, compare the declaration with the scene's `teleop_devices` and the options that register it (`--embodiment g1_locomanip_wbc_pink` for `keyboard_23d`). If keys do nothing, click inside the Isaac window; the catheter keyboard clears a stuck command with `L`. If no demo is saved, check that the success term can be reached in that scene and configuration.

## Prerequisites

Require `./setup.sh` to have built `common/.venv` and `arena/.venv`, a display for the visible session, the physical device (leader arm on its serial port, XR headset with CloudXR), any patient twin the scene needs, and a present human operator.

## Limitations

`record_demos.py` writes only demos that reach success; it cannot keep failed or aborted attempts, and its demos carry no per-node `segments`. Record rule-based or policy episodes, which keep every episode with its success flag, with `i4h-workflow-validate` instead.

## Examples

- `Record 5 keyboard teleop demonstrations for locomanip tray pick and place.` → run `./run.sh locomanip_tray_pick_and_place --teleop keyboard_23d --episodes 5` with a fresh run directory, which runs `record_demos.py` on `G1Tray`/`g1_tray` with `--embodiment g1_locomanip_wbc_pink`; require a human operator and inspect the saved demos.

## Completion gate

Report workflow, environment class and scene, device, options, and controls, requested and saved demo counts, HDF5 path, inspected frames, action width, and cameras, the visual result, how the session ended, and whether a human operator completed the task.
