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

Drive a workflow's IsaacLab-Arena environment with a human-input device through Isaac Lab's `record_demos.py`, which writes each demonstration that reaches the environment's success termination to an Isaac Lab HDF5 dataset.

## Instructions

1. Resolve the checkout, the workflow's environment, and its teleop device.
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
ls arena/i4h_arena/scenes/manifest
grep -rn -A4 "teleop_devices = DevicesCfg" arena/i4h_arena/scenes
grep -rln "isaac_teleop" arena/i4h_arena/scenes
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Map the workflow id to its environment class and scene name with the tables in `docs/workflows/<specialty>.md` (for example `scissor_pick_and_place` is `SoarmScissors`, `soarm_scissors`), and read that page's teleop command and controls. The device name `--teleop_device` takes must be a key of `teleop_devices` in the scene's `configure_env_cfg`, which the first grep lists per scene module; subclasses inherit them (`_locomanip.py` serves `g1_cart` and `g1_tray`, `endoluminal_navigation.py` also serves `endoluminal_navigation_arm`).

The tablecloth scenes (`_tablecloth.py`) declare Isaac Teleop instead and take no `--teleop_device`. A scene that declares neither does not support teleoperation: say so instead of relying on Isaac Lab's built-in `keyboard` or `spacemouse` fallback, whose actions do not match i4h embodiments.

## Choose input

- Use the named device when the scene declares it; otherwise use the device on the workflow's docs page.
- Copy the device's scene options from that page: G1 `keyboard_23d` needs `--embodiment g1_locomanip_wbc_pink`; SO-ARM `so101_leader` takes `--leader_port` (default `/dev/ttyACM1`); endoluminal `catheter_keyboard` needs `--presets newton` and a prepared `--patient_twin`; tablecloth XR needs `--presets newton`, `PXR_WORK_THREAD_LIMIT=1`, and CloudXR running as `docs/workflows/tablecloth.md` describes.
- Surface the device controls from the docs page and require the human operator to complete the task. Never pretend to provide human input in an unattended shell.

## Record

```bash
RUN_DIR="$PWD/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
arena/.venv/bin/python third_party/IsaacLab-e3f7d9f/scripts/tools/record_demos.py \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --external_environment_class_path i4h_arena.environments:<Class> --task <scene> \
  --teleop_device <device> --enable_cameras --viz kit \
  --dataset_file "$RUN_DIR/teleop.hdf5" --num_demos <N>
```

Add `--enable_cameras` for scenes whose docs command passes it (camera scenes); the endoluminal commands record fluoroscopy without it. Add the device's scene options from above. Teleop requires `--viz kit`. `--num_demos` counts saved demos; `0` records until the session is stopped. Use a new run directory for each session, because the recorder creates the dataset file afresh.

Keep the session visible and in the foreground. If the agent shell must yield, start it in the background with its output in `$RUN_DIR/record.log`, record its PID, and wait on that PID. Stop a foreground session with Ctrl+C and a background one only by its recorded PID, never by tool or module name.

`record_demos.py` saves a demo only after the environment's success term has held for `--num_success_steps` consecutive steps (default 10); time-outs and operator resets (`R`) are discarded. Tablecloth scenes have no success term: the operator accepts each demo through Isaac Teleop's controls. An endoluminal session without a patient twin has no goal, so it can save nothing.

## Verify

Require `Recording session completed with <N> successful demonstrations` and `Demonstrations saved to: <path>`. Then inspect content:

```bash
uv run --project tools/dataset i4h-dataset inspect "$RUN_DIR/teleop.hdf5"
uv run --project tools/dataset i4h-dataset actions "$RUN_DIR/teleop.hdf5"
```

Each `data/demo_N` holds Isaac Lab's `actions`, `processed_actions`, `states`, `initial_state`, and the pre-step observations under `obs/` (camera frames as `obs/<camera>`), plus the scene's own terms such as the endoluminal `obs/joint_pos`, `obs/fluoroscopy`, and `diagnostics/`. Visually confirm that the operator completes the task, robot motion follows the device, and every expected camera records it. Treat zero saved demos, a missing camera, or absent action motion as failure.

## Troubleshooting

If `--teleop_device` is rejected, compare the name with the scene's `teleop_devices` and the options that register it (`--embodiment g1_locomanip_wbc_pink` for `keyboard_23d`). If keys do nothing, click inside the Isaac window; the catheter keyboard clears a stuck command with `L`. If no demo is saved, check that the success term can be reached in that scene and configuration.

## Prerequisites

Require `./setup.sh` to have built `arena/.venv`, a display for `--viz kit`, the physical device (leader arm on its serial port, XR headset with CloudXR), any patient twin the scene needs, and a present human operator.

## Limitations

`record_demos.py` writes only demos that reach success; it cannot keep failed or aborted attempts, and its demos carry no per-node `segments`. Record rule-based or policy episodes, which keep every episode with its success flag, with `i4h-workflow-validate` instead.

## Examples

- `Record 5 keyboard teleop demonstrations for locomanip tray pick and place.` → run `record_demos.py` on `G1Tray`/`g1_tray` with `--embodiment g1_locomanip_wbc_pink --teleop_device keyboard_23d --enable_cameras --viz kit --num_demos 5`, require a human operator, and inspect the saved demos.

## Completion gate

Report workflow, environment class and scene, device, options, and controls, requested and saved demo counts, HDF5 path, inspected frames, action width, and cameras, the visual result, how the session ended, and whether a human operator completed the task.
