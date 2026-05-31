---
name: i4h-workflow-dataset-teleop
description: Record episodes for an agentic env via teleoperation (keyboard, SO-ARM leader, or VR) into HDF5. Use when the user wants to teleop or record human demos.
---

# i4h Workflow — Teleop Record

## Basics

- Teleop runs through `arena/run.sh --teleop`.
- Device support is env-specific. Check `arena/run.sh --env <env> --help` for valid `--teleop-device` values.

## Controls

Reserved keys (consistent across devices):

| Key | Action |
|---|---|
| `B` | Start episode |
| `N` | Mark success, save, advance |
| `R` | Discard, reset |
| `F` | Reserved by Isaac Sim — do not bind |

Device-specific keybindings (move, rotate, gripper, mode switches) are printed by the teleop process at startup and vary by `--teleop-device`. Report them to the user from the log; they cannot drive the sim without them. See "Surface Device Keybindings".

Stop from terminal:

```bash
workflows/agentic/stop.sh arena --env <env>
```

## Known Devices

| Env | Devices |
|---|---|
| `scissor_pick_and_place` | `keyboard`, `so101_leader` |
| `locomanip_tray_pick_and_place` | `keyboard_23d` |
| `locomanip_push_cart` | `keyboard_23d` |

For other envs, consult `arena/run.sh --env <env> --help`.

## Run

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
ENV_ID=scissor_pick_and_place
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/teleop_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"

"${REPO_ROOT}/workflows/agentic/arena/run.sh" \
  --env "${ENV_ID}" \
  --teleop \
  --teleop-device <device> \
  --episodes 3 \
  --record-to "${RUN_DIR}/data/demo.hdf5" \
  2>&1 | tee "${RUN_DIR}/logs/teleop.log"
```

## Surface Device Keybindings

The teleop process prints its keybinding table to stdout shortly after launch (look for sections such as `Keybindings`, `Controls`, `Key Map`, mode-switch lines, or any block enumerating keys → actions). Background the run, wait for the table to appear, extract it from the log, and report it to the user before they need to drive the sim.

```bash
# After launching teleop in the background and tailing the log:
sed -n '/Keybind\|Controls\|Key Map\|Current Mode\|Mode\] Switched/,/^$/p' "${RUN_DIR}/logs/teleop.log"
```

If the block is multi-section (e.g. `BOTH_HANDS`, `BASE_NAV`, `LEFT_HAND`, `RIGHT_HAND` modes), include every mode in the report. Append the reserved keys above so the user has one consolidated reference.

## Notes

- `--record-to` must be absolute. The recorder resolves relative paths against `workflows/agentic/arena` (its CWD) and writes to a nested orphan dir. `${RUN_DIR}/data/demo.hdf5` built from `${REPO_ROOT}` is absolute.
- Use `--save-all-episodes` only when failed attempts must be kept.

## Verify

- `${RUN_DIR}/data/demo.hdf5` exists.
- Log contains `run complete: N/M episodes succeeded`.

## Final Response

Report env, device, the device-specific keybinding table extracted from the log, requested vs saved episodes, HDF5 path, log path.
