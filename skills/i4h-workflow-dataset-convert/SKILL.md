---
name: i4h-workflow-dataset-convert
description: Convert Isaac Lab HDF5 recordings of i4h workflows to LeRobot datasets for training or browser inspection. Use for conversion; do not use for replay, augmentation, or raw-data repair.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-15"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - hdf5
    - lerobot
---

# Convert Workflow HDF5 to LeRobot

## Purpose

Preserve recorded actions, state, cameras, task text, and embodiment labels in a local LeRobot dataset with `tools/dataset`.

## Instructions

1. Run the checkout resolver and select the source HDF5 recordings.
2. Resolve the workflow, embodiment, task text, and any patient twins.
3. Run conversion for the selected successful episodes.
4. Inspect metadata, parquet, videos, and feature widths.

## Resolve and inspect

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
find runs -name '*.hdf5' -type f -printf '%T@ %p\n' | sort -nr | head
HDF5_PATH=/absolute/path/to/recording.hdf5
uv run --project tools/dataset i4h-dataset inspect "$HDF5_PATH"
uv run --project tools/dataset python -c 'import h5py, json, sys; d = h5py.File(sys.argv[1], "r")["data"]; print(json.loads(d.attrs["env_args"]).get("env_name")); demo = d[next(iter(d))]; print(sorted(demo)); print(sorted(demo.get("obs", {})))' "$HDF5_PATH"
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Use the explicit or current-chain recording; otherwise select the newest plausible HDF5 and state that choice. The second command prints the scene name `record_demos.py` stores in `env_args` (empty for policy-runner recordings, so take the scene from the run context), then one demo's datasets and observations. Map the scene to its workflow with `docs/workflows/<specialty>.md`.

Take `--robot` from the `embodiment` of the policy task the dataset will train, `tasks/<stack>/i4h_tasks/<stack>/manifest/<task>.yaml`, or else from the scene manifest `arena/i4h_arena/scenes/manifest/<scene>.yaml`. Take `--task` from that task manifest's `prompt`, or its `summary` when it has none, as `scripts/e2e/run.sh` does, unless the user supplies the instruction. The embodiment manifest `arena/i4h_arena/embodiments/manifest/<robot>.yaml` names the columns. Do not assume state width equals action width; the converter derives both from the recording.

The converter reads state from `obs/joint_pos`, which every i4h embodiment records into a dataset, and stores the action as the state for a demo without it. Check the printed datasets and report a missing `obs/joint_pos` rather than presenting the action copy as measured state.

## Convert

```bash
RUN_DIR="$(pwd)/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
DATASET_DIR="$RUN_DIR/lerobot/local/<name>"
mkdir -p "$(dirname "$DATASET_DIR")"
[ ! -e "$DATASET_DIR" ] || { echo "Destination already exists: $DATASET_DIR" >&2; exit 2; }
uv run --project tools/dataset i4h-dataset convert \
  "$HDF5_PATH" "$DATASET_DIR" \
  --robot <embodiment> \
  --repo-id "local/<name>" \
  --successful-only \
  --task "<instruction>"
```

Pass several recordings before the output path to write one dataset with shared normalization statistics; they must come from the same embodiment. Use `--fps` or `--skip-frames` only when the user requests it or source metadata justifies it. Keep the default H.264 video codec for compatibility with GR00T's decord loader; select another `--video-codec` only when the target consumer requires it.

Endoluminal embodiments (`catheter`, `franka_catheter`) append five navigation goal columns derived from each demo's `diagnostics/tip_world_m` and the patient twin's route. Pass `--patient-twin` with one manifest per recording, in the order the recordings are given; Isaac Lab recordings have no `run.json` beside them, which the converter reads only as a fallback for recordings from the removed `run.sh` launcher.

Conversion writes aggregate `meta/stats.json` for downstream policy loaders. It writes GR00T's semantic `meta/modality.json` when the embodiment manifest's `state_split` and `action_split` tile the converted widths, or for `--robot g1` with 43-D state and 50-D action (a camera recorded as `head` is renamed `ego_view`). A descriptor whose declared splits do not match the recording fails before any video is encoded. `--g1-wbc-policy-actions` maps a 23-D G1 keyboard recording (made with `--embodiment g1_locomanip_wbc_pink`) plus 43-D measured `obs/joint_pos` to the 50-D G1 policy action; it fails when that state is missing.

## Verify

Require:

- `meta/info.json`
- `meta/stats.json`
- `meta/modality.json` when the target trainer requires semantic modality slices
- episode parquet data
- video files for every recorded camera
- converted episode count matching the selected successful sources
- action/state feature widths and names matching the recording plus embodiment descriptor, and state taken from `obs/joint_pos`

Treat missing inputs or zero converted episodes as failure. If conversion leaves a partial destination, quarantine or remove that exact incomplete directory before retrying; never report it as usable.

## Troubleshooting

On dimension errors, resolve the source workflow and embodiment again. On missing videos, confirm the recording holds `obs/<camera>` frames; a camera scene recorded without `--enable_cameras` has none. On a patient-twin error, pass one existing `--patient-twin` per recording, and confirm each demo has `diagnostics/tip_world_m`.

## Prerequisites

Require `./setup.sh` to have built `tools/dataset`, a readable HDF5 recording, its embodiment manifest, and, for endoluminal recordings, the patient twin each was recorded with.

## Limitations

Conversion cannot reconstruct missing cameras, actions, state, task text, goal diagnostics, or successful episodes.

## Examples

- `Convert my scissor pick-and-place recording into a LeRobot dataset.` → resolve `so101` and the task prompt from the scissor task manifest, convert successful episodes, and verify metadata, parquet, and the `room` and `wrist` videos.

## Completion gate

Report source HDF5 and workflow, embodiment, task text, state source, patient twins when used, source/converted/skipped counts, action/state widths, output directory and repo id, aggregate-stats/modality/parquet/video checks, and any missing modality or state.
