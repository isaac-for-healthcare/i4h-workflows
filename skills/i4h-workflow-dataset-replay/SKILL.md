---
name: i4h-workflow-dataset-replay
description: Replay recorded HDF5 episodes in the workflow's Arena environment with replay_demos.py. Use for visual trajectory and recording verification; do not use for policy evaluation or LeRobot data.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-15"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - replay
    - hdf5
---

# Replay a Workflow Recording

## Purpose

Restore each selected episode's recorded initial state, apply its stored actions in the matching IsaacLab-Arena environment with `./run.sh <workflow> --replay`, which runs Isaac Lab's `replay_demos.py`, and inspect the result visibly.

## Instructions

1. Run the checkout resolver and select the exact HDF5.
2. Identify the workflow or scene and the options the recording was made with.
3. Map the requested episode to a replay index.
4. Run the replay visibly in the foreground through completion.

## Resolve the recording

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
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Use an explicit or current-chain recording first. Otherwise select the newest plausible HDF5 and state that choice. Never substitute an older file after a failed recording.

## Inspect before launch

```bash
HDF5_PATH=/absolute/path/to/recording.hdf5
uv run --project tools/dataset i4h-dataset inspect "$HDF5_PATH"
uv run --project tools/dataset python -c 'import h5py, json, sys; d = h5py.File(sys.argv[1], "r")["data"]; print(json.loads(d.attrs["env_args"]).get("env_name")); print(list(d))' "$HDF5_PATH"
```

The second command prints the scene name `record_demos.py` stores in `env_args` (policy-runner recordings leave it empty, so take the scene from the run context, `runs/<workflow>/...`) and the demo names in the order replay indexes them. `./run.sh list` maps the scene to its workflow id (`soarm_scissors` is `scissor_pick_and_place`); `run.sh` also accepts the bare scene name for replay. Confirm the requested episode exists and has a top-level `actions` dataset.

The workflow declaration supplies its launcher options, such as Newton and `PXR_WORK_THREAD_LIMIT=1` for the tablecloth scenes. Pass the options the recording was made with, from its run context or the workflow's docs page: `--presets newton` for any other Newton run, the same `--patient-twin` for endoluminal recordings, `-- --embodiment g1_locomanip_wbc_pink` for a G1 keyboard recording (23-D actions), and `--episode-steps` when the recording ran longer than the scene's default budget.

Interpret natural ordinals as zero-based demo numbers: first is `demo_0`, second is `demo_1`. `--episode`, which `run.sh` passes as `replay_demos.py --select_episodes`, takes positions in the printed order, which is lexicographic (`demo_0`, `demo_1`, `demo_10`, `demo_11`, `demo_2`, ...); with ten demos or fewer the position equals the demo number, otherwise use the demo's position in that list.

## Replay

```bash
./run.sh <workflow> --replay "$HDF5_PATH" --episode <index> -- --validate_success_rate
```

Omit `--episode` to replay every episode. Options after `--` go to `replay_demos.py`; add `--validate_states` there to compare the recorded `states` step by step in a single environment. `run.sh` replays without cameras: Arena's registration callback starts Kit before Isaac Lab can enable camera rendering, so camera sensors would fail. Replay is visible unless `--headless` is given; keep it visible for a visual check. `./run.sh show` prints the `replay_demos.py` command.

Keep the visible simulator and command in the foreground; `B` pauses and `N` resumes. Poll yielded execution until `run.sh` exits; do not detach or return while replay is still running. Stop it with Ctrl+C, or with `./stop.sh <workflow>` if it was started in the background.

## Verify

Observe the complete trajectory, relevant objects, and final state, and require `Finished replaying <N> episodes`. With `--validate_success_rate`, report `Successfully replayed: <X>/<Y>` and any failed demo IDs; scenes without a success term, such as the tablecloth scenes, print that none was found. If motion diverges, report the first diverging step or the action-width error. Do not change workflow, episode, or recording silently.

## Troubleshooting

On launch failure, verify the workflow or scene, its options, the episode index, and the action width against the environment the recording came from. A reach episode replays the same arm motion toward a newly sampled target, because targets are sampled commands rather than scene state. Cloth replay does not restore a deformed mesh, and contact-rich trajectories may diverge. Episodes produced by `tools/mimic` keep their source's `states`, so `--validate_states` reports their jitter as mismatches.

## Prerequisites

Require `common/.venv` and `arena/.venv` from `./setup.sh`, the workflow's assets and any patient twin it was recorded with, a display for the visible replay, and a readable episode with top-level `actions` whose width matches the environment.

## Limitations

Replay verifies stored actions in one environment; it does not evaluate a learned policy, re-render recorded frames, or guarantee transfer to another environment.

## Examples

- `Replay the second episode.` → select the current recording, map "second" to `demo_1` and its position in replay order, run `./run.sh <workflow> --replay <recording> --episode <index>`, and observe the visible replay.

## Completion gate

Report workflow, environment class and scene, options passed, HDF5 path, demo name and replay index, frame count, action width, the replay summary lines and exit status, and visible agreement or the first mismatch.
