---
name: i4h-workflow-dataset-replay
version: "0.6.0"
description: Replay a recorded HDF5 episode inside Isaac Sim for visual verification. Use when the user asks to replay, play back, or step through an HDF5 recording.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - replay
    - hdf5
---

# i4h Workflow — Replay Dataset

## Purpose

Replay a recorded HDF5 episode inside Isaac Sim for visual verification. Use when the user asks to replay, play back, or step through an HDF5 recording.

## Base Code

These steps drive the i4h-workflows base code (the `workflows/agentic/` tree). To reuse an existing checkout, set `I4H_WORKFLOWS` to its path (no clone happens). Otherwise this resolves the current repo, or clones to `~/i4h-workflows` — pick that default without prompting. Run every command below from the resolved root:

```bash
# Resolve the i4h-workflows base code (provides workflows/agentic/).
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/agentic" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/agentic" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

- Replay runs `arena/run.sh --replay` against the env that produced the HDF5.
- Use it to verify visual correctness before conversion or training.

## Run

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
HDF5_PATH="${REPO_ROOT}/workflows/agentic/runs/<run>/data/demo.hdf5"
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/replay_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"

"${REPO_ROOT}/workflows/agentic/arena/run.sh" \
  --env "${ENV_ID}" \
  --replay "${HDF5_PATH}" \
  --episode-index 0 \
  2>&1 | tee "${RUN_DIR}/logs/replay.log"
```

## Notes

- `--episode-index` selects the episode within the HDF5 (zero-based).
- Use the same env id as the env that produced the recording.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist).
- An existing HDF5 recording to replay.
- The env id that produced the recording.

## Limitations

- Visual verification only; replay does not modify or expand the recording.
- Replays one episode per invocation, selected by `--episode-index`.
- Runs inside Isaac Sim; the env id must match the one that produced the HDF5.

## Troubleshooting

- **Error:** `.venv` not found / replay fails to launch - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** recording fails to load - Cause: wrong or missing `--replay` HDF5 path. Fix: point to the existing recording file.
- **Error:** episode index out of range - Cause: `--episode-index` exceeds the episodes in the HDF5. Fix: use a valid zero-based index.
- **Error:** mismatched/garbled playback - Cause: `--env` differs from the env that produced the recording. Fix: use the same env id.

## Final Response

Report env, HDF5 path, episode index, launch outcome, visible mismatches.
