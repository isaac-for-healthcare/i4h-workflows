---
name: i4h-workflow-dataset-replay
description: Replay a recorded HDF5 episode inside Isaac Sim for visual verification. Use when the user asks to replay, play back, or step through an HDF5 recording.
---

# i4h Workflow — Replay Dataset

## Basics

- Replay runs `arena/run.sh --replay` against the env that produced the HDF5.
- Use it to verify visual correctness before conversion or training.

## Run

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
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

## Final Response

Report env, HDF5 path, episode index, launch outcome, visible mismatches.
