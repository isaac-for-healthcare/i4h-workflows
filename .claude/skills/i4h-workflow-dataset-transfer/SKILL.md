---
name: i4h-workflow-dataset-transfer
description: Augment HDF5 camera streams with NVIDIA Cosmos Transfer for visual diversity. Use when the user asks to apply Cosmos, domain-randomize videos, or make sim look photoreal. Requires Docker and GPU.
---

# i4h Workflow — Cosmos Transfer

## Basics

- Cosmos modifies videos only; robot states and actions are unchanged.
- Requires Docker with NVIDIA GPU support and accepted Cosmos model licenses.
- For action/state augmentation use [[i4h-workflow-dataset-mimic]] instead.

## Run

Component scripts live under `workflows/agentic/cosmos/`. Cosmos CLI flags change more often than the core workflow — consult `--help` for the current options.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
ENV_ID=scissor_pick_and_place
HDF5_PATH="${REPO_ROOT}/workflows/agentic/runs/<run>/data/demo.hdf5"
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/cosmos_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
OUT="${RUN_DIR}/data/demo_cosmos.hdf5"

"${REPO_ROOT}/workflows/agentic/cosmos/run.sh" --help
```

## Verify

- Output HDF5 preserves the original state/action groups.
- Camera streams are present and playable.
- Replay or convert + viz the output before declaring done ([[i4h-workflow-dataset-replay]], [[i4h-lerobot-viz]]).

## Final Response

Report input HDF5, output HDF5, camera streams modified, Docker or model-license blockers.
