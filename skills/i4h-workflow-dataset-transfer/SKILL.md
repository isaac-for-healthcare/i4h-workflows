---
name: i4h-workflow-dataset-transfer
version: "0.6.0"
description: Augment HDF5 camera streams with NVIDIA Cosmos Transfer for visual diversity. Use when the user asks to apply Cosmos, domain-randomize videos, or make sim look photoreal. Requires Docker and GPU.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - cosmos
    - domain-randomization
---

# i4h Workflow — Cosmos Transfer

## Purpose

Augment HDF5 camera streams with NVIDIA Cosmos Transfer for visual diversity. Use when the user asks to apply Cosmos, domain-randomize videos, or make sim look photoreal.

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

- **Env config (source of truth):** `workflows/agentic/config/environments/<env>.yaml` — the camera streams (`zenoh.camera_names`) Cosmos augments.
- Cosmos modifies videos only; robot states and actions are unchanged.
- Requires Docker with NVIDIA GPU support and accepted Cosmos model licenses.
- For action/state augmentation use [[i4h-workflow-dataset-mimic]] instead.

## Run

Component scripts live under `workflows/agentic/cosmos/`. Cosmos CLI flags change more often than the core workflow — consult `--help` for the current options.

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"

# Point HDF5_PATH at a real recording to augment (absolute path). Recordings come from teleop,
# mimic, or validate (data/verify.hdf5 under each runs/eval_* dir). List candidates newest-first:
#   find "${RUNS_ROOT}" -name '*.hdf5' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort -r | head
HDF5_PATH="${HDF5_PATH:-}"
if [ ! -f "${HDF5_PATH}" ]; then
  echo "cosmos: set HDF5_PATH to an existing .hdf5 (got '${HDF5_PATH:-<unset>}'). Candidates:" >&2
  find "${RUNS_ROOT}" -name '*.hdf5' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort -r | head
  exit 1
fi

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

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist).
- Docker with NVIDIA GPU support and accepted Cosmos model licenses.
- An existing HDF5 recording to augment (set `HDF5_PATH` to an absolute path; the Run block lists candidates if it's unset or wrong).
- Component scripts under `workflows/agentic/cosmos/`; check `run.sh --help` for current flags.

## Limitations

- Cosmos modifies videos only; robot states and actions are unchanged.
- Requires Docker with NVIDIA GPU support and accepted Cosmos model licenses.
- For action/state augmentation use [[i4h-workflow-dataset-mimic]] instead.
- Cosmos CLI flags change more often than the core workflow; consult `--help` for the current options.

## Troubleshooting

- **Error:** `.venv` not found / module import fails - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** Docker / GPU not available - Cause: Docker lacks NVIDIA GPU support. Fix: ensure Docker with NVIDIA GPU support is installed and running.
- **Error:** model license not accepted - Cause: Cosmos model licenses not accepted. Fix: accept the required Cosmos model licenses.
- **Error:** input HDF5 not found - Cause: wrong or missing `--hdf5-path`. Fix: point `HDF5_PATH` at an existing recording.

## Final Response

Report input HDF5, output HDF5, camera streams modified, Docker or model-license blockers.
