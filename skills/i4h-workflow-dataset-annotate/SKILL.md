---
name: i4h-workflow-dataset-annotate
version: "0.6.0"
description: Use a VLM to verify whether each episode satisfies the env's task description. Use when the user asks to annotate, label episodes, filter demos, or gate finetuning on a success classifier.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - annotation
    - vlm
---

# i4h Workflow — Annotate Dataset

## Purpose

Use a VLM to verify whether each episode satisfies the env's task description. Use when the user asks to annotate, label episodes, filter demos, or gate finetuning on a success classifier.

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

- Annotation is optional. Do not run it during validation unless the user requests labels.
- The annotator reads task text from env YAML. Pass `--task-description` to override.
- Default endpoint is an OpenAI-compatible vLLM server at `localhost:8000/v1`.

## Start VLM

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
if ! "${REPO_ROOT}/workflows/agentic/annotator/vllm.sh" status; then
  "${REPO_ROOT}/workflows/agentic/annotator/vllm.sh" start &
fi
until "${REPO_ROOT}/workflows/agentic/annotator/vllm.sh" status; do
  sleep 1
done
```

## Run (Offline HDF5)

```bash
ENV_ID=scissor_pick_and_place
HDF5_PATH="${REPO_ROOT}/workflows/agentic/runs/<run>/data/demo.hdf5"
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/annotate_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"

"${REPO_ROOT}/workflows/agentic/annotator/run.sh" \
  --env "${ENV_ID}" \
  --output "${RUN_DIR}/annotations.jsonl" \
  offline \
  --hdf5-path "${HDF5_PATH}" \
  --filter "${RUN_DIR}/data/filtered.hdf5"

"${REPO_ROOT}/workflows/agentic/annotator/vllm.sh" stop
```

## Live Mode

Use only when a policy/Arena session is already running and the user requests live judging.

## Verify

- `annotations.jsonl` exists.
- Filtered HDF5 exists when `--filter` was passed.
- Tally success/failure counts from the JSONL before reporting.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist).
- An existing HDF5 recording to annotate (e.g. `runs/<run>/data/demo.hdf5`).
- A running VLM server; start it with `annotator/vllm.sh start` (default endpoint `localhost:8000/v1`).
- Annotation is optional — only run it when the user requests labels.

## Limitations

- Annotation is optional and is not run during validation unless requested.
- Requires a reachable OpenAI-compatible vLLM server; defaults to `localhost:8000/v1`.
- Live mode applies only when a policy/Arena session is already running and the user requests live judging.
- The annotator reads task text from the env YAML; override per-run with `--task-description`.

## Troubleshooting

- **Error:** `.venv` not found / module import fails - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** connection refused at `localhost:8000/v1` - Cause: VLM server not running. Fix: start it with `annotator/vllm.sh start` and wait until `vllm.sh status` succeeds.
- **Error:** input HDF5 not found - Cause: wrong or missing `--hdf5-path`. Fix: point `HDF5_PATH` at an existing recording.
- **Error:** filtered HDF5 missing - Cause: `--filter` was not passed. Fix: add `--filter <path>` to write the filtered dataset.

## Final Response

Report env, input HDF5, annotations path, filtered HDF5 (if any), success/failure counts, VLM blockers.
