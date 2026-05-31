---
name: i4h-workflow-dataset-annotate
description: Use a VLM to verify whether each episode satisfies the env's task description. Use when the user asks to annotate, label episodes, filter demos, or gate finetuning on a success classifier.
---

# i4h Workflow — Annotate Dataset

## Basics

- Annotation is optional. Do not run it during validation unless the user requests labels.
- The annotator reads task text from env YAML. Pass `--task-description` to override.
- Default endpoint is an OpenAI-compatible vLLM server at `localhost:8000/v1`.

## Start VLM

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
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

## Final Response

Report env, input HDF5, annotations path, filtered HDF5 (if any), success/failure counts, VLM blockers.
