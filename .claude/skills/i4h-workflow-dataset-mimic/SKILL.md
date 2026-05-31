---
name: i4h-workflow-dataset-mimic
description: Expand an HDF5 recording by replicating trajectories with small action and state noise. Use when the user asks to mimic, expand, or augment a dataset without recording new episodes.
---

# i4h Workflow — Mimic Dataset

## Basics

- Mimic perturbs action/state, not visuals. For visual variation use [[i4h-workflow-dataset-transfer]].
- Default `--include-source` keeps the original demos in the output.

## Run

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
ENV_ID=scissor_pick_and_place
IN="${REPO_ROOT}/workflows/agentic/runs/<run>/data/demo.hdf5"
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/mimic_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
OUT="${RUN_DIR}/data/demo_mimic.hdf5"

"${REPO_ROOT}/workflows/agentic/mimic/run.sh" --env "${ENV_ID}" \
  --input "${IN}" \
  --output "${OUT}" \
  --episodes 3 \
  --noise-std 0.01 \
  --include-source \
  --overwrite \
  2>&1 | tee "${RUN_DIR}/logs/mimic.log"
```

## Verify

```bash
"${REPO_ROOT}/workflows/agentic/mimic/run.sh" inspect --input "${OUT}"
```

Confirm the output episode count and that state/action dimensions match the source.

## Final Response

Report input path, output path, generated episode count, noise std, whether source demos were included.
