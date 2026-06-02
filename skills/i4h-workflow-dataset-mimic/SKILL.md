---
name: i4h-workflow-dataset-mimic
version: "0.6.0"
description: Expand an HDF5 recording by cloning trajectories with action/state noise. Use when asked to mimic, expand, or augment a dataset; not for recording new demos (use [[i4h-workflow-dataset-teleop]]).
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - mimic
    - augmentation
---

# i4h Workflow — Mimic Dataset

## Purpose

Expand an HDF5 recording by replicating trajectories with small action and state noise. Use when the user asks to mimic, expand, or augment a dataset without recording new episodes.

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

- **Env config (source of truth):** `workflows/agentic/config/environments/<env>.yaml` defines the `<env>` robot and task the mimicked trajectories replay against.
- Mimic perturbs action/state, not visuals. For visual variation use [[i4h-workflow-dataset-transfer]].
- Default `--include-source` keeps the original demos in the output.

## Run

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"

# Point IN at a real recording to expand (absolute path). Recordings come from teleop or
# validate (which writes data/verify.hdf5 under each runs/eval_* dir). List candidates newest-first:
#   find "${RUNS_ROOT}" -name '*.hdf5' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort -r | head
IN="${IN:-}"
if [ ! -f "${IN}" ]; then
  echo "mimic: set IN to an existing .hdf5 (got '${IN:-<unset>}'). Candidates:" >&2
  find "${RUNS_ROOT}" -name '*.hdf5' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort -r | head
  exit 1
fi

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
uv --directory "${REPO_ROOT}/workflows/agentic/mimic" run python -c \
  "import h5py; print('episodes:', len(h5py.File('${OUT}','r')['data']))"
```

Confirm the output episode count equals the source demos plus `--episodes`.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist).
- An existing input HDF5 recording (`--input`).
- The env id that produced the recording.

## Limitations

- Perturbs action/state only, not visuals; for visual variation use [[i4h-workflow-dataset-transfer]].
- Augments an existing recording rather than recording new episodes.
- Output state/action dimensions must match the source.

## Troubleshooting

- **Error:** `.venv` not found / mimic fails to launch - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** input recording not found - Cause: wrong or missing `--input` HDF5 path. Fix: point to the existing recording file.
- **Error:** output already exists / write refused - Cause: `--output` path is occupied. Fix: choose a new path or pass `--overwrite`.
- **Error:** state/action dimension mismatch on inspect - Cause: `--env` differs from the env that produced the input. Fix: use the same env id as the source recording.

## Final Response

Report input path, output path, generated episode count, noise std, whether source demos were included.
