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
- **Env config (source of truth):** the annotator reads the success criterion (`policy.task_description`) from `workflows/agentic/config/environments/<env>.yaml`. Pass `--task-description` to override.
- Talks to an OpenAI-compatible vLLM via `--base-url` (default `http://localhost:8000/v1`) and `--model` (default `Qwen/Qwen3-VL-8B-Instruct`). Point both at any running vision-model server — including the local-agent one (`qwen3-vl-32b` on `:8000`).

## Start VLM

> **Skip this section if an OpenAI-compatible vLLM serving a vision model is already running** (e.g. the local-agent server on `:8000`) — just set `VLM_BASE_URL`/`VLM_MODEL` in Run to point at it. `annotator/vllm.sh` also defaults to port `8000`, so starting it on top of an existing server collides; don't.

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
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"

# LLM endpoint + model (OpenAI-compatible vLLM). Defaults match annotator/vllm.sh; override to use an
# external server — e.g. the local-agent one: VLM_BASE_URL=http://localhost:8000/v1 VLM_MODEL=qwen3-vl-32b
VLM_BASE_URL="${VLM_BASE_URL:-http://localhost:8000/v1}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

# Point HDF5_PATH at a real recording (absolute path). Recordings come from teleop, mimic, or
# validate (which writes data/verify.hdf5 under each runs/eval_* dir). List candidates newest-first:
#   find "${RUNS_ROOT}" -name '*.hdf5' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort -r | head
HDF5_PATH="${HDF5_PATH:-}"
if [ ! -f "${HDF5_PATH}" ]; then
  echo "annotate: set HDF5_PATH to an existing .hdf5 (got '${HDF5_PATH:-<unset>}'). Candidates:" >&2
  find "${RUNS_ROOT}" -name '*.hdf5' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort -r | head
  exit 1
fi

RUN_DIR="${RUNS_ROOT}/annotate_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"

"${REPO_ROOT}/workflows/agentic/annotator/run.sh" \
  --env "${ENV_ID}" \
  --base-url "${VLM_BASE_URL}" \
  --model "${VLM_MODEL}" \
  --output "${RUN_DIR}/annotations.jsonl" \
  offline \
  --hdf5-path "${HDF5_PATH}" \
  --filter "${RUN_DIR}/data/filtered.hdf5"

# Stop the annotator's own vLLM ONLY if you started it in "Start VLM" above.
# Skip this when using an external server (e.g. the local-agent one) — it would kill that server.
# "${REPO_ROOT}/workflows/agentic/annotator/vllm.sh" stop
```

## Live Mode

Annotate the latest camera frames from a **running** policy/Arena session over Zenoh (cameras default to the env config). Use only when such a session is already up and the user asks for live judging.

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
VLM_BASE_URL="${VLM_BASE_URL:-http://localhost:8000/v1}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
RUN_DIR="${RUNS_ROOT}/annotate_live_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

"${REPO_ROOT}/workflows/agentic/annotator/run.sh" \
  --env "${ENV_ID}" \
  --base-url "${VLM_BASE_URL}" \
  --model "${VLM_MODEL}" \
  --output "${RUN_DIR}/live.jsonl" \
  live \
  --count 5 \
  --interval 2.0 \
  --timeout 30.0
```

- `--count 0` runs forever; `--interval` is seconds between snapshots; `--timeout` is how long to wait for first frames from every camera.
- `--min-success-frames N` (needs a finite `--count`) exits non-zero unless at least N sampled snapshots pass — use it as a gate.
- `--dump-frames-dir DIR` saves sampled frames; add `--dump-frames-only` to dump without calling the VLM.
- `--cameras a,b` overrides the env's Zenoh camera names.

## Verify

- `annotations.jsonl` exists.
- Filtered HDF5 exists when `--filter` was passed.
- Tally success/failure counts from the JSONL before reporting.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist).
- An existing HDF5 recording to annotate (set `HDF5_PATH` to an absolute path; the Run block lists candidates if it's unset or wrong).
- A reachable OpenAI-compatible vLLM serving a vision model — either start the annotator's own (`annotator/vllm.sh start`) or point `VLM_BASE_URL`/`VLM_MODEL` at an existing one (e.g. the local-agent server).
- Annotation is optional — only run it when the user requests labels.

## Limitations

- Annotation is optional and is not run during validation unless requested.
- Requires a reachable OpenAI-compatible vLLM server; defaults to `localhost:8000/v1`.
- Live mode applies only when a policy/Arena session is already running and the user requests live judging.
- The annotator reads task text from the env YAML; override per-run with `--task-description`.

## Troubleshooting

- **Error:** `.venv` not found / module import fails - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** connection refused at `localhost:8000/v1` - Cause: no vLLM at `VLM_BASE_URL`. Fix: start one (`annotator/vllm.sh start`) or set `VLM_BASE_URL`/`VLM_MODEL` to a running server.
- **Error:** model not found / 404 from the endpoint - Cause: `VLM_MODEL` is not the id the server actually serves. Fix: set `VLM_MODEL` to the served name (e.g. `qwen3-vl-32b` for the local-agent server; check `curl ${VLM_BASE_URL}/models`).
- **Error:** input HDF5 not found - Cause: `HDF5_PATH` unset or not a real file. Fix: pick an absolute path from the candidates the Run block prints.
- **Error:** filtered HDF5 missing - Cause: `--filter` was not passed. Fix: add `--filter <path>` to write the filtered dataset.

## Final Response

Report env, input HDF5, annotations path, filtered HDF5 (if any), success/failure counts, VLM blockers.
