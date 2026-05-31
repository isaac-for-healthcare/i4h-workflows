---
name: i4h-workflow-setup
description: Verify host requirements and run `workflows/agentic/setup.sh`. Use when the user asks to set up, install, or bootstrap the agentic workflow, or hits missing `.venv`, third-party checkout, or engine errors.
---

# i4h Workflow — Setup

## Basics

- `workflows/agentic/setup.sh` is the idempotent setup entry point.
- Cosmos setup is separate; invoke only when the user asks for Cosmos or video transfer.

## Preflight

```bash
command -v uv
command -v git
nvidia-smi
df -h .
```

Required: Linux, `uv`, `git`, NVIDIA driver/GPU, disk space for third-party checkouts. Docker is optional unless using Cosmos or local VLM containers.

## Run

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/setup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"

"${REPO_ROOT}/workflows/agentic/setup.sh" 2>&1 | tee "${RUN_DIR}/logs/setup.log"
```

For component-specific retries:

```bash
"${REPO_ROOT}/workflows/agentic/third_party/setup.sh"
"${REPO_ROOT}/workflows/agentic/policy/gr00t_n16/setup.sh"
```

## Verify

```bash
"${REPO_ROOT}/workflows/agentic/policy/run.sh" --list-envs
"${REPO_ROOT}/workflows/agentic/arena/run.sh" --help
"${REPO_ROOT}/workflows/agentic/policy/run.sh" --env scissor_pick_and_place --dry-run
"${REPO_ROOT}/workflows/agentic/arena/run.sh" --env scissor_pick_and_place --dry-run
```

## Final Response

Report setup status, failed component (if any), relevant log path, next recommended smoke test.
