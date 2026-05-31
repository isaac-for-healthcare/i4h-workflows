---
name: i4h-workflow-finetune
description: Fine-tune a GR00T or openpi PI0 policy on a LeRobot dataset. Use when the user asks to finetune, train, or post-train a policy on recorded demos.
---

# i4h Workflow — Finetune

## Basics

- The dataset path must be an existing LeRobot directory with `meta/info.json`.
- Train support is determined by `policy.train_module` in `config/environments/<env>.yaml`. A null value means inference-only.
- `assemble_trocar` is inference-only.

## Stack Map

| Env | Stack | CLI |
|---|---|---|
| `scissor_pick_and_place` | `gr00t_n15` | `i4h-agentic-gr00t-n15-train` |
| `locomanip_tray_pick_and_place` | `gr00t_n16` | `i4h-agentic-gr00t-n16-train` |
| `locomanip_push_cart` | `gr00t_n16` | `i4h-agentic-gr00t-n16-train` |
| `ultrasound_liver_scan` | `openpi_pi0` | `i4h-agentic-openpi-pi0-train` |

N1.6 locomanip envs share `policy.locomanip.train`.

## Preflight

```bash
test -f "${DATASET_PATH}/meta/info.json"
nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
workflows/agentic/policy/<stack>/run.sh --list-envs
```

## Run

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
ENV_ID=scissor_pick_and_place
STACK_DIR=gr00t_n15
TRAIN_CLI=i4h-agentic-gr00t-n15-train
DATASET_PATH="${REPO_ROOT}/workflows/agentic/runs/<run>/lerobot/local/${ENV_ID}"
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/finetune_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
OUT="${RUN_DIR}/checkpoint"
mkdir -p "${OUT}" "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"

uv --directory "${REPO_ROOT}/workflows/agentic/policy/${STACK_DIR}" run "${TRAIN_CLI}" \
  --env "${ENV_ID}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUT}" \
  --max-steps 1000 \
  --save-steps 1000 \
  --num-gpus 1 \
  2>&1 | tee "${RUN_DIR}/logs/finetune.log"
```

Tyro flags use kebab case (`--max-steps`, not `--max_steps`).

## Common Flags

- `--dataset-path PATH` (required)
- `--output-dir PATH`
- `--base-model-path PATH_OR_REPO` overrides YAML `policy.model_repo`
- `--max-steps N`, `--save-steps N`
- `--batch-size N`, `--learning-rate FLOAT`
- `--no-tune-visual` — freeze the vision backbone (trains the action head + projector only): ~2× faster, ~half the memory, less overfitting. Good default for small datasets; unfreeze only with lots of data + a real visual domain gap.
- `--num-gpus N` — must not exceed visible GPUs
- `--report-to tensorboard|wandb`

## Verify

- Checkpoint directory `${OUT}/checkpoint-<N>` contains `model-0000*-of-*.safetensors`, `experiment_cfg/`, `processor/`.
- Log contains `train_loss` lines and a final `'train_runtime': ...` summary.

## Final Response

Report env, stack, dataset path, output checkpoint path, train_loss summary, and blockers.
