---
name: i4h-workflow-finetune
description: Fine-tune a manifest-backed GR00T or openpi policy task on compatible LeRobot data. Use for training; do not use for inference-only tasks or checkpoint rollout.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-15"
  tags:
    - isaac-for-healthcare
    - i4h
    - lerobot
    - gr00t
    - openpi
---

# Fine-tune a Workflow Policy Task

## Purpose

Resolve and run training for the policy task that serves the selected workflow, from its owning task manifest.

## Instructions

1. Resolve the base checkout, the workflow's policy stack, and its task.
2. Verify dataset compatibility and a `train:` block.
3. Dry-run the exact configuration.
4. Train in the foreground, verify checkpoint artifacts, and load-check the checkpoint with the stack's server.

## Resolve the workflow, task, and data

```bash
export I4H_WORKFLOWS_REPO_URL="${I4H_WORKFLOWS_REPO_URL:-https://github.com/isaac-for-healthcare/i4h-workflows}"
I4H_REPO_DIR_NAME="${I4H_WORKFLOWS_REPO_URL%/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*:}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME%.git}"
[ -n "$I4H_REPO_DIR_NAME" ] || { echo "Cannot derive a checkout name from I4H_WORKFLOWS_REPO_URL" >&2; exit 2; }
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/arena/i4h_arena/scenes/manifest" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/$I4H_REPO_DIR_NAME}"
  [ -d "$ROOT/arena/i4h_arena/scenes/manifest" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT"
ls tasks/*/i4h_tasks/*/manifest/
test -f /absolute/path/to/dataset/meta/info.json
nvidia-smi
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

The workflow's page in `docs/workflows/` names the stack and task that serve its policy: `tasks/<stack>/.venv/bin/python -m i4h_tasks.<stack>.serve <task>` means task id `<stack>/<task>` (GR00T N1.6 tasks live under `tasks/gr00t_n16`). When a workflow has two stacks, use the one the user names, otherwise its ordinary policy: `scissor_pick_and_place` trains `gr00t_n15/scissor_pick_and_place`, and `gr00t_n17/scissor_pick_and_place` only when N1.7 is requested. Open `tasks/<stack>/i4h_tasks/<stack>/manifest/<task>.yaml` and require `train:`; a task without it, such as `gr00t_n15/assemble_trocar`, is inference-only. Resolve the base model/config, output defaults, and modality contract from that manifest and the stack's `train.py`.

Use the current-chain LeRobot dataset when the prompt omits a path. Verify its embodiment, cameras, task text, feature widths, and episode count are compatible with the task. For GR00T N1.6 G1 tasks the manifest's `train.modality_config_path` requests only the head camera (`ego_view`); the trainer's `--dry-run` checks that every requested video exists in the dataset.

## Resolve configuration before a long run

All policy train entry points support `--dry-run`:

```bash
uv run --project "tasks/<stack>" "i4h-tasks-<stack-with-hyphens>-train" \
  --task <stack>/<task> \
  --dataset /absolute/path/to/dataset \
  --output-dir /absolute/path/to/checkpoints \
  --max-steps <N> \
  --batch-size <N> \
  --dry-run
```

Inspect the resolved config. Keep user-requested steps, batch size, model/config, and GPU count exact.

For GR00T, “turn off vision tuning” maps to `--no-tune-visual`. Do not pass that flag to openpi, whose CLI does not expose it. Use only flags present in the selected stack's current `train.py`.

No trainer needs a Weights & Biases account. GR00T logs to TensorBoard. openpi logs to W&B only with `--wandb` or `train.wandb: true` in the task manifest, never because of the environment, and the dry-run prints the resolved `wandb` value and its reason. Add `--wandb` only when the user asks for W&B logging.

## Train

Remove `--dry-run` and keep the command in the foreground:

```bash
uv run --project "tasks/<stack>" "i4h-tasks-<stack-with-hyphens>-train" \
  --task <stack>/<task> \
  --dataset /absolute/path/to/dataset \
  --output-dir /absolute/path/to/checkpoints \
  --max-steps <N> \
  --save-steps <N> \
  --batch-size <N> \
  --num-gpus <N>
```

Add backend-specific flags only after resolving them. Do not silently lower requested steps or batch size to make training fit.

## Verify

Require exit status 0, completed requested steps, saved training logs, and at least one loadable checkpoint artifact. Resolve the exact checkpoint path rather than calling an incomplete output directory a checkpoint: GR00T saves `checkpoint-<step>/` directories under the output directory, and openpi saves `<output>/<config>/<experiment>/<step>/` with one `assets/<asset_id>/norm_stats.json`.

Load-check the checkpoint with the stack's own server before reporting it usable. Start it in the background with its PID kept, wait for its ready line, then stop it by that PID, all in one foreground command:

```bash
CHECKPOINT=/absolute/path/to/checkpoint
LOG="/absolute/path/to/checkpoints/load-check.log"
PYTHONUNBUFFERED=1 tasks/<stack>/.venv/bin/python -m i4h_tasks.<stack>.serve <task> \
  --model-path "$CHECKPOINT" --port <free-port> >"$LOG" 2>&1 &
SERVER_PID=$!
READY=0
for _ in $(seq 360); do
  if grep -q "<ready line>" "$LOG"; then READY=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 5
done
kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
[ "$READY" = 1 ] && echo "checkpoint loaded" || tail -n 40 "$LOG"
```

Use the stack's actual server and flag, and only a task its `--help` lists: the GR00T N1.5 and N1.7 `serve` modules take `--model-path` and print `Server is ready`; the openpi `serve` module takes `--checkpoint`, compiles with a warm-up inference, and logs `server listening on`; GR00T N1.6 has no i4h server, so run GR00T's `third_party/Isaac-GR00T-1.6/gr00t/eval/run_gr00t_server.py --model-path "$CHECKPOINT" --embodiment-tag NEW_EMBODIMENT` from `tasks/gr00t_n16/.venv` and wait for `Server is ready`. Choose a port no other session is using (`ss -ltn`). Stop the server only by its PID, never by module name. A training exit alone proves that files were written, not that inference can load them.

Report `du -sh` for the task output and the selected checkpoint. Some trainers save both a final model at the output root and numbered checkpoints; identify that duplication, but do not delete either copy unless the user explicitly asks for cleanup.

Hand the exact load-checked checkpoint path to `i4h-workflow-validate`; do not evaluate unless the user requested it.

## Troubleshooting

Report the first dataset, manifest, model-access, GPU-memory, or backend error. Preserve logs and never silently change requested hyperparameters.

## Prerequisites

Require a compatible LeRobot dataset, the synced stack environment, model access, GPU capacity, and a task manifest with `train:`.

## Limitations

Inference-only tasks cannot be fine-tuned, and this skill does not claim rollout success from training alone.

## Examples

- `Fine-tune for 200 steps with a batch size of 32. Turn off vision tuning.` → preserve exact values, apply GR00T's supported vision flag, dry-run, train, load-check, and report the checkpoint.

## Completion gate

Report workflow, stack and task id and manifest, dataset compatibility, resolved config, requested/completed steps, batch/GPU/vision settings, checkpoint path, load-check result and ready line, output/checkpoint disk sizes, exact validation handoff, exit summary, and any inference-only or resource blocker.
