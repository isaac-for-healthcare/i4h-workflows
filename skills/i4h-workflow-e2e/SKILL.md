---
name: i4h-workflow-e2e
description: Run the maintained workflow data-to-policy pipeline from recording through checkpoint validation. Use for full end-to-end requests; do not use for one individual stage.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-15"
  tags:
    - isaac-for-healthcare
    - i4h
    - robotics
    - data-to-policy
---

# Run the Workflow End-to-End Pipeline

## Purpose

Use the maintained driver so stage resolution, artifacts, logs, and checkpoint handoff stay consistent with the current scene and task manifests.

## Instructions

1. Resolve the base checkout and a workflow the driver supports.
2. Require a successful driver dry-run.
3. Execute the maintained driver in the foreground.
4. Inspect every stage artifact before reporting completion.

## Resolve the checkout

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
```

Treat this resolver as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

The driver supports `scissor_pick_and_place` (GR00T N1.5 in `SoarmScissors`, `soarm_scissors`) and `ultrasound_liver_scan` (PI0 in `PandaPhantom`, `panda_phantom`); any other workflow exits with status 2. It reads the embodiment, task text, and trainability from the task manifest and the annotate settings (criterion, cameras, judge, frames) from the scene manifest's `annotate` block, falling back to the task text. The dry-run prints the resolved `[e2e] env=... scene=... stack=... task=... robot=... trainable=...` and `[e2e] annotate:` lines.

## Dry-run first

```bash
./scripts/e2e/run.sh --env <workflow> --dry-run
```

Require exit status 0 and inspect every printed command and artifact path. The dry-run executes nothing and creates no directories; it is the source of truth for current stages and backend ownership.

## Run in the foreground

```bash
./scripts/e2e/run.sh --env <workflow>
```

Use `--run-dir` only when the caller needs a specific location; the default is `runs/<workflow>/<YYYYMMDD_HHMMSS>/`. `--headless` drops the `--viz kit` viewport from the record and replay stages; checkpoint validation always runs headless. Apply `--skip-mimic`, `--skip-annotate`, `--skip-replay`, or `--skip-viz` only when the user explicitly omits that optional stage or a documented smoke profile requires it. Set `RECORD_EPISODES`, `MIMIC_EPISODES`, `FINETUNE_STEPS`, `FINETUNE_BATCH_SIZE`, `VALIDATE_EPISODES` (defaults 3, 3, 500, 32, 3), or `I4H_POLICY_PORT` (default 8000) only to honor a requested value.

Fine-tuning needs no Weights & Biases account. The PI0 trainer logs there only with `--wandb` or `train.wandb: true` in its task manifest; the driver passes neither, and the environment does not turn it on.

Keep the driver as this agent's foreground tool call. Do not use a subagent, monitor task, shell backgrounding, `nohup`, `tmux`, or a detached process. Poll until exit.

The driver performs full setup, then owns its stage sequence, run directory, and logs. It starts each policy server in the background for its stage, waits until the server is listening on its port (up to `I4H_E2E_SERVER_TIMEOUT_S`, default 3600 s, which covers a first checkpoint download), and stops it by its PID, and it stops the annotator's vLLM container only when it started it. Do not replace it with a manually assembled subset.

## Verify

On success, inspect the printed summary and artifacts under the run directory:

- `logs/workflow.log` and the policy-server logs `logs/server-record.log` and `logs/server-validate.log`
- policy recording `data/recording.hdf5` and per-episode results under `record/<time>/episode_results_rank0.jsonl`; the recording keeps every episode with its success flag
- expanded and filtered HDF5 (`data/expanded.hdf5`, `data/filtered.hdf5`) as applicable, with every annotate verdict and the pass count
- visible replay result when enabled
- LeRobot metadata, parquet, and videos under `lerobot/local/<workflow>_e2e`
- visualizer URL/content when enabled
- training logs and the exact checkpoint under `checkpoints/` when trainable
- checkpoint validation recording `validate/verify.hdf5`, its results under `validate/outputs/<time>/episode_results_rank0.jsonl`, and the `[e2e] <k>/<n> validation episodes succeeded` line; the stage passes when at least one episode succeeds, so report `k/n`

On failure, stop at the first failed stage, inspect `logs/workflow.log` and that stage's server log, preserve the run directory, and repair the owning stage before rerunning. Do not skip a required failure merely to obtain a green summary. The driver's exit handler stops its own servers. If one survives a hard kill, find it by the driver's port (`ss -ltnp`), confirm its start time matches this run, and stop it by that PID; never stop processes by tool or module name, because other sessions run the same servers.

## Troubleshooting

Use the first failed stage and its log to choose the owning stage skill. Preserve the run directory and rerun only after that stage verifies its output.

## Prerequisites

Require a supported workflow plus host, simulator, policy-stack, VLM, dataset, training, and visualization dependencies for every enabled stage.

## Limitations

The pipeline supports only `scissor_pick_and_place` and `ultrasound_liver_scan`; an inference-only task skips fine-tuning and checkpoint validation.

## Examples

- `Run end-to-end smoke pipeline for scissor pick-and-place.` → dry-run, execute the driver, and report each recording-to-validation stage.

## Completion gate

Report workflow, scene, stack/task/embodiment/trainability, dry-run result, run directory, every stage outcome and skip, dataset/visualizer/checkpoint/verification artifacts, validation successes out of episodes, final exit status, and cleanup state.
