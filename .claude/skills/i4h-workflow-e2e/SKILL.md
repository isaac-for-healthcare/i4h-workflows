---
name: i4h-workflow-e2e
description: Run the full end-to-end agentic pipeline (record → mimic → annotate/filter → replay → convert → visualize → finetune → validate). Use when the user asks to run the full pipeline, smoke the whole workflow, demo the workflow, or do an e2e run.
---

# i4h Workflow — End-to-End

## Basics

- Use the e2e script for full pipeline runs.
- For per-stage work, use the corresponding dataset/finetune/validate skills.
- `assemble_trocar` is inference-only; the e2e script skips finetune and checkpoint validation for it.

## Dry Run

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
"${REPO_ROOT}/workflows/agentic/scripts/e2e/run.sh" --dry-run --env <env>
```

## Run

```bash
"${REPO_ROOT}/workflows/agentic/scripts/e2e/run.sh" --env <env>
```

## Flags

- `--skip-mimic`, `--skip-annotate`, `--skip-replay`, `--skip-viz`
- `--from-stage <stage> --run-dir <existing-run>` resumes from a prior run.

Stages: `setup record mimic annotate replay convert viz finetune validate summary`.

## Outputs

The script prints `RUN_DIR`. Subdirs:

- `logs/`
- `data/`
- `lerobot/`
- `checkpoint/` (trainable envs only)
- `SUMMARY.txt`

## Stop

```bash
"${REPO_ROOT}/workflows/agentic/stop.sh" all --env <env>
```

## Final Response

Report env, run dir, skipped stages, per-stage success/failure, key artifact paths.
