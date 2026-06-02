---
name: i4h-workflow
version: "0.6.0"
description: Overview of `workflows/agentic/` (IsaacLab-Arena + GR00T/openpi). Use when the user asks what i4h workflow is, what's supported, or where to start.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - robotics
    - overview
---

# i4h Agentic Workflow

## Purpose

Orient on the agentic workflow before touching a specific stage: which envs/robots/policies are supported, how the `workflows/agentic/` subprojects fit together, and which per-stage skill to invoke next. This skill routes — it runs no pipeline stage itself.

```bash
# from repo root: confirm the workflow is set up, then jump to a stage skill
workflows/agentic/policy/run.sh --list-envs
```

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

- Env YAMLs at `workflows/agentic/config/environments/<env>.yaml` are the source of truth.
- Each pipeline stage has its own skill. Compose them or use [[i4h-workflow-e2e]] for full runs.

## Supported Envs

This table is the **curated, tested** set (robot + policy combos shipped with the repo) and is a static reference — it does **not** auto-update. Envs you add with [[i4h-workflow-create]] are registered on disk but will **not** appear here, so do not treat this table as the complete list.

**To list every registered env (including ones you just created), enumerate at runtime instead of reading this table:**

```bash
workflows/agentic/policy/run.sh --list-envs        # authoritative registry of all envs
ls workflows/agentic/config/environments/*.yaml    # env YAMLs on disk (source of truth; includes new ones)
```

| Env | Robot | Policy |
|---|---|---|
| `scissor_pick_and_place` | SO-ARM 101 | GR00T N1.5 (N1.7 alternative) |
| `locomanip_tray_pick_and_place` | Unitree G1 | GR00T N1.6 (shared `policy.locomanip.*`) |
| `locomanip_push_cart` | Unitree G1 | GR00T N1.6 (shared `policy.locomanip.*`) |
| `assemble_trocar` | Unitree G1 + Dex hands | GR00T N1.5 (inference-only) |
| `ultrasound_liver_scan` | Franka-style arm | openpi PI0 |

## Subprojects

| Directory | Purpose |
|---|---|
| `arena/` | IsaacLab-Arena envs, scenes, tasks, teleop, record, replay |
| `policy/` | Policy daemons and train dispatchers |
| `dataset/` | HDF5 → LeRobot conversion and visualization |
| `mimic/` | HDF5 trajectory expansion |
| `annotator/` | VLM success labels and filtering |
| `cosmos/` | Optional Cosmos Transfer video augmentation |
| `common/` | Shared config, messaging, robot constants |

## Skill Index

- [[i4h-workflow-setup]] — install / sync / check third-party deps.
- [[i4h-workflow-create]] — add a new env.
- [[i4h-workflow-scene-edit]] — edit an existing scene / task / camera.
- [[i4h-workflow-dataset-teleop]] — record human demos.
- [[i4h-workflow-dataset-replay]] — replay HDF5 episodes.
- [[i4h-workflow-dataset-mimic]] — expand HDF5 demos with noise.
- [[i4h-workflow-dataset-annotate]] — VLM label / filter episodes.
- [[i4h-workflow-dataset-convert]] — convert HDF5 to LeRobot.
- [[i4h-workflow-dataset-transfer]] — Cosmos Transfer video augmentation.
- [[i4h-workflow-finetune]] — train supported envs.
- [[i4h-workflow-validate]] — roll out / evaluate policy checkpoints.
- [[i4h-workflow-e2e]] — run the full pipeline.
- [[i4h-lerobot-viz]] — open the LeRobot HTML viewer.

## Prerequisites

- For any hands-on stage, set up the workflow first — see [[i4h-workflow-setup]] (`.venv` present, third-party checked out).
- Env YAMLs at `workflows/agentic/config/environments/<env>.yaml` are the source of truth.

## Limitations

- Overview/routing only — each pipeline stage has its own skill; this one performs no recording, training, or rollout.
- Supported envs/robots/policies are limited to those in **Supported Envs**; new robots or tasks require [[i4h-workflow-create]].

## Troubleshooting

- **Error: env not found / unsupported** — Cause: the env is not registered or has no YAML under `workflows/agentic/config/environments/`. Fix: run `workflows/agentic/policy/run.sh --list-envs` to see all registered envs (including newly-created ones) and pick one, or add a new one with [[i4h-workflow-create]].
- **Not sure which skill applies** — Cause: stage unclear. Fix: match the pipeline stage to the **Skill Index** above.

## Final Response

For overview questions, list the live envs via `workflows/agentic/policy/run.sh --list-envs` (not just the curated table above), summarize the subproject layout, and point to the next stage skill.
