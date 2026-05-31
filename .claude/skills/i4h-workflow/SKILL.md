---
name: i4h-workflow
description: Overview of `workflows/agentic/` (IsaacLab-Arena + GR00T/openpi). Use when the user asks what i4h workflow is, what's supported, or where to start.
---

# i4h Agentic Workflow

## Basics

- Env YAMLs at `workflows/agentic/config/environments/<env>.yaml` are the source of truth.
- Each pipeline stage has its own skill. Compose them or use [[i4h-workflow-e2e]] for full runs.

## Supported Envs

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

## Final Response

For overview questions, summarize the supported envs, the subproject layout, and which skill to invoke next.
