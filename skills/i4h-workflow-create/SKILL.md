---
name: i4h-workflow-create
description: Create a blank i4h workflow scene, containing only ground and light, that runs as an IsaacLab-Arena environment. Use for fast new workflow scaffolding; do not use to edit an existing scene.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-15"
  tags:
    - isaac-for-healthcare
    - i4h
    - isaac-sim
    - workflow-authoring
---

# Create a Blank i4h Workflow

## Purpose

Create a complete, runnable blank scene for a new workflow without searching for an existing workflow or booting Isaac Sim. The scene is an IsaacLab-Arena environment as soon as its manifest exists.

## Instructions

1. Resolve the checkout using `I4H_WORKFLOWS` when set; otherwise use the current git root. If neither contains `arena/i4h_arena/scenes/manifest`, use the checkout resolver from the repository `AGENTS.md`.
2. Use the requested workflow name as a lowercase snake_case scene id.
3. Check that the id is neither an existing scene nor a documented workflow id.
4. Generate the blank scene and run the static validation below.

A specialty named in the request, such as `hospital-automation-robotics`, is context only: the generator takes no specialty, so do not ask for one or infer one, and do not add the blank scene to a `docs/workflows/` page.

Check the id before generating:

```bash
ls arena/i4h_arena/scenes/manifest
grep -rn "\`<scene_id>\`" docs/workflows/
```

The generator rejects an existing scene id, but a documented workflow id usually runs a differently named scene: `ultrasound_liver_scan` runs `panda_phantom`. If either check finds the id, stop before writing files, report the existing workflow and its environment, and ask whether to edit that scene or choose a different id.

Run:

```bash
./scripts/create_blank_environment.py <scene_id>
```

This is the only authoring utility that generates scene-specific source. It writes the fixed blank templates that establish the design boundary; later scene, policy, and task code is written by the coding agent. Use `--dry-run` to preview its output, `--description TEXT` to customize the scene-manifest description, and `--validate` to run the focused static checks after creation. Do not combine `--dry-run` and `--validate`.

The script refuses to overwrite files and creates only:

```text
./
├── arena/i4h_arena/assets/<scene_id>.py
├── arena/i4h_arena/scenes/<scene_id>.py
├── arena/i4h_arena/scenes/manifest/<scene_id>.yaml
└── tests/test_<scene_id>_contract.py
```

The result contains a ground plane and a dome light; no embodiment (`embodiment: none`, `dof: 0`, `robots: []`); no task, so no success termination and episodes end at the manifest's `max_steps`; and no objects or cameras. `i4h_arena.environments` exposes it as `i4h_arena.environments:<CamelCase>` with scene name `<scene_id>`; for example, `my_workflow` becomes `MyWorkflow`. Its contract test verifies durable manifest invariants rather than asserting that the scene stays blank, so adding an embodiment, objects, cameras, policies, or devices later does not require rewriting the test. Do not inspect robot assets, choose a policy, create task manifests, load upstream Isaac Sim skills, or launch the simulator during blank creation.

## Utility

| Command | Purpose | Arguments |
|---|---|---|
| `./scripts/create_blank_environment.py` | Create the complete overwrite-safe blank scene | `<scene_id> [--description TEXT] [--dry-run \| --validate]` |

## Validate

Run only fast static checks:

```bash
./scripts/create_blank_environment.py <scene_id> --validate
```

When the scene already exists because it was created without `--validate`, run the equivalent checks directly:

```bash
cd "$(git rev-parse --show-toplevel)"
common/.venv/bin/python -m pytest tests/test_<scene_id>_contract.py -q
arena/.venv/bin/python -m pytest arena/tests/test_scene.py -q
```

`--validate` requires the `common` and `arena` environments; run `./setup.sh` first if either is missing. It verifies the generated manifest contract and that every scene, including the new one, is an Arena environment named after it, and that a zero-DOF scene opens no sensor panels, without launching Isaac Sim. If a component virtual environment is elsewhere, use that component's Python for its direct test command.

Do not run visible or dynamic validation unless the user explicitly requests it. A blank scene has no embodiment to hold, so open it with Arena's built-in zero-action policy and a step count:

```bash
arena/.venv/bin/python -m isaaclab_arena.evaluation.policy_runner \
  --policy_type zero_action --num_steps 300 --viz kit \
  --external_environment_class_path i4h_arena.environments:<CamelCase> <scene_id>
```

## Troubleshooting

- If the checkout cannot be resolved, set `I4H_WORKFLOWS` to the existing repository root or use the resolver in `AGENTS.md`.
- If the id check or the generator reports an existing scene or workflow, report its manifest or `docs/workflows/` page and ask whether to edit it or use a different id; never overwrite, relocate, or duplicate it.
- If `--validate` reports a missing component environment, run `./setup.sh` and repeat the same validation.

## Prerequisites

Require a writable root-level i4h-workflows checkout and Python 3.11 or newer. Static `--validate` also requires the `common` and `arena` component environments.

## Limitations

This skill creates only the four-file blank scene. It does not add an embodiment, success termination, policy, teleoperation device, cameras, or task-specific assets, and it does not visibly validate the scene.

## Continue

Treat creation as the first stage of an incremental workflow:

1. Create the blank scene with this skill: ground, light, scene manifest, and contract test.
2. Use `i4h-workflow-scene-edit` to add and visibly verify assets, robots, cameras, layout, and physics, then bake the confirmed scene.
3. Define behavior only after the user provides a concrete goal and success condition: a rule-based Arena policy under `arena/i4h_arena/policies/` and the success termination in the scene's task. Reuse a compatible policy; create a focused one only when the behavior is new. Do not create an empty placeholder policy.
4. Add teleoperation devices, a recording path, or a remote policy with its task manifest only when requested and only when their action contracts are runnable.
5. Validate every new way to run the workflow with the owning stage skill.

The blank scene is intentionally useful before behavior exists: Arena's zero-action policy and the live-edit bridge already open it for authoring, so a blank policy or task manifest would add no capability.

## Examples

- `Create a blank hospital-automation-robotics workflow named my_workflow.` → check the id, generate the blank `my_workflow` scene (`MyWorkflow`), and run static validation.
- `Make a new laparoscopic-robotics workflow called training_sandbox as fast as possible. Start blank.` → generate the same four-file blank scene as `training_sandbox` (`TrainingSandbox`).

## Completion gate

Report the created scene id and environment class, the four files, that the scene is blank (no embodiment or success termination), and the static validation results. Do not commit unless explicitly asked.
