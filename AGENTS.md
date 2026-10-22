# i4h Workflows Agent Guide

Run commands from the repository root.

## First Checks

Every workflow runs as an IsaacLab-Arena environment: an i4h scene, exposed by `arena/i4h_arena/environments.py` under its CamelCase name. The specialty pages in `docs/workflows/` map each workflow to its environment, policies, devices, and commands. List the scenes with:

```bash
ls arena/i4h_arena/scenes/manifest
```

If a component environment is missing, run `./setup.sh`. Run a scene with Arena's policy runner; launcher options such as `--presets newton`, `--enable_cameras`, and `--viz kit` go before the environment, the environment's own options after it:

```bash
arena/.venv/bin/python -m isaaclab_arena.evaluation.policy_runner \
  --policy_type i4h_arena.policies.hold.HoldPolicy --num_episodes 1 \
  --external_environment_class_path i4h_arena.environments:PsmReach psm_reach
```

Record teleoperation with Isaac Lab's `scripts/tools/record_demos.py` and replay with `replay_demos.py`, both through `--external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback --external_environment_class_path i4h_arena.environments:<Class> --task <scene>`. Serve a policy checkpoint from its own stack with `tasks/<stack>/.venv/bin/python -m i4h_tasks.<stack>.serve <task>` (GR00T N1.6 uses GR00T's own server; see `docs/workflows/hospital-automation-robotics.md`).

Stop a foreground run with Ctrl+C. Stop a background run or policy server by the PID you started it with, never by a tool or module name that other sessions may also be running. Stop a live scene session with `./local-agent/bridge.sh stop <scene>`.

Create a blank scene with `./scripts/create_blank_environment.py <scene_id>`. Pass `--dry-run` to preview the generated files or `--validate` to run its contract and scene tests after creation.

## Resolve the Checkout

Skills may also run from a central catalog outside this repository. Resolve or clone the repository before using repository-relative paths:

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

`I4H_WORKFLOWS_REPO_URL` chooses the clone source and `I4H_WORKFLOWS` chooses or reuses a specific checkout. Never replace an existing checkout.

To edit a scene live, open it with `./local-agent/bridge.sh start <scene>`, which runs it as an Arena environment with its timeline paused and Isaac Sim's Python server open, then apply one edit at a time with `arena/.venv/bin/python scripts/live_scene_edit.py`. Export confirmed edits with `scripts/live_scene_edit.py export-scene --workflow <scene> --output-path <run-dir>/live_scene.json` and resolve reusable facts with `arena/.venv/bin/python scripts/authoring_info.py snapshot <scene> <run-dir>/live_scene.json`. These utilities do not generate code; the coding agent patches the owning scene, asset source, and manifest while the snapshot remains a run artifact.

Keep generated runtime artifacts under `runs/<workflow>/<YYYYMMDD_HHMMSS>/`: pass that directory as `--output_base_dir` and `--recorder_dataset_export_dir_path` to the policy runner, and put `record_demos.py --dataset_file` inside it.

## Source of Truth

- Workflows, their environments and commands: `docs/workflows/<specialty>.md`
- Scenes as Arena environments: `arena/i4h_arena/environments.py`
- Scene construction: `arena/i4h_arena/scenes/`
- Scene capabilities and step budget: `arena/i4h_arena/scenes/manifest/<scene>.yaml`
- Embodiment facts: `arena/i4h_arena/embodiments/manifest/<robot>.yaml`
- Rule-based and remote policies: `arena/i4h_arena/policies/`
- Teleoperation devices: `arena/i4h_arena/devices/`
- Policy stacks, their servers and task manifests: `tasks/<stack>/i4h_tasks/<stack>/`
- Shared HDF5 contract: `common/i4h_common/episode.py`
- Online RL profiles, backends, and environment adapters: `rl/`

## Change Discipline

- Do what an Isaac Lab or IsaacLab-Arena user would: a policy is an Arena `PolicyBase`, a device an Isaac Lab `DeviceBase`, a success criterion the environment's termination. Add i4h code only where neither offers the concept.
- Keep online RL in `rl/`; it owns vectorized stepping and hands checkpoints back to policy validation under the policy runner.
- Keep each policy stack in its own environment behind its server; do not import policy packages into arena.
- Do not increase a scene's validated step budget to hide inference or controller issues.
- Give every task manifest a concise `summary`; add `prompt` only when it needs more detail. Keep model, observation, and training defaults in the owning task manifest.
- Add CPU tests for policy, device, and recorder behavior and perform a visible simulator validation for scene changes.

For generic Isaac Sim physics, cameras, USD, rendering, or spatial authoring, route through `skills/i4h-workflow-scene-edit/references/isaacsim-skill-routing.md` and use the selected upstream skill. Keep i4h guidance focused on scene and policy integration, manifests, recording, and validation.

## Skill Routing

- Start with `skills/i4h-workflow/` for architecture, support, or where-to-start questions.
- Use `skills/i4h-workflow-setup/` for installation and missing component environments.
- Use `skills/i4h-workflow-create/` for a new blank scene and `skills/i4h-workflow-scene-edit/` for an existing scene or task contract.
- Use the matching `skills/i4h-workflow-dataset-*` skill for teleoperation, replay, mimic, annotation, or conversion.
- Use `skills/i4h-workflow-finetune/` for offline policy training, `skills/i4h-workflow-train-rl/` for online RL, and `skills/i4h-workflow-validate/` for rollouts.
- Use `skills/i4h-workflow-e2e/` only for the full maintained data-to-policy pipeline and `skills/i4h-lerobot-viz/` for browser inspection of converted data.

## Skill Validation

After changing `skills/`, skill routing, or skill-backed examples, run the Recommended Local Validation and Eval Dataset Schema Check in `TESTING.md`. Treat that file as the validation source of truth; lighter checks do not replace its NV-BASE report.
