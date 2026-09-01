# i4h Workflows Agent Guide

Run commands from the repository root.

## First Checks

```bash
./run.sh list
./run.sh lint <workflow>
./run.sh show <workflow>
```

If a component environment is missing, run `./setup.sh`. Use `./stop.sh all` for process cleanup.

Create a blank idle-only Workflow with `./scripts/create_blank_environment.py <workflow_id> --specialty <specialty>`. Choose one of `laparoscopic-robotics`, `ultrasound-robotics`, `endoluminal-robotics`, or `hospital-automation-robotics`. Pass `--dry-run` to preview the generated files or `--validate` to run focused static checks after creation.

## Resolve the Checkout

Skills may also run from a central catalog outside this repository. Resolve or clone the root-level workflow runtime before using repository-relative paths:

```bash
export I4H_WORKFLOWS_REPO_URL="${I4H_WORKFLOWS_REPO_URL:-https://github.com/isaac-for-healthcare/i4h-workflows}"
I4H_REPO_DIR_NAME="${I4H_WORKFLOWS_REPO_URL%/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*:}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME%.git}"
[ -n "$I4H_REPO_DIR_NAME" ] || { echo "Cannot derive a checkout name from I4H_WORKFLOWS_REPO_URL" >&2; exit 2; }
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/i4h_workflows" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/$I4H_REPO_DIR_NAME}"
  [ -d "$ROOT/workflows/i4h_workflows" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT"
```

`I4H_WORKFLOWS_REPO_URL` chooses the clone source and `I4H_WORKFLOWS` chooses or reuses a specific checkout. Never replace an existing checkout.

For a live scene, export confirmed edits with `arena/.venv/bin/python scripts/live_scene_edit.py export-scene --workflow <workflow> --output-path <run-dir>/live_scene.json`, then resolve reusable facts with `arena/.venv/bin/python scripts/authoring_info.py snapshot <workflow> <run-dir>/live_scene.json`. These utilities do not generate workflow code; the coding agent patches the owning Scene, asset source, and manifest while the snapshot remains a run artifact.

Keep generated runtime artifacts under `runs/<workflow>/<YYYYMMDD_HHMMSS>/`. Pass `run.sh --run-dir <that-directory>` when a skill or driver must place launcher logs and recordings together; do not create flat stage-prefixed run directories.

## Source of Truth

- Workflow layout and specialty catalog: `workflows/README.md`
- Author-facing Workflow value: `engine/i4h_engine/interface.py`
- TaskGraph-building API: `engine/i4h_engine/graph.py`
- Workflow and run modes: `workflows/i4h_workflows/<specialty>/<workflow>.py`
- Standard run-mode vocabulary and shared builders: `workflows/i4h_workflow_modes/README.md`
- Scene construction: `arena/i4h_arena/scenes/`
- Scene capabilities and step cap: `arena/i4h_arena/scenes/manifest/<scene>.yaml`
- Embodiment facts: `arena/i4h_arena/embodiments/manifest/<robot>.yaml`
- Task implementation and manifest: `tasks/<project>/i4h_tasks/<project>/`
- Exported RSL-RL policy Tasks: `tasks/rsl_rl/`
- Shared HDF5 contract: `common/i4h_common/episode.py`
- Online RL profiles, backends, and workflow adapters: `rl/`

Use `--rule-based` for local controller graphs. Do not add mode aliases.

## Change Discipline

- A task reads `ctx.scene`, writes `ctx.act`, and never calls `env.step`.
- Keep online RL outside Workflow modes; `rl` owns vectorized stepping and hands checkpoints back to normal policy validation.
- Keep incompatible policy stacks isolated behind `i4h_common.server.PolicyServer`; a simulator-compatible exported TorchScript actor may be an in-process Task under `tasks/rsl_rl`.
- Do not import policy packages into arena or Isaac packages into the light discovery path.
- Do not increase a workflow's validated step cap to hide inference or controller issues.
- Give every task manifest a concise `summary`; add `prompt` only when it needs more detail. Keep model, observation, and training defaults in the owning remote-task manifest.
- Add CPU tests for graph/task behavior and perform a visible simulator validation for scene changes.

For generic Isaac Sim physics, cameras, USD, rendering, or spatial authoring, route through `skills/i4h-workflow-scene-edit/references/isaacsim-skill-routing.md` and use the selected upstream skill. Keep i4h guidance focused on workflow/task integration, manifests, policy wiring, recording, and validation.

## Skill Routing

- Start with `skills/i4h-workflow/` for architecture, support, or where-to-start questions.
- Use `skills/i4h-workflow-setup/` for installation and missing component environments.
- Use `skills/i4h-workflow-create/` for a new blank workflow and `skills/i4h-workflow-scene-edit/` for an existing Scene or task contract.
- Use the matching `skills/i4h-workflow-dataset-*` skill for teleoperation, replay, mimic, annotation, or conversion.
- Use `skills/i4h-workflow-finetune/` for offline policy training, `skills/i4h-workflow-train-rl/` for online RL, and `skills/i4h-workflow-validate/` for rollouts.
- Use `skills/i4h-workflow-e2e/` only for the full maintained data-to-policy pipeline and `skills/i4h-lerobot-viz/` for browser inspection of converted data.

## Skill Validation

After changing `skills/`, skill routing, or skill-backed examples, run the Recommended Local Validation and Eval Dataset Schema Check in `TESTING.md`. Treat that file as the validation source of truth; lighter checks do not replace its NV-BASE report.
