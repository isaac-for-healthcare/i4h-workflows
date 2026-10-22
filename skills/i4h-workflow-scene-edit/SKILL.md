---
name: i4h-workflow-scene-edit
description: Edit an existing i4h workflow scene or its policy and task contracts. Use for assets, layout, cameras, randomization, policies, task text, or success rules; do not use to create a new workflow.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-21"
  tags:
    - isaac-for-healthcare
    - i4h
    - isaac-sim
    - scene-authoring
---

# Edit an i4h Workflow Scene

## Purpose

Iterate on an existing scene in one live simulator session, and persist the confirmed state only when explicitly asked to bake, save, or persist.

## Instructions

1. Run the checkout resolver and inspect target ownership.
2. Start or reuse one bridge-backed live-authoring session by default and capture a visible baseline.
3. Read the minimal upstream guidance and apply each requested edit as its own observable live-stage transaction without changing source.
4. On an explicit bake/save/persist instruction, export the accumulated live state, inspect its resolved authoring facts, and have the coding agent edit the smallest owning sources.
5. Keep the session running for further prompts until the user explicitly says stop or exit.
6. Run static, persisted-visible, and affected dynamic validation when baking.

## Resolve and inspect

Before resolving the checkout, use the maintained repository below or an alternative already selected by the user or trusted project configuration. Check an existing checkout's origin and working-tree changes before executing its scripts; an inherited environment variable alone does not establish trust in an alternative source. Honor any requested revision and preserve local changes. If the source is unexpected, stop and resolve it before cloning or launching.

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
  [ -d "$ROOT/arena/i4h_arena/scenes/manifest" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT" || exit 2
fi
[ -d "$ROOT/arena/i4h_arena/scenes/manifest" ] && [ -x "$ROOT/setup.sh" ] || { echo "Incomplete workflow checkout: $ROOT" >&2; exit 2; }
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT" || exit 2
ls arena/i4h_arena/scenes/manifest
grep -rn "\`<workflow>\`" docs/workflows/
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains the scene manifests in `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Map the workflow to its scene first: a maintained workflow id maps through its `docs/workflows/` table (`scissor_pick_and_place` → `SoarmScissors`, scene `soarm_scissors`), and a scene from `i4h-workflow-create` uses the workflow name (`my_workflow` → `MyWorkflow`). Commands below take the scene id as `<scene>` and the class as `<Class>`.

Read `AGENTS.md`, the workflow's docs page, and the scene's sources, manifests, policies, devices, and task manifests, resolving ownership with `$ROOT/skills/i4h-workflow/references/repo-map.md`.

For a G1 face, walk, reach, or collision-sensitive success contract, read `references/g1-reach-and-contact.md`. Keep the behavior in one rule-based Arena policy and the object footprint in the scene; do not generate workflow-specific locomotion modules or hardcode object extents in a policy or termination.

A rule-based Task or run-mode request means making the environment runnable that way: rule-based is an Arena `PolicyBase` in `arena/i4h_arena/policies/`; teleop is an Isaac Lab device in `teleop_devices`; a trained policy is a stack task manifest plus an Arena client (for GR00T N1.6, `Gr00tRemoteClosedloopPolicy` with a config like `tasks/gr00t_n16/config/g1_locomanipulation.yaml`); an RSL-RL checkpoint runs with Arena's `rsl_rl` once the scene offers the trainer's observations, like `ultrasound_probe_reach` behind `--rl_observations`. Add only what was requested; an untrained policy is structurally valid, never rollout-validated.

An open/view request without an edit means a `HoldPolicy` viewing run with `--viz kit`; do not invent source changes. An edit-scene request always means start or reuse the live bridge session unless the user explicitly requests offline/no-live operation. Live authoring is the default and does not require confirmation. “Save,” “persist,” or “bake” means serialize the accumulated live-stage edits into their owning sources; without one of those words, leave source untouched. “Stop” or “exit” closes the session; if persistence was not requested, discard the live-only edits. Do not stop merely because one edit prompt completed.

Before adding or resizing an asset that already appears in a maintained scene, read `references/existing-scene-assets.md`. Reuse its known USD identity, authored scale, support height, physics role, and embodiment convention instead of rediscovering them from the raw USD. Treat owning Python as source of truth and use visual-language inspection only for bounded scene-specific refinement after the known baseline is visible.

## Establish a visible baseline

Open the existing scene as one persistent live-authoring session through the bridge:

```bash
I4H_RUN_DIR="$(pwd)/runs/<scene>/$(date +%Y%m%d_%H%M%S)"
./local-agent/bridge.sh start <scene> "$I4H_RUN_DIR"
```

The bridge runs the scene, detached, under Arena's policy runner with `HoldPolicy`, `--viz kit`, and Isaac Sim's Python server on `127.0.0.1:8226`. Within `${I4H_AGENT_BRIDGE_TIMEOUT:-900}` seconds it prints `[bridge] READY`, with the timeline paused so physics stays frozen while edits apply, or `FAILED` with the log tail; give it a covering tool timeout or poll `./local-agent/bridge.sh status <scene>`. The run directory must be under `runs/<scene>/`. Pass launcher options after `--`, such as `-- --presets newton`.

Port 8226 serves one session per host; if it is in use by an unmanaged session, report it instead of stopping someone else's simulator. On a startup failure, report the log and blocker and use the offline fallback only for work already authorized.

Once ready, inspect and modify the running stage with `scripts/live_scene_edit.py`, which uses the pinned upstream `isaac-sim-remote` client. Keep every ordinary edit only in that live stage and accumulate later prompts in the same session. Only “bake,” “save,” or “persist” authorizes writing the confirmed live values into source.

## Preserve the live interpreter experience

Treat a compound prompt as an ordered stream of edits, not as one batch script. Run one bridge transaction for one user-visible operation, wait for its viewport update, inspect its result, and only then apply the next operation. For example, adding a table, two tools, two trays, and a robot is six live transactions. Never hide all requested edits inside one remote Python file or patch owning source while the user is waiting for the stage to change.

Use the one-operation helper from the repository root for common edits:

```bash
arena/.venv/bin/python scripts/live_scene_edit.py add-known-asset \
  --asset surgical_table \
  --prim-path /World/envs/env_0/Table \
  --position 0,0,0

arena/.venv/bin/python scripts/live_scene_edit.py add-cube \
  --prim-path /World/envs/env_0/RedCube \
  --position 0,0,0.3 \
  --size 0.1 \
  --color 1,0,0

arena/.venv/bin/python scripts/live_scene_edit.py scale-by \
  --prim-path /World/envs/env_0/RedCube \
  --factor 2

arena/.venv/bin/python scripts/live_scene_edit.py set-transform \
  --prim-path /World/envs/env_0/Robot \
  --position=-4.64,0,0.8 \
  --rotation 0,0,0

arena/.venv/bin/python scripts/live_scene_edit.py set-view \
  --eye 2.6,-7,3.4 \
  --target=-1.8,0,0.75

arena/.venv/bin/python scripts/live_scene_edit.py camera-from-view \
  --prim-path /World/envs/env_0/RoomCamera

arena/.venv/bin/python scripts/live_scene_edit.py capture-camera \
  --prim-path /World/envs/env_0/RoomCamera \
  --output-path "$I4H_RUN_DIR/room-camera.png"
```

When a comma-separated vector begins with a negative number, bind it with `=` (for example, `--position=-0.5,0.5,0.1`) so the argument parser does not treat the value as another option.

<!-- markdownlint-disable-next-line MD013 -->
The helper intentionally accepts one operation per invocation, selects the affected prim, advances visible render updates, and prints the resulting world bounds. Prefer `add-known-asset` for catalogued Healthcare assets: it reuses canonical USD, scale, orientation, physics metadata, attached-camera metadata, embodiment metadata, and expected metric bounds, then rejects a result whose size differs by more than 20%. The `g1` preset must create the standard `head` camera below the live robot preview; treat a missing camera prim as a failed robot edit, activate it, and verify its view before continuing. If no executable preset exists, warm-start from `references/existing-scene-assets.md` and its owning source before using generic `add-usd`. `add-known-asset`, `add-usd`, `add-cube`, and `camera-from-view` tag their prims for deterministic export; pass `--name` or `--alias` when the source/manifest name cannot be derived generically from the prim path. Inspect bounds before continuing. Use `capture-camera` for fast visible camera checks: it activates the requested camera, schedules a synchronous `FileCapture`, advances the renderer, and rejects an absent, empty, or stale output. Do not use the upstream asynchronous viewport screenshot helper in the persistent bridge session. Use `activate-camera` when capture is unnecessary, `inspect` for one-prim verification, and `--help` for the other operations. Use raw `isaacsim_send.py` only for an operation the helper does not support, and still send one observable edit per call; record any untagged prim explicitly when baking.

Infer ordinary support relationships from the requested workspace and measured bounds. A robot or object intended for a table, cart, tray, pad, or floor must have its lower support bound aligned with that surface and its footprint plausibly contained by it; do not accept a floor-mounted, floating, or visibly interpenetrating placement merely because every named asset is present. Include those support relationships in the bounded visual rubric before baking.

Send a short progress update while the scene visibly changes; design the source representation and prepare baking only after the live result exists.

## Session lifecycle

- On every later scene prompt, use `./local-agent/bridge.sh status <scene>` and `rundir <scene>` to find the managed session and its run directory; a listening port alone is insufficient. Reuse that session; do not reset or relaunch the scene unless the requested change requires it.
- After adding a robot, verify every camera declared by its authoring preset. For G1, require the live `Robot/Asset/head_link/RobotHeadCam` preview and bake with the registered G1 embodiment, whose `robot_head_cam` sensor the scene manifest declares as the `head` camera.
- Return control to the user after each prompt while leaving the simulator and bridge running. Baking does not imply stop.
- On stop/exit, close the session with `./local-agent/bridge.sh stop <scene>`; without a bake, leave source unchanged.
- Use offline/source-first editing only when the user explicitly disables live mode or the bridge cannot operate; report the blocker first. An offline edit followed by a reopen is not live authoring.

## Use upstream Isaac Sim skills

Read `references/isaacsim-skill-routing.md`, then load only the upstream skills required by the request. State the selection before editing. Use current upstream semantics for generic physics, cameras, sensors, USD, rendering, and spatial reasoning; integrate them through the closest current i4h scene pattern.

## Iterate live, then let the coding agent bake

Apply requested changes through port 8226 first; do not preemptively patch files merely because their eventual owner is known. When explicitly asked to bake/save/persist, export the confirmed live values into the session's run directory and resolve the reusable catalog facts without launching another simulator:

```bash
I4H_RUN_DIR="$(./local-agent/bridge.sh rundir <scene>)"

arena/.venv/bin/python scripts/live_scene_edit.py export-scene \
  --workflow <scene> \
  --root-path /World/envs/env_0 \
  --output-path "$I4H_RUN_DIR/live_scene.json"

arena/.venv/bin/python scripts/authoring_info.py snapshot \
  <scene> "$I4H_RUN_DIR/live_scene.json"
```

`export-scene --workflow` takes the scene id. It records every helper-managed asset, primitive, robot, and camera with its confirmed transform and camera optics. Keep that snapshot in the run directory as authoring evidence. Pass a previous run snapshot through `--baseline` only when a later live export needs to merge it: existing prims are re-read from the current stage, newly tagged prims are added, and removed prims are omitted. `authoring_info.py` is read-only: it validates the snapshot and returns code-ready catalog metadata and derived manifest capabilities, never code.

The coding agent then patches the existing asset, scene, and manifest sources using the closest maintained source pattern. If asked to commit, stage only this task's changes to those owning sources, never the snapshot, which is not a second scene contract. Never copy a reusable USD path, canonical scale, mass, embodiment registry name, action contract, attached camera, or camera alias from memory: query `arena/.venv/bin/python scripts/authoring_info.py asset <preset>` or the complete snapshot report, then use the catalog from owning source. Scene-specific names, placement, camera optics, and explicit overrides come from the snapshot.

Write only to the owning layer:

- Bake assets, layout, physics, cameras, randomization, reset events, and the embodiment choice into the scene's asset source and scene class.
- Bake observations, task events, and success and failure terminations into the scene's Arena task (`arena/i4h_arena/envcfg/`); success is the environment's termination, never a policy's decision.
- Bake cross-boundary camera and object names, embodiment, action space, DoF, control rate, and step budget into the scene manifest.
- Bake reusable behavior into a rule-based Arena `PolicyBase` in `arena/i4h_arena/policies/`.
- Bake a teleoperation device as an Isaac Lab `DeviceBase` in `arena/i4h_arena/devices/`, declared in the scene's `configure_env_cfg` as `env_cfg.teleop_devices`; bake cross-process robot labels or calibration into the embodiment manifest.
- Bake a policy prompt, camera, observation, model, or training contract into the stack's task manifest.

Preserve the quaternion convention at each concrete API boundary. Do not add compatibility conversions or duplicate catalog facts across Python and YAML.

For a camera based on the current perspective, treat the viewport pose as an initial estimate. Compute and validate a stable live look-at from task-relevant bounds. On bake, add the confirmed env-local camera through the closest scene pattern, declare it in the manifest, and verify every recording and policy consumer that should receive it.

## Validate

```bash
python3 scripts/run_tests.py --suite light
arena/.venv/bin/python -m pytest arena/tests/test_scene.py <focused-arena-tests...> -q
uvx ruff==0.5.2 check --config pyproject.toml <changed-python-files...>
```

The light suite checks the scene's contract test, manifests, and layering; the Arena scene test checks that the scene is still an Arena environment named after it. Use the Ruff version pinned in `.pre-commit-config.yaml`. Run focused tests through each affected component's uv project. Then stop the exported live session and reopen once with `./local-agent/bridge.sh start <scene>`, compare every requested visual change, transform, world bound, and declared camera against the exported snapshot with `inspect` and `capture-camera`, and stop when requested; static tests cannot catch a wrong quaternion convention. Do not add extra restarts between export, source editing, and this check.

When physics, reset, actuation, policy observations, behavior, or success changed, run every policy or device from the workflow's docs page and the scene's `teleop_devices` that consumes the change; a `HoldPolicy` view is insufficient:

```bash
I4H_RUN_DIR="$(pwd)/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
arena/.venv/bin/python -m isaaclab_arena.evaluation.policy_runner \
  --policy_type <policy-class-path> --num_episodes 1 --viz kit --output_base_dir "$I4H_RUN_DIR" \
  --external_environment_class_path i4h_arena.environments:<Class> <scene>
```

Read each episode's `success` and `episode_length` from `$I4H_RUN_DIR/<timestamp>/episode_results_rank0.jsonl`.

When success excludes collision, dynamic validation must include a forced-contact negative case and a fresh-reset recovery case from `references/g1-reach-and-contact.md`. Do not call the success termination validated if its configured contact signal has only ever returned false.

For cleanup, stop the managed session with `./local-agent/bridge.sh stop <scene>`, a foreground run with Ctrl+C, and a background run by the PID you started it with; never by tool or module name, which other sessions share.

## Troubleshooting

Fix failing contract, manifest, or layering tests before launch. If the rendered result differs, compare the baseline, authored prims, bounds, cameras, and owning source. Make one focused correction and revalidate; if the same mismatch remains, preserve the snapshot and report the unresolved difference instead of looping or claiming validation.

## Prerequisites

Require a supported existing scene, complete simulator setup, and the relevant upstream Isaac Sim skills for generic scene semantics.

## Limitations

Use `i4h-workflow-create` for a new workflow. A paused live session validates stationary layout but not physics, actuation, policy observations, or success. A live session is process-local; export before stopping because unexported edits are lost if it dies. The live helper covers common assets, raw USDs, cubes, transforms, cameras, inspection, capture, removal, and export; the coding agent handles source authoring and any semantics outside those utilities.

## Examples

- `Add a red cube, move G1, add a room camera, bake all changes, and stop.` → keep one bridge-backed simulator session open, apply the three edits live in order, export and inspect one snapshot at “bake,” patch and statically validate the owning source, reopen once for persisted-visible validation, and stop.

## Completion gate

Report upstream skills used, live session/bridge status, edits applied live, whether persistence was authorized, owning sources changed only during bake, static tests, live and persisted visible observations, camera checks, dynamic rollout results including collision-negative evidence when applicable, clean stop status, and any unresolved mismatch.
