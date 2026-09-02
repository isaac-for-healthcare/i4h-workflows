# Local i4h agent

You drive the root-level Isaac for Healthcare workflow runtime. The repository skills contain the maintained procedures, paths, commands, manifests, and validation expectations. Load the matching skill before reading or changing repository files, and follow it instead of reconstructing an older layout from memory.

## Mandatory routing gate

Before the first repository `Read`, shell command, or edit for a user request:

1. Classify the requested outcome using the table below.
2. Invoke the matching repository skill with the Skill tool.
3. Treat that skill's procedure and completion gate as authoritative; do not replace them with ad hoc package imports, remembered commands, or generic environment checks.

If no specialized row matches, invoke `i4h-workflow` for discovery and routing. The only no-skill exception is the exact operational request `Stop all`, handled as documented below.

## Load the matching skill first

| Request | Skill |
| --- | --- |
| Overview, architecture, or routing | `i4h-workflow` |
| Install, synchronize, or repair the runtime | `i4h-workflow-setup` |
| Create a blank workflow | `i4h-workflow-create` |
| Edit a Scene, camera, task contract, or success rule | `i4h-workflow-scene-edit` |
| Record teleoperation demonstrations | `i4h-workflow-dataset-teleop` |
| Replay an HDF5 episode | `i4h-workflow-dataset-replay` |
| Expand demonstrations with action jitter | `i4h-workflow-dataset-mimic` |
| Annotate or filter episodes with a VLM | `i4h-workflow-dataset-annotate` |
| Convert HDF5 to LeRobot | `i4h-workflow-dataset-convert` |
| Fine-tune GR00T or openpi | `i4h-workflow-finetune` |
| Train, evaluate, or export an online-RL policy | `i4h-workflow-train-rl` |
| Validate a policy or rule-based rollout | `i4h-workflow-validate` |
| Run the maintained end-to-end pipeline | `i4h-workflow-e2e` |
| Inspect a converted LeRobot dataset | `i4h-lerobot-viz` |

For `Stop all`, do not load a stage skill. Run `./stop.sh all` from the repository root and report what stopped. Never invent `run.sh stop`.

## Root layout and commands

- Stay at the repository root unless a skill explicitly changes directory.
- The runtime is rooted at `common/`, `engine/`, `workflows/`, `tasks/`, `arena/`, `rl/`, `tools/`, and `scripts/`; do not add an extra nesting prefix.
- Use `./setup.sh`, `./run.sh`, `./train.sh`, and `./stop.sh` as the root entry points.
- Put run artifacts under `runs/<workflow>/<timestamp>/` or pass `./run.sh --run-dir` when stages must share a directory.
- Never write scratch output to `/tmp`; use the active run directory or `.run/`.
- Do not edit `third_party/` or copy upstream Isaac Sim skills into `skills/`.
- Do not use destructive git commands. Undo your own work with targeted file edits.

## Live scene editing

An edit request is live-only unless the user explicitly asks to bake, save, or persist it.

1. Start the workflow with `./local-agent/bridge.sh start <workflow>`. It launches `./run.sh <workflow> --live` in the background and prints the run directory.
2. Apply one observable operation at a time with `python scripts/live_scene_edit.py ...`. Use `--help` to select `add-known-asset`, `set-transform`, `set-view`, `camera-from-view`, `capture-camera`, `inspect`, `export-scene`, or another supported operation.
3. Keep scripts, exports, captures, and logs in the run directory printed by `bridge.sh`.
4. Bake only when explicitly requested, following `i4h-workflow-scene-edit` and validating the resulting source through the normal workflow contracts and a fresh simulator launch.
5. Stop the managed live session with `./local-agent/bridge.sh stop <workflow>`. Use `./stop.sh all` only for broad cleanup.

Opening a workflow is not permission to modify it. If the user asks only to open it, start the live session, report readiness, and wait for an explicit edit.

This coding model cannot inspect images itself. When a loaded skill requires a visual observation, capture the viewport or declared camera into the active run directory, then run `python3 local-agent/vlcheck.py --image <capture> --prompt <bounded-rubric>`. Treat the returned verdict as visual evidence, not as a replacement for static checks or an affected dynamic rollout. If the configured VLM is unavailable, report that visual validation is deferred instead of claiming that the scene looks correct.

## Completion standard

Run every verification required by the loaded skill. Static lint, dry-run, and compilation checks are not substitutes for simulator validation when the request requires a real build or visual result. Repair failures and rerun the relevant checks before reporting completion. State exactly what ran, what passed, and what remains unavailable.

Do not manufacture a missing prerequisite with a different workflow mode, an unattended interactive/teleop process, or a synthetic artifact. If a requested stage needs an input that does not exist and the loaded skill does not define a safe way to create it, report the missing input and the precise preceding stage that must produce it.

After the final tool result, always return a concise user-facing outcome. Never end a turn on raw command output or an unfinished investigation.
