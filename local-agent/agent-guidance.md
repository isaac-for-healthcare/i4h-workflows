# Local i4h agent

You drive the Isaac for Healthcare workflows: i4h scenes that run as IsaacLab-Arena environments. The repository skills contain the maintained procedures, paths, commands, manifests, and validation expectations. Load the matching skill before reading or changing repository files, and follow it instead of reconstructing an older layout from memory.

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
| Install, synchronize, or repair the environments | `i4h-workflow-setup` |
| Create a blank workflow scene | `i4h-workflow-create` |
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

For `Stop all`, do not load a stage skill. Stop each workflow you started with `./stop.sh <workflow>`, any live session with `./local-agent/bridge.sh stop <scene>`, and anything else you started by the process ID you recorded, and report what stopped. Never stop processes by tool or module name: other sessions run the same tools.

## Root layout and commands

- Stay at the repository root unless a skill explicitly changes directory.
- The repository is rooted at `common/`, `arena/`, `tasks/`, `rl/`, `tools/`, `docs/`, and `scripts/`; do not add an extra nesting prefix.
- Use `./setup.sh`, `./run.sh`, `./stop.sh` and `./train.sh` as the root entry points. `./run.sh show <workflow> [mode]` prints the Arena policy runner, Isaac Lab recording or replay, and policy server commands a run executes.
- Put run artifacts under `runs/<workflow>/<timestamp>/`, and pass that directory to every stage that must share it.
- Never write scratch output to `/tmp`; use the active run directory or `.run/`.
- Do not edit `third_party/` or copy upstream Isaac Sim skills into `skills/`.
- Do not use destructive git commands. Undo your own work with targeted file edits.

## Live scene editing

An edit request is live-only unless the user explicitly asks to bake, save, or persist it.

1. Start the scene with `./local-agent/bridge.sh start <scene>`. It launches the scene as an IsaacLab-Arena environment in a Kit window, pauses its timeline so nothing moves while you edit, and prints the run directory.
2. Apply one observable operation at a time with `python scripts/live_scene_edit.py ...`. Use `--help` to select `add-known-asset`, `set-transform`, `set-view`, `camera-from-view`, `capture-camera`, `inspect`, `export-scene`, or another supported operation.
3. Keep scripts, exports, captures, and logs in the run directory printed by `bridge.sh`.
4. Bake only when explicitly requested, following `i4h-workflow-scene-edit` and validating the resulting source with its tests and a fresh simulator launch.
5. Stop the managed live session with `./local-agent/bridge.sh stop <scene>`.

Opening a workflow is not permission to modify it. If the user asks only to open it, start the live session, report readiness, and wait for an explicit edit.

This coding model cannot inspect images itself. When a loaded skill requires a visual observation, capture the viewport or declared camera into the active run directory, then run `python3 local-agent/vlcheck.py --image <capture> --prompt <bounded-rubric>`. Treat the returned verdict as visual evidence, not as a replacement for static checks or an affected dynamic rollout. If the configured VLM is unavailable, report that visual validation is deferred instead of claiming that the scene looks correct.

## Completion standard

Run every verification required by the loaded skill. Static lint, dry-run, and compilation checks are not substitutes for simulator validation when the request requires a real build or visual result. Repair failures and rerun the relevant checks before reporting completion. State exactly what ran, what passed, and what remains unavailable.

Do not manufacture a missing prerequisite with a different policy, an unattended interactive/teleop process, or a synthetic artifact. If a requested stage needs an input that does not exist and the loaded skill does not define a safe way to create it, report the missing input and the precise preceding stage that must produce it.

After the final tool result, always return a concise user-facing outcome. Never end a turn on raw command output or an unfinished investigation.
