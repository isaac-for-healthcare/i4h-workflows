---
name: i4h-workflow
description: Orient users to the i4h IsaacLab-Arena workflows and route them to the correct stage skill. Use for architecture, support, or where-to-start questions; do not use to execute a known stage.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-15"
  tags:
    - isaac-for-healthcare
    - i4h
    - robotics
    - onboarding
---

# i4h Workflows

## Purpose

Orient the user from live repository facts, then hand execution to the narrowest stage skill.

## Instructions

1. Run the base-checkout resolver.
2. Read `AGENTS.md`, `DESIGN.md`, and the live workflow tables in `docs/workflows/`.
3. Use only current architecture facts in the answer.
4. Use the narrowest stage skill for execution.

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

Treat this resolver as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains the scene manifests in `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

## Inspect before answering

Read `./AGENTS.md` and `./DESIGN.md` for architecture and `skills/i4h-workflow/references/repo-map.md` for ownership. Discover current support instead of copying a static table:

```bash
ls docs/workflows
ls arena/i4h_arena/scenes/manifest
```

Read the Workflows table of each relevant `docs/workflows/<specialty>.md` page. It maps every workflow id, such as `scissor_pick_and_place`, to its environment class and scene, such as `SoarmScissors` and `soarm_scissors`, and lists its policies, teleoperation devices, and exact commands. Map a natural-language name to the workflow id first, then to its environment.

Discovery needs no component environment. If `arena/.venv` is missing, report that running a workflow needs setup and route to `i4h-workflow-setup`.

## Explain the design

Keep the summary precise:

- A scene owns the simulated world (assets, cameras, randomization, reset events), its embodiment, and its task, whose success termination is the workflow's success criterion. `arena/i4h_arena/environments.py` exposes every scene manifest as an IsaacLab-Arena external environment under its CamelCase name (`psm_reach` → `PsmReach`).
- The scene manifest declares the scene's capabilities and the step budget its workflows were validated with; episodes end on the success termination or time out at that budget.
- A workflow id is the user-facing name for one environment plus the policies, devices, and commands its `docs/workflows/` page documents. There is no separate workflow module or run mode.
- Policies are IsaacLab-Arena `PolicyBase` classes. Rule-based controllers and remote-policy clients live in `arena/i4h_arena/policies/`; Arena also provides its own, such as `zero_action`, `rsl_rl`, and its GR00T client. Arena's policy runner steps the environment and writes per-episode results.
- Foundation-model checkpoints run out of process, each in its policy stack's own environment under `tasks/<stack>/`, behind a server that an Arena client policy connects to. The simulator environment never imports a policy stack.
- Teleoperation devices are Isaac Lab `DeviceBase` classes that a scene declares in `env_cfg.teleop_devices`. Isaac Lab's `record_demos.py` and `replay_demos.py` record and replay through Arena's registration callback, and recordings use Isaac Lab's recorder and HDF5 dataset format.
- Online RL in `rl/` is a separate training lifecycle: it owns vectorized stepping and hands a checkpoint back for evaluation under the policy runner.
- Offline tools in `tools/` (mimic, dataset conversion, annotation, Cosmos, patient twin) remain independent of the simulator.
- Python owns behavior. Scene, embodiment, and task manifests carry facts across dependency boundaries.

Do not describe the retired i4h runtime (`run.sh`, `stop.sh`, run modes, Workflow modules, TaskGraphs, the Engine, or Zenoh). If a user asks for it, map the request to the current Arena or Isaac Lab command from the workflow's docs page.

## Route the next action

| Goal | Skill |
|---|---|
| Install, sync, or repair dependencies | `i4h-workflow-setup` |
| Create a new blank workflow scene | `i4h-workflow-create` |
| Edit an existing scene, camera, policy, task, or success rule | `i4h-workflow-scene-edit` |
| Record demonstrations | `i4h-workflow-dataset-teleop` |
| Replay HDF5 | `i4h-workflow-dataset-replay` |
| Augment HDF5 | `i4h-workflow-dataset-mimic` |
| Grade/filter HDF5 with a VLM | `i4h-workflow-dataset-annotate` |
| Convert HDF5 to LeRobot | `i4h-workflow-dataset-convert` |
| Inspect LeRobot in a browser | `i4h-lerobot-viz` |
| Fine-tune a manifest-backed policy task | `i4h-workflow-finetune` |
| RL post-train a supported policy in simulation | `i4h-workflow-train-rl` |
| Run policy or rule-based rollouts | `i4h-workflow-validate` |
| Run the maintained complete pipeline | `i4h-workflow-e2e` |

For `Stop all`, do not load a stage skill. From the repository root, stop the processes this agent started: interrupt a foreground run with Ctrl+C and stop a background run or policy server by the PID recorded when it started. Stop each live scene session that has a `<scene>.pid` file in `.run/local-agent-bridge/` with `./local-agent/bridge.sh stop <scene>`. Never stop processes by tool or module name, because other sessions run the same tools. Report what was stopped.

## Troubleshooting

If discovery fails, verify the resolved checkout. If a workflow, policy, or teleoperation device is absent from its `docs/workflows/` page and the scene manifests, report it as unsupported.

## Prerequisites

Require a readable base checkout or network access to clone it.

## Limitations

This router does not install, author, simulate, process data, train, or evaluate.

## Examples

- `What does the i4h workflow include, and where should I start?` → inspect the live workflow tables and scene manifests, summarize the architecture, and recommend one stage skill.

## Completion gate

Answer with the live workflow list (workflow id → environment), a short architecture summary, and one concrete next skill. If the requested workflow, policy, or device is absent from the `docs/workflows/` pages and scene manifests, say it is unsupported instead of inventing a command.
