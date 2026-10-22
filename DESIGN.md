# Workflow Architecture

Every i4h workflow is an IsaacLab-Arena environment. i4h contributes the healthcare content an environment is built from: the scene, the robot, the success criterion, the controllers and policy clients that drive it, and the devices an operator uses. Running, recording, replaying, and training use the tools Isaac Lab and IsaacLab-Arena users already have:

```text
i4h scene (arena/i4h_arena/scenes)
  → IsaacLabArenaEnvironment: embodiment + task + scene assets
  → Arena's ArenaEnvBuilder → Isaac Lab ManagerBasedRLEnv
  → stepped by Arena's policy runner, Isaac Lab's record_demos.py and replay_demos.py,
    or an RL trainer in rl/
```

`./run.sh` runs a workflow's modes with those tools, as declared in `workflows/<specialty>/<workflow>.yaml`, and `./run.sh show` prints the commands it would run.

All paths in this document are relative to `./`.

## Architecture at a Glance

| Component | Owns | Location |
| --- | --- | --- |
| Workflow | Its scene and, per mode, the policy, policy server, or teleoperation devices; data only | `workflows/<specialty>/<workflow>.yaml` |
| Scene | One simulated world: assets, cameras, randomization, its embodiment and task, CLI options, operator panels, and the Isaac Lab cfg tuning the world needs | `arena/i4h_arena/scenes/` |
| Environment | Each scene as an Arena external environment, named after it in CamelCase; time-out at the scene's step budget, re-rendering after reset, recording rules | `arena/i4h_arena/environments.py` |
| Embodiment | The robot: articulation, action terms, proprioception, cameras, recorder terms | `arena/i4h_arena/embodiments/` |
| Task | The goal: observations, rewards, success and other terminations, reset events, the task description a policy is prompted with | `arena/i4h_arena/envcfg/` |
| Policies | Rule-based controllers and clients for policy servers, as Arena policies | `arena/i4h_arena/policies/` |
| Devices | Teleoperation devices, as Isaac Lab devices a scene declares in `teleop_devices` | `arena/i4h_arena/devices/` |
| Operator UI | Docked sensor panels, such as the fluoroscopy view, opened by a startup event in any Kit window | `arena/i4h_arena/ui/` |
| Policy stacks | One environment per model family: its server, fine-tuning, and task manifests | `tasks/<stack>/` |
| Online RL | Profiles, trainer configuration, and environment adapters | `rl/` |
| Offline tools | Mimic, annotation, conversion, visualization, Cosmos | `tools/` |
| Shared contracts | Scene and task manifests, robot descriptors, recording schema | `common/i4h_common/` |

A scene's `build()` returns an `IsaacLabArenaEnvironment`. `Scene.arena_environment()` composes the task's and embodiment's `tune_env_cfg` hooks into Arena's one `env_cfg_callback`, then applies the simulation and rendering defaults below. `i4h_arena.environments` registers one Arena external environment per scene manifest, so a new scene is runnable as soon as its manifest exists.

## Isaac Lab and Arena First

Use Isaac Lab for anything it expresses, IsaacLab-Arena for a concept Isaac Lab has no place for, and an i4h abstraction only when neither can express the function cleanly.

| Need | Isaac Lab or Arena mechanism | i4h addition, and why |
| --- | --- | --- |
| Assets, sensors, physics backend | Isaac Lab asset, sensor and simulation cfgs; `ConfigAsset` hands a cfg to Arena's scene unchanged | `i4h_arena.simulation` turns off `use_newton_actuators` and PhysX `enable_external_forces_every_iteration`, which Isaac Lab 3 turned on and which change the dynamics i4h policies were trained under |
| Rendering | Isaac Lab's `IsaacRtxRendererGlobalSettingsCfg`, applied by each camera renderer and by a `startup` event term | `i4h_arena.rendering` layers scene defaults, task, embodiment and scene settings. Isaac Lab 3 changed its default rendering and Arena zeroes the ambient light, so the defaults keep the look i4h policies were trained under |
| Robot | Arena `EmbodimentBase` subclasses, including subclasses of Arena's own G1 embodiments | None |
| Goal | Arena `TaskBase` subclasses returning `TaskTerminationCfg`, built from Isaac Lab manager term cfgs | None |
| Running an environment | Arena's `policy_runner`: `get_action`, then `env.step`, until each episode ends on a termination | `environments.py` times episodes out at the step budget each scene was validated with |
| Launching a workflow | The policy runner, `record_demos.py`, `replay_demos.py`, and each stack's server, each a separate command | `run.sh` (`scripts/run_workflow.py`) composes a mode's commands from its workflow declaration, starts its policy server and waits for it, and stops what it started; `stop.sh` stops a background run. It runs nothing those tools cannot, and `run.sh show` prints the commands |
| Rule-based controllers | Arena `PolicyBase` | Controllers for each procedure: reach, lift, probe sweep, scissor keyframes, catheter sweeps, holding still |
| Policies whose dependencies conflict with Isaac Sim | Arena's remote policies and clients (GR00T's `PolicyClient`, openpi's websocket client) | Client policies that build each i4h checkpoint's request from the environment's observations, and `serve` modules that start the stack's own server with i4h's modality configs and fixes |
| Teleoperation | Isaac Lab devices, `record_demos.py`, Isaac Teleop | Devices for the SO-ARM, the G1 and the catheter where Isaac Lab has none |
| Recording | Isaac Lab `RecorderManager` and its HDF5 format; `InteractiveScene.get_state` and `reset_to` capture and restore the initial state | Recorder terms for what conversion needs beyond actions and states, such as fluoroscopy frames and catheter diagnostics |
| Replay | Isaac Lab's `replay_demos.py` | None |
| Online RL | Isaac Lab's RSL-RL `train.py` with an external registration callback | `rl/` profiles select trainer, model and hyperparameters per workflow; RLinf runs GR00T post-training, which Isaac Lab does not ship |
| Live scene editing | Isaac Sim's Python server; Isaac Lab pauses its loop while the timeline is paused | `local-agent/bridge.sh` opens a scene paused, and `scripts/live_scene_edit.py` applies one observable edit at a time |

`third_party/setup.sh` pins the upstream checkouts and applies the few patches i4h carries against them. Prefer an upstream fix, a subclass, or an Arena registration hook to a new patch.

## Dependency Boundaries

These boundaries answer two questions: which code runs in the same process, and which layers that code may import. They exist because Isaac Sim, GR00T releases, and openpi require incompatible dependency sets.

### Process Separation

```text
┌─ Simulator process ─────────┐  stack's own protocol  ┌─ Policy server process ──┐
│ arena                       │◄──────────────────────►│ tasks/gr00t_* or openpi  │
│ common                      │  (GR00T ZMQ, openpi    │ common                    │
└─────────────────────────────┘   websocket)           └───────────────────────────┘

┌─ Offline process ───────────┐                        ┌─ Online RL ───────────────┐
│ tools/*                     │                        │ rl, arena, trainer stack  │
│ common                      │                        │ (see below)               │
└─────────────────────────────┘                        └───────────────────────────┘
```

A policy stack never loads into the simulator process. Arena's runner waits for each inference, as it does for any Arena remote policy, so the world advances in lock step with the policy and a slow policy shows up as wall time, not as a moving world.

### Online RL Training

```text
scene + embodiment + RL objective
  → rl/ resolves the profile and builds vectorized Isaac Lab environments
  → the selected trainer runs rollouts and updates
  → RL checkpoint, exported
  → evaluated in the environment under Arena's policy runner
```

The scene and its Arena task remain the source of assets, observations, actions, resets, rewards, and terminations. `rl/profiles/<workflow>.yaml` declares the supported workflows and `rl/config/*.yaml` the backend hyperparameters; `rl/i4h_rl/adapters/<workflow>.py` exists only when a scene needs workflow-specific observation, action, registration, or evaluation mapping. `train.sh rl` resolves and validates these contracts without starting Isaac Sim, and the heavy launcher starts only after its preflight.

A compatible trainer such as RSL-RL runs with Arena in one process; its checkpoint is evaluated with Arena's `rsl_rl` policy. An incompatible stack such as GR00T with RLinf runs its model separately and exchanges observations and actions through a local bridge; its exported checkpoint is served like any other. This follows the [IsaacLab-Arena reinforcement-learning workflow](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/reinforcement_learning_workflows/index.html).

### Offline Tools

Tools consume recordings or derived datasets in their own uv environments. They do not import Arena, advance the simulator, define task success, or own policy inference.

| Component | Entry point | Primary input | Output or effect |
| --- | --- | --- | --- |
| Trajectory mimic | `uv run --project tools/mimic i4h-mimic` | HDF5 recording | Expanded HDF5 recording with bounded action jitter |
| VLM annotator | `uv run --project tools/annotator i4h-annotator offline` | HDF5 recording | Episode labels and an optional filtered HDF5 |
| Dataset utility | `uv run --project tools/dataset i4h-dataset` | HDF5 recording | Inspection reports, decoded actions, or a LeRobot dataset |
| LeRobot viewer | `tools/dataset/scripts/viz.sh` | LeRobot dataset | Local HTML viewer |
| Cosmos adapter | `uv run --project tools/cosmos i4h-cosmos` | HDF5 recording or augmented MP4 files | Per-camera MP4 exports or an HDF5 recording with imported visual variants |

`tools/annotator/scripts/vllm.sh` manages the default local Qwen VLM container for annotation. `tools/cosmos/scripts/serve.sh` manages the optional local Cosmos service.

### Imports

- `common` imports no other i4h layer.
- `arena` imports `common` and never imports a policy stack; only the clients Arena itself uses cross over.
- Policy stacks and offline tools import `common` only among i4h layers.
- `rl` imports `arena` and its trainer stacks only after its preflight.

Each component is an independent uv project, so its environment contains only the dependencies its process needs. `tests/test_layering.py` enforces these rules.

## Manifests

Manifests make facts available without importing the code or environment that owns them. Python remains the source of executable behavior; use YAML for a fact that must cross a process or dependency boundary, and do not repeat a fact in both.

| Manifest | Owns | Main consumers |
| --- | --- | --- |
| `arena/i4h_arena/scenes/manifest/*.yaml` | Scene implementation path, embodiment, action space, DOF, cameras, objects, robots, control rate, step budget, and the VLM `annotate` criterion | `environments.py`, RL, e2e, annotation |
| `arena/i4h_arena/embodiments/manifest/*.yaml` | Cross-process robot metadata such as state/action labels, joint names, calibration, and home pose | Devices, policy clients, dataset conversion |
| `tasks/*/i4h_tasks/*/manifest/*.yaml` | A policy task's summary and optional prompt, embodiment, cameras, requirements, model, observation, and training configuration | The stack's `serve` and `train` modules, e2e |

The scene manifest's `annotate` block tells the VLM annotator what success looks like on camera: the criterion, the cameras to show, and whether to judge the final state or the motion. It belongs to the scene because it holds for every recording of it, whichever controller, checkpoint, or operator produced it.

## Policies

A rule-based controller is an Arena policy that computes each action from the environment's observations and state, on the schedule the procedure needs: Cartesian reach and lift for the surgical arms, a probe sweep, the scissor keyframes, the catheter sweeps, and a hold policy that keeps every robot still for viewing a scene.

A trained checkpoint runs behind its stack's server. The Arena client policy builds the checkpoint's request from the environment's observations, converts units where the checkpoint was trained in another calibration, and replays each returned action chunk for its open-loop horizon. Arena's runner hands the policy the environment's task description as its prompt, so each scene's task description is the instruction its checkpoints were trained on.

## Recording

Isaac Lab's `RecorderManager` writes every recording, in Isaac Lab's HDF5 format (`format_version` 1):

```text
/data                              attrs: env_args, total
  /demo_N                          attrs: success, num_samples
    initial_state                  scene state the episode started from
    actions, processed_actions     (T, action_width)
    states/...                     articulation and rigid object state
    obs/<term>                     observations, cameras, scene-specific frames
    diagnostics/<name>             scene-specific measurements, such as the catheter's tip
```

Each step stores the action beside the state and observations it was chosen from. Embodiments add Isaac Lab's action and state recorder terms; a scene adds what conversion needs beyond them, such as the 256 px fluoroscopy frame and its attenuation signal. The policy runner records only when asked for a dataset (`--recorder_dataset_export_dir_path`), and then records every episode with its success flag. `record_demos.py` records with the same terms and keeps the demonstrations that reach the environment's success termination. Isaac Lab holds an episode on the device until it resets and writes it then.

| Consumer | Use of the recording |
| --- | --- |
| Replay | `replay_demos.py` restores the initial state and reapplies the stored actions |
| Mimic | Perturbs whole episodes |
| Annotation | Grades episodes against the scene's `annotate` criterion |
| Dataset conversion | Converts arrays and embodiment labels into a LeRobot dataset |

The converter derives state and action widths from each recording and uses the embodiment manifest for column labels, which supports controllers whose action vector contains fields that are not articulation joints, including the G1 WBC command tail. Recordings made by the earlier `run.sh` runtime remain readable: they also carry node `segments`, which mimic and annotation can scope to, and their catheter recordings stored the procedure state after each action rather than before it.

## Authoring

Start by deciding which component owns the change:

| Change needed | Edit |
| --- | --- |
| New assets, cameras, layout, or randomization | Scene and scene manifest |
| New robot, action terms, or proprioception | Embodiment and embodiment manifest |
| New goal, success rule, reward, or reset event | Task under `arena/i4h_arena/envcfg/` |
| New scripted behavior | A policy under `arena/i4h_arena/policies/` |
| New teleoperation input | A device under `arena/i4h_arena/devices/`, declared in the scene's `teleop_devices` |
| New checkpoint family or model | A policy stack under `tasks/<stack>/` and its Arena client policy |
| New online RL trainer, model mapping, or hyperparameters | `rl/` profile, declarative config, backend, and optional workflow adapter |

### To Add a Workflow

1. **Create the scene.** `./scripts/create_blank_environment.py <scene>` writes a blank scene, its assets, its manifest, and a contract test. It is an Arena environment from then on, as `i4h_arena.environments:<CamelCase> <scene>`.
2. **Build the world.** Add assets, cameras, and layout to the scene, live with `local-agent/bridge.sh` and `scripts/live_scene_edit.py` or directly in source. Declare its capabilities and step budget in the manifest.
3. **Add the robot and the goal.** Reuse or add an Arena embodiment, and an Arena task whose success termination ends an episode when the goal is met.
4. **Add the ways to run it.** A rule-based controller as an Arena policy, a teleoperation device, and, for a trained checkpoint, a task manifest in its stack with an Arena client policy.
5. **Declare it.** Add `workflows/<specialty>/<workflow>.yaml` naming the scene and each mode's policy, policy server, or devices, so `./run.sh <workflow>` runs it.
6. **Document it.** Add the workflow to its specialty page in `docs/workflows/` with its environment and commands.
7. **Test the owning layer.** Test policies, devices, and recorder terms on CPU; `tests/test_run_workflow.py` checks the declaration. If assets, physics, or cameras changed, finish with a visible simulator run of each mode with `./run.sh`.

Use upstream Isaac Sim skills for generic USD, camera, physics, rendering, and spatial authoring. Keep the i4h layer focused on scene integration, policies, devices, recording, and workflow-specific validation.
