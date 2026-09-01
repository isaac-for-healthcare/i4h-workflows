# Workflow Architecture

A workflow describes **what should happen** without taking ownership of **how the simulator advances**. The same architecture runs learned policies, rule-based controllers, teleoperation, replay, and idle scene inspection:

```text
selected workflow run mode
  → TaskGraph
  → Engine decides which tasks run
  → tasks read the SceneView and write actuation
  → SimulationRunner advances and records the simulator
```

This separation provides one simulator lifecycle for every workflow, reusable tasks across scenes, static validation before Isaac Sim starts, and process isolation for incompatible policy dependencies.

All paths in this document are relative to `./`.

## Architecture at a Glance

The design separates authored definitions from shared runtime services.

### Authoring Concepts

| Concept | Defines | Does not own | Location |
| --- | --- | --- | --- |
| Scene | One simulated world: assets, embodiment, cameras, randomization, scene-specific view/actuation adapters, and reset hooks | Episode loops, workflow execution, or goal semantics | `arena/i4h_arena/scenes/` |
| Task | One reusable capability with typed inputs, typed outputs, and Scene requirements | Simulator stepping or episode lifecycle | `tasks/<project>/` |
| Workflow | The selected Scene, exposed run modes, run-mode-specific `TaskGraph` builders, and goal semantics | Scene construction, graph execution, or simulator stepping | `workflows/i4h_workflows/<specialty>/` |

### Runtime Services

| Service | Owns | Does not own | Location |
| --- | --- | --- | --- |
| Engine | `TaskGraph` node activation, typed data wiring, task status, node-level retries, timeouts, and workflow events | Simulator reset, rendering, or `env.step` | `engine/` |
| Simulation Runner | Whole-episode attempts, simulator reset/step/render loop, recording, publication, and run summaries | Assets, scene-specific mappings, task behavior, or graph decisions | `arena/i4h_arena/runner.py` |

`arena/` is the simulator integration component. It contains the Scene implementations, adapters, command-line entry point, recorder, and shared `SimulationRunner`; “Arena” is not another authoring concept.

There is one author-facing `Workflow` concept. Each workflow module exports a `WORKFLOW` value that names its Scene, lists its run modes, and optionally exposes a goal predicate. Selecting a run mode builds a `TaskGraph`; the graph contains only nodes and edges and carries no workflow name, Scene, or mode.

The loader combines the filename-derived name, selected run mode, Scene, and built `TaskGraph` into an internal `ResolvedWorkflow`. This runtime value transports already-authored information to lint, rendering, recording, and `SimulationRunner`; authors never construct it.

A Task reads the lightweight `SceneView` adapter in `ctx.scene`, writes through the `Actuation` adapter in `ctx.act`, and returns a status. It does not receive the authored Scene object and never advances the simulator. `SimulationRunner` is the only owner of `env.step` during normal Workflow execution, which gives every Scene, Workflow, and run mode the same timing, recording, and episode-attempt behavior. Online RL is a separate training lifecycle whose trainer owns vectorized stepping; it is not a Workflow run mode.

### Execution Sequence

Loading and validation happen before any simulator dependency is imported:

```text
CLI selects a workflow and mode
  → loader imports the module's WORKFLOW value
  → selected run-mode builder creates a TaskGraph
  → Registry resolves Scene and Task specifications from manifests
  → lint validates the graph and matches registered Task requirements to the Scene
  → Arena loads the Scene and builds the simulator environment
```

One episode attempt then follows the same loop for every mode:

```text
SimulationRunner resets the environment
  → Scene reset hooks run and the view/actuation adapters are refreshed
  → Engine activates and ticks workflow tasks
  → tasks read the scene view and write actuation
  → SimulationRunner advances env.step only when the Engine requests it
  → SimulationRunner records and publishes the resulting frame
  → repeat until the workflow succeeds, fails, or is aborted
```

Scene is the per-world extension point; `SimulationRunner` is shared infrastructure. Change a Scene when a world needs different assets, cameras, action layout, observation aliases, or reset preparation. Change `SimulationRunner` only when the episode or simulator lifecycle must change consistently for every Scene and mode.

Retries exist at two different levels: the Engine applies a node's `FailurePolicy`, while `SimulationRunner` starts a fresh episode attempt after a failed workflow.

## Dependency Boundaries

These boundaries answer two questions: which code runs in the same process, and which layers that code may import. They exist because Isaac Sim, GR00T releases, and openpi require incompatible dependency sets.

### Process Separation

```text
┌─ Simulator process ─────────┐       Zenoh       ┌─ Policy backend process ─┐
│ arena                       │◄─────────────────►│ tasks/gr00t_* or openpi  │
│ workflows                   │                   │ common                    │
│ tasks/basic, ik, teleop     │                   └───────────────────────────┘
│ tasks/rsl_rl (TorchScript)  │
│ engine                      │
│ common                      │                   ┌─ Offline process ─────────┐
└─────────────────────────────┘                   │ tools/*                   │
                                                  │ common                    │
                                                  └───────────────────────────┘
```

Incompatible policy backends never load into the simulator process. The simulator sees a generic `RemoteTask`; observations and actions cross the Zenoh boundary using contracts from `common`. A small exported actor that is compatible with the simulator's existing Torch runtime may instead be an in-process Task under `tasks/rsl_rl`; it still reads `ctx.scene`, writes `ctx.act`, and never steps the simulator. Offline dataset tools are independent of the simulator runtime.

### Online RL Training

Online RL deliberately sits outside the normal Workflow runner:

```text
Scene + embodiment + RL objective
  → rl/ resolves the profile and builds vectorized Isaac Lab environments
  → selected trainer runs policy rollouts and updates
  → RL checkpoint
  → exported in-process or remote policy Task
  → SimulationRunner validation
```

The Scene and Arena task configuration remain the source of simulator assets, observations, actions, resets, rewards, and terminations. The RL layer owns trainer selection, model mapping, hyperparameters, and the adapter between that maintained environment and the trainer; each profile selects those declarations for one Workflow.

A compatible trainer such as RSL-RL may run with Arena in one isolated process. An incompatible stack such as GR00T N1.5/RLinf runs its model controller separately from the current Isaac Sim/Arena process and exchanges only observation/action data through a local bridge.

Incompatible model imports remain forbidden in the simulator process.

`rl/profiles/<workflow>.yaml` declares supported workflows, and `rl/config/*.yaml` owns declarative backend hyperparameters. Every profile uses the same schema and config location. Generic backend code translates those declarations into its runtime; `rl/i4h_rl/adapters/<workflow>.py` exists only when a Scene needs workflow-specific observation, action, registration, or evaluation mapping. `train.sh rl` resolves and validates these contracts without starting Isaac Sim, and the heavy launcher starts only after runtime preflight.

`ultrasound_probe_reach` uses RSL-RL PPO from scratch and exports a simulator-compatible TorchScript Task. Trocar uses GR00T N1.5 online PPO post-training through RLinf and returns to its existing remote policy Task. These are different trainer profiles, not changes to the Workflow abstraction.

This lifecycle follows the [IsaacLab-Arena reinforcement-learning workflow](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/reinforcement_learning_workflows/index.html): build the Arena environment for training, let the selected RL framework own optimization, and evaluate the trained policy through Arena. The compact ultrasound example uses conventional RSL-RL; the Trocar profile substitutes RLinf VLA post-training.

### Offline Tools

Tools consume recordings or derived datasets in their own uv environments. They do not import Arena, advance the simulator, define task success, or own policy inference.

| Component | Entry point | Primary input | Output or effect | Responsibility |
| --- | --- | --- | --- | --- |
| Trajectory mimic | `uv run --project tools/mimic i4h-mimic` | Workflow HDF5 | Expanded workflow HDF5 | Clone successful demonstrations with bounded action jitter, optionally scoped to one recorded node segment |
| VLM annotator | `uv run --project tools/annotator i4h-annotator` | Workflow HDF5 in `offline` mode, or live Arena frames in `live` mode | Episode labels and optional filtered HDF5 | Judge visible task outcomes through an OpenAI-compatible vision endpoint |
| Dataset utility | `uv run --project tools/dataset i4h-dataset` | Workflow HDF5 | Inspection reports, decoded actions, or a LeRobot dataset | Inspect the shared recording contract and convert successful episodes for training or visualization |
| LeRobot viewer | `tools/dataset/scripts/viz.sh` | Completed LeRobot dataset | Local HTML viewer | Serve videos and state/action timelines without modifying the dataset |
| Cosmos adapter | `uv run --project tools/cosmos i4h-cosmos` | Workflow HDF5 or augmented MP4 files | Per-camera MP4 exports or an HDF5 recording with imported visual variants | Bridge recordings to and from visual augmentation without inventing actions or states |

`tools/annotator/scripts/vllm.sh` manages the default local Qwen VLM container for annotation. `tools/cosmos/scripts/serve.sh` manages the optional local Cosmos service. These helpers own only their external service lifecycle; the corresponding CLI owns validation and data transformation.

### Imports Inside the Simulator Process

An arrow means “may import.” A layer may also import any layer below it, but never a layer above it:

```text
arena
  ↓
workflows
  ↓
tasks/basic, tasks/ik, tasks/teleop
  ↓
engine
  ↓
common
```

The practical rules are:

- `common` imports no other i4h layer.
- `engine` defines workflow and task interfaces, but knows no concrete workflows, tasks, Arena, or Isaac Sim.
- In-process tasks never import workflows or Arena.
- Workflows compose tasks but never import Arena, Isaac Sim, or policy stacks.
- Arena may use every simulator-side layer, but never imports GR00T or openpi packages.
- Among i4h layers, remote policy backends and offline tools import `common` only.

Each component is an independent uv project so its environment contains only the dependencies required for that process. `tests/test_layering.py` enforces the import rules and keeps workflow discovery and linting usable without Isaac Sim or a policy stack.

## Task Graph

A task graph answers two separate questions:

1. **When may a task run?** Control edges define ordering.
2. **What data does the task receive?** Data edges connect typed ports.

| Graph element | Meaning | Authoring API |
| --- | --- | --- |
| Node | One invocation of an in-process or remote task | `task("project/name", ...)` or `node(Task(...))` |
| Control edge | The source must succeed before the destination may start | `source >> destination` |
| Data edge | One source output supplies one destination input | `.wire(source.out.value, destination.in_.value)` |
| Failure policy | Abort the workflow or retry only that node | `.on_failure(node, "retry", times=2)` |

```python
locate = node(Locate("needle", name="locate"))
approach = node(Approach(name="approach"))
grasp = node(Grasp(object="needle", name="grasp"))

graph = (
    TaskGraph()
    .flow(locate >> approach >> grasp)
    .wire(locate.out.pose, approach.in_.target)
)
```

Control and data are deliberately independent. `locate >> approach` orders the nodes but does not imply which output should feed which input. The Engine performs a narrow automatic wiring only when there is exactly one compatible choice; ambiguous cases must use `.wire(...)`.

### Node Lifecycle

```text
predecessors succeed
  → Engine resolves typed inputs
  → on_enter(ctx, inputs) runs once
  → tick(ctx) runs once per Engine tick
      RUNNING → keep the node active
      WAITING → hold until external work arrives
      SUCCESS → call on_exit(), publish outputs, unlock successors
      FAILURE → apply the node's failure policy
```

The workflow succeeds when all terminal nodes in its selected `TaskGraph` succeed. A node failure aborts the workflow unless that node has a retry policy.

Goal predicates do not replace graph completion. `WORKFLOW.success` is an author-owned predicate that mode builders may pass to a policy or teleop Task as an `until` condition. The Engine does not call it automatically. A `TaskGraph` may also define `timeout_success`, which the Engine evaluates only when the graph exhausts its step budget.

An Engine tick is not automatically a simulation step. If every active node returns `WAITING`, the Engine asks `SimulationRunner` to update the application without advancing physics or consuming the workflow step budget. This is how a remote task can wait for its backend without freezing the UI or moving the world.

Parallel branches are allowed. They may write to different robots or actuation channels in the same tick; two nodes writing the same channel is a hard error.

## Run Modes

A run mode answers **how the workflow should run**. In code it is a named builder for a task graph, stored in `WORKFLOW.modes` and selected with `--mode`. It is not a separate Runner and does not create a separate scene implementation.

```text
CLI mode
  → WORKFLOW.modes[mode]
  → builder returns a TaskGraph
  → scene capabilities and task requirements are linted
  → the shared SimulationRunner executes it
```

| Run mode | Graph normally contains | External requirement | Runtime behavior |
| --- | --- | --- | --- |
| `idle` | A local wait task | None | Render the scene without advancing physics |
| `policy` | A learned-policy task | A compatible checkpoint and, for isolated stacks, its policy backend | Run a compatible exported actor in-process or exchange observations/actions with a remote backend |
| `rule-based` | Local controller tasks | None | Execute a deterministic task graph in the simulator process |
| `teleop` | A local teleoperation task | A supported input device | Convert operator input into scene actuation |
| `replay` | A local recorded-action task | An HDF5 episode | Apply stored actions through the current scene adapter |

A workflow exposes only the run modes it can actually execute. `WORKFLOW.default_mode` selects the run mode used when the CLI does not provide one and defaults to `idle`. `rule-based` is the standard run mode for local controller graphs. The canonical vocabulary and friendly labels live in `workflows/i4h_workflow_modes/README.md`.

Every run mode of a workflow selects the same scene name, although the scene manifest may provide mode-specific capability overrides such as action space, cameras, or control frequency. Policy tasks remain ordinary graph nodes; an incompatible model stack crosses a process boundary while a simulator-compatible exported actor may run in-process.

## Manifests

Manifests make facts available without importing the code or environment that owns those facts. Python remains the source of executable behavior.

| Manifest | Owns | Main consumers |
| --- | --- | --- |
| `arena/i4h_arena/scenes/manifest/*.yaml` | Scene implementation path, embodiment, action space, DOF, cameras, objects, robots, control rate, and mode overrides | Registry, lint, Arena |
| `arena/i4h_arena/embodiments/manifest/*.yaml` | Cross-process robot metadata such as state/action labels, joint names, calibration, and supported teleop devices | Arena adapters, teleop, dataset conversion |
| `tasks/*/i4h_tasks/*/manifest/*.yaml` | Every task's summary and optional richer prompt; implementation path for in-process tasks; requirements, observation/model configuration, and backend declaration for remote tasks | Registry, lint, remote backend, training tools |

Every task manifest owns a concise `summary` and may add a `prompt` only when the runtime instruction needs more detail. An in-process task's Python class owns its ports, requirements, and behavior; its manifest points to the implementation. A remote task declares the remainder of its contract in YAML because its implementation cannot be imported into the simulator environment.

```text
manifests
  → Registry discovers task and scene specifications
  → workflow builders refer to those specifications by ID
  → lint compares registered task requirements with the selected scene
  → only a valid workflow proceeds to Isaac Sim or a policy backend
```

Use YAML for a fact that must cross a process or dependency boundary. Keep scene construction, goal predicates, controller graphs, and task behavior in Python. Do not repeat the same fact in both places.

Lint always checks graph structure and typed ports. The requirements-versus-provides check currently applies to registry-backed nodes; a directly instantiated in-process `Task` supplies its ports from the class but does not receive that capability check. Prefer a registry ID when full static compatibility checking is important.

## Remote Policy Protocol

A learned-policy stack runs outside the simulator because its dependencies may conflict with Isaac Sim. The simulator represents every such stack with the same generic `RemoteTask`; task manifests select the concrete backend and model.

| Participant | Responsibility |
| --- | --- |
| `RemoteTask` in the simulator | Start a task session, publish observations, wait without stepping when necessary, validate returned actions, and write them to `ctx.act` |
| Policy backend | Load the checkpoint, reset per-task session state, run inference, and publish action chunks and status |
| `common` | Define bus keys and typed messages understood by both processes |

```text
simulator                                       policy backend
  task spec + run/episode identity  ──────────► load or reset session
                                    ◄────────── ready + action contract
  observation + camera frames       ──────────► infer
                                    ◄────────── action chunk
  apply one action per sim step
  repeat observations/actions       ◄─────────► until terminal status
```

The backend reports the action space, layout, robots, gripper convention, and width after it has loaded the checkpoint. `RemoteTask` compares that live contract with the scene before applying an action. A mismatch fails the node instead of sending ambiguous commands to the robot.

Within one run, each whole-episode attempt gets a distinct task UID. Observations, actions, and status use that UID, so a delayed or cached action from the preceding attempt cannot enter the next one. Runs of the same workflow that execute concurrently must use different `--namespace` values. Runtime prompts and backend extras apply only to remote nodes; checkpoint overrides also apply to model-backed in-process Tasks, which receive the simulator device selected by the CLI.

## Recording

Recording has three inputs with distinct owners:

```text
Engine node events ───────► identify the active task segment
SimulationRunner frames ──► capture action, state, and cameras after a step
episode result ───────────► commit or discard the buffered attempt
```

The shared HDF5 schema is defined in `common/i4h_common/episode.py` so every writer and offline reader agrees on the layout:

```text
/data                             attrs: workflow, mode, scene, total
  /demo_N                         attrs: success, status, episode_index, attempt_index
    actions                       (T, action_width)
    obs/joint_pos                 (T, state_width)
    obs/<camera>                  (T, height, width, 3)
    segments                      (node, task_id, start, end)
```

Segments use half-open frame ranges `[start, end)` and associate each range with a workflow node and registry task ID. They are optional for compatibility with older recordings, and all readers must tolerate their absence.

| Consumer | Use of the recording |
| --- | --- |
| Replay | Reapply the stored action sequence through the original workflow |
| Mimic | Perturb an entire episode or a selected node segment |
| Annotation | Grade an episode or selected capability segment |
| Dataset conversion | Convert arrays and embodiment labels into a LeRobot dataset |

The converter derives state and action widths from each recording rather than assuming they are equal. It uses the embodiment manifest for column labels, which supports controllers whose action vector contains fields that are not articulation state joints, including the G1 WBC command tail.

The current recorder stores environment 0 only. The Engine can execute parallel nodes, but the recorder currently tracks one open node segment at a time; workflows that need accurate overlapping segment attribution require a per-node recorder extension.

## Authoring

Start by deciding which concept owns the change:

| Change needed | Edit |
| --- | --- |
| New assets, cameras, layout, randomization, view aliases, or actuation mapping | Scene and scene manifest |
| New reusable behavior | Task and task manifest |
| New composition, supported run mode, or success rule | Workflow definition |
| New cross-process robot labels, calibration, or teleop support | Embodiment manifest |
| New episode-wide simulator lifecycle behavior for every workflow | `SimulationRunner` |
| New graph execution semantics for every workflow | Engine |
| New online RL trainer, model mapping, or hyperparameters | `rl/` profile, declarative config, backend, and optional workflow adapter |

The author-facing workflow contract is intentionally separate from its implementation:

| File | Purpose |
| --- | --- |
| `engine/i4h_engine/interface.py` | The small author-facing `Workflow` value: `scene`, `modes`, optional `success`, and `default_mode` |
| `engine/i4h_engine/graph.py` | The graph-building API: `TaskGraph`, `node`, `task`, control edges, data wiring, and failure policies |
| `workflows/i4h_workflows/<specialty>/<name>.py` | One specialty-grouped authored module exporting exactly one `WORKFLOW = Workflow(...)` value |
| `workflows/README.md` | Workflow source layout and specialty catalog |
| `workflows/i4h_workflow_modes/README.md` | Standard run-mode vocabulary, friendly labels, shared builders, and extension rules |

### To Add a Workflow

1. **Choose the world.** Reuse a Scene when the required robot, objects, cameras, and action space already exist. Otherwise add the Scene first and declare those capabilities in its manifest.
2. **Choose the capabilities.** Search the Task registry for reusable actions such as locate, approach, grasp, or policy inference. Add a Task only when the behavior itself is new.
3. **Compose one graph per run mode.** A mode builder creates Tasks, orders them with `>>`, wires typed data, and returns a `TaskGraph`.
4. **Export the Workflow.** Add `workflows/i4h_workflows/<specialty>/<name>.py` with one `WORKFLOW` value that selects the Scene and maps run-mode names to graph builders. Use one of the approved product specialties documented in `workflows/README.md`.

For example, a new reach workflow can be understood from one file:

```python
from i4h_engine.graph import TaskGraph, task
from i4h_engine.interface import Workflow
from i4h_tasks.basic.predicates import near_object

from i4h_workflow_modes.idle import idle


def success(ctx):
    return near_object(ctx.scene, "reach_target", radius=0.01)


def rule_based() -> TaskGraph:
    locate = task("basic/locate", id="locate", object="reach_target")
    reach = task("ik/move_to_pose", id="reach")
    return (
        TaskGraph(description="Locate and reach the sampled target.")
        .flow(locate >> reach)
        .wire(locate.out.pose, reach.in_.target)
    )


WORKFLOW = Workflow(
    scene="psm_reach",
    success=success,
    modes={
        "rule-based": rule_based,
        "idle": idle,
    },
)
```

Reading from the bottom upward: `WORKFLOW` says what users can run, `rule_based()` says how that run mode is composed, and `success()` defines the goal a run mode may use for early completion. The Scene owns the world; the Tasks own the behavior; this module owns only their composition.

1. **Inspect before running.** From the repository root, run:

   ```bash
   ./run.sh show <name> --mode rule-based
   ./run.sh lint <name> --mode rule-based
   ./run.sh lint --all
   ```

2. **Test the owning layer.** Test new Task or graph behavior on CPU. If assets, physics, cameras, or Scene adapters changed, finish with a visible simulator validation.

A new workflow should never need a custom `SimulationRunner`. If it appears to, first check whether the behavior belongs in a Scene reset hook, a Task, a success predicate, or a reusable Engine feature.

Use upstream Isaac Sim skills for generic USD, camera, physics, rendering, and spatial authoring. Keep the i4h layer focused on workflow/task wiring, scene integration, policy contracts, recording, and workflow-specific validation.
