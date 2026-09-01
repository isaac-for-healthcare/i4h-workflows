# Workflow Run Modes

A run mode answers one simple question: **How should this workflow run?**

The code and command-line interface use the shorter name `mode`, including `WORKFLOW.modes` and `--mode`. Documentation and other beginner-facing text use **run mode** to make the meaning clear.

## Standard Run Modes

| Friendly label | Mode name | What it does | Shared implementation |
| --- | --- | --- | --- |
| View only | `idle` | Opens and renders the Scene without driving the robot. | [`idle.py`](idle.py) |
| Manual control | `teleop` | Lets a person drive the robot with a supported input device. | [`teleop.py`](teleop.py) |
| Rule-based | `rule-based` | Runs a deterministic task graph in the simulator. | Workflow-specific graph, with reusable Cartesian patterns in [`rule_based.py`](rule_based.py) |
| AI policy | `policy` | Runs a learned policy Task. | [`policy.py`](policy.py) |
| Playback | `replay` | Applies actions from a recorded episode. | [`replay.py`](replay.py) |

A workflow exposes only the run modes it can actually execute. For example:

```python
WORKFLOW = Workflow(
    scene="example_scene",
    modes={
        "idle": idle,
        "rule-based": rule_based,
        "teleop": teleop,
    },
)
```

All run modes use the same Scene and shared `SimulationRunner`. Selecting a run mode changes the `TaskGraph`; it does not select a different runner or create another Scene.

## Workflow-Specific Run Modes

A workflow may expose a specialized run mode when the standard five do not describe a genuinely different graph. Current examples include:

- `policy_n17` for the scissor workflow's GR00T N1.7 policy variant.
- `demo` for the endoluminal workflow's deterministic demonstration.
- `validate_fluoroscopy` for the endoluminal workflow's imaging diagnostic.

These names are extensions for one workflow, not additional standard run modes. Prefer a standard mode when it accurately describes the execution path, and use checkpoint or runtime configuration instead of creating a new mode for ordinary parameter changes.

## Ownership

- Keep reusable run-mode builders in this directory.
- Keep workflow-specific graph builders in the owning workflow module.
- Keep each workflow's supported run modes in its `WORKFLOW.modes` mapping.
- Keep the Engine independent of concrete run-mode names; it executes the selected `TaskGraph`.
