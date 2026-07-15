# Arena State Machines

Scripted, policy-free rollouts that drive an env to task success. They also generate data: `--state-machine --record-to <file.hdf5>` saves the successful episodes.

```bash
workflows/agentic/arena/run.sh --env scissor_pick_and_place \
    --state-machine --episodes 3 --record-to "$PWD/workflows/agentic/runs/sm/data/demo.hdf5"
```

## Design

Each env's state machine is a `StateMachine` subclass in `statemachine/<env>.py`; the shared base and reusable behaviors live in `statemachine/core/`.

A rollout runs its scripted stages to completion — with `hold_last`, the final stage repeats to fill the reference episode length — and **latches success**: an episode is successful if the task condition held at any step, never just from reaching the timeout. A subclass implements `build_stages(env)` (the scripted actions) and `succeeded(env)` (the success check).

`statemachine/core/` holds `machine.py` (the `StateMachine` base, `Stage`, episode loop, HDF5 recording), `common.py` (scene/tensor helpers), `dispatch.py` (the `--state-machine` CLI glue), and `reach.py` / `lift.py` (the reusable `ReachStateMachine` / `DualReachStateMachine` / `LiftStateMachine` behaviors the surgical envs subclass).

## Envs

- `scissor_pick_and_place` — SO-ARM joint-space keyframes with a live shoulder-pan correction.
- `ultrasound_liver_scan` — Franka relative-IK probe servo along the organ-local scan line.
- `surgical_reach_psm`, `surgical_reach_star`, `surgical_reach_dual_psm` — IK servo to the commanded tool-tip pose(s).
- `surgical_lift_block`, `surgical_lift_needle`, `surgical_lift_needle_organs` — approach, grasp, and lift the object.

## Add a new env

Write `statemachine/<env>.py` with a `StateMachine` subclass (implement `build_stages` + `succeeded`) plus a module-level `run_state_machine(*, args, env, app, controller)` that calls `.run(...)`, then set `state_machine_module = "arena.statemachine.<env>"` on the env class.
