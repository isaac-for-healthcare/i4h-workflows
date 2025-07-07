# IsaacLab simulation for robotic ultrasound

This folder contains the scripts to run the simulation in the IsaacSim environment.
- [PI0 Policy Evaluation in IsaacSim](./imitation_learning/README.md)
- [Sim with DDS](./environments/README.md)
- [Data Recording and Replaying with State Machine](./environments/state_machine/README.md)
- [Teleoperation](./environments/teleoperation/README.md)
- [Ultrasound Raytracing Simulation](./examples/README.md)
- [Trajectory Evaluation](./evaluation/README.md)

## Task setting

Isaac-Lab scripts usually receive a `--task <task_name>` argument. This argument is used to select the environment/task to be run. In the robotic ultrasound workflow, the task is set to `Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0` by default, but you can change it to other tasks by passing the `--task` argument.

## Jump to Section

- [Policy Runner with DDS](../policy_runner/README.md)
- [Task setting for the simulation environment](./exts/robotic_us_ext/README.md)
