# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-env task definitions.

Two file shapes, mirroring IsaacLab-Arena's ``tasks/`` layout:

* ``<env>_task.py`` — a :class:`isaaclab_arena.tasks.task_base.TaskBase`
  subclass that builds the termination / event / reward / mimic config for
  one env (scissor, tray-pick, push-cart, assemble-trocar).
* ``<env>_mdp.py``  — task-specific MDP terms (events, observations,
  rewards) re-exporting ``isaaclab.envs.mdp`` so callers can use one
  ``mdp.xxx`` symbol regardless of where it's defined (assemble_trocar,
  ultrasound).
"""
