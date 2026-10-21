# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endoluminal navigation whose catheter drive is carried on a robot flange."""

from i4h_engine.graph import TaskGraph, task
from i4h_engine.interface import Workflow
from i4h_tasks.basic.predicates import env_success
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.teleop import teleop

WORKFLOW = Workflow(
    scene="endoluminal_navigation_arm",
    modes={
        "idle": idle,
        # ``until`` ends the episode on arrival -- the catheter embodiment's termination
        # term, all-false without a patient twin -- so a recorded demonstration is
        # goal-terminated and carries a success label. Without it teleop only
        # ever stops at the step cap, which is why recording used to need
        # ``--record-failures`` to keep anything at all.
        "teleop": lambda device="catheter_keyboard", **kwargs: teleop(
            device, until=env_success, max_seconds=float("inf"), **kwargs
        ),
        # Serves ``catheter_navigation_arm`` rather than ``catheter_navigation``
        # only because a task manifest names one embodiment and lint matches it
        # by name. The checkpoint is the same one: both scenes accept the same
        # four velocity channels, and the arm appends ``arm.panda_joint*``
        # columns as a third state group outside the ``catheter``/``carm``
        # groups the task reads. Train in either scene, roll out in either.
        "policy_n17": lambda: policy("gr00t_n17/catheter_navigation_arm", until=env_success),
        "replay": replay,
        "demo": lambda: TaskGraph(description="Deterministic sweep exercising the arm-borne catheter drive.").flow(
            task("basic/catheter_sweep")
        ),
    },
)
