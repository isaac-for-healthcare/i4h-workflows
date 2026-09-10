# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endoluminal navigation whose catheter drive is carried on a robot flange."""

from i4h_engine.graph import TaskGraph, task
from i4h_engine.interface import Workflow
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.teleop import teleop


def success(ctx) -> object:
    """Catheter tip held at the distal end of the vessel centerline.

    The tip lives in a Newton particle buffer that only the simulator side can
    read, so the criterion is evaluated by the termination term the catheter
    embodiment installs and read back here. It is all-false when the workflow
    runs without a patient twin, which leaves no centerline to navigate.
    """
    return ctx.scene.termination("success")


WORKFLOW = Workflow(
    scene="endoluminal_navigation_arm",
    success=success,
    modes={
        "idle": idle,
        # ``until`` ends the episode on arrival, so a recorded demonstration is
        # goal-terminated and carries a success label. Without it teleop only
        # ever stops at the step cap, which is why recording used to need
        # ``--record-failures`` to keep anything at all.
        "teleop": lambda device="catheter_keyboard", **kwargs: teleop(
            device, until=success, max_seconds=float("inf"), **kwargs
        ),
        # No policy mode here on purpose. ``gr00t_n17/catheter_navigation``
        # names embodiment ``catheter`` and this scene provides
        # ``franka_catheter``, so lint refuses the pairing. Recordings made
        # here still train that task -- the arm only appends
        # ``arm.panda_joint*`` state columns, which sit outside the
        # ``catheter``/``carm`` modality groups the task reads -- so roll a
        # checkpoint out in ``endoluminal_navigation`` instead.
        "replay": replay,
        "demo": lambda: TaskGraph(description="Deterministic sweep exercising the arm-borne catheter drive.").flow(
            task("basic/catheter_sweep")
        ),
    },
)
