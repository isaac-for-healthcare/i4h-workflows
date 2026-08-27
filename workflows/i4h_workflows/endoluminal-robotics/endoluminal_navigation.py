# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endoluminal navigation workflow with synchronized fluoroscopy."""

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
    scene="endoluminal_navigation",
    success=success,
    modes={
        "idle": idle,
        "teleop": lambda device="catheter_keyboard", **kwargs: teleop(device, max_seconds=float("inf"), **kwargs),
        "replay": replay,
        "demo": lambda: TaskGraph(description="Deterministic catheter/fluoroscopy demonstration.").flow(
            task("basic/catheter_sweep")
        ),
        "validate_fluoroscopy": lambda: TaskGraph(
            description="Autonomous C-arm motion and patient-backed fluoroscopy image validation."
        ).flow(task("basic/fluoroscopy_carm_sweep")),
    },
)
