# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endoluminal navigation workflow with synchronized fluoroscopy."""

from i4h_engine.graph import TaskGraph, task
from i4h_engine.interface import Workflow
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.teleop import teleop

WORKFLOW = Workflow(
    scene="endoluminal_navigation",
    modes={
        "idle": idle,
        "teleop": lambda device="catheter_keyboard", **kwargs: teleop(device, max_seconds=float("inf"), **kwargs),
        "demo": lambda: TaskGraph(description="Deterministic catheter/fluoroscopy demonstration.").flow(
            task("basic/catheter_sweep")
        ),
        "validate_fluoroscopy": lambda: TaskGraph(
            description="Autonomous C-arm motion and patient-backed fluoroscopy image validation."
        ).flow(task("basic/fluoroscopy_carm_sweep")),
    },
)
