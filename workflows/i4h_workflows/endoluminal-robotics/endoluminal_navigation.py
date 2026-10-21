# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endoluminal navigation workflow with synchronized fluoroscopy."""

from i4h_engine.graph import TaskGraph, task
from i4h_engine.interface import Workflow
from i4h_tasks.basic.predicates import env_success
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.teleop import teleop

WORKFLOW = Workflow(
    scene="endoluminal_navigation",
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
        "policy_n17": lambda: policy("gr00t_n17/catheter_navigation", until=env_success),
        "replay": replay,
        "demo": lambda: TaskGraph(description="Deterministic catheter/fluoroscopy demonstration.").flow(
            task("basic/catheter_sweep")
        ),
        "validate_fluoroscopy": lambda: TaskGraph(
            description="Autonomous C-arm motion and patient-backed fluoroscopy image validation."
        ).flow(task("basic/fluoroscopy_carm_sweep")),
    },
)
