# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author, train, and run a torso-surface ultrasound probe reach."""

from i4h_engine.interface import Workflow
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy


def success(ctx) -> object:
    return ctx.scene.termination("success")


WORKFLOW = Workflow(
    scene="ultrasound_probe_reach",
    success=success,
    modes={
        "idle": idle,
        "policy": lambda: policy("rsl_rl/ultrasound_probe_reach", until=success),
    },
)
