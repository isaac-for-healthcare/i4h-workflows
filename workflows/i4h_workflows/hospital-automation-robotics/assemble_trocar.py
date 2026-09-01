# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 with dex hands installs a trocar from the box."""

from __future__ import annotations

from i4h_engine.interface import Workflow
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy
from i4h_workflow_modes.replay import replay


def success(ctx) -> object:
    """Use the env's five-stage lift/align/insert/place/release criterion."""
    return ctx.scene.termination("success")


WORKFLOW = Workflow(
    scene="g1_trocar",
    success=success,
    modes={
        "policy": lambda: policy("gr00t_n15/assemble_trocar", until=success),
        "replay": replay,
        "idle": idle,
    },
)
