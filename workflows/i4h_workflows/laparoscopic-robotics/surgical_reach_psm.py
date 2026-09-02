# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dVRK PSM moves its tool tip to a sampled target pose."""

from __future__ import annotations

from i4h_engine.interface import Workflow
from i4h_tasks.basic.predicates import near_object
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.rule_based import cartesian_reach


def success(ctx) -> object:
    return near_object(ctx.scene, "reach_target", radius=0.01)


WORKFLOW = Workflow(
    scene="psm_reach",
    success=success,
    modes={
        "rule-based": lambda: cartesian_reach("reach_target", pre_hold_s=0.5),
        "replay": replay,
        "idle": idle,
    },
)
