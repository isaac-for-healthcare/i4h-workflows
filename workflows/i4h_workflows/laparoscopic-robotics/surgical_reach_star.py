# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""STAR arm reaches a sampled target pose."""

from __future__ import annotations

from i4h_engine.interface import Workflow
from i4h_tasks.basic.predicates import near_object
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.rule_based import cartesian_reach


def success(ctx) -> object:
    return near_object(ctx.scene, "reach_target", radius=0.03)


WORKFLOW = Workflow(
    scene="star_reach",
    success=success,
    modes={
        # STAR's Cartesian controller needs a brief initialized hold before
        # its first target command.
        "rule-based": lambda: cartesian_reach(
            "reach_target",
            pre_hold_s=0.5,
            duration_s=0.01,
            position_tolerance=0.03,
            # Let the scene's 150-step budget be authoritative. STAR can
            # need nearly the full rollout to converge for edge targets.
            settle_timeout_s=10.0,
            timeout_success=success,
        ),
        "replay": replay,
        "idle": idle,
    },
)
