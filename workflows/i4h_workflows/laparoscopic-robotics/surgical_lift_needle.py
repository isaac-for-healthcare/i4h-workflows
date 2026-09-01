# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dVRK PSM lifts a suture needle with its rule-based controller."""

from __future__ import annotations

from i4h_engine.interface import Workflow
from i4h_tasks.basic.predicates import object_above
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.rule_based import cartesian_lift


def success(ctx) -> object:
    # Cartesian surgical SceneViews expose object poses in the PSM-root frame.
    # The root is at world z=0.15 m, so this is world z > 0.065 m.
    return object_above(ctx.scene, "needle", -0.085)


WORKFLOW = Workflow(
    scene="psm_needle",
    success=success,
    modes={
        "rule-based": lambda: cartesian_lift("needle", success=success),
        "replay": replay,
        "idle": idle,
    },
)
