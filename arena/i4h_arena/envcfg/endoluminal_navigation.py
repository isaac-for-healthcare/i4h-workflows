# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IsaacLab term configs for the catheter navigation goal.

The predicates live in :mod:`i4h_arena.medical.navigation_goal`; this module is
only the wiring that turns them into a ``success`` termination term and the
reset event that clears its hold counter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from i4h_arena.medical.navigation_goal import (
    ARRIVAL_HOLD_STEPS,
    ARRIVAL_TOLERANCE_M,
    reached_navigation_target,
    reset_arrival_progress,
)


@configclass
class CatheterNavigationTerminationsCfg:
    """Arrival only.

    No ``time_out`` term on purpose: the runner already enforces the step
    budget, and it runs teleop without one. Adding a time out here would start
    resetting an interactive session that is expected to keep going.
    """

    success: TerminationTermCfg = MISSING


@configclass
class CatheterNavigationEventsCfg:
    reset_arrival_progress = EventTermCfg(func=reset_arrival_progress, mode="reset")


def navigation_terminations_cfg(
    target_world_m: Iterable[float],
    *,
    tolerance_m: float = ARRIVAL_TOLERANCE_M,
    hold_steps: int = ARRIVAL_HOLD_STEPS,
) -> CatheterNavigationTerminationsCfg:
    """Build the ``success`` term for a target in Isaac world metres."""
    return CatheterNavigationTerminationsCfg(
        success=TerminationTermCfg(
            func=reached_navigation_target,
            time_out=False,
            params={
                "target_world_m": tuple(float(value) for value in target_world_m),
                "tolerance_m": float(tolerance_m),
                "hold_steps": int(hold_steps),
            },
        )
    )


__all__ = [
    "CatheterNavigationEventsCfg",
    "CatheterNavigationTerminationsCfg",
    "navigation_terminations_cfg",
]
