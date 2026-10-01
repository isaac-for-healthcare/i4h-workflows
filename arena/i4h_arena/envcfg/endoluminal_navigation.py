# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IsaacLab term configs for the catheter navigation goal.

The predicates live in :mod:`i4h_arena.medical.navigation_goal` and the reward
quantities in :mod:`i4h_arena.medical.navigation_reward`; this module is only
the wiring that turns them into term configs, plus the reset events that clear
their per-episode state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass

from i4h_arena.medical.navigation_goal import (
    ARRIVAL_HOLD_STEPS,
    ARRIVAL_TOLERANCE_M,
    reached_navigation_target,
    reset_arrival_progress,
)
from i4h_arena.medical.navigation_observation import (
    drive_state,
    fluoroscopy_image,
    route_state,
    target_offset,
    tip_direction,
    tip_position,
)
from i4h_arena.medical.navigation_reward import (
    approach_reward,
    arrival_reward,
    fold_penalty,
    lateral_offset_penalty,
    reset_approach_potential,
    reset_route_progress,
    route_progress_reward,
    wall_penetration_penalty,
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
    # Both shaping potentials are differenced across a step, so both have to
    # forget the previous episode or the reset itself is scored as a move.
    reset_route_progress = EventTermCfg(func=reset_route_progress, mode="reset")
    reset_approach_potential = EventTermCfg(func=reset_approach_potential, mode="reset")


@configclass
class CatheterNavigationObservationsCfg:
    """Navigation state and the fluoroscopy view. Bound by :func:`navigation_observations_cfg`.

    Terms are left unconcatenated so each one is addressable by name. The
    RLinf bridge composes GR00T's modality dict out of named keys -- see
    :mod:`i4h_rl.adapters.endoluminal_navigation` -- and a flat vector would
    make the image and the state indistinguishable to it. A trainer that
    wants one vector concatenates on its own side, which is cheap; recovering
    named slices from a concatenation is not.
    """

    @configclass
    class NavigationObsCfg(ObservationGroupCfg):
        fluoroscopy_rgb: ObservationTermCfg = MISSING
        tip_position: ObservationTermCfg = MISSING
        tip_direction: ObservationTermCfg = MISSING
        target_offset: ObservationTermCfg = MISSING
        route_state: ObservationTermCfg = MISSING
        drive_state: ObservationTermCfg = MISSING

        def __post_init__(self) -> None:
            # Corruption off: these are simulated instrument readings, and the
            # noise that matters for transfer is in the physics, not here.
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: NavigationObsCfg = MISSING


def navigation_observations_cfg(
    target_world_m: Iterable[float],
    *,
    route_world_m: Iterable[Iterable[float]],
) -> CatheterNavigationObservationsCfg:
    """Bind the navigation observation group to one scene's route and target."""
    return CatheterNavigationObservationsCfg(
        policy=CatheterNavigationObservationsCfg.NavigationObsCfg(
            # Not the stock image term: it renders on read, and the slang
            # backend cannot render until the scene binds a C-arm, which is
            # after the observation manager probes this term's shape.
            fluoroscopy_rgb=ObservationTermCfg(
                func=fluoroscopy_image,
                params={
                    "sensor_cfg": SceneEntityCfg("fluoroscopy"),
                    "data_type": "rgb",
                },
            ),
            tip_position=ObservationTermCfg(func=tip_position),
            tip_direction=ObservationTermCfg(func=tip_direction),
            target_offset=ObservationTermCfg(
                func=target_offset,
                params={"target_world_m": tuple(float(value) for value in target_world_m)},
            ),
            route_state=ObservationTermCfg(
                func=route_state,
                params={"route_world_m": tuple(tuple(float(value) for value in point) for point in route_world_m)},
            ),
            drive_state=ObservationTermCfg(func=drive_state),
        )
    )


#: Pay per metre of vessel closed.
PROGRESS_WEIGHT = 150.0

#: Decay length of the approach potential, about five arrival tolerances.
APPROACH_SCALE_M = 0.025

#: Chosen so the two scales hand off smoothly rather than by taste. The
#: approach potential's slope at the target is ``1 / APPROACH_SCALE_M`` per
#: metre, so this weight makes the last millimetre pay what ``progress`` pays
#: for a millimetre of arc -- which matters because ``progress`` goes flat
#: inside the final route sample and this term is what takes over there. Any
#: smaller and the handoff is a cliff the policy coasts off.
APPROACH_WEIGHT = PROGRESS_WEIGHT * APPROACH_SCALE_M


@configclass
class CatheterNavigationRewardsCfg:
    """Dense navigation objective. Every term is bound by :func:`navigation_rewards_cfg`.

    The weights are a starting point sized against one episode rather than a
    tuned result, and they are the part most likely to need moving. Over the
    600-step cap and the 0.66 m s0011 route: a full traverse pays about 99
    through ``progress``, the fifteen-step hold pays 75 through ``arrival``, and
    a tip pinned against the wall gives up roughly 1.0 per step across
    ``lateral`` and ``penetration``. That ordering -- arriving worth more than
    traversing, traversing worth more than any amount of loitering -- is the
    intent; the exact numbers are not load-bearing.

    Loitering being worth nothing is load-bearing, and is now structural rather
    than a property of the numbers. Every positive term is either a difference
    of a potential, which a stationary tip cannot collect, or ``arrival``,
    which is gated on the tolerance and terminates. No weight choice
    reintroduces a payout for holding still short of the target.

    ``penetration`` is the one term that breaks the 1.0-per-step scale, and
    deliberately. It reads zero until the rod is actually through the wall,
    then 0.2 per step per millimetre of depth, so perforating throughout an
    episode costs more than the traverse pays. Its balance against ``lateral``
    has not been retuned since it changed from a mean to a worst-point.
    """

    progress: RewardTermCfg = MISSING
    approach: RewardTermCfg = MISSING
    arrival: RewardTermCfg = MISSING
    lateral: RewardTermCfg = MISSING
    penetration: RewardTermCfg = MISSING
    fold: RewardTermCfg = MISSING
    action_rate = RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01)


def navigation_rewards_cfg(
    target_world_m: Iterable[float],
    *,
    route_world_m: Iterable[Iterable[float]],
    lumen_radii_m: Iterable[float] | None = None,
    tolerance_m: float = ARRIVAL_TOLERANCE_M,
) -> CatheterNavigationRewardsCfg:
    """Bind the dense reward to one scene's route, lumen widths and target.

    Without ``lumen_radii_m`` the lateral and penetration terms have no wall to
    measure against and are wired at zero weight rather than dropped, so the
    term set stays the same shape across scenes and a log comparing two runs
    lines up.
    """
    route = tuple(tuple(float(value) for value in point) for point in route_world_m)
    radii = None if lumen_radii_m is None else tuple(float(value) for value in lumen_radii_m)
    walled = radii is not None
    return CatheterNavigationRewardsCfg(
        progress=RewardTermCfg(
            func=route_progress_reward,
            weight=PROGRESS_WEIGHT,
            params={"route_world_m": route},
        ),
        # Coarse and fine in one term rather than two: remaining arc already
        # covers the approach at route scale, so what is missing is only the
        # last centimetre the 5 mm tolerance is judged on.
        approach=RewardTermCfg(
            func=approach_reward,
            weight=APPROACH_WEIGHT,
            params={"target_world_m": tuple(float(value) for value in target_world_m), "scale_m": APPROACH_SCALE_M},
        ),
        arrival=RewardTermCfg(
            func=arrival_reward,
            weight=5.0,
            params={
                "target_world_m": tuple(float(value) for value in target_world_m),
                "tolerance_m": float(tolerance_m),
            },
        ),
        lateral=RewardTermCfg(
            func=lateral_offset_penalty,
            weight=-2.0 if walled else 0.0,
            params={"route_world_m": route, "lumen_radii_m": radii},
        ),
        penetration=RewardTermCfg(
            func=wall_penetration_penalty,
            weight=-200.0 if walled else 0.0,
            params={"route_world_m": route, "lumen_radii_m": radii},
        ),
        fold=RewardTermCfg(func=fold_penalty, weight=-1.0),
    )


def navigation_terminations_cfg(
    target_world_m: Iterable[float],
    *,
    tolerance_m: float = ARRIVAL_TOLERANCE_M,
    hold_steps: int = ARRIVAL_HOLD_STEPS,
    route_world_m: Iterable[Iterable[float]] | None = None,
) -> CatheterNavigationTerminationsCfg:
    """Build the ``success`` term for a target in Isaac world metres.

    ``route_world_m`` is the planned centerline the target sits at the end of.
    It rides on the term so the operator readout can report remaining vessel
    from the same place the criterion reads its target, and does not change
    when the episode ends.
    """
    return CatheterNavigationTerminationsCfg(
        success=TerminationTermCfg(
            func=reached_navigation_target,
            time_out=False,
            params={
                "target_world_m": tuple(float(value) for value in target_world_m),
                "tolerance_m": float(tolerance_m),
                "hold_steps": int(hold_steps),
                "route_world_m": (
                    None
                    if route_world_m is None
                    else tuple(tuple(float(value) for value in point) for point in route_world_m)
                ),
            },
        )
    )


__all__ = [
    "CatheterNavigationEventsCfg",
    "CatheterNavigationObservationsCfg",
    "CatheterNavigationRewardsCfg",
    "CatheterNavigationTerminationsCfg",
    "navigation_observations_cfg",
    "navigation_rewards_cfg",
    "navigation_terminations_cfg",
]
