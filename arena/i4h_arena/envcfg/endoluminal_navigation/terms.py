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
    TerminationTermCfg,
)
from isaaclab.utils.configclass import configclass

from i4h_arena.medical.navigation_goal import (
    ARRIVAL_HOLD_STEPS,
    reached_navigation_target,
    reset_arrival_progress,
    resolve_arrival_tolerance_m,
)
from i4h_arena.medical.navigation_observation import (
    drive_state,
    fluoroscopy_rgb,
    route_state,
    target_offset,
    tip_direction,
    tip_position,
)
from i4h_arena.medical.navigation_reward import (
    MAX_STEP_ADVANCE_M,
    lateral_offset_penalty,
    reset_route_progress,
    reset_tip_route_state,
    route_progress_reward,
    wall_penetration_penalty,
)


@configclass
class CatheterNavigationTerminationsCfg:
    """Arrival, and running out of time.

    ``time_out`` is declared ``time_out=True`` so Isaac Lab reports it as
    truncation rather than termination, which is what a value function needs to
    tell a step budget apart from an outcome: ``rsl_rl.algorithms.ppo`` adds
    ``gamma * value`` for every environment flagged in ``extras["time_outs"]``,
    and that flag is the ``time_out=True`` terms and nothing else. Without the
    term it is false forever, so an episode whose only other ending is arrival
    does not end at all -- and a policy that has not learned to arrive yet
    never sees a reset.

    The RLinf bridge caps episodes itself, so it does not need this. The stock
    RSL-RL runner does not, and ``app.py`` strips the term for interactive runs
    where the workflow runner owns the budget. Declared here so every consumer
    gets the signal and the one that must not have it removes it explicitly.
    """

    success: TerminationTermCfg = MISSING
    time_out: TerminationTermCfg = TerminationTermCfg(func=base_mdp.time_out, time_out=True)


@configclass
class CatheterNavigationEventsCfg:
    reset_arrival_progress = EventTermCfg(func=reset_arrival_progress, mode="reset")
    # Route progress is differenced across a step, so it must forget the
    # previous episode or the reset itself would be scored as a move.
    reset_route_progress = EventTermCfg(func=reset_route_progress, mode="reset")
    # IsaacLab rewards, then resets, then observes, all under one step
    # counter, so the step's cached tip projection has to go with the reset
    # or the new episode's first observation describes the old tip.
    reset_tip_route_state = EventTermCfg(func=reset_tip_route_state, mode="reset")


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
            # Shape discovery runs before make_view(), so the image term
            # binds the sensor's live sources before attempting to render.
            fluoroscopy_rgb=ObservationTermCfg(func=fluoroscopy_rgb),
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


#: Normalize one physically reachable step along the curved route to one.
ROUTE_PROGRESS_WEIGHT = 1.0 / MAX_STEP_ADVANCE_M

#: Centerline tracking is guidance, not the primary objective.
LATERAL_WEIGHT = -0.1

#: Penetration depth, in metres, that scores a penalty of one. Equal to
#: ``MAX_STEP_ADVANCE_M`` on purpose, so a full-scale wall violation cancels a
#: full step of advance, but its own constant because the two measure
#: different things: a depth into the vessel wall against a ceiling on drive
#: speed divided by the control rate. Retuning the clamp should not reweight
#: wall contact behind the author's back.
PENETRATION_SCALE_M = 0.0025

#: Normalize a full-scale penetration to a penalty of one.
PENETRATION_WEIGHT = -1.0 / PENETRATION_SCALE_M

#: Endoluminal demonstrations are labelled successful within 8 mm of the
#: target, so the RL termination defaults to that while the shared medical
#: navigation default stays at the stricter 5 mm.
ENDOLUMINAL_DEFAULT_ARRIVAL_TOLERANCE_M = 0.008

#: Resolved through the same environment variable as every other arrival
#: criterion. ``catheter.py`` builds the termination without naming a
#: tolerance, so a constant here would make ``I4H_CATHETER_ARRIVAL_MM`` a no-op
#: for the one workflow it exists to serve: the collection session would log
#: and label at the tolerance it asked for while the episode ended at 8 mm.
ENDOLUMINAL_ARRIVAL_TOLERANCE_M = resolve_arrival_tolerance_m(default_m=ENDOLUMINAL_DEFAULT_ARRIVAL_TOLERANCE_M)


@configclass
class CatheterNavigationRewardsCfg:
    """Three terms: advance along the route, track it, and avoid walls."""

    route_progress: RewardTermCfg = MISSING
    lateral: RewardTermCfg = MISSING
    penetration: RewardTermCfg = MISSING


def navigation_rewards_cfg(
    target_world_m: Iterable[float],
    *,
    route_world_m: Iterable[Iterable[float]],
    lumen_radii_m: Iterable[float] | None = None,
) -> CatheterNavigationRewardsCfg:
    """Bind the simple reward to one scene's curved route and vessel wall.

    Without ``lumen_radii_m`` the lateral and penetration terms have no wall to
    measure against and are wired at zero weight rather than dropped, so the
    term set stays the same shape across scenes and a log comparing two runs
    lines up.
    """
    route = tuple(tuple(float(value) for value in point) for point in route_world_m)
    radii = None if lumen_radii_m is None else tuple(float(value) for value in lumen_radii_m)
    walled = radii is not None
    return CatheterNavigationRewardsCfg(
        route_progress=RewardTermCfg(
            func=route_progress_reward,
            weight=ROUTE_PROGRESS_WEIGHT,
            params={"route_world_m": route},
        ),
        lateral=RewardTermCfg(
            func=lateral_offset_penalty,
            weight=LATERAL_WEIGHT if walled else 0.0,
            params={"route_world_m": route, "lumen_radii_m": radii},
        ),
        penetration=RewardTermCfg(
            func=wall_penetration_penalty,
            weight=PENETRATION_WEIGHT if walled else 0.0,
            params={"route_world_m": route, "lumen_radii_m": radii},
        ),
    )


def navigation_terminations_cfg(
    target_world_m: Iterable[float],
    *,
    tolerance_m: float = ENDOLUMINAL_ARRIVAL_TOLERANCE_M,
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
    "ENDOLUMINAL_ARRIVAL_TOLERANCE_M",
    "ENDOLUMINAL_DEFAULT_ARRIVAL_TOLERANCE_M",
    "navigation_observations_cfg",
    "navigation_rewards_cfg",
    "navigation_terminations_cfg",
]
