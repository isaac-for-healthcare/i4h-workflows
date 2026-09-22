# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When catheter navigation has succeeded: the tip held at the distal centerline.

The arithmetic lives here rather than in the env cfg because env cfg modules
cannot be imported before Kit starts, and this is the part worth testing on CPU.
The env cfg beside it only wraps these functions in IsaacLab term configs.

Insertion moves the rod's kinematic root particle along its local tangent and
XPBD carries that down the chain, so the tip genuinely advances through the
lumen; arriving at the far end of the centerline is something a controller has
to achieve rather than something initialization hands it.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from i4h_arena.medical.route_progress import RouteProgress, route_progress

_LOGGER = logging.getLogger(__name__)

DRIFT_LOG_ENV_VAR = "I4H_CATHETER_DRIFT"

#: Tip-to-target distance that counts as arrival. Lumens in the shipped twins
#: are a few millimetres across, so this stays inside one radius; a looser
#: tolerance would accept a tip parked in the neighbouring branch.
ARRIVAL_TOLERANCE_M = 0.005

#: Consecutive control steps the tip has to stay inside the tolerance. Controls
#: advance at 30 Hz, so this is about half a second, which keeps a tip that
#: merely swings through the target on one step from registering as arrival.
ARRIVAL_HOLD_STEPS = 15

#: Attribute on the env holding the per-environment hold counter.
HOLD_COUNTER_ATTR = "_catheter_arrival_steps"


def catheter_tip_world_m(env: Any) -> torch.Tensor:
    """Distal end of each environment's catheter polyline, in world metres.

    Returns infinities before Newton finalizes its model, when there are no
    particles to read. That reads as "nowhere near the target" and keeps an
    unknown tip from matching a goal that happens to sit near the origin.
    """
    positions = env.scene["catheter"].data.positions_world_m
    if positions is None:
        return torch.full((int(env.num_envs), 3), float("inf"), device=env.device)
    return positions[:, -1, :]


def tip_distance_to_target_m(env: Any, target_world_m: Iterable[float]) -> torch.Tensor:
    """Distance from each catheter tip to the navigation target.

    The target is deliberately not offset by ``env.scene.env_origins``. Every
    environment's rod is seeded from the same absolute centerline positions, so
    the cloned environments share one world-space goal.
    """
    target = torch.as_tensor(tuple(target_world_m), device=env.device, dtype=torch.float32)
    return torch.linalg.norm(catheter_tip_world_m(env) - target, dim=-1)


def hold_counter(env: Any) -> torch.Tensor:
    """Per-environment consecutive-steps-within-tolerance counter."""
    counter = getattr(env, HOLD_COUNTER_ATTR, None)
    if counter is None or int(counter.shape[0]) != int(env.num_envs):
        counter = torch.zeros(int(env.num_envs), dtype=torch.long, device=env.device)
        setattr(env, HOLD_COUNTER_ATTR, counter)
    return counter


def arrival_readout(
    distance_m: float,
    held_steps: int = 0,
    *,
    tolerance_m: float = ARRIVAL_TOLERANCE_M,
    hold_steps: int = ARRIVAL_HOLD_STEPS,
    route: RouteProgress | None = None,
) -> str:
    """One line telling the operator what still stands between them and arrival.

    Both halves of the criterion are visible because either one can be the thing
    holding an episode open: the tip can be far away, or it can be close enough
    and drifting back out before the hold completes.

    Given a route, the vessel still ahead leads and the straight line follows.
    The straight line is what ends the episode so it cannot be dropped, but it
    is not monotonic on a vessel that doubles back, and an operator steering on
    it alone will undo a correct insertion when the arch makes the number grow.
    Once inside the tolerance the arc is spent and the hold is the whole story,
    so the line reverts to the criterion it is about to satisfy.
    """
    if not math.isfinite(distance_m):
        return "Target: waiting for the catheter"
    millimetres = distance_m * 1000.0
    if distance_m <= tolerance_m:
        return f"Target: {millimetres:.1f} mm away -- holding {min(held_steps, hold_steps)}/{hold_steps}"
    within = f"(arrive within {tolerance_m * 1000.0:.0f} mm)"
    if route is None:
        return f"Target: {millimetres:.1f} mm away {within}"
    if not route.on_route:
        # Naming the straight line rather than an arc figure that projection
        # cannot stand behind: off route, the tip may be measured against a
        # stretch of vessel it is not in.
        return f"Target: {millimetres:.1f} mm direct, tip off route {within}"
    return f"Target: {route.remaining_m * 1000.0:.0f} mm of vessel ahead -- {millimetres:.1f} mm direct {within}"


@dataclass(frozen=True)
class ArrivalProgress:
    """Everything the readout needs about one environment's approach.

    Attributes:
        distance_m: Straight-line tip-to-target distance, the arrival quantity.
        held_steps: Consecutive steps already spent inside the tolerance.
        tolerance_m: Distance counting as arrival.
        hold_steps: Steps the tip must hold before the episode succeeds.
        route: Progress along the planned vessel, or ``None`` when the scene
            configured no route. Advisory: nothing terminates on it.
    """

    distance_m: float
    held_steps: int
    tolerance_m: float
    hold_steps: int
    route: RouteProgress | None = None


def arrival_progress(env: Any) -> ArrivalProgress | None:
    """Live approach state for one environment.

    The target, the route and both thresholds are read back off the configured
    ``success`` term rather than re-derived, so a readout built from this cannot
    disagree with the criterion that actually ends the episode. Returns ``None``
    for a scene that terminates on something else.
    """
    manager = getattr(env, "termination_manager", None)
    if manager is None or "success" not in getattr(manager, "active_terms", ()):
        return None
    params = manager.get_term_cfg("success").params
    target = params["target_world_m"]
    tolerance_m = float(params.get("tolerance_m", ARRIVAL_TOLERANCE_M))
    hold_steps = int(params.get("hold_steps", ARRIVAL_HOLD_STEPS))
    # Read the counter the term maintains; advancing it here would let the
    # readout consume part of the hold the operator still has to earn.
    distance_m = float(tip_distance_to_target_m(env, target)[0])
    return ArrivalProgress(
        distance_m=distance_m,
        held_steps=int(hold_counter(env)[0]),
        tolerance_m=tolerance_m,
        hold_steps=hold_steps,
        route=_route_progress(env, params.get("route_world_m"), distance_m),
    )


def _route_progress(env: Any, route_world_m: Any, distance_m: float) -> RouteProgress | None:
    """Arc progress for the configured route, or ``None`` when unavailable.

    Skipped before there are particles, since the tip is then infinite and has
    no projection.     A malformed route is reported once and dropped rather than
    raised: this feeds a status line, and a scene that still simulates and
    still terminates correctly should not be brought down by its caption.
    """
    if route_world_m is None or not math.isfinite(distance_m):
        return None
    try:
        tip = catheter_tip_world_m(env)[0].detach().cpu().numpy()
        return route_progress(route_world_m, tip)
    except (ValueError, IndexError, RuntimeError):
        global _route_warned
        if not _route_warned:
            # Once: this runs every step, and the condition is a fixed property
            # of the configured route rather than something a frame can fix.
            _route_warned = True
            _LOGGER.warning("navigation route unusable; falling back to the straight-line readout", exc_info=True)
        return None


_route_warned = False


def drift_log_seconds(environ: Any = None) -> float:
    """Seconds between tip-distance reports, from ``I4H_CATHETER_DRIFT``.

    Off by default. Reading the tip is free here -- the arrival term already
    does it every step -- but a line per frame would bury a teleop log.
    """
    raw = (environ if environ is not None else os.environ).get(DRIFT_LOG_ENV_VAR, "")
    try:
        interval = float(str(raw).strip())
    except ValueError:
        return 0.0
    return interval if interval > 0.0 else 0.0


_drift_logged_at = 0.0


def _log_tip_drift(distance_m: float) -> None:
    """Report the tip-to-target distance on a wall clock, to measure creep.

    The tip is meant to hold still when nothing is driving it. It does not:
    containment injects excess arc length that the cleanup sweeps then pull back
    toward rest, and with the proximal end latched at the roller the only place
    that shortening can go is the tip walking backward. Eyeballing the on-screen
    readout cannot separate that from ordinary jitter, so the interval is a wall
    clock and the log carries one, which makes the creep a slope in mm/s.
    """
    interval = drift_log_seconds()
    if interval <= 0.0:
        return
    global _drift_logged_at
    now = time.monotonic()
    if now - _drift_logged_at < interval:
        return
    _drift_logged_at = now
    _LOGGER.info("catheter drift: t=%.2f s  tip_to_target=%.2f mm", now, 1000.0 * float(distance_m))


def arrival_status(env: Any) -> str:
    """``arrival_readout`` for the live env, or empty for a scene without the term."""
    progress = arrival_progress(env)
    if progress is None:
        return ""
    _log_tip_drift(progress.distance_m)
    return arrival_readout(
        progress.distance_m,
        progress.held_steps,
        tolerance_m=progress.tolerance_m,
        hold_steps=progress.hold_steps,
        route=progress.route,
    )


def reset_arrival_progress(env: Any, env_ids: Any = None) -> None:
    """Clear the hold counter for the environments being reset."""
    counter = hold_counter(env)
    if env_ids is None:
        counter.zero_()
        return
    counter[env_ids] = 0


def reached_navigation_target(
    env: Any,
    target_world_m: Iterable[float],
    tolerance_m: float = ARRIVAL_TOLERANCE_M,
    hold_steps: int = ARRIVAL_HOLD_STEPS,
    route_world_m: Iterable[Iterable[float]] | None = None,
) -> torch.Tensor:
    """True once the tip has stayed within ``tolerance_m`` for ``hold_steps`` steps.

    ``route_world_m`` is carried on this term so the readout and the criterion
    share one source, and is deliberately not part of the test. Arrival stays a
    straight-line question: remaining arc is only defined while the projection
    is unambiguous, and making success depend on it would let a tip that
    wandered off the route end an episode on a guess. On the shipped s0011
    aorta the route's closest approach to its own endpoint from elsewhere is
    44 mm, well outside the 5 mm tolerance, so the straight line cannot be
    satisfied early by the arch doubling back.
    """
    counter = hold_counter(env)
    within = tip_distance_to_target_m(env, target_world_m) <= float(tolerance_m)
    counter = torch.where(within, counter + 1, torch.zeros_like(counter))
    setattr(env, HOLD_COUNTER_ATTR, counter)
    return counter >= int(hold_steps)


__all__ = [
    "ARRIVAL_HOLD_STEPS",
    "ARRIVAL_TOLERANCE_M",
    "HOLD_COUNTER_ATTR",
    "ArrivalProgress",
    "arrival_progress",
    "arrival_readout",
    "arrival_status",
    "catheter_tip_world_m",
    "hold_counter",
    "reached_navigation_target",
    "reset_arrival_progress",
    "tip_distance_to_target_m",
]
