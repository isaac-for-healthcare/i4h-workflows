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
from typing import Any

import torch

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
) -> str:
    """One line telling the operator what still stands between them and arrival.

    Both halves of the criterion are visible because either one can be the thing
    holding an episode open: the tip can be far away, or it can be close enough
    and drifting back out before the hold completes.
    """
    if not math.isfinite(distance_m):
        return "Target: waiting for the catheter"
    millimetres = distance_m * 1000.0
    if distance_m > tolerance_m:
        return f"Target: {millimetres:.1f} mm away (arrive within {tolerance_m * 1000.0:.0f} mm)"
    return f"Target: {millimetres:.1f} mm away -- holding {min(held_steps, hold_steps)}/{hold_steps}"


def arrival_progress(env: Any) -> tuple[float, int, float, int] | None:
    """Live ``(distance_m, held_steps, tolerance_m, hold_steps)`` for one environment.

    The target and both thresholds are read back off the configured ``success``
    term rather than re-derived, so a readout built from this cannot disagree
    with the criterion that actually ends the episode. Returns ``None`` for a
    scene that terminates on something else.
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
    return distance_m, int(hold_counter(env)[0]), tolerance_m, hold_steps


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
    distance_m, held_steps, tolerance_m, hold_steps = progress
    _log_tip_drift(distance_m)
    return arrival_readout(distance_m, held_steps, tolerance_m=tolerance_m, hold_steps=hold_steps)


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
) -> torch.Tensor:
    """True once the tip has stayed within ``tolerance_m`` for ``hold_steps`` steps."""
    counter = hold_counter(env)
    within = tip_distance_to_target_m(env, target_world_m) <= float(tolerance_m)
    counter = torch.where(within, counter + 1, torch.zeros_like(counter))
    setattr(env, HOLD_COUNTER_ATTR, counter)
    return counter >= int(hold_steps)


__all__ = [
    "ARRIVAL_HOLD_STEPS",
    "ARRIVAL_TOLERANCE_M",
    "HOLD_COUNTER_ATTR",
    "arrival_progress",
    "arrival_readout",
    "arrival_status",
    "catheter_tip_world_m",
    "hold_counter",
    "reached_navigation_target",
    "reset_arrival_progress",
    "tip_distance_to_target_m",
]
