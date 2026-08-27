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

from collections.abc import Iterable
from typing import Any

import torch

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
    "catheter_tip_world_m",
    "hold_counter",
    "reached_navigation_target",
    "reset_arrival_progress",
    "tip_distance_to_target_m",
]
