# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a catheter navigation policy is allowed to see, per environment, every step.

The reward in :mod:`i4h_arena.medical.navigation_reward` already projects the
tip onto the route to decide what to pay for; these terms re-expose the same
projection as something the policy can read, so the quantity being optimized
and the quantity being observed cannot drift apart.

Two choices here are worth stating.

Positions are relative rather than absolute. The tip is offset by the cloned
environment's origin and the target is given as the vector *from* the tip, so a
policy learns "steer toward the goal" rather than memorizing one patient's
world coordinates. That the cloned environments currently share one absolute
route (see :func:`~i4h_arena.medical.navigation_goal.tip_distance_to_target_m`)
makes the offset a no-op today, but it is what the terms will need the moment
the envs are actually spread apart.

The drive state is included, and it is not redundant with the tip pose. The tip
bend is about the catheter's local X axis and axial twist is what aims that
bend at a branch, so a policy that cannot observe its own twist cannot steer
deliberately -- it can only wiggle and see what happens. Insertion depth is
likewise distinct from tip position: they disagree exactly when the shaft is
buckling, which is the failure the operator most needs the policy to feel.

Every term returns zeros of its declared width when the rod is not readable
yet, rather than propagating the infinities the tip reads before Newton
finalizes its model. IsaacLab concatenates observation terms into a fixed-width
vector, so a term that changed shape or emitted a nan would take the run down.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from i4h_arena.medical.navigation_reward import tip_route_state

#: Width of each term, for sizing ``state_dof`` in an RL profile without
#: starting a simulator to measure it.
TIP_POSITION_DIM = 3
TIP_DIRECTION_DIM = 3
TARGET_OFFSET_DIM = 3
ROUTE_STATE_DIM = 2
DRIVE_STATE_DIM = 4

#: Total width of :func:`navigation_observations_cfg`'s concatenated group.
NAVIGATION_STATE_DIM = TIP_POSITION_DIM + TIP_DIRECTION_DIM + TARGET_OFFSET_DIM + ROUTE_STATE_DIM + DRIVE_STATE_DIM


def _zeros(env: Any, width: int) -> torch.Tensor:
    return torch.zeros((int(env.num_envs), width), dtype=torch.float32, device=env.device)


def _polyline(env: Any) -> torch.Tensor | None:
    """The catheter polyline as finite float32, or ``None`` when unreadable.

    ``None`` only for a shape this module cannot read at all, which really is
    a property of the whole batch. A rod that diverges is not, and testing it
    batch-wide meant one of them zeroed the tip readings for all eight. Its
    points are replaced with its environment origin instead, which leaves the
    three terms below reading as parked at the entry pointing nowhere with the
    target still the full distance away -- the same thing their zeros already
    mean for an unbound rod, and never mistakable for arrival.
    """
    positions = env.scene["catheter"].data.positions_world_m
    if positions is None:
        return None
    points = torch.as_tensor(positions, dtype=torch.float32, device=env.device)
    if points.ndim != 3 or points.shape[1] < 2:
        return None
    valid = torch.isfinite(points).flatten(1).all(dim=-1)
    origin = _env_origins(env, points[:, 0, :]).unsqueeze(1)
    return torch.where(valid.view(-1, 1, 1), points, origin)


def _env_origins(env: Any, like: torch.Tensor) -> torch.Tensor:
    """Cloned-env origins, or zeros for a scene that does not define them."""
    origins = getattr(getattr(env, "scene", None), "env_origins", None)
    if origins is None:
        return torch.zeros_like(like)
    return torch.as_tensor(origins, dtype=like.dtype, device=like.device)


def tip_position(env: Any) -> torch.Tensor:
    """``(N, 3)`` catheter tip relative to its cloned environment origin, metres."""
    points = _polyline(env)
    if points is None:
        return _zeros(env, TIP_POSITION_DIM)
    tip = points[:, -1, :]
    return tip - _env_origins(env, tip)


def tip_direction(env: Any) -> torch.Tensor:
    """``(N, 3)`` unit vector the tip is pointing along.

    Taken from the last two nodes rather than from a solver frame, so it stays
    defined for any rod the physics can produce. A degenerate final segment --
    two coincident particles, which the solver does emit transiently -- yields
    a zero vector rather than a nan, reading as "direction unknown".
    """
    points = _polyline(env)
    if points is None:
        return _zeros(env, TIP_DIRECTION_DIM)
    span = points[:, -1, :] - points[:, -2, :]
    length = torch.linalg.norm(span, dim=-1, keepdim=True)
    return torch.where(length > 0.0, span / length.clamp_min(1e-12), torch.zeros_like(span))


def target_offset(env: Any, target_world_m: Iterable[float]) -> torch.Tensor:
    """``(N, 3)`` vector from the tip to the navigation target, metres.

    Relative rather than absolute so the policy reads a direction to steer in.
    Translation invariance also means this term needs no env-origin correction:
    the offset between two world points is already origin-free.
    """
    points = _polyline(env)
    if points is None:
        return _zeros(env, TARGET_OFFSET_DIM)
    target = torch.as_tensor(tuple(target_world_m), dtype=torch.float32, device=env.device)
    return target - points[:, -1, :]


def fluoroscopy_image(env: Any, sensor_cfg: Any, data_type: str = "rgb") -> torch.Tensor:
    """``(N, H, W, 3)`` uint8 detector frame, zeros until the C-arm is bound.

    IsaacLab's stock image term cannot serve this sensor. It reads through
    ``sensor.data``, which renders on demand, and the slang backend refuses to
    render until the scene binds a C-arm provider -- which happens after the
    environment is built, so the observation manager's one shape-probing read
    always precedes it. The synthetic backend tolerates an unbound provider,
    which is why this only bites with ``--patient-twin``.

    Shape has to be right on that first read even so, because IsaacLab fixes
    each term's width from it. The sensor has already allocated correctly
    shaped zero buffers by then, so serve their dimensions.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    if not getattr(sensor, "is_renderable", True):
        return torch.zeros(
            (int(env.num_envs), int(sensor.cfg.height), int(sensor.cfg.width), 3),
            dtype=torch.uint8,
            device=env.device,
        )
    return sensor.data.output[data_type].clone()


def route_state(env: Any, route_world_m: Iterable[Iterable[float]]) -> torch.Tensor:
    """``(N, 2)`` remaining route arc and lateral offset from the centerline, metres.

    The pair the reward is built on. Remaining arc says how much vessel is
    left; lateral offset says whether the tip is threading the lumen or riding
    its wall. They are independent, and a policy given only the first cannot
    tell a good approach from one pinned against the outside of a curve.

    Not zeroed when the tip is unreadable, which is what this used to do: a
    remaining arc of zero is the signature of a perfect arrival, so an
    exploded rod -- or any rod, since the test was batch-wide -- reported the
    task complete. The substitution in :func:`tip_route_state` puts an
    unreadable tip at the vessel entrance instead, which reads as the whole
    route still ahead.
    """
    state = tip_route_state(env, route_world_m)
    return torch.stack((state.remaining_m, state.lateral_m), dim=-1).to(dtype=torch.float32)


def drive_state(env: Any) -> torch.Tensor:
    """``(N, 4)`` insertion depth, axial twist, tip bend, and C-arm orbit angle.

    SI throughout: metres for the first, radians for the rest. Read through the
    action terms' public accessors rather than from the raw action buffer,
    because three of the four are integrated state the drive owns and the last
    command says nothing about where they ended up.
    """
    manager = getattr(env, "action_manager", None)
    if manager is None:
        return _zeros(env, DRIVE_STATE_DIM)
    catheter = _action_term(manager, "catheter")
    carm = _action_term(manager, "carm_orbit")
    columns = (
        _channel(env, catheter, "insertion_depth_m"),
        _channel(env, catheter, "twist_rad"),
        _channel(env, catheter, "tip_bend_angle"),
        _channel(env, carm, "angle_rad"),
    )
    return torch.stack(columns, dim=-1)


def _action_term(manager: Any, name: str) -> Any:
    """One action term by name, or ``None`` when the scene does not have it."""
    getter = getattr(manager, "get_term", None)
    if getter is None:
        return None
    try:
        return getter(name)
    except (KeyError, ValueError):
        return None


def _channel(env: Any, term: Any, attribute: str) -> torch.Tensor:
    """``(N,)`` float32 column from an action term, or zeros when absent."""
    value = None if term is None else getattr(term, attribute, None)
    if value is None:
        return torch.zeros(int(env.num_envs), dtype=torch.float32, device=env.device)
    column = torch.as_tensor(value, dtype=torch.float32, device=env.device).reshape(-1)
    return torch.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0)


__all__ = [
    "DRIVE_STATE_DIM",
    "NAVIGATION_STATE_DIM",
    "ROUTE_STATE_DIM",
    "TARGET_OFFSET_DIM",
    "TIP_DIRECTION_DIM",
    "TIP_POSITION_DIM",
    "drive_state",
    "fluoroscopy_image",
    "route_state",
    "target_offset",
    "tip_direction",
    "tip_position",
]
