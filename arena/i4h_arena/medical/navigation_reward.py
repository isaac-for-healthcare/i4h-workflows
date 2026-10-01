# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What to pay a catheter controller for, per environment, every step.

The quantities are the ones :mod:`i4h_arena.medical.route_progress` already
defines for the operator readout -- arc travelled, arc remaining, lateral offset
from the route -- re-expressed batched in torch because rewards are read for
every environment on every step and the readout version is a single-point numpy
projection behind an ``lru_cache``.

Three things here are deliberate rather than incidental, and each one exists
because the obvious version was measured and found exploitable.

Both positive shaping terms pay a *change*, not a level, which is
potential-based shaping and so leaves the optimal policy untouched. Progress
differences the remaining arc and approach differences ``exp(-distance/scale)``.
Approach was a level until it was priced: at one per step for hovering, and
hovering being unbounded in time, the best stationary spot just outside the
arrival tolerance discounted to roughly 150 against roughly 84 for holding the
arrival and terminating, so the task paid better unfinished. A level-valued
positive term is collectable by standing still, and standing still is always
available.

Progress is also clamped to what insertion can physically deliver in one
control step. Nearest-point projection onto a route that doubles back is not
continuous in the tip position: on a recorded s0011 episode the projected arc
jumped by up to 31 mm across five steps and six times over the episode,
crediting 153 mm of travel against 112 mm actually made. Unclamped, the surplus
is free return for wiggling the tip across the arch rather than advancing
through it.

Wall contact and folding are paid as depths and curvatures, not as booleans. A
boolean fold flag fires on roughly nine frames in ten of a recorded episode,
which makes it a constant offset the advantage estimator subtracts away rather
than a gradient pointing anywhere.

Being *off the axis* is charged for separately from being *short of the end*.
Distance along the vessel and distance from its centerline are independent, and
a recorded episode that never arrived ended with 7.1 mm of arc left and 7.2 mm
of lateral offset inside a 4.5 mm-radius lumen: nearly all the length, none of
the alignment. With arc progress as the only positive term, that episode scores
as near-total success.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

from i4h_arena.medical.navigation_goal import (
    ARRIVAL_TOLERANCE_M,
    catheter_tip_world_m,
    tip_distance_to_target_m,
)

#: Attribute holding the previous step's remaining arc, for the progress term.
REMAINING_ARC_ATTR = "_catheter_remaining_arc_m"

#: Attribute holding the previous step's approach potential, for the same
#: reason and handled the same way.
APPROACH_POTENTIAL_ATTR = "_catheter_approach_potential"

#: Cache for the route tensors, keyed by device so a term does not rebuild the
#: polyline on every call. The route is fixed for the life of the scene.
ROUTE_CACHE_ATTR = "_catheter_route_cache"

#: Cache for the per-segment lumen radii, separate from the route so a term
#: that needs only the route cannot deny the wall to one that needs both.
RADII_CACHE_ATTR = "_catheter_lumen_radii_cache"

#: Largest advance one control step can make, in metres. The drive clamps
#: insertion to 0.05 m/s (``CatheterDriveSpec.max_insertion_velocity_mps``) and
#: controls advance at 30 Hz, so 1.67 mm is the physical ceiling; the margin
#: covers the tip travelling slightly further than the root it is fed from
#: while the shaft straightens.
MAX_STEP_ADVANCE_M = 0.0025

#: Bend radius at or below which a node counts as folded, matching
#: :func:`~i4h_arena.medical.newton_catheter_physics.containment_report`. The
#: s0011 route's own tightest curve is 13.1 mm, so anatomy cannot trip it.
FOLD_RADIUS_M = 0.010


def _route_tensors(
    env: Any,
    route_world_m: Iterable[Iterable[float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(starts, spans, start_arc)`` for the route, built once per device.

    ``start_arc`` is the cumulative arc at each segment's start, so a projection
    landing a fraction along segment *i* has arc ``start_arc[i] + fraction *
    |spans[i]|``. Cached on the env because the route is fixed for the life of
    the scene and every term projects onto it on every step.
    """
    cache = getattr(env, ROUTE_CACHE_ATTR, None)
    if cache is not None and cache[0] == str(env.device):
        return cache[1]
    path = torch.as_tensor(
        [tuple(float(value) for value in point) for point in route_world_m],
        dtype=torch.float32,
        device=env.device,
    )
    if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
        raise ValueError(f"route must be at least two 3-D points, got shape {tuple(path.shape)}")
    starts, ends = path[:-1], path[1:]
    spans = ends - starts
    lengths = torch.linalg.norm(spans, dim=-1)
    start_arc = torch.cat((torch.zeros(1, device=env.device), torch.cumsum(lengths, dim=0)[:-1]))
    tensors = (starts, spans, start_arc)
    setattr(env, ROUTE_CACHE_ATTR, (str(env.device), tensors))
    return tensors


def _segment_radii(env: Any, lumen_radii_m: Iterable[float] | None, segments: int) -> torch.Tensor | None:
    """Lumen radius at each route segment's start, or ``None`` without widths.

    Cached separately from the route so that a term reading the route alone
    cannot poison the cache for one that needs the wall too.
    """
    if lumen_radii_m is None:
        return None
    cache = getattr(env, RADII_CACHE_ATTR, None)
    if cache is not None and cache[0] == str(env.device):
        return cache[1]
    radii = torch.as_tensor([float(value) for value in lumen_radii_m], dtype=torch.float32, device=env.device)
    if radii.shape[0] < segments:
        raise ValueError(f"lumen radii cover {radii.shape[0]} of {segments} route segments")
    radii = radii[:segments]
    setattr(env, RADII_CACHE_ATTR, (str(env.device), radii))
    return radii


def project_to_route(
    points: torch.Tensor,
    starts: torch.Tensor,
    spans: torch.Tensor,
    start_arc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(arc_m, lateral_m, segment)`` of the nearest route point to each input.

    ``points`` is ``(..., 3)``; the leading shape is preserved. Mirrors
    :func:`i4h_arena.medical.route_progress.project_to_route` including the
    zero-length-segment guard, which matters because extracted centerlines do
    contain duplicated samples and a nan would win the ``argmin``.
    """
    lengths_sq = torch.einsum("sj,sj->s", spans, spans)
    safe = torch.where(lengths_sq > 0.0, lengths_sq, torch.ones_like(lengths_sq))
    offset = points.unsqueeze(-2) - starts
    fraction = torch.clamp(torch.einsum("...sj,sj->...s", offset, spans) / safe, 0.0, 1.0)
    projected = starts + fraction.unsqueeze(-1) * spans
    distances = torch.linalg.norm(points.unsqueeze(-2) - projected, dim=-1)
    lateral_m, segment = torch.min(distances, dim=-1)
    chosen = fraction.gather(-1, segment.unsqueeze(-1)).squeeze(-1)
    arc_m = start_arc[segment] + chosen * torch.sqrt(safe)[segment]
    return arc_m, lateral_m, segment


def tip_route_state(
    env: Any,
    route_world_m: Iterable[Iterable[float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """``(remaining_m, lateral_m, total_m)`` for every environment's tip.

    ``None`` before Newton finalizes its model, when there are no particles and
    the tip reads as infinite. Every term treats that as "pay nothing" rather
    than guessing, since a projection of an infinite point is meaningless.
    """
    tip = catheter_tip_world_m(env)
    if not torch.isfinite(tip).all():
        return None
    starts, spans, start_arc = _route_tensors(env, route_world_m)
    total_m = start_arc[-1] + torch.linalg.norm(spans[-1])
    arc_m, lateral_m, _ = project_to_route(tip, starts, spans, start_arc)
    return torch.clamp(total_m - arc_m, min=0.0), lateral_m, total_m


def remaining_arc_state(env: Any) -> torch.Tensor | None:
    """The remaining arc the progress term last saw, or ``None`` before step one."""
    return getattr(env, REMAINING_ARC_ATTR, None)


def reset_route_progress(env: Any, env_ids: Any = None) -> None:
    """Drop the stored remaining arc so a reset environment earns no phantom step.

    Without this the first step after a reset differences the new episode's
    remaining arc against the old one's, which on a successful reset is the
    whole route and would pay out the entire task for doing nothing.
    """
    stored = getattr(env, REMAINING_ARC_ATTR, None)
    if stored is None:
        return
    if env_ids is None:
        delattr(env, REMAINING_ARC_ATTR)
        return
    stored[env_ids] = float("nan")


def route_progress_reward(
    env: Any,
    route_world_m: Iterable[Iterable[float]],
    max_step_advance_m: float = MAX_STEP_ADVANCE_M,
) -> torch.Tensor:
    """Metres of vessel closed since the previous step, clamped to what is reachable.

    Symmetric: backing out costs exactly what advancing the same distance pays,
    which is what keeps the shaping potential-based and the optimal policy
    unchanged. The clamp is applied to both directions for the same reason --
    a one-sided clamp would pay more for a round trip than for standing still.
    """
    state = tip_route_state(env, route_world_m)
    if state is None:
        return torch.zeros(int(env.num_envs), device=env.device)
    remaining_m, _, _ = state
    previous = getattr(env, REMAINING_ARC_ATTR, None)
    setattr(env, REMAINING_ARC_ATTR, remaining_m.clone())
    if previous is None or previous.shape != remaining_m.shape:
        return torch.zeros_like(remaining_m)
    limit = abs(float(max_step_advance_m))
    advance = torch.clamp(previous - remaining_m, -limit, limit)
    # A reset environment carries nan until it takes its first step.
    return torch.nan_to_num(advance, nan=0.0)


def lateral_offset_penalty(
    env: Any,
    route_world_m: Iterable[Iterable[float]],
    lumen_radii_m: Iterable[float] | None,
    free_fraction: float = 0.5,
) -> torch.Tensor:
    """How far off-axis the tip sits, as a fraction of the room it has.

    The term the recorded failure argues for. Arc progress alone cannot
    distinguish a tip threading the lumen from one pressed against the outer
    wall at the same station, and it is the second that stops being steerable:
    the episode that never arrived ended 7.2 mm off-axis in a 4.5 mm-radius
    lumen with 7.1 mm of arc left.

    A penalty rather than a reward for being centred, because a per-step payout
    for sitting on the axis is collectable without going anywhere, and the
    inner ``free_fraction`` of the lumen is free so that hugging the inside of
    a curve -- which is what a real wire does -- costs nothing. Normalized by
    the local radius because the same offset is harmless in the aorta and
    against the wall in a branch.
    """
    state = tip_route_state(env, route_world_m)
    if state is None:
        return torch.zeros(int(env.num_envs), device=env.device)
    _, lateral_m, _ = state
    starts, spans, start_arc = _route_tensors(env, route_world_m)
    radii = _segment_radii(env, lumen_radii_m, int(starts.shape[0]))
    if radii is None:
        return torch.zeros(int(env.num_envs), device=env.device)
    _, _, segment = project_to_route(catheter_tip_world_m(env), starts, spans, start_arc)
    allowed = radii[segment].clamp_min(1e-6)
    return torch.clamp(lateral_m - float(free_fraction) * allowed, min=0.0) / allowed


def wall_penetration_penalty(
    env: Any,
    route_world_m: Iterable[Iterable[float]],
    lumen_radii_m: Iterable[float] | None,
) -> torch.Tensor:
    """Mean depth, in metres, by which the rod sits outside the lumen wall.

    Every particle, not just the tip: a tip that threads the arch while the
    shaft behind it cuts the corner is the failure this is for. Depth rather
    than a count, so easing off a deep contact pays before the contact clears.
    """
    positions = env.scene["catheter"].data.positions_world_m
    if positions is None:
        return torch.zeros(int(env.num_envs), device=env.device)
    points = torch.as_tensor(positions, dtype=torch.float32, device=env.device)
    if not torch.isfinite(points).all():
        return torch.zeros(int(env.num_envs), device=env.device)
    starts, spans, start_arc = _route_tensors(env, route_world_m)
    radii = _segment_radii(env, lumen_radii_m, int(starts.shape[0]))
    if radii is None:
        return torch.zeros(int(env.num_envs), device=env.device)
    _, lateral_m, segment = project_to_route(points, starts, spans, start_arc)
    return torch.clamp(lateral_m - radii[segment], min=0.0).mean(dim=-1)


def bend_radius_m(positions: torch.Tensor) -> torch.Tensor:
    """Radius of curvature at every interior node, batched over environments.

    The circumradius of each consecutive triple, matching
    :func:`~i4h_arena.medical.newton_catheter_physics.bend_radii_m` so a reward
    and a diagnostic cannot disagree about what counts as a fold. Straight and
    coincident runs come back as ``inf``, which reads as no bend.
    """
    back = positions[:, 1:-1] - positions[:, :-2]
    forward = positions[:, 2:] - positions[:, 1:-1]
    span = positions[:, 2:] - positions[:, :-2]
    twice_area = torch.linalg.norm(torch.cross(back, forward, dim=-1), dim=-1)
    sides = torch.linalg.norm(back, dim=-1) * torch.linalg.norm(forward, dim=-1) * torch.linalg.norm(span, dim=-1)
    return torch.where(
        twice_area > 0.0,
        sides / (2.0 * twice_area.clamp_min(1e-12)),
        torch.full_like(twice_area, float("inf")),
    )


def fold_penalty(env: Any, fold_radius_m: float = FOLD_RADIUS_M) -> torch.Tensor:
    """How hard the rod is folded, as mean excess curvature past the fold radius.

    ``relu(fold_radius / radius - 1)`` per interior node: zero for any bend
    gentler than the threshold, and growing without bound as a node creases.
    Dimensionless, so it does not change meaning if the segment count changes,
    and continuous, which the boolean version this replaces was not.
    """
    positions = env.scene["catheter"].data.positions_world_m
    if positions is None:
        return torch.zeros(int(env.num_envs), device=env.device)
    points = torch.as_tensor(positions, dtype=torch.float32, device=env.device)
    if points.shape[1] < 3 or not torch.isfinite(points).all():
        return torch.zeros(int(env.num_envs), device=env.device)
    radii = bend_radius_m(points)
    excess = torch.clamp(float(fold_radius_m) / radii - 1.0, min=0.0)
    return torch.nan_to_num(excess, nan=0.0, posinf=0.0).mean(dim=-1)


def arrival_reward(
    env: Any,
    target_world_m: Iterable[float],
    tolerance_m: float = ARRIVAL_TOLERANCE_M,
) -> torch.Tensor:
    """One for every step the tip spends inside the arrival tolerance.

    Paid per step rather than once at termination so that the hold the success
    criterion requires is itself worth something. A single terminal bonus leaves
    the fifteen steps of holding unpaid, and an agent that has already banked
    the approach has no reason to spend them.
    """
    within = tip_distance_to_target_m(env, target_world_m) <= float(tolerance_m)
    return within.to(dtype=torch.float32)


def approach_potential(
    env: Any,
    target_world_m: Iterable[float],
    scale_m: float,
) -> torch.Tensor:
    """``exp(-distance / scale)``: one at the target, decaying over ``scale_m``.

    The potential itself, which is not the reward. Separate from
    :func:`approach_reward` so a test and a readout can ask what the shaping is
    built on without going through a difference that needs two steps to mean
    anything.
    """
    distance_m = tip_distance_to_target_m(env, target_world_m)
    return torch.nan_to_num(torch.exp(-distance_m / float(scale_m)), nan=0.0, posinf=0.0)


def approach_reward(
    env: Any,
    target_world_m: Iterable[float],
    scale_m: float,
) -> torch.Tensor:
    """Change in straight-line closeness to the target, for the last few millimetres.

    Remaining arc goes flat once the tip is within one route sample of the end,
    so it cannot guide the final approach that the 5 mm tolerance is decided on.
    This is the fine-scale companion, mirroring the two-scale position reward
    the ultrasound probe reach task uses.

    The *change*, for the reason progress pays a change: paid as a level this
    term rewards sitting still near the target. At the previous weight of 1.0
    the level form paid up to 1.0 every step for hovering, and hovering has no
    end, so at the configured discount the best stationary spot just outside
    the tolerance was worth roughly 150 against roughly 84 for holding the
    arrival and terminating. Finishing the task was a pay cut. Differenced, a
    stationary tip earns exactly nothing wherever it is parked, and the only
    way to collect is to close distance.

    Undiscounted, where strict policy invariance wants ``gamma * phi' - phi``.
    The omission leaves a residual per-step payout of
    ``weight * (1 - gamma) * phi``, which is about 0.019 at the target against
    the 5.0 arrival pays there, so it cannot recreate the inversion. Taking
    ``gamma`` as a parameter was the alternative and is worse: it would be a
    second copy of the trainer's discount, free to drift from it, and a wrong
    ``gamma`` breaks the invariance it was added to guarantee.
    """
    potential = approach_potential(env, target_world_m, scale_m)
    previous = getattr(env, APPROACH_POTENTIAL_ATTR, None)
    setattr(env, APPROACH_POTENTIAL_ATTR, potential.clone())
    if previous is None or previous.shape != potential.shape:
        return torch.zeros_like(potential)
    # A reset environment carries nan until it takes its first step.
    return torch.nan_to_num(potential - previous, nan=0.0)


def reset_approach_potential(env: Any, env_ids: Any = None) -> None:
    """Drop the stored potential so a reset environment earns no phantom step.

    The counterpart of :func:`reset_route_progress`, and needed for the same
    reason: an episode that ends at the target and resets to the vessel entry
    would otherwise difference a potential near one against a potential near
    zero and be charged the whole approach for the reset itself.
    """
    stored = getattr(env, APPROACH_POTENTIAL_ATTR, None)
    if stored is None:
        return
    if env_ids is None:
        delattr(env, APPROACH_POTENTIAL_ATTR)
        return
    stored[env_ids] = float("nan")


def route_length_m(route_world_m: Iterable[Iterable[float]]) -> float:
    """Total arc length of a route, for sizing progress weights against the task."""
    path = [tuple(float(value) for value in point) for point in route_world_m]
    return float(sum(math.dist(previous, current) for previous, current in zip(path[:-1], path[1:], strict=True)))


__all__ = [
    "APPROACH_POTENTIAL_ATTR",
    "FOLD_RADIUS_M",
    "MAX_STEP_ADVANCE_M",
    "REMAINING_ARC_ATTR",
    "approach_potential",
    "approach_reward",
    "arrival_reward",
    "bend_radius_m",
    "fold_penalty",
    "lateral_offset_penalty",
    "project_to_route",
    "remaining_arc_state",
    "reset_approach_potential",
    "reset_route_progress",
    "route_length_m",
    "route_progress_reward",
    "tip_route_state",
    "wall_penetration_penalty",
]
