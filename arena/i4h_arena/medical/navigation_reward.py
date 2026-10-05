# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What to pay a catheter controller for, per environment, every step.

The quantities are the ones :mod:`i4h_arena.medical.route_progress` already
defines for the operator readout -- arc travelled, arc remaining, lateral offset
from the route -- re-expressed batched in torch because rewards are read for
every environment on every step and the readout version is a single-point numpy
projection behind an ``lru_cache``.

:class:`~i4h_arena.envcfg.endoluminal_navigation.CatheterNavigationRewardsCfg`
configures three terms and no others: ``route_progress``, ``lateral`` and
``penetration``, defined below in that order. That is the whole reward PPO
optimizes; this module holds nothing that is not part of it.

Three things about it are deliberate rather than incidental, and each one
exists because the obvious version was measured and found exploitable.

Progress pays a *change*, not a level, which is potential-based shaping: it
differences the remaining arc, so a stationary tip collects nothing wherever it
is parked. It is a difference of a potential only while its clamp below is
slack; a clamped step is not, and the invariance theorem does not cover it.
That is a deliberate trade against the projection discontinuity, and is worth
stating in that direction rather than the flattering one.

There is no level-valued positive term, which is why an earlier approach bonus
was dropped rather than given a smaller weight. Paid at one per step for
hovering, and hovering being unbounded in time, the best stationary spot just
outside the arrival tolerance out-valued holding the arrival and terminating
under the weights of the time: the task paid better unfinished. Anything
collectable by standing still will be collected, because standing still is
always available.

Progress is also clamped to what insertion can physically deliver in one
control step. Nearest-point projection onto a route that doubles back is not
continuous in the tip position: on a recorded s0011 episode the projected arc
jumped by up to 31 mm across five steps and six times over the episode,
crediting 153 mm of travel against 112 mm actually made. Unclamped, the surplus
is free return for wiggling the tip across the arch rather than advancing
through it.

Wall contact is paid as a depth, not a boolean. A flag that fires on most
frames is a constant offset the advantage estimator subtracts away rather than
a gradient pointing anywhere -- the fold flag this reward used to carry fired
on roughly nine frames in ten of a recorded episode.

Being *off the axis* is charged for separately from being *short of the end*.
Distance along the vessel and distance from its centerline are independent, and
a recorded episode that never arrived ended with 7.1 mm of arc left and 7.2 mm
of lateral offset inside a 4.5 mm-radius lumen: nearly all the length, none of
the alignment. With arc progress as the only positive term, that episode scores
as near-total success.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from typing import Any, NamedTuple

import torch

from i4h_arena.medical.navigation_goal import catheter_tip_world_m

#: Attribute holding the previous step's remaining arc, for the progress term.
REMAINING_ARC_ATTR = "_catheter_remaining_arc_m"

#: Cache for the route tensors, so a term does not rebuild the polyline on
#: every call. Keyed by the route itself and not only by device: one entry per
#: route is what makes a second route on the same device a miss rather than a
#: silent substitution of the first.
ROUTE_CACHE_ATTR = "_catheter_route_cache"

#: Cache for the per-segment lumen radii, separate from the route so a term
#: that needs only the route cannot deny the wall to one that needs both.
RADII_CACHE_ATTR = "_catheter_lumen_radii_cache"

#: Cache for the step's tip projection, as ``(step, state)``. Dropped on
#: reset by :func:`reset_tip_route_state`.
TIP_STATE_CACHE_ATTR = "_catheter_tip_route_state"

#: Largest advance one control step can make, in metres. The drive clamps
#: insertion to 0.060 m/s (``max_insertion_velocity_mps`` on
#: ``CatheterVelocityActionCfg`` and on its arm-driven counterpart
#: ``ArmDrivenCatheterActionCfg``) and controls advance at 30 Hz, from a
#: ``sim.dt`` of 1/120 with ``decimation`` of 4, so 2 mm is the physical
#: ceiling; the margin above it covers the tip travelling slightly further
#: than the root it is fed from while the shaft straightens.
#:
#: Named rather than derived because this module is the shared medical layer
#: and the limit belongs to two specific embodiments, so deriving it would
#: invert the dependency and would have to pick between them.
#: ``test_step_advance_clears_the_drive_ceiling`` holds the number to the two
#: they declare.
MAX_STEP_ADVANCE_M = 0.0025


def _cache_key(values: Any) -> Hashable:
    """A key that tells one route, or one set of radii, from another.

    The configs pass tuples of floats, which are hashable as they stand, so
    the hot path costs a hash and no conversion. Anything else is normalized,
    which costs no more than building the tensors would have.
    """
    try:
        hash(values)
    except TypeError:
        return tuple(tuple(float(value) for value in point) for point in values)
    return values


def _route_tensors(
    env: Any,
    route_world_m: Iterable[Iterable[float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(starts, spans, start_arc)`` for the route, built once per route.

    ``start_arc`` is the cumulative arc at each segment's start, so a projection
    landing a fraction along segment *i* has arc ``start_arc[i] + fraction *
    |spans[i]|``. Cached on the env because every term projects onto the route
    on every step.

    One entry per route rather than one per device. A single slot keyed on the
    device alone is correct only while the env never sees a second route, which
    holds today because the reward, observation and termination configs are all
    built from the same ``rod_spec.initial_path_world_m`` -- and stops holding
    the moment a twin is randomized per episode, at which point the stale route
    is returned silently. The dict grows by the number of distinct routes, so
    twelve twins is twelve entries.
    """
    key = (str(env.device), _cache_key(route_world_m))
    cache = getattr(env, ROUTE_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(env, ROUTE_CACHE_ATTR, cache)
    cached = cache.get(key)
    if cached is not None:
        return cached
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
    cache[key] = tensors
    return tensors


def _segment_radii(env: Any, lumen_radii_m: Iterable[float] | None, segments: int) -> torch.Tensor | None:
    """Lumen radius at each route segment's start, or ``None`` without widths.

    Cached separately from the route so that a term reading the route alone
    cannot poison the cache for one that needs the wall too.

    ``segments`` is part of the key, not just the widths: what is stored is
    already truncated to it, so a slot keyed on the device alone would hand a
    shorter route's slice to a longer one. Same reason the route cache keys on
    the route -- see :func:`_route_tensors`.
    """
    if lumen_radii_m is None:
        return None
    key = (str(env.device), _cache_key(lumen_radii_m), int(segments))
    cache = getattr(env, RADII_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(env, RADII_CACHE_ATTR, cache)
    cached = cache.get(key)
    if cached is not None:
        return cached
    radii = torch.as_tensor([float(value) for value in lumen_radii_m], dtype=torch.float32, device=env.device)
    if radii.shape[0] < segments:
        raise ValueError(f"lumen radii cover {radii.shape[0]} of {segments} route segments")
    radii = radii[:segments]
    cache[key] = radii
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


class TipRouteState(NamedTuple):
    """One projection of every tip onto the route, shared by every term."""

    remaining_m: torch.Tensor
    lateral_m: torch.Tensor
    total_m: torch.Tensor
    segment: torch.Tensor
    #: Per environment, false where the tip is not a finite point.
    valid: torch.Tensor


def reset_tip_route_state(env: Any, env_ids: Any = None) -> None:
    """Drop the cached projection so a reset environment is re-projected.

    Isaac Lab computes rewards, then resets, then observations, all under one
    ``common_step_counter``. Without this the observation handed to a reset
    environment is the projection of where its tip was before the rod was
    re-seeded, which is the staleness the upstream comment at
    ``manager_based_rl_env.py`` ("done after reset to get the correct
    observations for reset envs") exists to avoid.

    The whole cache goes rather than the reset rows: it is one projection to
    rebuild, against a partial update that has to agree with ``env_ids``.
    """
    if hasattr(env, TIP_STATE_CACHE_ATTR):
        delattr(env, TIP_STATE_CACHE_ATTR)


def tip_route_state(env: Any, route_world_m: Iterable[Iterable[float]]) -> TipRouteState:
    """Where every environment's tip sits on the route.

    ``valid`` is per environment. Before Newton finalizes its model there are
    no particles and it is false everywhere; a rod that diverges on its own
    makes it false for that one environment. The distinction is the point: a
    batch-wide test zeroed progress, the lateral penalty and the route
    observation for all eight environments whenever one of them exploded, and
    a zeroed route observation reads as remaining arc zero, which is the
    signature of a perfect arrival.

    A non-finite tip is substituted with the route's first point *before*
    projecting rather than masked after. Projecting an infinity gives an
    infinite offset, and ``inf * 0`` is ``nan``, so the order is the
    difference between a zero and a poisoned batch. It also leaves the
    unreadable environment reading as parked at the vessel entrance with the
    whole route ahead of it -- a guess, but not one that can be mistaken for
    arrival.

    Cached for the step because Isaac Lab computes each reward term and each
    observation separately and all of them want this one projection. The mask
    stays a tensor throughout, so nothing here forces a host synchronization;
    the batch-wide ``torch.isfinite(tip).all()`` it replaces was called in a
    Python conditional three times per step.
    """
    step = int(getattr(env, "common_step_counter", -1))
    cached = getattr(env, TIP_STATE_CACHE_ATTR, None)
    if cached is not None and cached[0] == step:
        return cached[1]
    starts, spans, start_arc = _route_tensors(env, route_world_m)
    total_m = start_arc[-1] + torch.linalg.norm(spans[-1])
    tip = catheter_tip_world_m(env)
    valid = torch.isfinite(tip).all(dim=-1)
    arc_m, lateral_m, segment = project_to_route(
        torch.where(valid.unsqueeze(-1), tip, starts[0]), starts, spans, start_arc
    )
    state = TipRouteState(torch.clamp(total_m - arc_m, min=0.0), lateral_m, total_m, segment, valid)
    setattr(env, TIP_STATE_CACHE_ATTR, (step, state))
    return state


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
    remaining_m = state.remaining_m
    # An unreadable tip is stored as nan rather than as the arc its substituted
    # position projects to, which is the whole route: differencing against that
    # on the step it becomes readable again would pay out the entire task.
    previous = getattr(env, REMAINING_ARC_ATTR, None)
    setattr(env, REMAINING_ARC_ATTR, torch.where(state.valid, remaining_m, torch.full_like(remaining_m, float("nan"))))
    if previous is None or previous.shape != remaining_m.shape:
        return torch.zeros_like(remaining_m)
    limit = abs(float(max_step_advance_m))
    advance = torch.clamp(previous - remaining_m, -limit, limit)
    # A reset environment, and one whose tip was unreadable last step, carry nan.
    return torch.nan_to_num(advance, nan=0.0) * state.valid


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
    starts, _spans, _start_arc = _route_tensors(env, route_world_m)
    radii = _segment_radii(env, lumen_radii_m, int(starts.shape[0]))
    if radii is None:
        return torch.zeros(int(env.num_envs), device=env.device)
    # The segment comes off the shared projection. This used to project the
    # same tip onto the same route a second time to recover it.
    allowed = radii[state.segment].clamp_min(1e-6)
    offset = torch.clamp(state.lateral_m - float(free_fraction) * allowed, min=0.0) / allowed
    return offset * state.valid


def wall_penetration_penalty(
    env: Any,
    route_world_m: Iterable[Iterable[float]],
    lumen_radii_m: Iterable[float] | None,
) -> torch.Tensor:
    """Deepest point, in metres, at which the rod sits outside the lumen wall.

    Every particle, not just the tip: a tip that threads the arch while the
    shaft behind it cuts the corner is the failure this is for. Depth rather
    than a count, so easing off a deep contact pays before the contact clears.

    The worst particle, not the average of them: averaging divides a local
    perforation by the particle count, so across this rod's 121 particles a
    tip 1 mm through the wall came to 0.99 over a full episode against a
    traverse worth 99. The max also keeps the weight meaningful for a rod with
    a different particle count, and stops the cost drifting with insertion
    depth as particles parked at the entry stop padding the denominator.

    The trade is that extent no longer registers: one particle 2 mm out scores
    the same as twenty, and nothing else in the configured reward charges for
    the difference.
    """
    positions = env.scene["catheter"].data.positions_world_m
    if positions is None:
        return torch.zeros(int(env.num_envs), device=env.device)
    points = torch.as_tensor(positions, dtype=torch.float32, device=env.device)
    starts, spans, start_arc = _route_tensors(env, route_world_m)
    radii = _segment_radii(env, lumen_radii_m, int(starts.shape[0]))
    if radii is None:
        return torch.zeros(int(env.num_envs), device=env.device)
    # Per environment and substituted before projecting, for the reason
    # :func:`tip_route_state` is: one diverged rod used to zero the wall cost
    # for every environment in the batch, and masking an infinity afterwards
    # yields nan rather than zero.
    valid = torch.isfinite(points).flatten(1).all(dim=-1)
    _, lateral_m, segment = project_to_route(
        torch.where(valid.view(-1, 1, 1), points, starts[0]), starts, spans, start_arc
    )
    return torch.clamp(lateral_m - radii[segment], min=0.0).amax(dim=-1) * valid


def route_length_m(route_world_m: Iterable[Iterable[float]]) -> float:
    """Total arc length of a route, for sizing progress weights against the task."""
    path = [tuple(float(value) for value in point) for point in route_world_m]
    return float(sum(math.dist(previous, current) for previous, current in zip(path[:-1], path[1:], strict=True)))


__all__ = [
    "MAX_STEP_ADVANCE_M",
    "REMAINING_ARC_ATTR",
    "TIP_STATE_CACHE_ATTR",
    "TipRouteState",
    "lateral_offset_penalty",
    "project_to_route",
    "remaining_arc_state",
    "reset_route_progress",
    "reset_tip_route_state",
    "route_length_m",
    "route_progress_reward",
    "tip_route_state",
    "wall_penetration_penalty",
]
