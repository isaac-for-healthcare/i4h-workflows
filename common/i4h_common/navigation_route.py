# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The route a catheter navigates, and the goal columns derived from it.

The catheter descriptor declares five state columns that no actuator reports:
the tip-to-target offset and the remaining-arc and lateral-offset pair. They are
derived rather than recorded because they are facts about the anatomy and the
goal, not readings from the drive. Deriving them means rebuilding the same route
the Scene navigated and projecting the recorded tip onto it.

Here rather than in arena because ``tools/dataset`` cannot import arena without
pulling in Isaac Lab, and because the derivation has to agree with the Scene
exactly. A route rebuilt at a different sample spacing gives different arc
figures for the same episode, and nothing would raise -- the dataset would
simply be mislabelled. :data:`ROUTE_SPACING_MM` is therefore defined once, here,
and read by the embodiment rather than repeated in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from i4h_common.centerline import ordered_centerline_lumen
from i4h_common.patient_twin import PatientTwin

#: Arc spacing the centerline is resampled to, in millimetres.
#:
#: Owned here because two consumers have to agree on it: the Scene seeds the rod
#: and drives the root along this polyline, and conversion projects the recorded
#: tip onto it to label the goal columns. A disagreement is silent -- both sides
#: produce a plausible route and plausible arc figures that mean different
#: things -- so the constant has one home and the embodiment imports it.
ROUTE_SPACING_MM = 7.5

#: The derived state columns, in the order ``catheter.yaml`` declares them.
#:
#: Matches ``navigation_observation.route_state`` for the trailing pair, which is
#: the same order the reward reads them in. Asserted against the descriptor by
#: the dataset tests rather than trusted.
GOAL_COLUMN_NAMES = (
    "target_offset_x_m",
    "target_offset_y_m",
    "target_offset_z_m",
    "route_remaining_m",
    "route_lateral_m",
)


@dataclass(frozen=True, slots=True)
class NavigationRoute:
    """A resolved centerline route, its lumen widths, and where it ends.

    Attributes:
        path_world_m: ``(N, 3)`` route samples in Isaac world metres.
        lumen_radii_m: ``(N,)`` vessel radius at each sample, or ``None`` when
            the twin carries no radii. Conversion does not need them, but a
            caller checking whether the tip was ever inside the lumen does.
        target_world_m: ``(3,)`` navigation target, which is the route's last
            sample. The rod is seeded over the first part of the path, so the
            tip starts short of this and insertion has to cover the remainder.
    """

    path_world_m: np.ndarray
    lumen_radii_m: np.ndarray | None
    target_world_m: np.ndarray

    @property
    def total_length_m(self) -> float:
        """Arc length of the whole route, which is what ``remaining`` counts down from."""
        return float(np.sum(np.linalg.norm(np.diff(self.path_world_m, axis=0), axis=1)))


def resolve_navigation_route(twin_manifest: str | Path) -> NavigationRoute:
    """Rebuild the route a catheter Scene navigates, from its patient twin.

    Mirrors ``CatheterEmbodiment._apply_patient_twin``: read the skeleton the
    twin points at, order and resample it, then carry it into world metres. The
    steps are duplicated rather than shared because the Scene does far more with
    the result -- seeding the rod, placing the drive root, sizing the C-arm --
    and only the path, the radii and the endpoint are wanted here.

    Raises:
        KeyError: when the twin declares no centerline artifacts. Raised rather
            than returning ``None`` because a caller asking for a route has
            nothing to fall back on, and a twin without a centerline cannot
            have produced a catheter recording in the first place.
    """
    twin = PatientTwin.load(twin_manifest)
    try:
        points_patient_mm = np.load(twin.artifacts["centerline_points"])
        edges = np.load(twin.artifacts["centerline_edges"])
    except KeyError as exc:
        raise KeyError(
            f"{twin.source}: no centerline artifacts, so this twin describes no navigable route. "
            "A catheter recording cannot have come from it."
        ) from exc
    radii_path = twin.artifacts.get("centerline_radii")
    radii_mm = np.load(radii_path) if radii_path is not None else None

    path_patient_mm, lumen_radii_mm = ordered_centerline_lumen(
        points_patient_mm,
        edges,
        target_spacing_mm=ROUTE_SPACING_MM,
        radii_mm=radii_mm,
    )
    path_world_m = np.asarray(twin.patient_mm_to_world(path_patient_mm), dtype=np.float64)
    return NavigationRoute(
        path_world_m=path_world_m,
        lumen_radii_m=None if lumen_radii_mm is None else np.asarray(lumen_radii_mm, dtype=np.float64) / 1000.0,
        target_world_m=path_world_m[-1].copy(),
    )


def project_to_route(path_world_m: np.ndarray, points_world_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(arc_m, lateral_m)`` of the nearest route point to each input point.

    Batched over the leading dimension, and deliberately the same computation as
    the torch version in ``navigation_reward.project_to_route`` -- including the
    zero-length-segment guard, which matters because extracted centerlines do
    contain duplicated samples and a nan would win the ``argmin`` and report the
    tip as nowhere. A reward and the labels trained against it disagreeing about
    where the tip was would be the worst kind of bug to find later.
    """
    path = np.asarray(path_world_m, dtype=np.float64)
    points = np.asarray(points_world_m, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
        raise ValueError(f"route must be at least two 3-D points, got shape {path.shape}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    starts, ends = path[:-1], path[1:]
    spans = ends - starts
    lengths_sq = np.einsum("sj,sj->s", spans, spans)
    safe = np.where(lengths_sq > 0.0, lengths_sq, 1.0)
    lengths = np.sqrt(lengths_sq)
    start_arc = np.concatenate(([0.0], np.cumsum(lengths)[:-1]))

    offset = points[:, None, :] - starts
    fraction = np.clip(np.einsum("nsj,sj->ns", offset, spans) / safe, 0.0, 1.0)
    projected = starts + fraction[..., None] * spans
    distances = np.linalg.norm(points[:, None, :] - projected, axis=-1)
    segment = np.argmin(distances, axis=-1)
    rows = np.arange(points.shape[0])
    arc_m = start_arc[segment] + fraction[rows, segment] * lengths[segment]
    return arc_m, distances[rows, segment]


def goal_columns(tip_world_m: np.ndarray, route: NavigationRoute) -> np.ndarray:
    """``(N, 5)`` goal columns for a recorded tip trajectory.

    The columns :data:`GOAL_COLUMN_NAMES` names, in that order: the offset from
    tip to target in world axes, then remaining arc and lateral offset.

    World axes rather than tip-local, matching the ``target_offset`` observation
    the Scene publishes, so a policy needs its own heading to act on it. Changing
    that here would make the recorded columns disagree with what a policy sees at
    rollout, which is the one thing this function exists to prevent.

    Non-finite tip samples yield zeros for that frame rather than propagating.
    The solver does emit transient non-finite positions, and a nan reaching the
    dataset poisons GR00T's normalization statistics for every episode in it.
    """
    tip = np.asarray(tip_world_m, dtype=np.float64)
    if tip.ndim != 2 or tip.shape[1] != 3:
        raise ValueError(f"tip_world_m must have shape (N, 3), got {tip.shape}")

    finite = np.isfinite(tip).all(axis=1)
    columns = np.zeros((tip.shape[0], len(GOAL_COLUMN_NAMES)), dtype=np.float32)
    if not finite.any():
        return columns

    usable = tip[finite]
    arc_m, lateral_m = project_to_route(route.path_world_m, usable)
    # Clamped because a tip past the final sample projects onto the last
    # segment's end, and a negative remainder would read as an overshoot the
    # route cannot express.
    remaining_m = np.maximum(route.total_length_m - arc_m, 0.0)
    columns[finite, 0:3] = route.target_world_m - usable
    columns[finite, 3] = remaining_m
    columns[finite, 4] = lateral_m
    return columns


__all__ = [
    "GOAL_COLUMN_NAMES",
    "ROUTE_SPACING_MM",
    "NavigationRoute",
    "goal_columns",
    "project_to_route",
    "resolve_navigation_route",
]
