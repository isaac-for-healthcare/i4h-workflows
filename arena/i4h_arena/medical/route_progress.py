# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How far along the planned route the tip has come.

Arrival is judged on straight-line distance to the far end of the centerline,
and on a vessel that doubles back that is not a measure of progress. On the
shipped s0011 aorta a tip advancing correctly along the lumen watches the
straight-line distance fall to 53 mm, climb back to 79 mm through the arch, and
only then close. An operator reading that number sees their own correct
insertion as a mistake and backs it out.

Remaining arc along the route falls monotonically under the same motion, which
is what makes it drivable. The two answer different questions and both belong
on screen: arc says how much vessel is left, straight-line says what the
success term is about to do.

Nearest-point projection only identifies where the tip is if the tip is closer
to its own stretch of route than to any other, so :func:`unambiguous_radius_m`
measures that from the route itself and progress is reported with whether it
holds. Assuming it would be the same mistake in a different place: a tip in the
descending aorta could otherwise be reported as most of the way along the arch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

#: Arc separation beyond which landing on the wrong stretch of route would be a
#: real mistake rather than a rounding one. It also sets how much of the route's
#: own length is excluded when measuring self-approach: any curve is close to
#: itself a short way along, so a small value measures that continuity instead
#: of the fold it is meant to find. On the shipped s0011 aorta 100 mm leaves the
#: arch limbs as the closest genuine approach, at 52 mm.
DEFAULT_MIN_ARC_GAP_M = 0.1


@dataclass(frozen=True)
class RouteProgress:
    """Where the tip sits along the route, and whether that is knowable.

    Attributes:
        arc_m: Arc length from the route's start to the tip's projection.
        remaining_m: Arc length from that projection to the route's end.
        lateral_m: Distance from the tip to the route itself.
        on_route: Whether ``lateral_m`` is inside the route's ambiguity radius.
            False means the arc figures name a place the tip may not be at, so
            a reader should prefer the straight-line distance.
    """

    arc_m: float
    remaining_m: float
    lateral_m: float
    on_route: bool


def _as_path(path_world_m: Any) -> np.ndarray:
    path = np.asarray(path_world_m, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError(f"route must be at least two 3-D points, got shape {path.shape}")
    if not np.isfinite(path).all():
        raise ValueError("route contains non-finite points")
    return path


def route_arc_m(path_world_m: Any) -> np.ndarray:
    """Cumulative arc length at each route point, starting at zero."""
    path = _as_path(path_world_m)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _project_to_segments(path: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment ``(distance, arc)`` for the closest point on each segment."""
    starts, ends = path[:-1], path[1:]
    spans = ends - starts
    lengths_sq = np.einsum("ij,ij->i", spans, spans)
    # A repeated route point gives a zero-length segment; clamping the divisor
    # collapses it onto its start rather than producing a nan that would win
    # the argmin and report the tip as nowhere.
    travel = np.clip(
        np.einsum("ij,ij->i", point - starts, spans) / np.where(lengths_sq > 0.0, lengths_sq, 1.0),
        0.0,
        1.0,
    )
    closest = starts + travel[:, None] * spans
    arcs = route_arc_m(path)[:-1] + travel * np.sqrt(lengths_sq)
    return np.linalg.norm(point - closest, axis=1), arcs


def project_to_route(path_world_m: Any, point_world_m: Any) -> tuple[float, float]:
    """``(arc_m, lateral_m)`` of the route point closest to ``point_world_m``."""
    path = _as_path(path_world_m)
    point = np.asarray(point_world_m, dtype=np.float64).reshape(3)
    distances, arcs = _project_to_segments(path, point)
    nearest = int(np.argmin(distances))
    return float(arcs[nearest]), float(distances[nearest])


@lru_cache(maxsize=8)
def _unambiguous_radius_m(path_key: tuple[tuple[float, float, float], ...], min_arc_gap_m: float) -> float:
    path = _as_path(path_key)
    arcs = route_arc_m(path)
    closest = np.inf
    for index, point in enumerate(path):
        distances, segment_arcs = _project_to_segments(path, point)
        # Only stretches genuinely elsewhere on the route can be confused for
        # this one; a neighbouring segment is near because the route is
        # continuous, and projection onto it is the right answer, not a mix-up.
        far = np.abs(segment_arcs - arcs[index]) > float(min_arc_gap_m)
        if far.any():
            closest = min(closest, float(distances[far].min()))
    # Halved: at exactly half the separation the tip is equidistant from both
    # stretches and the nearest-point answer is a coin toss.
    return closest / 2.0


def unambiguous_radius_m(path_world_m: Any, *, min_arc_gap_m: float = DEFAULT_MIN_ARC_GAP_M) -> float:
    """How far off-route the tip may stray before its arc position is guesswork.

    Half the closest approach between two stretches of route more than
    ``min_arc_gap_m`` apart along it. A tip within this distance of the route
    is guaranteed to project onto a point within ``min_arc_gap_m`` of where it
    truly is, which is the claim the readout needs: not that the arc figure is
    exact, but that it is not naming the other limb of the arch.

    The guarantee is therefore conservative, and deliberately so. A straight
    route gets half the arc gap rather than infinity, because that is all this
    measurement can witness -- projection onto a straight route is in fact
    exact at any offset, but establishing that needs the curve's reach, which
    is far more machinery than a status line justifies. Infinite only when the
    route is too short for any pair to be ``min_arc_gap_m`` apart.

    Measured from route vertices against route segments, so it is a sample
    rather than a bound: it assumes the vertices resolve the curve, which for
    the shipped twins they do at 7.5 mm spacing.
    """
    path = _as_path(path_world_m)
    return _unambiguous_radius_m(tuple(map(tuple, path)), float(min_arc_gap_m))


def route_progress(
    path_world_m: Any,
    tip_world_m: Any,
    *,
    min_arc_gap_m: float = DEFAULT_MIN_ARC_GAP_M,
) -> RouteProgress:
    """Arc travelled, arc remaining, and how far off the route the tip is."""
    path = _as_path(path_world_m)
    arc_m, lateral_m = project_to_route(path, tip_world_m)
    total_m = float(route_arc_m(path)[-1])
    return RouteProgress(
        arc_m=arc_m,
        # Clamped because a tip past the final point projects onto the last
        # segment's end, and a negative remainder would read as overshoot the
        # route cannot express.
        remaining_m=max(0.0, total_m - arc_m),
        lateral_m=lateral_m,
        on_route=lateral_m <= unambiguous_radius_m(path, min_arc_gap_m=min_arc_gap_m),
    )


__all__ = [
    "DEFAULT_MIN_ARC_GAP_M",
    "RouteProgress",
    "project_to_route",
    "route_arc_m",
    "route_progress",
    "unambiguous_radius_m",
]
