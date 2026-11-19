# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Insertion measurements with separate simulation and wall clocks."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Containment fields worth a per-frame column. The report also carries constants
#: such as ``num_particles`` and ``rest_length_mm``, which would be the same value
#: on every frame of every episode and are better read off the spec.
RECORDED_CONTAINMENT_FIELDS = (
    "worst_penetration_mm",
    "particles_outside",
    "chord_min_pct",
    "chord_max_pct",
    "arc_length_mm",
    "arc_excess_mm",
    "min_bend_radius_mm",
    "min_bend_radius_node",
    "bend_radius_p05_mm",
    "kinked_nodes",
    "first_kinked_node",
    "last_kinked_node",
    "live_worst_penetration_mm",
    "live_samples_outside",
)


class CatheterEpisodeDiagnostics:
    """Per-frame catheter state for the episode recording.

    What the recording otherwise holds -- the four commanded joints and the
    fluoroscopy image -- cannot distinguish a clean run from a prolapse. A
    catheter that buckles into a loop still reports plausible insertion and
    rotation, and the projection hides the fold behind the anatomy. So an
    episode can reach the target, be labelled a success, and be a demonstration
    of the wire coiling rather than of navigating. These columns are what make
    that difference visible without replaying the episode.

    ``tip_target_distance_m`` is the same quantity the operator's readout shows
    and the same one arrival is judged on, recorded per frame so that the
    approach can be read back rather than only its final verdict. Paired with
    ``min_bend_radius_mm`` it separates the two ways an episode stalls: a tip
    that stops advancing with the rod straight is anatomy, and one that stops
    while the bend radius collapses is the rod folding.

    Reading particles costs a device-to-host copy, which is why the standalone
    probe is off by default. Here it is paid once per recorded frame, against a
    recording that is already writing images.
    """

    def __init__(self, catheter: Any, *, target_world_m: Iterable[float] | None = None) -> None:
        self._catheter = catheter
        self._target_world_m = None if target_world_m is None else np.asarray(tuple(target_world_m), dtype=np.float64)

    def diagnostics(self) -> dict[str, float | np.ndarray]:
        """Measurements for the current frame, or ``{}`` before there are particles."""
        points = self._points_world_m()
        if points is None:
            return {}
        values: dict[str, float | np.ndarray] = {
            "tip_world_m": points[-1].astype(np.float32),
            "root_world_m": points[0].astype(np.float32),
        }
        if self._target_world_m is not None:
            values["tip_target_distance_m"] = float(np.linalg.norm(points[-1] - self._target_world_m))
        values.update(self._containment(points))
        return values

    def _points_world_m(self) -> np.ndarray | None:
        positions = getattr(getattr(self._catheter, "data", None), "positions_world_m", None)
        if positions is None:
            return None
        array = np.asarray(_to_numpy(positions), dtype=np.float64)
        # The asset stores ``(num_envs, num_points, 3)``. Env 0 only, matching
        # the recorder, because the HDF5 schema is one trajectory per demo --
        # flattening the batch instead would splice every environment's rod into
        # one polyline and put the tip and the bend measurements somewhere
        # between them.
        if array.ndim == 3:
            array = array[0]
        points = array.reshape(-1, 3)
        return points if len(points) >= 2 else None

    @staticmethod
    def _containment(points: np.ndarray) -> Mapping[str, float]:
        """Containment, chord and bend fields, or ``{}`` without a centerline.

        Routed through the installed rod handle rather than recomputed so the
        recorded numbers are the ones the live probe prints, and so a scene on a
        straight track -- which has no lumen to measure against -- records the
        tip alone instead of failing.
        """
        from i4h_arena.medical.newton_catheter_physics import active_handle

        handle = active_handle()
        if handle is None:
            return {}
        report = handle.report_containment(points)
        if report is None:
            return {}
        return {name: float(report[name]) for name in RECORDED_CONTAINMENT_FIELDS if name in report}


def _to_numpy(values: Any) -> Any:
    """Host copy of a torch tensor or anything already array-like."""
    detach = getattr(values, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    return values


@dataclass
class InsertionSample:
    simulation_s: float
    wall_s: float
    commanded_m: float
    points_m: np.ndarray


def _projected_displacement(displacement: np.ndarray, tangent: np.ndarray) -> float:
    length = float(np.linalg.norm(tangent))
    return float(np.dot(displacement, tangent) / length) if length > 0.0 else 0.0


def route_coordinate_m(point: np.ndarray, path: np.ndarray) -> float:
    """Arc coordinate of a point's nearest projection on the ordered route."""
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError("route must contain at least two finite 3D points")
    edges = np.diff(path, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if not np.any(lengths > 0.0):
        raise ValueError("route must have nonzero length")
    fraction = np.clip(np.einsum("ij,ij->i", point - path[:-1], edges) / np.maximum(lengths**2, 1.0e-30), 0.0, 1.0)
    projected = path[:-1] + fraction[:, None] * edges
    edge = int(np.linalg.norm(projected - point, axis=1).argmin())
    return float(lengths[:edge].sum() + fraction[edge] * lengths[edge])


def insertion_report(
    previous: InsertionSample, current: InsertionSample, route_m: np.ndarray | None = None
) -> dict[str, float]:
    """Compare integrated feed and endpoint travel over the same sim interval.

    Root displacement uses the interval's initial proximal tangent. Tip
    progress uses route arc coordinates when available, otherwise the initial
    distal tangent. Neither speed is divided by wall time; wall time is used
    only for the real-time factor.
    """
    simulation_s = current.simulation_s - previous.simulation_s
    wall_s = current.wall_s - previous.wall_s
    if simulation_s <= 0.0 or wall_s <= 0.0:
        raise ValueError("simulation and wall time must both advance")
    old, new = previous.points_m, current.points_m
    if old.shape != new.shape or old.ndim != 2 or old.shape[1] != 3 or len(old) < 2:
        raise ValueError("endpoint samples must have matching Nx3 shapes with at least two points")
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("endpoint samples must be finite")
    root_m = _projected_displacement(new[0] - old[0], old[1] - old[0])
    if route_m is None:
        tip_m = _projected_displacement(new[-1] - old[-1], old[-1] - old[-2])
    else:
        tip_m = route_coordinate_m(new[-1], route_m) - route_coordinate_m(old[-1], route_m)
    command_m = current.commanded_m - previous.commanded_m
    return {
        "simulation_s": simulation_s,
        "wall_s": wall_s,
        "real_time_factor": simulation_s / wall_s,
        "commanded_m": command_m,
        "root_m": root_m,
        "tip_m": tip_m,
        "commanded_mps": command_m / simulation_s,
        "root_mps": root_m / simulation_s,
        "tip_mps": tip_m / simulation_s,
    }


def tube_surface_gaps_m(
    catheter: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray,
    catheter_radius_m: float,
    *,
    open_root: int = -1,
    open_root_neighbor: int = -1,
) -> np.ndarray:
    """Surface gaps using the collision model's tapered-tube union.

    Positive means outside. Includes the same three interior samples per rod
    edge as collision, with the same tapered-radius projection and open inlet.
    Samples outside that inlet are excluded. This is a read-only diagnostic
    of the final committed state, not the depth before a contact correction.
    """
    catheter = np.asarray(catheter, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    radii = np.asarray(radii, dtype=np.float64)
    fractions = np.array([0.25, 0.5, 0.75])
    interior = catheter[:-1, None, :] + fractions[None, :, None] * np.diff(catheter, axis=0)[:, None, :]
    samples = np.concatenate((catheter, interior.reshape(-1, 3)))
    if open_root >= 0 and open_root_neighbor >= 0:
        inward = nodes[open_root_neighbor] - nodes[open_root]
        samples = samples[(samples - nodes[open_root]) @ inward >= 0.0]
    best = np.full(len(samples), np.inf)
    for start, end in edges:
        a, b = nodes[start], nodes[end]
        axis = b - a
        length = float(np.linalg.norm(axis))
        if length <= 1.0e-8:
            continue
        fraction = np.clip((samples - a) @ axis / length**2, 0.0, 1.0)
        radial = np.linalg.norm(samples - a - fraction[:, None] * axis, axis=1)
        dr = radii[end] - radii[start]
        if abs(dr) < length - 1.0e-8:
            fraction = np.clip(fraction + dr * radial / (length * np.sqrt(max(length**2 - dr**2, 1.0e-12))), 0.0, 1.0)
        distance = np.linalg.norm(samples - a - fraction[:, None] * axis, axis=1)
        radius = radii[start] + fraction * dr
        best = np.minimum(best, distance - radius + catheter_radius_m)
    return best
