# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small self-contained centerline utilities for guided catheter initialization."""

from __future__ import annotations

import heapq

import numpy as np


def _arc_length_samples(points: np.ndarray, spacing: float) -> np.ndarray:
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(segments))
    if total <= 0.0:
        raise ValueError("centerline path has zero length")
    return np.linspace(0.0, total, max(2, int(np.ceil(total / spacing)) + 1))


def _resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    return sample_polyline(points, _arc_length_samples(points, spacing))


def sample_polyline_scalar(points: np.ndarray, values: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Sample per-vertex scalars along a polyline at arc-length distances.

    The scalar counterpart to :func:`sample_polyline`, so a quantity carried at
    the centerline's vertices -- the lumen radius, in practice -- can be
    resampled onto exactly the same stations as the path itself.
    """
    path = np.asarray(points, dtype=np.float64)
    scalars = np.asarray(values, dtype=np.float64).reshape(-1)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3:
        raise ValueError("points must have shape (N, 3) with N >= 2")
    if scalars.shape[0] != path.shape[0]:
        raise ValueError(f"values must have one entry per point, got {scalars.shape[0]} for {path.shape[0]}")

    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    clipped = np.clip(np.asarray(distances, dtype=np.float64), 0.0, cumulative[-1])
    indices = np.clip(np.searchsorted(cumulative, clipped, side="right") - 1, 0, path.shape[0] - 2)
    start = cumulative[indices]
    width = cumulative[indices + 1] - start
    alpha = np.divide(clipped - start, width, out=np.zeros_like(clipped), where=width > 1e-12)
    return ((1.0 - alpha) * scalars[indices] + alpha * scalars[indices + 1]).astype(np.float32)


def sample_polyline(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Sample a polyline at arc-length distances, clamped to its endpoints."""
    path = np.asarray(points, dtype=np.float64)
    requested = np.asarray(distances, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3:
        raise ValueError("points must have shape (N, 3) with N >= 2")
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    sampled = np.empty((*requested.shape, 3), dtype=np.float64)
    clipped = np.clip(requested, 0.0, cumulative[-1])
    indices = np.searchsorted(cumulative, clipped, side="right") - 1
    indices = np.clip(indices, 0, path.shape[0] - 2)
    start = cumulative[indices]
    width = cumulative[indices + 1] - start
    alpha = np.divide(clipped - start, width, out=np.zeros_like(clipped), where=width > 1e-12)
    sampled[...] = (1.0 - alpha[..., None]) * path[indices] + alpha[..., None] * path[indices + 1]
    return sampled.astype(np.float32)


CENTERLINE_SMOOTHING_MM = 8.8
"""Arc length over which to smooth an extracted centerline.

Skeletonizing a segmentation leaves curvature the anatomy does not have. On
``s0011`` the raw path bends to a 10.5 mm radius at the solver's own sampling,
and a rod of 40 segments over 660 mm would have to turn 104 degrees at a single
joint to follow it, which it cannot: the segment length is comparable to the
diameter of the bend. The rod deviates instead, most visibly at the tip, where
the last joint has a neighbour on one side only and is the least constrained.

At 8.8 mm the tightest radius becomes 22.4 mm and the median 148 mm, while no
point of the path moves more than 3.4 mm -- well inside a lumen of radius 9 to
14 mm, so tidying the path does not push it into a wall. Smoothing halves the
turn a joint must make but does not remove it; segment count is the other term.
"""

_SMOOTHING_LUMEN_FRACTION = 0.5
"""How far into the lumen radius a sample may be moved to tidy the path.

Half leaves the smoothed path comfortably inside the vessel wall even where the
original was already off centre. Without a cap, one sample on ``s0011`` moved
3.46 mm where the iliac radius is 3.31 mm, so the seeded rod would have started
outside the vessel it is meant to be threaded through.
"""

_SMOOTHING_SAMPLE_MM = 1.0
"""Spacing to smooth at, before resampling to whatever the caller asked for.

The width has to mean millimetres of vessel rather than a number of samples,
and the raw skeleton's spacing is not uniform. Smoothing at the caller's
spacing would also be self-defeating for coarse requests: at 7.5 mm an 8.8 mm
kernel is barely one sample wide, and the noise worth removing has already been
aliased into the samples that survive.
"""


def _smooth_along_arc(
    path: np.ndarray,
    *,
    spacing_mm: float,
    width_mm: float,
    radii_mm: np.ndarray | None = None,
    lumen_fraction: float = _SMOOTHING_LUMEN_FRACTION,
) -> np.ndarray:
    """Gaussian-smooth a uniformly sampled path, holding both ends fixed.

    The ends are pinned because they are not merely samples: the first is the
    access point the rod is seeded from and aims its track down, and the last is
    the navigation target that success is measured against. Neither may move
    because the path in between was tidied.

    The correction is also capped against the local lumen. One width cannot suit
    the whole route: on ``s0011`` a straight 8.8 mm moved the path 3.46 mm where
    the iliac radius is 3.31 mm, putting the seeded rod outside the vessel it is
    meant to start inside. Capping keeps the smoothing strong in the wide aorta,
    which is where the rod's coarse segments actually struggle, and gentle in the
    narrow access vessels, where there is no room to be moved.
    """
    count = int(path.shape[0])
    if width_mm <= 0.0 or count < 3:
        return path
    sigma = float(width_mm) / float(spacing_mm)
    if sigma < 0.5:
        return path
    half = int(max(1, round(3.0 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1, dtype=np.float64) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.vstack([np.repeat(path[:1], half, axis=0), path, np.repeat(path[-1:], half, axis=0)])
    smoothed = np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)], axis=1)
    # Restore the ends with a correction that is linear in arc length. Ramping
    # the correction in over the kernel's reach instead would leave the last
    # ~3 sigma barely smoothed -- and the distal end is exactly where the rod
    # deviates worst, so that is the one stretch that must not be exempt. A
    # linear field has no second derivative, so it pins both ends without
    # putting back any of the curvature just removed.
    fraction = np.linspace(0.0, 1.0, count)[:, None]
    smoothed = smoothed + (1.0 - fraction) * (path[0] - smoothed[0]) + fraction * (path[-1] - smoothed[-1])

    delta = smoothed - path
    if radii_mm is not None:
        limit = max(0.0, float(lumen_fraction)) * np.asarray(radii_mm, dtype=np.float64).reshape(-1)
        distance = np.linalg.norm(delta, axis=1)
        excess = distance > limit
        delta[excess] *= (limit[excess] / np.maximum(distance[excess], 1e-12))[:, None]
    return path + delta


def ordered_centerline_path(
    points_mm: np.ndarray,
    edges: np.ndarray,
    *,
    target_spacing_mm: float,
    radii_mm: np.ndarray | None = None,
    smoothing_mm: float = CENTERLINE_SMOOTHING_MM,
) -> np.ndarray:
    """Recover and uniformly sample the reference viewport's primary vessel path."""
    return ordered_centerline_lumen(
        points_mm,
        edges,
        target_spacing_mm=target_spacing_mm,
        radii_mm=radii_mm,
        smoothing_mm=smoothing_mm,
    )[0]


def ordered_centerline_lumen(
    points_mm: np.ndarray,
    edges: np.ndarray,
    *,
    target_spacing_mm: float,
    radii_mm: np.ndarray | None = None,
    smoothing_mm: float = CENTERLINE_SMOOTHING_MM,
) -> tuple[np.ndarray, np.ndarray | None]:
    """The vessel path and how wide the vessel is at each sample along it.

    The radii were already being read to weight the graph search; returning them
    resampled as well is what lets a caller ask whether a point is actually
    inside the lumen rather than merely near the path. Without a width, "on the
    centerline" has no tolerance, and a prescribed point can sit centimetres
    outside a vessel with no one to object.

    Args:
        smoothing_mm: Arc length to smooth the path over, in millimetres. See
            :data:`CENTERLINE_SMOOTHING_MM` for why the default is not zero.
            Pass ``0.0`` for the raw skeleton.

    Returns:
        ``(path_mm, radii_mm)``, the path uniformly resampled and the lumen
        radius at each of its samples. The radii are ``None`` when the caller
        supplied none, or supplied one per point that did not match the
        centerline.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    edge_array = np.asarray(edges, dtype=np.int64)
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] != 3:
        raise ValueError("centerline points must have shape (N, 3) with N >= 4")
    if edge_array.ndim != 2 or edge_array.shape[0] < 1 or edge_array.shape[1] != 2:
        raise ValueError("centerline edges must have shape (M, 2)")
    if np.any(edge_array < 0) or np.any(edge_array >= points.shape[0]):
        raise ValueError("centerline edges contain an out-of-range node")
    if not np.isfinite(target_spacing_mm) or target_spacing_mm <= 0.0:
        raise ValueError("target_spacing_mm must be positive and finite")

    radii = None
    if radii_mm is not None:
        candidate = np.asarray(radii_mm, dtype=np.float64).reshape(-1)
        if candidate.shape == (points.shape[0],):
            radii = np.maximum(candidate, 1e-3)

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(points.shape[0])]
    degree = np.zeros(points.shape[0], dtype=np.int32)
    for first, second in edge_array:
        distance = float(np.linalg.norm(points[first] - points[second]))
        if not np.isfinite(distance) or distance <= 1e-6:
            continue
        weight = distance
        if radii is not None:
            weight /= max(float(np.sqrt(0.5 * (radii[first] + radii[second]))), 1e-3)
        adjacency[int(first)].append((int(second), weight))
        adjacency[int(second)].append((int(first), weight))
        degree[first] += 1
        degree[second] += 1

    endpoints = np.flatnonzero(degree == 1)
    start = int(endpoints[np.argmin(points[endpoints, 2])]) if endpoints.size else int(np.argmin(points[:, 2]))
    distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    previous = np.full(points.shape[0], -1, dtype=np.int64)
    distance[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance > distance[current]:
            continue
        for neighbor, weight in adjacency[current]:
            candidate = current_distance + weight
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    reachable = np.isfinite(distance)
    if not np.any(reachable):
        raise RuntimeError("centerline graph contains no reachable nodes")
    end = int(np.argmax(np.where(reachable, distance, -1.0)))

    indices: list[int] = []
    current = end
    while current >= 0:
        indices.append(current)
        if current == start:
            break
        current = int(previous[current])
    indices.reverse()
    if len(indices) < 2:
        raise RuntimeError("recovered centerline path is too short")
    ordered = np.asarray(indices)
    path = points[ordered]
    path_radii = None if radii is None else radii[ordered]

    # Smooth on a uniform fine grid, so the width is millimetres of vessel and
    # not a sample count. The 3-tap pass this replaced was about 1 mm wide,
    # which left the skeleton's spurious curvature essentially intact.
    fine_spacing = min(float(target_spacing_mm), _SMOOTHING_SAMPLE_MM)
    fine_samples = _arc_length_samples(path, fine_spacing)
    fine_radii = None if path_radii is None else sample_polyline_scalar(path, path_radii, fine_samples)
    fine_path = _smooth_along_arc(
        sample_polyline(path, fine_samples),
        spacing_mm=fine_spacing,
        width_mm=float(smoothing_mm),
        radii_mm=fine_radii,
    )

    samples = _arc_length_samples(fine_path, float(target_spacing_mm))
    resampled_radii = None if fine_radii is None else sample_polyline_scalar(fine_path, fine_radii, samples)
    return sample_polyline(fine_path, samples), resampled_radii
