# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small self-contained centerline utilities for guided catheter initialization."""

from __future__ import annotations

import heapq

import numpy as np


def _resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total = float(cumulative[-1])
    if total <= 0.0:
        raise ValueError("centerline path has zero length")
    samples = np.linspace(0.0, total, max(2, int(np.ceil(total / spacing)) + 1))
    return sample_polyline(points, samples)


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


def ordered_centerline_path(
    points_mm: np.ndarray,
    edges: np.ndarray,
    *,
    target_spacing_mm: float,
    radii_mm: np.ndarray | None = None,
) -> np.ndarray:
    """Recover and uniformly sample the reference viewport's primary vessel path."""
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
    path = points[np.asarray(indices)]
    if path.shape[0] > 2:
        smoothed = path.copy()
        smoothed[1:-1] = 0.25 * path[:-2] + 0.5 * path[1:-1] + 0.25 * path[2:]
        path = smoothed
    return _resample_polyline(path, float(target_spacing_mm))
