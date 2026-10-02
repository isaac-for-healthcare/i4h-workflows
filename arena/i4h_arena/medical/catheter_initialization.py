# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initial particle/frame state for the catheter's local-Z XPBD rod.

Reset pose and material rest curvature are separate: placing a catheter in a
curved vessel must not manufacture that curve into its unloaded shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions."""
    return np.concatenate((a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), [a[3] * b[3] - a[:3] @ b[:3]]))


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if dot < -1.0 + 1.0e-12:
        # A half-turn needs an axis perpendicular to this source, not a fixed
        # world axis (which could itself be parallel to the source).
        basis = np.eye(3)[np.argmin(np.abs(source))]
        axis = np.cross(source, basis)
        return np.concatenate((axis / np.linalg.norm(axis), [0.0]))
    quat = np.concatenate((np.cross(source, target), [1.0 + dot]))
    return quat / np.linalg.norm(quat)


def rod_frames_along_polyline(positions_world_m: np.ndarray) -> np.ndarray:
    """Parallel-transport xyzw frames with local +Z along the node tangents.

    The XPBD stretch constraint uses half-length local-Z offsets at both ends
    of each edge. These frames avoid an arbitrary straight initial orientation;
    the curved discrete constraint may still have a small geometric residual.
    This constructs initial frames, not a stress-free material rest shape.
    """
    points = np.asarray(positions_world_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("positions must have shape (N, 3) with at least two points")
    if not np.isfinite(points).all():
        raise ValueError("positions must be finite")
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("consecutive rod positions must be distinct")
    directions = edges / lengths[:, None]
    tangents = np.empty_like(points)
    tangents[0], tangents[-1] = directions[0], directions[-1]
    tangents[1:-1] = directions[:-1] + directions[1:]
    norms = np.linalg.norm(tangents, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("a reversing polyline has an undefined node tangent")
    tangents /= norms[:, None]

    frames = np.empty((len(points), 4), dtype=np.float64)
    frames[0] = _rotation_between(np.array([0.0, 0.0, 1.0]), tangents[0])
    for index in range(1, len(points)):
        transport = _rotation_between(tangents[index - 1], tangents[index])
        frame = _multiply(transport, frames[index - 1])
        frame /= np.linalg.norm(frame)
        # Equivalent quaternion signs must not create a false bend jump.
        frames[index] = -frame if frame @ frames[index - 1] < 0.0 else frame
    return frames.astype(np.float32)


def initialize_rod_state(solver: Any, positions_world_m: np.ndarray) -> None:
    """Seed current, predicted and reset state before stepping or graph capture.

    Arena owns the patient placement. Both solver workspaces receive that same
    placement and transported frames, while lengths, stiffness and rest Darboux
    values retain the configured device properties. Subsequent resets use the
    solver's existing in-place, per-environment reset kernels.
    """
    points = np.asarray(positions_world_m, dtype=np.float32)
    if points.shape != (solver.num_points, 3):
        raise ValueError(f"positions must have shape ({solver.num_points}, 3), got {points.shape}")
    frames = rod_frames_along_polyline(points)
    for workspace in (solver._ws, solver._bws):
        if workspace is None:
            continue
        copies = workspace.positions.shape[0] // len(points)
        positions = np.tile(points, (copies, 1))
        orientations = np.tile(frames, (copies, 1))
        for name in ("positions", "predicted_positions", "rest_positions"):
            getattr(workspace, name).assign(positions)
        for name in ("orientations", "predicted_orientations", "prev_orientations", "rest_orientations"):
            getattr(workspace, name).assign(orientations)
        for name in ("velocities", "angular_velocities", "forces", "torques", "lambda_sum"):
            getattr(workspace, name).zero_()


def publish_reset_state(solver: Any, particle_range: Any, states: tuple[Any, ...], env_ids: Any) -> None:
    """Publish only reset environments to both Newton buffers before readback.

    The rod's reset restores its internal state. Newton remains authoritative
    for incoming particle positions, so leaving old values there would undo
    the reset on the next step. Other particles/environments are untouched.
    """
    import warp as wp

    ids = solver._resolve_env_ids(env_ids)
    for state in states:
        if state is None:
            continue
        for env in ids:
            source = int(env) * solver.num_points
            destination = particle_range.offset + source
            wp.copy(state.particle_q, solver.position_array, destination, source, solver.num_points)
            if state.particle_qd is not None:
                wp.copy(state.particle_qd, solver.velocity_array, destination, source, solver.num_points)
