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


def solver_workspaces(solver: Any) -> tuple[Any, ...]:
    """The rod solver's live constraint buffers, batched one first.

    ``XPBDRodSolver`` keeps its state on two workspaces and exposes neither.
    ``_ws`` always exists and is sized for a single rod; ``_bws`` is built only
    when the solver was asked for more than one environment, holds
    ``num_envs * num_edges_per_rod``, and is seeded by tiling the single-rod
    values. They are separate allocations from then on, and the batched solve
    reads only the batched ones, so anything that writes material properties
    has to write both or it writes nothing that matters.

    Collected here because getting that wrong is quiet. ``bend_stiffness`` was
    tapered on ``_ws`` alone, which left the taper inert in every multi-rod run
    and, because the per-environment edge count was recovered by dividing a
    single-rod buffer by the environment count, stamped the profile once per
    environment down one rod. A rod of 120 edges divides evenly by the 8 and 4
    environments the profiles ask for, so no size check caught it.

    One reach rather than four, and the one place to change when physics-sim
    grows a public accessor. It already has the shape for one: ``velocities``
    is ``self._bws.velocities if self._bws is not None else self._ws.velocities``.
    The three buffers this layer writes -- ``bend_stiffness``,
    ``inv_inertia_local_diag`` and ``rest_darboux`` -- are not among the ones
    exposed that way.

    Batched first so a caller that only needs one buffer reads the one the
    solve will actually use.
    """
    workspaces = (getattr(solver, "_bws", None), getattr(solver, "_ws", None))
    live = tuple(workspace for workspace in workspaces if workspace is not None)
    if not live:
        raise RuntimeError(
            "rod solver exposes no workspace; `_ws`/`_bws` are private and may have been renamed upstream"
        )
    return live


def workspace_edges_per_env(workspace: Any) -> int:
    """Edges of one rod in ``workspace``, whichever name it files them under.

    The batched workspace counts ``num_edges_per_rod`` and the single-rod one
    ``num_edges``. Same quantity, and asking for the wrong one returns zero
    rather than failing, so both names are tried here instead of at each site.
    """
    for name in ("num_edges_per_rod", "num_edges"):
        edges = int(getattr(workspace, name, 0) or 0)
        if edges > 0:
            return edges
    return 0


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
    for workspace in solver_workspaces(solver):
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
