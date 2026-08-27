# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State providers that read from Isaac Lab's physics and scene-data layers.

These exist so image formation stops reaching into a solver's private buffers.
The fluoroscopy sensor consumes :class:`CatheterStateProvider` and
:class:`CArmStateProvider`, and both are satisfied here from data Isaac Lab
already publishes: catheter geometry from Newton's particle buffers, and C-arm
poses through :class:`~isaaclab.scene_data.SceneDataProvider`, which is the
backend-agnostic read path for body transforms.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from i4h_arena.medical.carm import CArmState
from i4h_arena.medical.catheter import CatheterState


class NewtonRodCatheterStateProvider:
    """Catheter polyline read from Newton's particle buffers.

    The rod solver owns catheter dynamics, but Newton's ``particle_q`` is where
    the result is published for the rest of the scene, so reading it here keeps
    image formation independent of which solver produced the motion.

    Nodal positions are read from the Newton state directly rather than through
    ``SceneDataProvider``. The provider's deformable path
    (``SceneDataFormat.Points`` and ``get_points()``) does not exist in the
    pinned Isaac Lab and is only available on newer builds; when the pin
    advances, this class is the single place that changes.

    Args:
        state_getter: Returns the current Newton state, i.e. anything exposing
            ``particle_q``. Passed as a callable because the state object is
            swapped between substeps.
        offset: Index of the rod's first particle in Newton's particle arrays.
        num_points: Particles per environment.
        num_envs: Environments covered by this rod.
        radius_m: Catheter radius handed to the renderer.
        origin_world_m: Added to every position. Use when rod particles are
            authored in a frame offset from the Isaac world origin.
    """

    def __init__(
        self,
        state_getter: Callable[[], Any],
        *,
        offset: int,
        num_points: int,
        num_envs: int,
        radius_m: float,
        origin_world_m: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> None:
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        self._state_getter = state_getter
        self._offset = int(offset)
        self._num_points = int(num_points)
        self._num_envs = int(num_envs)
        self._radius_m = float(radius_m)
        self._origin_world_m = np.asarray(origin_world_m, dtype=np.float32).reshape(3)

    @classmethod
    def from_particle_range(
        cls,
        state_getter: Callable[[], Any],
        particle_range: Any,
        *,
        radius_m: float,
        origin_world_m: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> "NewtonRodCatheterStateProvider":
        """Build from the ``RodParticleRange`` returned by the rod builder."""
        num_envs = int(particle_range.num_envs)
        return cls(
            state_getter,
            offset=int(particle_range.offset),
            num_points=int(particle_range.count) // num_envs,
            num_envs=num_envs,
            radius_m=radius_m,
            origin_world_m=origin_world_m,
        )

    def snapshot(self, num_envs: int) -> CatheterState:
        if num_envs != self._num_envs:
            raise ValueError(f"catheter spans {self._num_envs} environment(s), requested {num_envs}")
        state = self._state_getter()
        buffer = getattr(state, "particle_q", None)
        if buffer is None:
            raise RuntimeError(
                "Newton state has no 'particle_q'. The rod's particles must be added to the "
                "ModelBuilder before finalize, on the PhysicsEvent.MODEL_INIT callback."
            )
        count = self._num_points * self._num_envs
        end = self._offset + count
        positions = np.asarray(buffer.numpy() if hasattr(buffer, "numpy") else buffer, dtype=np.float32)
        if len(positions) < end:
            raise RuntimeError(
                f"rod particle range [{self._offset}, {end}) does not fit in 'particle_q' of "
                f"length {len(positions)}"
            )
        polylines = positions[self._offset : end].reshape(self._num_envs, self._num_points, 3)
        return CatheterState(
            positions_world_m=polylines + self._origin_world_m,
            valid_nodes=np.full(self._num_envs, self._num_points, dtype=np.int32),
            radius_m=self._radius_m,
        )


class SceneDataCArmStateProvider:
    """C-arm source and detector poses read through ``SceneDataProvider``.

    Replaces per-asset ``get_world_poses()`` calls. The provider exposes one
    Warp-native read path for body transforms whichever physics backend is
    active, so image formation no longer depends on the asset view a particular
    backend happens to offer.

    Args:
        provider: Result of ``SimulationContext.instance().get_scene_data_provider()``.
        source_paths: One USD prim path per environment for the X-ray source.
        detector_paths: One USD prim path per environment for the detector.
        detector_size_m: Physical detector width and height.
    """

    def __init__(
        self,
        provider: Any,
        *,
        source_paths: list[str],
        detector_paths: list[str],
        detector_size_m: tuple[float, float],
    ) -> None:
        if len(source_paths) != len(detector_paths):
            raise ValueError(
                f"got {len(source_paths)} source path(s) and {len(detector_paths)} detector "
                "path(s); one of each per environment is required"
            )
        if not source_paths:
            raise ValueError("at least one environment's prim paths are required")
        self._provider = provider
        self._num_envs = len(source_paths)
        self._detector_size_m = detector_size_m
        # One mapping for both bodies keeps this to a single provider read per
        # frame; sources occupy the first half of the output, detectors the second.
        self._mapping = provider.create_mapping(list(source_paths) + list(detector_paths))
        self._output = None

    def snapshot(self, num_envs: int) -> CArmState:
        if num_envs != self._num_envs:
            raise ValueError(f"C-arm spans {self._num_envs} environment(s), requested {num_envs}")
        from isaaclab.scene_data import SceneDataFormat

        if self._output is None:
            self._output = SceneDataFormat.Transform()
        # Passthrough is refused because a mapping has to be applied, and a
        # zero-copy view of the backend array would ignore it.
        if not self._provider.get_transforms(self._output, self._mapping, allow_passthrough=False):
            raise RuntimeError("SceneDataProvider could not convert body transforms to SceneDataFormat.Transform")
        transforms = self._output.transforms.numpy()
        if len(transforms) < 2 * self._num_envs:
            raise RuntimeError(f"expected {2 * self._num_envs} C-arm transforms, got {len(transforms)}")
        source = np.asarray(transforms[: self._num_envs, :3], dtype=np.float64)
        detector = np.asarray(transforms[self._num_envs :, :3], dtype=np.float64)
        # warp transforms carry XYZW quaternions, matching the detector frame
        # convention the renderer expects.
        detector_quat = np.asarray(transforms[self._num_envs :, 3:7], dtype=np.float64)
        x_axis = _rotate_xyzw(detector_quat, np.array([1.0, 0.0, 0.0]))
        return CArmState(source, detector, x_axis, self._detector_size_m)


def _rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate ``vector`` by a batch of XYZW quaternions."""
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + w * vector)


__all__ = ["NewtonRodCatheterStateProvider", "SceneDataCArmStateProvider"]
