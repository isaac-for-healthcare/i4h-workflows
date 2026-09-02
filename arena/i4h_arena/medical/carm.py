# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""C-arm state boundary between Isaac scene assets and image formation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .patient_volume import PatientVolume


def _numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return np.asarray(value)


def _quat_xyzw_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate one vector per environment by an Isaac Lab XYZW quaternion."""
    xyz = quaternion[:, :3]
    w = quaternion[:, 3:4]
    vec = np.broadcast_to(np.asarray(vector, dtype=np.float64), xyz.shape)
    return vec + 2.0 * np.cross(xyz, np.cross(xyz, vec) + w * vec)


@dataclass(frozen=True, slots=True)
class CArmState:
    """World-space source and detector geometry for every environment."""

    source_world_m: np.ndarray
    detector_center_world_m: np.ndarray
    detector_x_axis_world: np.ndarray
    detector_size_m: tuple[float, float]

    def __post_init__(self) -> None:
        source = np.asarray(self.source_world_m, dtype=np.float64)
        detector = np.asarray(self.detector_center_world_m, dtype=np.float64)
        x_axis = np.asarray(self.detector_x_axis_world, dtype=np.float64)
        if source.ndim != 2 or source.shape[-1] != 3:
            raise ValueError("source_world_m must have shape (num_envs, 3)")
        if detector.shape != source.shape or x_axis.shape != source.shape:
            raise ValueError("detector position and x axis must match source shape")
        if not np.isfinite(source).all() or not np.isfinite(detector).all() or not np.isfinite(x_axis).all():
            raise ValueError("C-arm state must contain only finite values")
        norms = np.linalg.norm(x_axis, axis=-1)
        if np.any(norms < 1e-9):
            raise ValueError("detector x axis must be non-zero")
        if len(self.detector_size_m) != 2 or min(self.detector_size_m) <= 0.0:
            raise ValueError("detector_size_m must contain positive width and height")
        object.__setattr__(self, "source_world_m", source)
        object.__setattr__(self, "detector_center_world_m", detector)
        object.__setattr__(self, "detector_x_axis_world", x_axis / norms[:, None])

    @property
    def num_envs(self) -> int:
        return int(self.source_world_m.shape[0])


@runtime_checkable
class CArmStateProvider(Protocol):
    def snapshot(self, num_envs: int) -> CArmState:
        """Return current C-arm geometry without advancing the simulator."""


class SceneCArmStateProvider:
    """Read C-arm poses from Isaac Lab scene assets."""

    def __init__(self, source_asset: Any, detector_asset: Any, *, detector_size_m: tuple[float, float]) -> None:
        self._source_asset = source_asset
        self._detector_asset = detector_asset
        self._detector_size_m = detector_size_m

    def snapshot(self, num_envs: int) -> CArmState:
        source_pos, _source_quat = self._source_asset.get_world_poses()
        detector_pos, detector_quat = self._detector_asset.get_world_poses()
        source = _numpy(source_pos)[..., :3]
        detector = _numpy(detector_pos)[..., :3]
        quaternion = _numpy(detector_quat)
        if source.shape[0] != num_envs or detector.shape[0] != num_envs:
            raise ValueError(f"C-arm provider returned {source.shape[0]} environments; expected {num_envs}")
        x_axis = _quat_xyzw_rotate(quaternion, np.array([1.0, 0.0, 0.0]))
        return CArmState(source, detector, x_axis, self._detector_size_m)


class ReferenceProjectionCArmStateProvider:
    """Calibrate a visible C-arm angle to the reference xray_simulator projections.

    The visible assembly keeps its intuitive patient-surrounding motion. The
    shared orbit angle independently defines the renderer's AP/LAO/lateral/RAO
    coordinate frame, avoiding a visual equipment pose dictated by CT storage
    axes.
    """

    def __init__(
        self,
        patient: PatientVolume,
        orbit_action: Any,
        *,
        detector_size_m: tuple[float, float],
        source_to_detector_m: float = 1.020,
    ) -> None:
        self._patient = patient
        self._orbit_action = orbit_action
        self._detector_size_m = detector_size_m
        self._half_sdd_m = 0.5 * float(source_to_detector_m)

    def snapshot(self, num_envs: int) -> CArmState:
        angles = _numpy(self._orbit_action.angle_rad).reshape(-1)
        if angles.shape != (num_envs,):
            raise ValueError(f"C-arm orbit action returned {angles.shape[0]} environments; expected {num_envs}")
        volume_to_world = np.array(self._patient.volume_xyz_mm_to_world_m[:3, :3], copy=True)
        volume_to_world /= np.linalg.norm(volume_to_world, axis=0, keepdims=True)
        isocenter = self._patient.volume_mm_to_world(self._patient.center_xyz_mm)
        source = np.zeros((num_envs, 3), dtype=np.float64)
        detector = np.zeros_like(source)
        detector_x = np.zeros_like(source)
        for index, angle in enumerate(angles):
            cosine = np.cos(float(angle))
            sine = np.sin(float(angle))
            renderer_orbit = np.array(
                [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
                dtype=np.float64,
            )
            local_to_world = volume_to_world @ renderer_orbit
            beam_axis = local_to_world[:, 2]
            source[index] = isocenter - self._half_sdd_m * beam_axis
            detector[index] = isocenter + self._half_sdd_m * beam_axis
            detector_x[index] = local_to_world[:, 0]
        return CArmState(source, detector, detector_x, self._detector_size_m)

    def select_angle(self, angle_rad: float) -> float:
        """Select one reference projection on the shared visible C-arm action."""
        setter = getattr(self._orbit_action, "set_orbit_angle", None)
        if not callable(setter):
            raise TypeError("the C-arm orbit action does not support named projections")
        return float(setter(angle_rad))
