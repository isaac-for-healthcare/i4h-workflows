# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate-safe manifest for patient-specific simulation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_MM_TO_M = np.diag((0.001, 0.001, 0.001, 1.0)).astype(np.float64)
_SUPPORTED_COORDINATE_FRAMES = frozenset(("DICOM_LPS", "NIFTI_RAS"))


def _affine(value: Any, name: str, *, rigid: bool = False) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix or a flat list of 16 values")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{name} must be an affine transform with final row [0, 0, 0, 1]")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-12:
        raise ValueError(f"{name} must have a non-singular linear transform")
    if rigid and not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must contain a rigid rotation without scale or shear")
    return matrix


@dataclass(frozen=True, slots=True)
class PatientTwin:
    """Resolved patient geometry and the transforms that align all consumers.

    ``voxel_to_patient_mm`` preserves CT spacing, origin, and direction cosines.
    ``world_from_patient_m`` places patient-space metres in the Isaac world.
    The explicit millimetre-to-metre conversion prevents a renderer, collision
    solver, and USD visualization from silently choosing different units.
    """

    patient_id: str
    coordinate_frame: str
    voxel_to_patient_mm: np.ndarray
    world_from_patient_m: np.ndarray
    artifacts: dict[str, Path]
    source: Path
    schema_version: int = 1

    @property
    def voxel_to_world_m(self) -> np.ndarray:
        return self.world_from_patient_m @ _MM_TO_M @ self.voxel_to_patient_mm

    def patient_mm_to_world(self, points_patient_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_patient_mm, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points_patient_mm must end in an xyz dimension")
        homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1), dtype=np.float64)), axis=-1)
        return (homogeneous @ (_MM_TO_M @ self.world_from_patient_m.T))[..., :3]

    def voxels_to_world(self, points_voxel: np.ndarray) -> np.ndarray:
        points = np.asarray(points_voxel, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points_voxel must end in an ijk dimension")
        homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1), dtype=np.float64)), axis=-1)
        return (homogeneous @ self.voxel_to_world_m.T)[..., :3]

    @classmethod
    def load(cls, path: str | Path, *, require_artifacts: bool = True) -> PatientTwin:
        source = Path(path).expanduser().resolve()
        try:
            raw = yaml.safe_load(source.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"{source}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise TypeError(f"{source}: expected a mapping")
        if int(raw.get("schema_version", 0)) != 1:
            raise ValueError(f"{source}: unsupported or missing schema_version")
        patient_id = str(raw.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError(f"{source}: patient_id is required")
        coordinate_frame = str(raw.get("coordinate_frame", ""))
        if coordinate_frame not in _SUPPORTED_COORDINATE_FRAMES:
            raise ValueError(
                f"{source}: coordinate_frame must be one of {sorted(_SUPPORTED_COORDINATE_FRAMES)}, "
                f"got {coordinate_frame!r}"
            )
        transforms = raw.get("transforms")
        if not isinstance(transforms, dict):
            raise TypeError(f"{source}: transforms must be a mapping")
        voxel_to_patient_mm = _affine(transforms.get("voxel_to_patient_mm"), "voxel_to_patient_mm")
        world_from_patient_m = _affine(transforms.get("world_from_patient_m"), "world_from_patient_m", rigid=True)
        artifact_values = raw.get("artifacts")
        if not isinstance(artifact_values, dict) or "attenuation_volume" not in artifact_values:
            raise ValueError(f"{source}: artifacts.attenuation_volume is required")
        artifacts: dict[str, Path] = {}
        for name, value in artifact_values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{source}: artifact {name!r} must be a non-empty path")
            artifact_path = (
                (source.parent / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
            )
            if require_artifacts and not artifact_path.exists():
                raise FileNotFoundError(f"{source}: artifact {name!r} does not exist: {artifact_path}")
            artifacts[str(name)] = artifact_path
        return cls(
            patient_id=patient_id,
            coordinate_frame=coordinate_frame,
            voxel_to_patient_mm=voxel_to_patient_mm,
            world_from_patient_m=world_from_patient_m,
            artifacts=artifacts,
            source=source,
        )
