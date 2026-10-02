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
    meters_per_scan_unit: float = 0.001
    array_axes: str = "kji"

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
        if int(raw.get("schema_version", 0)) not in (1, 3):
            raise ValueError(f"{source}: unsupported or missing schema_version")
        patient_id = str(raw.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError(f"{source}: patient_id is required")
        version = int(raw["schema_version"])
        coordinate_frame = str(raw.get("coordinate_frame", ""))
        if version == 3:
            coordinate_frame = {"RAS": "NIFTI_RAS", "LPS": "DICOM_LPS"}.get(coordinate_frame, coordinate_frame)
        if coordinate_frame not in _SUPPORTED_COORDINATE_FRAMES:
            raise ValueError(
                f"{source}: coordinate_frame must be one of {sorted(_SUPPORTED_COORDINATE_FRAMES)}, "
                f"got {coordinate_frame!r}"
            )
        transforms = raw.get("transforms")
        if not isinstance(transforms, dict):
            raise TypeError(f"{source}: transforms must be a mapping")
        units = 0.001
        axes = "kji"
        if version == 3:
            units = float(raw["meters_per_unit"])
            if not np.isfinite(units) or units <= 0:
                raise ValueError("meters_per_unit must be positive and finite")
            metadata_path = source.parent / raw["artifacts"]["volume_metadata"]
            volume = yaml.safe_load(metadata_path.read_text())["output"]
            expected_frame = {"RAS": "NIFTI_RAS", "LPS": "DICOM_LPS"}.get(volume["world_frame"])
            expected_units = {"m": 1.0, "mm": 0.001, "micron": 1e-6}.get(volume["world_unit"])
            if (
                expected_frame != coordinate_frame
                or expected_units != units
                or volume["world_unit"] != raw["spatial_unit"]
            ):
                raise ValueError("Patient manifest and volume YAML disagree on frame or units")
            axes = volume["array_axes"]
            if sorted(axes) != list("ijk"):
                raise ValueError("Invalid native volume array axes")
            voxel_to_patient_mm = _affine(transforms["voxel_to_scan"], "voxel_to_scan").copy()
            voxel_to_patient_mm[:3] *= units * 1000
            if "world_from_patient_m" in transforms:
                world_from_patient_m = _affine(transforms["world_from_patient_m"], "world_from_patient_m", rigid=True)
            else:
                # Simulator placement belongs here, never in the patient exporter.
                world_from_patient_m = np.eye(4)
                lps_from_scan = np.diag([-1.0, -1.0, 1.0]) if coordinate_frame == "NIFTI_RAS" else np.eye(3)
                world_from_patient_m[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) @ lps_from_scan
                size = np.array([volume["shape"][axes.index(c)] for c in "ijk"])
                center = (voxel_to_patient_mm @ np.r_[(size - 1) / 2, 1])[:3] * 0.001
                world_from_patient_m[:3, 3] = [0, 0, 0.85] - world_from_patient_m[:3, :3] @ center
        else:
            voxel_to_patient_mm = _affine(transforms.get("voxel_to_patient_mm"), "voxel_to_patient_mm")
            world_from_patient_m = _affine(transforms.get("world_from_patient_m"), "world_from_patient_m", rigid=True)
        artifact_values = raw.get("artifacts")
        required = "hu_volume" if version == 3 else "attenuation_volume"
        if not isinstance(artifact_values, dict) or required not in artifact_values:
            raise ValueError(f"{source}: schema {version} requires artifacts.{required}")
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
            schema_version=version,
            meters_per_scan_unit=units,
            array_axes=axes,
        )
