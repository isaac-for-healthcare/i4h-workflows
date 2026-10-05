# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate-safe manifest for patient-specific simulation artifacts.

``patient-digital-twin`` bundles keep the scan exactly as acquired: native array order,
any orientation, the scanner's frame (RAS or LPS) and units. :meth:`PatientTwin.load`
resolves all of that once, so every consumer works in a single patient frame of LPS
millimetres and never sees source frames, units, or array orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_MM_TO_M = np.diag((0.001, 0.001, 0.001, 1.0)).astype(np.float64)
_RAS_TO_LPS = np.diag((-1.0, -1.0, 1.0, 1.0)).astype(np.float64)
_METERS_PER_UNIT = {"m": 1.0, "mm": 0.001, "micron": 1e-6}
_SCHEMA_VERSION = 2

# Patient LPS to Isaac world: +X toward the head, +Y toward the patient's right, +Z
# anterior, which lays a supine patient along the table facing up.
_WORLD_FROM_LPS_ROTATION = np.asarray(((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)), dtype=np.float64)

# A working-height isocenter keeps the patient, catheter, table, and C-arm aligned in a
# recognizable clinical layout.
_ISOCENTER_WORLD_M = np.asarray((0.0, 0.0, 0.85), dtype=np.float64)


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


def _homogeneous(points: np.ndarray, matrix: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError(f"{name} must end in an xyz dimension")
    homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1), dtype=np.float64)), axis=-1)
    return (homogeneous @ matrix.T)[..., :3]


def default_world_from_patient_m(center_patient_mm: np.ndarray) -> np.ndarray:
    """Place a supine patient on the table with the scan centre at the isocenter."""
    world_from_patient = np.eye(4, dtype=np.float64)
    world_from_patient[:3, :3] = _WORLD_FROM_LPS_ROTATION
    world_from_patient[:3, 3] = _ISOCENTER_WORLD_M - _WORLD_FROM_LPS_ROTATION @ (np.asarray(center_patient_mm) * 0.001)
    return world_from_patient


@dataclass(frozen=True, slots=True)
class _ScanHeader:
    """Geometry from a ``volume.yaml`` sidecar, without loading the HU array."""

    voxel_to_scan: np.ndarray
    shape_ijk: tuple[int, int, int]
    array_axes: str
    frame: str
    unit: str

    @classmethod
    def read(cls, path: Path) -> _ScanHeader:
        output = yaml.safe_load(path.read_text(encoding="utf-8"))["output"]
        axes = str(output["array_axes"])
        if sorted(axes) != list("ijk"):
            raise ValueError(f"{path}: array_axes must be a permutation of 'ijk', got {axes!r}")
        if output["world_frame"] not in ("RAS", "LPS") or output["world_unit"] not in _METERS_PER_UNIT:
            raise ValueError(f"{path}: unsupported world frame or unit")
        # Columns select native i, j, k from the recorded array-axis affine.
        permutation = np.eye(4)
        permutation[:3, :3] = np.eye(3)[:, [axes.index(axis) for axis in "ijk"]]
        array_to_scan = _affine(output["array_index_to_world"], f"{path}: array_index_to_world")
        return cls(
            voxel_to_scan=array_to_scan @ permutation,
            shape_ijk=tuple(int(output["shape"][axes.index(axis)]) for axis in "ijk"),
            array_axes=axes,
            frame=str(output["world_frame"]),
            unit=str(output["world_unit"]),
        )

    @property
    def patient_mm_from_scan(self) -> np.ndarray:
        """Scan coordinates in scan units to LPS millimetres."""
        scale = np.diag((*(3 * (_METERS_PER_UNIT[self.unit] * 1000.0,)), 1.0))
        return (_RAS_TO_LPS if self.frame == "RAS" else np.eye(4)) @ scale


@dataclass(frozen=True, slots=True)
class Centerline:
    """Vessel centerline graph in patient LPS millimetres."""

    points_mm: np.ndarray
    edges: np.ndarray
    radii_mm: np.ndarray | None


@dataclass(frozen=True, slots=True)
class PatientTwin:
    """Resolved patient geometry and the transforms that align all consumers.

    ``voxel_to_patient_mm`` maps voxel-centre ``(i, j, k)`` indices to patient LPS
    millimetres, preserving the scan's spacing, origin, and direction cosines.
    ``world_from_patient_m`` places patient-space metres in the Isaac world.
    ``patient_mm_from_scan`` and ``array_axes`` describe how the bundle's own files are
    stored; use the accessor methods rather than applying them by hand.
    """

    patient_id: str
    voxel_to_patient_mm: np.ndarray
    world_from_patient_m: np.ndarray
    shape_ijk: tuple[int, int, int]
    artifacts: dict[str, Path]
    source: Path
    patient_mm_from_scan: np.ndarray
    array_axes: str

    @property
    def voxel_to_world_m(self) -> np.ndarray:
        return self.world_from_patient_m @ _MM_TO_M @ self.voxel_to_patient_mm

    @property
    def center_patient_mm(self) -> np.ndarray:
        """Centre of the voxel grid."""
        return _homogeneous((np.asarray(self.shape_ijk) - 1) / 2, self.voxel_to_patient_mm, "center")

    @property
    def isocenter_world_m(self) -> np.ndarray:
        return self.patient_mm_to_world(self.center_patient_mm)

    @property
    def width_mm(self) -> float:
        """Left-right extent of the scanned volume, whatever its array order."""
        return float(np.abs(self.voxel_to_patient_mm[0, :3]) @ np.asarray(self.shape_ijk))

    def patient_mm_to_world(self, points_patient_mm: np.ndarray) -> np.ndarray:
        return _homogeneous(points_patient_mm, self.world_from_patient_m @ _MM_TO_M, "points_patient_mm")

    def world_to_patient_mm(self, points_world_m: np.ndarray) -> np.ndarray:
        return _homogeneous(points_world_m, np.linalg.inv(self.world_from_patient_m @ _MM_TO_M), "points_world_m")

    def voxels_to_world(self, points_voxel: np.ndarray) -> np.ndarray:
        return _homogeneous(points_voxel, self.voxel_to_world_m, "points_voxel")

    def centerline(self) -> Centerline | None:
        """Navigation centerline in patient LPS millimetres, if the bundle has one."""
        points_path = self.artifacts.get("centerline_points")
        edges_path = self.artifacts.get("centerline_edges")
        if points_path is None or edges_path is None:
            return None
        radii_path = self.artifacts.get("centerline_radii")
        scale = float(np.linalg.norm(self.patient_mm_from_scan[:3, 0]))
        return Centerline(
            points_mm=_homogeneous(np.load(points_path), self.patient_mm_from_scan, "centerline_points"),
            edges=np.load(edges_path),
            radii_mm=None if radii_path is None else np.load(radii_path) * scale,
        )

    def vessel_mask_kji(self) -> np.ndarray | None:
        """Vessel mask in the attenuation volume's ``(k, j, i)`` array order."""
        path = self.artifacts.get("vessel_mask")
        if path is None:
            return None
        mask = np.asarray(np.load(path), dtype=np.uint8)
        return mask.transpose([self.array_axes.index(axis) for axis in "kji"])

    def anatomy_world_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        """World translation, rotation, and uniform scale for the anatomy USD.

        The USD is authored in scan coordinates and units, so the RAS flip (a proper
        rotation) folds into the rotation and the unit conversion into the scale.
        """
        world_from_scan = self.world_from_patient_m @ _MM_TO_M @ self.patient_mm_from_scan
        scale = float(np.linalg.norm(world_from_scan[:3, 0]))
        return world_from_scan[:3, 3].copy(), world_from_scan[:3, :3] / scale, scale

    @classmethod
    def load(cls, path: str | Path) -> PatientTwin:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"patient twin manifest does not exist: {source}")
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"{source}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise TypeError(f"{source}: expected a mapping")
        if raw.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                f"{source}: unsupported schema_version {raw.get('schema_version')!r}; expected {_SCHEMA_VERSION}. "
                "Rebuild the bundle with ./tools/patient_twin/run.sh"
            )
        patient_id = str(raw.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError(f"{source}: patient_id is required")
        transforms = raw.get("transforms") or {}
        if not isinstance(transforms, dict):
            raise TypeError(f"{source}: transforms must be a mapping")
        artifact_values = raw.get("artifacts")
        if not isinstance(artifact_values, dict) or not {"hu_volume", "volume_metadata"} <= artifact_values.keys():
            raise ValueError(f"{source}: artifacts.hu_volume and artifacts.volume_metadata are required")
        artifacts: dict[str, Path] = {}
        for name, value in artifact_values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{source}: artifact {name!r} must be a non-empty path")
            artifact_path = (source.parent / value).resolve()
            if not artifact_path.exists():
                raise FileNotFoundError(f"{source}: artifact {name!r} does not exist: {artifact_path}")
            artifacts[str(name)] = artifact_path

        # volume.yaml is the source of truth for scan geometry; the manifest repeats it.
        header = _ScanHeader.read(artifacts["volume_metadata"])
        if (raw.get("coordinate_frame"), raw.get("spatial_unit")) != (header.frame, header.unit) or (
            "voxel_to_scan" in transforms
            and not np.allclose(_affine(transforms["voxel_to_scan"], "voxel_to_scan"), header.voxel_to_scan)
        ):
            raise ValueError(
                f"{source}: manifest and {artifacts['volume_metadata'].name} disagree on the scan geometry"
            )
        voxel_to_patient_mm = header.patient_mm_from_scan @ header.voxel_to_scan
        if "world_from_patient_m" in transforms:
            # The exporter's placement hint is expressed in the scan frame.
            world_from_scan_m = _affine(transforms["world_from_patient_m"], "world_from_patient_m", rigid=True)
            world_from_patient_m = world_from_scan_m @ (_RAS_TO_LPS if header.frame == "RAS" else np.eye(4))
        else:
            center = _homogeneous((np.asarray(header.shape_ijk) - 1) / 2, voxel_to_patient_mm, "center")
            world_from_patient_m = default_world_from_patient_m(center)
        return cls(
            patient_id=patient_id,
            voxel_to_patient_mm=voxel_to_patient_mm,
            world_from_patient_m=world_from_patient_m,
            shape_ijk=header.shape_ijk,
            artifacts=artifacts,
            source=source,
            patient_mm_from_scan=header.patient_mm_from_scan,
            array_axes=header.array_axes,
        )
