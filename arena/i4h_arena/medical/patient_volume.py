# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patient attenuation volume resolved through the coordinate-safe twin manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .patient_twin import PatientTwin


@dataclass(frozen=True, slots=True)
class PatientVolume:
    """Attenuation volume plus transforms between renderer millimetres and Isaac metres."""

    twin: PatientTwin
    mu_volume: np.ndarray
    spacing_zyx_mm: tuple[float, float, float]
    volume_xyz_mm_to_world_m: np.ndarray
    world_m_to_volume_xyz_mm: np.ndarray
    voxel_to_volume_mm: np.ndarray | None = None

    @classmethod
    def load(cls, twin: PatientTwin, *, hu_to_mu_preset: str | None = None) -> PatientVolume:
        if twin.schema_version == 3:
            from xray_simulator import HuToMuMapping, PreprocessingSettings, VolumePreprocessor
            from xray_simulator.scan_volume import load_artifact

            scan = load_artifact(twin.artifacts["volume_metadata"])
            native = scan.ijk_to_world.copy()
            native[:3] *= scan.meters_per_unit * 1000
            if not np.allclose(native, twin.voxel_to_patient_mm, atol=1e-6):
                raise ValueError("Patient manifest and volume YAML disagree on the scan affine")
            settings = PreprocessingSettings(hu_to_mu=HuToMuMapping.preset(hu_to_mu_preset or "linear"))
            volume = VolumePreprocessor.from_scan(scan, settings=settings).preprocess()
            to_world = twin.world_from_patient_m @ np.diag([0.001, 0.001, 0.001, 1.0])
            return cls(twin, volume.mu_volume, volume.spacing_zyx_mm, to_world, np.linalg.inv(to_world), native)
        # Preserve legacy cached attenuation unless the caller explicitly remaps HU.
        use_hu = twin.schema_version == 2 or "attenuation_volume" not in twin.artifacts or hu_to_mu_preset is not None
        if use_hu and "hu_volume" not in twin.artifacts:
            raise ValueError("Changing attenuation requires artifacts.hu_volume; this bundle only contains mu")
        volume_path = twin.artifacts["hu_volume" if use_hu else "attenuation_volume"]
        metadata_path = twin.artifacts.get("volume_metadata", volume_path.with_name("metadata.json"))
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"{twin.source}: volume metadata does not exist: {metadata_path}; "
                "declare artifacts.volume_metadata or place metadata.json beside the volume"
            )
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        spacing = tuple(float(value) for value in raw["spacing_zyx_mm"])
        if len(spacing) != 3 or min(spacing) <= 0.0 or not np.isfinite(spacing).all():
            raise ValueError(f"{metadata_path}: spacing_zyx_mm must contain three positive finite values")
        volume = np.load(volume_path, mmap_mode="r")
        if volume.ndim != 3:
            raise ValueError(f"{volume_path}: volume must be ZYX rank 3, got {volume.shape}")
        declared_shape = tuple(int(value) for value in raw.get("shape_zyx", volume.shape))
        if declared_shape != volume.shape:
            raise ValueError(f"{metadata_path}: declared shape {declared_shape} does not match {volume.shape}")

        if not np.isfinite(volume).all():
            raise ValueError(f"{volume_path}: volume contains non-finite values")
        if use_hu:
            if (
                raw.get("intensity_units", "HU" if twin.schema_version == 1 else None) != "HU"
                or raw.get("array_order", "ZYX" if twin.schema_version == 1 else None) != "ZYX"
            ):
                raise ValueError(f"{metadata_path}: HU volume must use HU intensities and ZYX array order")
            from xray_simulator import DEFAULT_HU_TO_MU_PRESET, HuToMuMapping, PreprocessingSettings, VolumePreprocessor

            mapping = HuToMuMapping.preset(hu_to_mu_preset or DEFAULT_HU_TO_MU_PRESET)
            volume = (
                VolumePreprocessor(
                    hu_volume=volume,
                    spacing_zyx_mm=spacing,
                    origin_xyz_mm=tuple(raw["origin_xyz_mm"]) if raw.get("origin_xyz_mm") else None,
                    source=str(volume_path),
                    anatomical_frame=raw.get("anatomical_frame"),
                    settings=PreprocessingSettings(hu_to_mu=mapping),
                )
                .preprocess()
                .mu_volume
            )

        spacing_xyz = np.asarray(spacing[::-1], dtype=np.float64)
        voxel_from_volume_mm = np.diag((*np.reciprocal(spacing_xyz), 1.0))
        volume_to_world = twin.voxel_to_world_m @ voxel_from_volume_mm
        return cls(
            twin=twin,
            mu_volume=np.asarray(volume, dtype=np.float32),
            spacing_zyx_mm=spacing,
            volume_xyz_mm_to_world_m=volume_to_world,
            world_m_to_volume_xyz_mm=np.linalg.inv(volume_to_world),
        )

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.mu_volume.shape)

    @property
    def spacing_xyz_mm(self) -> tuple[float, float, float]:
        return self.spacing_zyx_mm[::-1]

    @property
    def center_xyz_mm(self) -> np.ndarray:
        if self.voxel_to_volume_mm is not None:
            return (self.voxel_to_volume_mm @ np.r_[(np.asarray(self.shape_zyx[::-1]) - 1) / 2, 1])[:3]
        return 0.5 * np.asarray(self.shape_zyx[::-1]) * np.asarray(self.spacing_xyz_mm)

    def world_to_volume_mm(self, points_world_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world_m, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("points_world_m must end in an xyz dimension")
        homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
        return (homogeneous @ self.world_m_to_volume_xyz_mm.T)[..., :3]

    def volume_mm_to_world(self, points_volume_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_volume_mm, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("points_volume_mm must end in an xyz dimension")
        homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
        return (homogeneous @ self.volume_xyz_mm_to_world_m.T)[..., :3]
