# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patient attenuation volume resolved through the coordinate-safe twin manifest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .patient_twin import PatientTwin

# Preserve navigation contrast from the original patient-twin pipeline.
DEFAULT_HU_TO_MU_PRESET = "interventional"


@dataclass(frozen=True, slots=True)
class PatientVolume:
    """Attenuation volume on the twin's voxel grid.

    The renderer's volume frame is the twin's patient frame (LPS millimetres), so every
    transform here comes from the twin. Geometry-only consumers should use the twin
    directly and skip the HU-to-attenuation conversion.
    """

    twin: PatientTwin
    mu_volume: np.ndarray

    @classmethod
    def load(cls, twin: PatientTwin, *, hu_to_mu_preset: str | None = None) -> PatientVolume:
        from xray_simulator import HuToMuMapping, PreprocessingSettings, VolumePreprocessor
        from xray_simulator.scan_volume import load_artifact

        # Keep high-HU contrast and implants, matching the previous patient pipeline.
        settings = PreprocessingSettings(
            hu_to_mu=HuToMuMapping.preset(hu_to_mu_preset or DEFAULT_HU_TO_MU_PRESET), clip_hu=False
        )
        scan = load_artifact(twin.artifacts["volume_metadata"])
        volume = VolumePreprocessor.from_scan(scan, settings=settings).preprocess()
        if not np.allclose(volume.metadata.voxel_to_lps_mm, twin.voxel_to_patient_mm, atol=1e-6):
            raise ValueError(f"{twin.source}: xray_simulator and the patient twin disagree on the scan affine")
        return cls(twin, np.asarray(volume.mu_volume, dtype=np.float32))

    @property
    def voxel_to_volume_mm(self) -> np.ndarray:
        return self.twin.voxel_to_patient_mm

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.mu_volume.shape)

    @property
    def spacing_xyz_mm(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.linalg.norm(self.voxel_to_volume_mm[:3, :3], axis=0))

    @property
    def spacing_zyx_mm(self) -> tuple[float, float, float]:
        return self.spacing_xyz_mm[::-1]

    @property
    def center_xyz_mm(self) -> np.ndarray:
        return self.twin.center_patient_mm

    def world_to_volume_mm(self, points_world_m: np.ndarray) -> np.ndarray:
        return self.twin.world_to_patient_mm(points_world_m)

    def volume_mm_to_world(self, points_volume_mm: np.ndarray) -> np.ndarray:
        return self.twin.patient_mm_to_world(points_volume_mm)
