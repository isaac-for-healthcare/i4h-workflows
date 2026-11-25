# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patient attenuation volume resolved through the coordinate-safe twin manifest."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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
    #: The HU-to-attenuation preset ``mu_volume`` was built with.
    hu_to_mu_preset: str = DEFAULT_HU_TO_MU_PRESET

    @classmethod
    def load(cls, twin: PatientTwin, *, hu_to_mu_preset: str | None = None) -> PatientVolume:
        """Attenuation for ``twin``; sensors and environments sharing a bundle share one array.

        The returned ``mu_volume`` is read-only and cached per bundle file and preset, so N
        environments on one patient cost one volume. Copy it before changing it.
        """
        metadata = Path(twin.artifacts["volume_metadata"])
        hu_path = Path(twin.artifacts["hu_volume"])
        preset = hu_to_mu_preset or DEFAULT_HU_TO_MU_PRESET
        mu_volume, voxel_to_lps_mm = _attenuation(
            str(metadata),
            _file_state(metadata) + _file_state(hu_path),
            preset,
        )
        if not np.allclose(voxel_to_lps_mm, twin.voxel_to_patient_mm, atol=1e-6):
            raise ValueError(f"{twin.source}: xray_simulator and the patient twin disagree on the scan affine")
        return cls(twin, mu_volume, preset)

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


def _file_state(path: Path) -> tuple[int, int]:
    """Modification time and size, so a rewritten bundle is not served from the cache."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _attenuation(metadata: str, file_state: tuple[int, ...], preset: str) -> tuple[np.ndarray, np.ndarray]:
    """HU-to-attenuation for one bundle and preset; the HU array is released on return."""
    del file_state  # Part of the cache key only.
    from xray_simulator import HuToMuMapping, PreprocessingSettings, VolumePreprocessor
    from xray_simulator.scan_volume import load_artifact

    # Keep high-HU contrast and implants, matching the previous patient pipeline.
    settings = PreprocessingSettings(hu_to_mu=HuToMuMapping.preset(preset), clip_hu=False)
    volume = VolumePreprocessor.from_scan(load_artifact(metadata), settings=settings).preprocess()
    mu_volume = np.array(volume.mu_volume, dtype=np.float32)
    mu_volume.setflags(write=False)
    voxel_to_lps_mm = np.array(volume.metadata.voxel_to_lps_mm, dtype=np.float64)
    voxel_to_lps_mm.setflags(write=False)
    return mu_volume, voxel_to_lps_mm
