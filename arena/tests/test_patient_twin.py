# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import yaml

from i4h_arena.embodiments.catheter import reference_initial_catheter_length_m
from i4h_arena.medical.patient_twin import PatientTwin


def _write_manifest(tmp_path, *, voxel_to_patient_mm=None):
    attenuation = tmp_path / "mu_volume.npy"
    np.save(attenuation, np.zeros((2, 2, 2), dtype=np.float32))
    manifest = tmp_path / "patient_twin.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "patient_id": "synthetic_patient",
                "coordinate_frame": "DICOM_LPS",
                "transforms": {
                    "voxel_to_patient_mm": voxel_to_patient_mm
                    or [
                        [0.0, -2.0, 0.0, 10.0],
                        [1.0, 0.0, 0.0, 20.0],
                        [0.0, 0.0, 3.0, 30.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "world_from_patient_m": [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                "artifacts": {"attenuation_volume": attenuation.name},
            },
            sort_keys=False,
        )
    )
    return manifest


def test_patient_twin_preserves_direction_and_units(tmp_path) -> None:
    twin = PatientTwin.load(_write_manifest(tmp_path))

    np.testing.assert_allclose(twin.voxels_to_world([[0.0, 0.0, 0.0]]), [[1.01, 2.02, 3.03]])
    np.testing.assert_allclose(twin.voxels_to_world([[1.0, 1.0, 1.0]]), [[1.008, 2.021, 3.033]])


def test_patient_twin_rejects_singular_direction(tmp_path) -> None:
    singular = np.eye(4)
    singular[2, 2] = 0.0

    with pytest.raises(ValueError, match="non-singular"):
        PatientTwin.load(_write_manifest(tmp_path, voxel_to_patient_mm=singular.tolist()))


def test_reference_initial_catheter_length_uses_ct_x_extent(tmp_path) -> None:
    manifest = _write_manifest(tmp_path)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        '{"shape_zyx": [431, 311, 311], "spacing_zyx_mm": [1.5, 1.5, 1.5]}',
        encoding="utf-8",
    )
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw["artifacts"]["volume_metadata"] = metadata.name
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    twin = PatientTwin.load(manifest)

    assert np.isclose(reference_initial_catheter_length_m(twin, fallback_m=0.46), 0.303225)
