# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import yaml

from i4h_arena.embodiments.catheter import route_initial_catheter_length_m
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


def test_route_initial_catheter_length_leaves_the_allowance_to_insert() -> None:
    assert np.isclose(route_initial_catheter_length_m(0.6463, allowance_m=0.12), 0.5263)


def test_route_initial_catheter_length_fits_the_scenes_step_budget() -> None:
    """The default has to leave less route than one episode can insert.

    600 steps at 30 Hz is twenty seconds, and the insertion slider defaults to
    9 mm/s, so 180 mm is the ceiling. This is the property the old CT-width
    rule violated on ``s0011``, where it asked for 343 mm.
    """
    insertable_m = (600 / 30.0) * 0.009

    remaining_m = 0.6463 - route_initial_catheter_length_m(0.6463)

    assert remaining_m < insertable_m


def test_route_initial_catheter_length_keeps_a_short_route_seeded() -> None:
    """A route shorter than the allowance still seeds a shaft to insert along."""
    length_m = route_initial_catheter_length_m(0.05, allowance_m=0.12)

    assert length_m > 0.0
    assert np.isclose(length_m, 0.005)


@pytest.mark.parametrize("route_length_m", [0.0, -0.1, float("nan")])
def test_route_initial_catheter_length_rejects_an_unusable_route(route_length_m) -> None:
    with pytest.raises(ValueError, match="route_length_m"):
        route_initial_catheter_length_m(route_length_m)


def test_route_initial_catheter_length_rejects_an_unusable_allowance() -> None:
    with pytest.raises(ValueError, match="allowance_m"):
        route_initial_catheter_length_m(0.6463, allowance_m=0.0)
