# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native HU bundles through sensor-simulation attenuation, without Isaac Sim."""

import numpy as np
import pytest
from conftest import to_scan

from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_volume import PatientVolume

HU = np.array([-1500, -300, 100, 300, 900, 3000, 6000, 9000], dtype=np.float32)
LINEAR = [0, 0.0035, 0.0055, 0.0065, 0.0095, 0.02, 0.02, 0.02]
INTERVENTIONAL = [0, 0, 0.0008, 0.0028, 0.009, 0.02, 0.0344, 0.044]
VOXEL_TO_LPS_MM = np.array([[0.8, -0.6, 0.0, 10.0], [0.6, 0.8, 0.0, 20.0], [0.0, 0.0, 2.0, 30.0], [0, 0, 0, 1.0]])


def _volume(write_patient_bundle, *, preset=None, frame="LPS", unit="mm", array_axes="ijk", name="bundle"):
    manifest = write_patient_bundle(
        HU.reshape(2, 2, 2),
        to_scan(VOXEL_TO_LPS_MM, frame=frame, unit=unit),
        frame=frame,
        unit=unit,
        array_axes=array_axes,
        name=name,
    )
    return PatientVolume.load(PatientTwin.load(manifest), hu_to_mu_preset=preset)


@pytest.mark.parametrize(
    "preset, expected", [(None, INTERVENTIONAL), ("interventional", INTERVENTIONAL), ("linear", LINEAR)]
)
def test_hu_maps_through_the_selected_preset(write_patient_bundle, preset, expected) -> None:
    volume = _volume(write_patient_bundle, preset=preset)

    # The attenuation volume is (k, j, i); the HU fixture was written in (i, j, k).
    np.testing.assert_allclose(volume.mu_volume, np.reshape(expected, (2, 2, 2)).transpose(2, 1, 0), atol=1e-8)


def test_volume_frame_is_the_patient_frame(write_patient_bundle) -> None:
    volume = _volume(write_patient_bundle)

    np.testing.assert_allclose(volume.voxel_to_volume_mm, VOXEL_TO_LPS_MM)
    np.testing.assert_allclose(volume.spacing_xyz_mm, (1.0, 1.0, 2.0))
    np.testing.assert_allclose(volume.volume_mm_to_world(volume.center_xyz_mm), [0.0, 0.0, 0.85])
    points = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    np.testing.assert_allclose(volume.world_to_volume_mm(volume.volume_mm_to_world(points)), points)


def test_scan_storage_does_not_change_the_attenuation_volume(write_patient_bundle) -> None:
    reference = _volume(write_patient_bundle)
    other = _volume(write_patient_bundle, frame="RAS", unit="m", array_axes="kji", name="ras")

    np.testing.assert_array_equal(other.mu_volume, reference.mu_volume)
    np.testing.assert_allclose(other.voxel_to_volume_mm, reference.voxel_to_volume_mm, atol=1e-9)
    np.testing.assert_allclose(other.center_xyz_mm, reference.center_xyz_mm, atol=1e-9)


def test_unknown_preset_is_rejected(write_patient_bundle) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        _volume(write_patient_bundle, preset="invalid")
