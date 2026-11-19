# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native HU bundles through sensor-simulation attenuation, without Isaac Sim."""

import numpy as np
import pytest
import yaml
from conftest import to_scan
from xray_simulator.scan_volume import digest

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


def test_environments_sharing_a_bundle_share_one_read_only_volume(write_patient_bundle) -> None:
    """N environments on one patient cost one attenuation volume, not N."""
    twin = PatientTwin.load(
        write_patient_bundle(HU.reshape(2, 2, 2), to_scan(VOXEL_TO_LPS_MM, frame="LPS", unit="mm"), frame="LPS")
    )

    first, second = PatientVolume.load(twin), PatientVolume.load(PatientTwin.load(twin.source))

    assert second.mu_volume is first.mu_volume
    assert not first.mu_volume.flags.writeable
    assert PatientVolume.load(twin, hu_to_mu_preset="linear").mu_volume is not first.mu_volume


def test_a_rewritten_bundle_is_not_served_from_the_cache(write_patient_bundle) -> None:
    twin = PatientTwin.load(
        write_patient_bundle(HU.reshape(2, 2, 2), to_scan(VOXEL_TO_LPS_MM, frame="LPS", unit="mm"), frame="LPS")
    )
    before = PatientVolume.load(twin).mu_volume.copy()
    hu_path = twin.artifacts["hu_volume"]
    np.save(hu_path, np.full_like(np.load(hu_path), 900.0))
    metadata_path = twin.artifacts["volume_metadata"]
    recipe = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    recipe["output"]["sha256"] = digest(hu_path)
    metadata_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")

    after = PatientVolume.load(twin).mu_volume

    assert not np.array_equal(after, before)
    np.testing.assert_allclose(after, 0.009, atol=1e-6)
