# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import yaml
from conftest import to_scan

from i4h_arena.embodiments.catheter import reference_initial_catheter_length_m
from i4h_arena.medical.patient_twin import PatientTwin

SHAPE_IJK = (4, 5, 6)
# An oblique grid with anisotropic spacing, in LPS millimetres.
_ANGLE = np.radians(30.0)
VOXEL_TO_LPS_MM = np.array(
    [
        [2.0 * np.cos(_ANGLE), -1.5 * np.sin(_ANGLE), 0.0, 10.0],
        [2.0 * np.sin(_ANGLE), 1.5 * np.cos(_ANGLE), 0.0, 20.0],
        [0.0, 0.0, 3.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
CENTERLINE_LPS_MM = np.array([[10.0, 20.0, 30.0], [12.0, 21.0, 36.0], [13.0, 23.0, 42.0]])
RADII_MM = np.array([1.0, 1.5, 2.0])
EDGES = np.array([[0, 1], [1, 2]])


def _bundle(write_patient_bundle, *, frame="LPS", unit="mm", array_axes="ijk", name="bundle", **kwargs):
    lps_to_scan = to_scan(np.eye(4), frame=frame, unit=unit)
    mask = np.zeros(SHAPE_IJK, dtype=np.uint8)
    mask[1, 2, 3] = mask[0, 4, 5] = 1
    points_scan = CENTERLINE_LPS_MM @ lps_to_scan[:3, :3].T
    radii_scan = RADII_MM * abs(lps_to_scan[0, 0])
    return write_patient_bundle(
        np.arange(np.prod(SHAPE_IJK), dtype=np.float32).reshape(SHAPE_IJK),
        to_scan(VOXEL_TO_LPS_MM, frame=frame, unit=unit),
        frame=frame,
        unit=unit,
        array_axes=array_axes,
        name=name,
        vessel_mask_ijk=mask,
        centerline_scan=(points_scan, EDGES, radii_scan),
        **kwargs,
    )


def test_every_scan_frame_unit_and_array_order_resolves_to_the_same_patient(write_patient_bundle) -> None:
    reference = PatientTwin.load(_bundle(write_patient_bundle))
    other = PatientTwin.load(_bundle(write_patient_bundle, frame="RAS", unit="m", array_axes="kji", name="ras"))

    for twin in (reference, other):
        np.testing.assert_allclose(twin.voxel_to_patient_mm, VOXEL_TO_LPS_MM, atol=1e-9)
        assert twin.shape_ijk == SHAPE_IJK
        np.testing.assert_allclose(twin.isocenter_world_m, [0.0, 0.0, 0.85], atol=1e-12)
    np.testing.assert_allclose(other.world_from_patient_m, reference.world_from_patient_m, atol=1e-12)
    assert other.width_mm == pytest.approx(reference.width_mm)

    centerlines = [twin.centerline() for twin in (reference, other)]
    for centerline in centerlines:
        np.testing.assert_allclose(centerline.points_mm, CENTERLINE_LPS_MM, atol=1e-9)
        np.testing.assert_allclose(centerline.radii_mm, RADII_MM, atol=1e-9)
        np.testing.assert_array_equal(centerline.edges, EDGES)
    np.testing.assert_array_equal(other.vessel_mask_kji(), reference.vessel_mask_kji())
    assert reference.vessel_mask_kji()[3, 2, 1] == 1


def test_anatomy_pose_places_scan_coordinates_at_their_world_position(write_patient_bundle) -> None:
    point_lps_mm = np.array([15.0, -4.0, 22.0])
    for frame, unit in (("LPS", "mm"), ("RAS", "m")):
        twin = PatientTwin.load(_bundle(write_patient_bundle, frame=frame, unit=unit, name=frame))
        position, rotation, scale = twin.anatomy_world_pose()
        point_scan = to_scan(np.eye(4), frame=frame, unit=unit)[:3, :3] @ point_lps_mm

        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        np.testing.assert_allclose(position + scale * rotation @ point_scan, twin.patient_mm_to_world(point_lps_mm))


def test_a_placement_hint_is_read_in_the_scan_frame(write_patient_bundle) -> None:
    world_from_scan = np.eye(4)
    world_from_scan[:3, 3] = (1.0, 2.0, 3.0)
    twin = PatientTwin.load(_bundle(write_patient_bundle, frame="RAS", world_from_scan_m=world_from_scan))

    # RAS (1, 2, 3) mm is LPS (-1, -2, 3) mm.
    np.testing.assert_allclose(twin.patient_mm_to_world([[-1.0, -2.0, 3.0]]), [[1.001, 2.002, 3.003]])
    np.testing.assert_allclose(twin.world_to_patient_mm([[1.001, 2.002, 3.003]]), [[-1.0, -2.0, 3.0]])


def test_width_is_the_left_right_extent_of_the_grid(write_patient_bundle) -> None:
    twin = PatientTwin.load(_bundle(write_patient_bundle, array_axes="kji"))
    expected = 4 * 2.0 * np.cos(_ANGLE) + 5 * 1.5 * np.sin(_ANGLE)

    assert twin.width_mm == pytest.approx(expected)
    assert reference_initial_catheter_length_m(twin, fallback_m=1.0) == pytest.approx(0.65 * expected * 0.001)
    assert reference_initial_catheter_length_m(twin, fallback_m=0.001) == 0.001


def test_legacy_schema_one_bundles_are_rejected_with_a_rebuild_hint(write_patient_bundle) -> None:
    manifest = _bundle(write_patient_bundle)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw["schema_version"] = 1
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="tools/patient_twin/run.sh"):
        PatientTwin.load(manifest)


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda raw: raw["artifacts"].pop("hu_volume"), "hu_volume and artifacts.volume_metadata are required"),
        (lambda raw: raw.update(coordinate_frame="RAS"), "disagree on the scan geometry"),
        (lambda raw: raw.update(spatial_unit="m"), "disagree on the scan geometry"),
        (lambda raw: raw["transforms"]["voxel_to_scan"][0].__setitem__(3, 99.0), "disagree on the scan geometry"),
        (lambda raw: raw["artifacts"].update(vessel_mask="missing.npy"), "does not exist"),
    ],
)
def test_inconsistent_manifests_are_rejected(write_patient_bundle, edit, match) -> None:
    manifest = _bundle(write_patient_bundle)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    edit(raw)
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError if match != "does not exist" else FileNotFoundError, match=match):
        PatientTwin.load(manifest)
