# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HU bundle loading and legacy attenuation compatibility, without Isaac Sim."""

import json

import numpy as np
import pytest
import yaml

from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_volume import PatientVolume


@pytest.fixture
def bundle(tmp_path):
    hu = np.array([-1500, -300, 100, 300, 900, 3000, 6000, 9000], dtype=np.float32).reshape(2, 2, 2)
    np.save(tmp_path / "hu.npy", hu)
    metadata = {
        "shape_zyx": [2, 2, 2],
        "spacing_zyx_mm": [2, 1, 1],
        "origin_xyz_mm": [10, 20, 30],
        "anatomical_frame": "LPS",
        "intensity_units": "HU",
        "array_order": "ZYX",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    affine[:3, 3] = [10, 20, 30]
    manifest = {
        "schema_version": 2,
        "patient_id": "test",
        "coordinate_frame": "DICOM_LPS",
        "transforms": {"voxel_to_patient_mm": affine.tolist(), "world_from_patient_m": np.eye(4).tolist()},
        "artifacts": {"hu_volume": "hu.npy", "volume_metadata": "metadata.json"},
    }
    path = tmp_path / "patient_twin.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path, manifest, metadata, hu


@pytest.mark.parametrize(
    "preset,expected",
    [
        (None, [0, 0.0035, 0.0055, 0.0065, 0.0095, 0.02, 0.02, 0.02]),
        ("linear", [0, 0.0035, 0.0055, 0.0065, 0.0095, 0.02, 0.02, 0.02]),
        ("interventional", [0, 0, 0.0008, 0.0028, 0.009, 0.02, 0.0344, 0.044]),
    ],
)
def test_hu_mapping_and_spatial_transform(bundle, preset, expected):
    path, _, _, hu = bundle
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    twin = PatientTwin.load(path)
    assert twin.schema_version == 2
    volume = PatientVolume.load(twin, hu_to_mu_preset=preset)
    np.testing.assert_allclose(volume.mu_volume.ravel(), expected, atol=1e-8)
    np.testing.assert_array_equal(np.load(path.parent / "hu.npy"), hu)
    np.testing.assert_allclose(volume.volume_mm_to_world([[0, 0, 0]]), [[0.01, 0.02, 0.03]])
    points = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    np.testing.assert_allclose(volume.world_to_volume_mm(volume.volume_mm_to_world(points)), points)
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}


def test_legacy_mu_is_preserved_and_explicit_remap_uses_hu(bundle):
    path, manifest, _, _ = bundle
    manifest["schema_version"] = 1
    manifest["artifacts"]["attenuation_volume"] = "mu.npy"
    np.save(path.parent / "mu.npy", np.full((2, 2, 2), 0.123, np.float32))
    path.write_text(yaml.safe_dump(manifest))
    twin = PatientTwin.load(path)
    np.testing.assert_allclose(PatientVolume.load(twin).mu_volume, 0.123)
    assert PatientVolume.load(twin, hu_to_mu_preset="interventional").mu_volume.flat[1] == 0
    del manifest["artifacts"]["hu_volume"]
    path.write_text(yaml.safe_dump(manifest))
    twin = PatientTwin.load(path)
    np.testing.assert_allclose(PatientVolume.load(twin).mu_volume, 0.123)
    with pytest.raises(ValueError, match="requires artifacts.hu_volume"):
        PatientVolume.load(twin, hu_to_mu_preset="linear")


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("intensity_units", "mu", "HU intensities"),
        ("array_order", "XYZ", "ZYX array order"),
        ("intensity_units", None, "HU intensities"),
        ("spacing_zyx_mm", [0, 1, 1], "positive finite"),
        ("shape_zyx", [2, 2, 3], "does not match"),
    ],
)
def test_invalid_hu_metadata(bundle, key, value, match):
    path, _, metadata, _ = bundle
    metadata[key] = value
    (path.parent / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        PatientVolume.load(PatientTwin.load(path))


def test_nonfinite_hu_and_unknown_preset_rejected(bundle):
    path, _, _, hu = bundle
    with pytest.raises(ValueError, match="Unknown"):
        PatientVolume.load(PatientTwin.load(path), hu_to_mu_preset="invalid")
    hu.flat[0] = np.nan
    np.save(path.parent / "hu.npy", hu)
    with pytest.raises(ValueError, match="non-finite"):
        PatientVolume.load(PatientTwin.load(path))


def test_schema_two_requires_hu(bundle):
    path, manifest, _, _ = bundle
    manifest["artifacts"]["attenuation_volume"] = manifest["artifacts"].pop("hu_volume")
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="schema 2 requires"):
        PatientTwin.load(path)


@pytest.mark.parametrize("axes,unit,scale", [("ijk", "mm", 1.0), ("kji", "m", 0.001)])
def test_native_bundle_geometry_and_default_placement(tmp_path, axes, unit, scale):
    from xray_simulator.scan_volume import from_array

    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    affine = np.array([[0.8, -0.6, 0.0, 10.0], [0.6, 0.8, 0.0, 20.0], [0.0, 0.0, 2.0, 30.0], [0.0, 0.0, 0.0, 1.0]])
    affine[:3] *= scale
    permutation = np.eye(4)
    permutation[:3, :3] = np.eye(3)[:, ["ijk".index(c) for c in axes]]
    scan = from_array(
        values.transpose(["ijk".index(c) for c in axes]), affine @ permutation, array_axes=axes, world_unit=unit
    )
    folder = tmp_path / "scan"
    scan.save(folder)
    manifest = {
        "schema_version": 3,
        "patient_id": "native",
        "coordinate_frame": "RAS",
        "meters_per_unit": scan.meters_per_unit,
        "spatial_unit": unit,
        "transforms": {"voxel_to_scan": affine.tolist()},
        "artifacts": {"hu_volume": "volume.npy", "volume_metadata": "volume.yaml"},
    }
    path = folder / "patient_twin.yaml"
    path.write_text(yaml.safe_dump(manifest))
    twin = PatientTwin.load(path)
    volume = PatientVolume.load(twin)
    np.testing.assert_allclose(volume.volume_mm_to_world(volume.center_xyz_mm), [0.0, 0.0, 0.85])
    idx = np.array([1.0, 2.0, 3.0, 1.0])
    point = (volume.voxel_to_volume_mm @ idx)[:3]
    np.testing.assert_allclose(volume.volume_mm_to_world(point), twin.voxels_to_world(idx[:3]))
    np.testing.assert_allclose(volume.world_to_volume_mm(twin.voxels_to_world(idx[:3])), point)
    assert volume.shape_zyx == values.shape[::-1]
