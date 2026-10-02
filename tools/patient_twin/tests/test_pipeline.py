# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real library exporter through the workflow tool; mock only inference."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml
from patient_digital_twin.importers import NVSegmentImporter, segmentation_anatomy

from i4h_tools.patient_twin import cli
from i4h_tools.patient_twin.pipeline import build_patient_twin

SHAPE = (29, 29, 41)


@pytest.fixture
def inference(monkeypatch):
    def segment(importer, *, names):
        image = importer.image
        mask = (np.asarray(image.dataobj) == 300).astype(np.uint8)
        return segmentation_anatomy(nib.Nifti1Image(mask, image.affine), {1: "aorta"}, names=names)

    monkeypatch.setattr(NVSegmentImporter, "to_anatomy_collection", segment)


def _ct(root, *, flipped=False):
    root.mkdir(parents=True)
    x, y, z = np.indices(SHAPE)
    mask = ((x - 14) ** 2 + (y - 14) ** 2 < 25) & (z > 3) & (z < 37)
    hu = np.full(SHAPE, -1000.0, dtype=np.float32)
    hu[2:-2, 2:-2, 2:-2] = 40
    hu[mask] = 300
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    if flipped:
        hu = np.flip(hu, axis=0)
        affine[0, 0] = -1
        affine[0, 3] = SHAPE[0] - 1
    path = root / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.ascontiguousarray(hu), affine), path)
    return path


def _build(ct, output, **options):
    return build_patient_twin(source="nvsegment", input=ct, classes=["aorta"], output=output, **options)


def test_complete_bundle_preserves_patient_placement(tmp_path, inference):
    ct = _ct(tmp_path / "subject")
    output = tmp_path / "bundle"
    path = _build(ct, output)
    manifest = yaml.safe_load(path.read_text())
    assert manifest["coordinate_frame"] == "RAS"
    assert manifest["patient_id"] == "subject"
    assert len(manifest["artifacts"]) == 7
    for relative in manifest["artifacts"].values():
        assert (output / relative).is_file()
    center = np.append((np.asarray(SHAPE) - 1) * 0.5, 1.0)
    np.testing.assert_allclose(
        np.asarray(manifest["transforms"]["voxel_to_scan"]) @ center, nib.load(ct).affine @ center
    )
    assert "world_from_patient_m" not in manifest["transforms"]
    assert np.load(output / "centerline_edges.npy").shape[1] == 2
    assert set(manifest["anatomy"]["structures"]) == {"aorta"}
    assert "centerlines" not in manifest


def test_stored_slice_order_is_preserved(tmp_path, inference):
    for flipped in (False, True):
        ct = _ct(tmp_path / str(flipped), flipped=flipped)
        path = _build(ct, tmp_path / f"out_{flipped}")
        np.testing.assert_array_equal(np.load(path.parent / "volume.npy"), nib.load(ct).get_fdata())
        metadata = yaml.safe_load((path.parent / "volume.yaml").read_text())["output"]
        np.testing.assert_allclose(metadata["array_index_to_world"], nib.load(ct).affine)
    np.testing.assert_array_equal(
        np.load(tmp_path / "out_False/vessel_mask.npy"), np.load(tmp_path / "out_True/vessel_mask.npy")[:, :, ::-1]
    )


def test_preparation_exports_hu_without_attenuation(tmp_path, inference):
    output = tmp_path / "output"
    ct = _ct(tmp_path / "subject")
    path = _build(ct, output)
    metadata = yaml.safe_load((output / "volume.yaml").read_text())["output"]
    assert metadata["intensity_unit"] == "HU"
    assert "hu_to_mu" not in metadata
    assert not (output / "mu_volume.npy").exists()
    np.testing.assert_array_equal(
        np.sort(np.load(output / "volume.npy").ravel()), np.sort(nib.load(ct).get_fdata().ravel())
    )
    assert yaml.safe_load(path.read_text())["schema_version"] == 3


def test_oblique_input_keeps_native_grid(tmp_path, inference):
    ct = _ct(tmp_path / "subject")
    image = nib.load(ct)
    affine = np.eye(4)
    angle = np.pi / 6
    affine[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    nib.save(nib.Nifti1Image(image.get_fdata(), affine), ct)
    path = _build(ct, tmp_path / "output")
    np.testing.assert_array_equal(np.load(path.parent / "volume.npy"), image.get_fdata())
    metadata = yaml.safe_load((path.parent / "volume.yaml").read_text())["output"]
    np.testing.assert_allclose(metadata["array_index_to_world"], nib.load(ct).affine)


def test_inference_failure_has_no_fallback_or_partial_bundle(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr(NVSegmentImporter, "to_anatomy_collection", failure)
    with pytest.raises(RuntimeError, match="model failed"):
        _build(_ct(tmp_path / "subject"), tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_cli_passes_source_image_classes_and_output(monkeypatch, tmp_path):
    captured = {}

    def build(**options):
        captured.update(options)
        return options["output"] / "patient_twin.yaml"

    monkeypatch.setattr(cli, "build_patient_twin", build)
    assert (
        cli.main(
            [
                "--source",
                "nvsegment",
                "--input",
                "ct.nii.gz",
                "--classes",
                "aorta",
                "--output",
                str(tmp_path / "output"),
                "--python",
                "/model/python",
            ]
        )
        == 0
    )
    assert captured["format"] == "bundle"
    assert captured["source"] == "nvsegment"
    assert captured["input"] == Path("ct.nii.gz")
    assert captured["classes"] == ["aorta"]
    assert captured["python_executable"] == "/model/python"


def test_existing_output_is_not_overwritten(tmp_path, inference):
    ct = _ct(tmp_path / "subject")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        _build(ct, output)
