# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real library exporter through the workflow tool; mock only inference."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml
from patient_digital_twin.importers import NVSegmentImporter
from patient_digital_twin.importers._common import segmentation_anatomy
from patient_digital_twin.legacy_ct import INTERVENTIONAL, LINEAR, hu_to_mu

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
    assert manifest["coordinate_frame"] == "DICOM_LPS"
    assert manifest["patient_id"] == "subject"
    assert len(manifest["artifacts"]) == 8
    for relative in manifest["artifacts"].values():
        assert (output / relative).is_file()
    center = np.append((np.asarray(SHAPE) - 1) * 0.5, 1.0)
    patient_mm = np.asarray(manifest["transforms"]["voxel_to_patient_mm"]) @ center
    patient_mm[:3] *= 0.001
    world = np.asarray(manifest["transforms"]["world_from_patient_m"]) @ patient_mm
    np.testing.assert_allclose(world[:3], [0.0, 0.0, 0.85], atol=1e-8)
    assert np.load(output / "centerline_edges.npy").shape[1] == 2
    assert set(manifest["anatomy"]["structures"]) == {"aorta"}
    assert set(manifest["centerlines"]) == {"aorta"}


def test_stored_slice_order_does_not_change_artifacts(tmp_path, inference):
    for flipped in (False, True):
        _build(_ct(tmp_path / str(flipped), flipped=flipped), tmp_path / f"out_{flipped}")
    for filename in ("mu_volume.npy", "vessel_mask.npy", "centerline_points_mm.npy"):
        np.testing.assert_allclose(
            np.load(tmp_path / "out_False" / filename), np.load(tmp_path / "out_True" / filename), atol=1e-5
        )


@pytest.mark.parametrize("preset,mapping", [("interventional", INTERVENTIONAL), ("linear", LINEAR)])
def test_attenuation_matches_recorded_preset(tmp_path, inference, preset, mapping):
    output = tmp_path / "output"
    _build(_ct(tmp_path / "subject"), output, hu_to_mu_preset=preset)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["hu_to_mu"]["preset"] == preset
    np.testing.assert_allclose(np.load(output / "mu_volume.npy"), hu_to_mu(np.load(output / "hu_volume.npy"), mapping))


def test_oblique_input_fails_before_inference(tmp_path, monkeypatch):
    ct = _ct(tmp_path / "subject")
    image = nib.load(ct)
    affine = np.eye(4)
    angle = np.pi / 6
    affine[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    nib.save(nib.Nifti1Image(image.get_fdata(), affine), ct)
    monkeypatch.setattr(NVSegmentImporter, "to_anatomy_collection", lambda *a, **kw: pytest.fail("model ran"))
    with pytest.raises(ValueError, match="oblique"):
        _build(ct, tmp_path / "output")
    assert not (tmp_path / "output").exists()


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
