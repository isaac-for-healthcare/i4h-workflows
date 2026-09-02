# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml
from vasculature_digital_twin import hu_to_mu

from i4h_tools.patient_twin import cli
from i4h_tools.patient_twin.pipeline import INTERVENTIONAL, LINEAR, build_patient_twin

SHAPE = (28, 28, 28)


def _anatomy() -> tuple[np.ndarray, np.ndarray]:
    """A soft-tissue block with a contrast-bright vessel through it, in nibabel ijk order.

    The vessel is wide enough to survive the package's minimum-component filter and thin
    enough for scikit-image to thin it to a curve.
    """
    hu = np.full(SHAPE, -1000.0, dtype=np.float32)
    hu[3:25, 3:25, 3:25] = 40.0
    vessel = np.zeros(SHAPE, dtype=np.uint8)
    vessel[11:16, 11:16, 4:26] = 1
    hu[vessel > 0] = 300.0
    return hu, vessel


def _write_subject(root: Path, *, flip_first_axis: bool = False) -> Path:
    """Write a TotalSegmentator-shaped subject.

    ``flip_first_axis`` stores the same patient in the opposite slice order along i, with an
    affine that compensates, so both spellings describe identical anatomy.
    """
    subject = root / "subject-1"
    segmentations = subject / "segmentations"
    segmentations.mkdir(parents=True)
    hu, vessel = _anatomy()
    affine = np.eye(4)
    if flip_first_axis:
        hu = np.flip(hu, axis=0)
        vessel = np.flip(vessel, axis=0)
        affine[0, 0] = -1.0
        affine[0, 3] = float(SHAPE[0] - 1)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(hu), affine), subject / "ct.nii.gz")
    nib.save(nib.Nifti1Image(np.ascontiguousarray(vessel), affine), segmentations / "aorta.nii.gz")
    return subject


def test_build_patient_twin_creates_complete_arena_bundle(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path)
    output = tmp_path / "output"

    manifest_path = build_patient_twin(
        subject,
        output,
        close_iterations=0,
        surface_step=1,
    )

    expected = {
        "mu_volume.npy",
        "hu_volume.npy",
        "metadata.json",
        "vessel_mask.npy",
        "centerline_points_mm.npy",
        "centerline_edges.npy",
        "centerline_radii_mm.npy",
        "patient_anatomy.usdc",
        "patient_twin.yaml",
    }
    assert expected <= {path.name for path in output.iterdir()}
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["patient_id"] == "subject-1"
    assert manifest["coordinate_frame"] == "DICOM_LPS"
    assert manifest["artifacts"]["anatomy_usd"] == "patient_anatomy.usdc"
    voxel_to_patient = np.asarray(manifest["transforms"]["voxel_to_patient_mm"])
    world_from_patient = np.asarray(manifest["transforms"]["world_from_patient_m"])
    center_voxel = np.append(0.5 * (np.asarray(SHAPE) - 1.0), 1.0)
    center_patient_m = (voxel_to_patient @ center_voxel)[:3] * 0.001
    center_world = world_from_patient @ np.append(center_patient_m, 1.0)
    np.testing.assert_allclose(center_world[:3], (0.0, 0.0, 0.85), atol=1e-8)
    assert np.load(output / "centerline_edges.npy").shape[1] == 2


def test_volume_lands_in_the_canonical_patient_frame(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path)

    build_patient_twin(subject, tmp_path / "output", close_iterations=0, surface_step=1)

    metadata = json.loads((tmp_path / "output" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["anatomical_frame"] == "LPS"
    assert metadata["source_orientation"] == "SAR"
    np.testing.assert_allclose(metadata["direction_row_major_3x3"], np.eye(3).reshape(-1), atol=1e-8)


def test_a_flipped_acquisition_reorients_onto_the_same_patient_geometry(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    flipped = tmp_path / "flipped"

    build_patient_twin(_write_subject(canonical), canonical / "output", close_iterations=0, surface_step=1)
    build_patient_twin(
        _write_subject(flipped, flip_first_axis=True),
        flipped / "output",
        close_iterations=0,
        surface_step=1,
    )

    for artifact in ("mu_volume.npy", "vessel_mask.npy", "centerline_points_mm.npy"):
        np.testing.assert_array_equal(
            np.load(canonical / "output" / artifact),
            np.load(flipped / "output" / artifact),
            err_msg=f"{artifact} depends on the stored slice order",
        )
    manifests = [
        yaml.safe_load((root / "output" / "patient_twin.yaml").read_text(encoding="utf-8"))["transforms"]
        for root in (canonical, flipped)
    ]
    assert manifests[0] == manifests[1]


def test_an_oblique_acquisition_is_rejected(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path)
    hu, _ = _anatomy()
    angle = np.radians(30.0)
    affine = np.eye(4)
    affine[:3, :3] = ((np.cos(angle), -np.sin(angle), 0.0), (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0))
    nib.save(nib.Nifti1Image(hu, affine), subject / "ct.nii.gz")

    with pytest.raises(ValueError, match="oblique"):
        build_patient_twin(subject, tmp_path / "output", close_iterations=0, surface_step=1)


def test_segmentation_on_a_different_grid_is_rejected(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path)
    _, vessel = _anatomy()
    shifted = np.eye(4)
    shifted[2, 3] = 5.0
    nib.save(nib.Nifti1Image(vessel, shifted), subject / "segmentations" / "aorta.nii.gz")

    with pytest.raises(ValueError, match="patient origin does not match"):
        build_patient_twin(subject, tmp_path / "output", close_iterations=0, surface_step=1)


def test_segment_vessels_builds_without_label_files(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path)
    for path in (subject / "segmentations").iterdir():
        path.unlink()
    output = tmp_path / "output"

    build_patient_twin(subject, output, close_iterations=0, surface_step=1, segment_vessels=True)

    assert np.load(output / "vessel_mask.npy").any()


def test_attenuation_volume_records_the_curve_that_produced_it(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path)
    output = tmp_path / "output"

    build_patient_twin(subject, output, close_iterations=0, surface_step=1)

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["hu_to_mu"]["preset"] == "interventional"
    assert metadata["hu_to_mu"]["control_points"] == [list(point) for point in INTERVENTIONAL.control_points]
    hu = np.load(output / "hu_volume.npy")
    np.testing.assert_allclose(np.load(output / "mu_volume.npy"), hu_to_mu(hu, INTERVENTIONAL))


def test_linear_preset_rebuilds_the_earlier_attenuation_volume(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path)
    output = tmp_path / "output"

    build_patient_twin(subject, output, close_iterations=0, surface_step=1, hu_to_mu_preset="linear")

    hu = np.load(output / "hu_volume.npy")
    np.testing.assert_allclose(np.load(output / "mu_volume.npy"), hu_to_mu(hu, LINEAR))


def test_unknown_attenuation_preset_fails_before_any_work(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path)
    output = tmp_path / "output"

    with pytest.raises(KeyError):
        build_patient_twin(subject, output, close_iterations=0, surface_step=1, hu_to_mu_preset="clinical")

    assert not output.exists()


def test_build_patient_twin_requires_subject_layout(tmp_path: Path) -> None:
    subject = tmp_path / "incomplete"
    subject.mkdir()

    try:
        build_patient_twin(subject, tmp_path / "output")
    except FileNotFoundError as error:
        assert "ct.nii.gz" in str(error)
    else:
        raise AssertionError("missing subject CT should fail")


def test_cli_writes_into_subject_directory_by_default(tmp_path: Path, monkeypatch) -> None:
    subject = tmp_path / "s0011"
    captured: dict[str, Path] = {}

    def fake_build(input_subject: Path, output: Path, **_kwargs) -> None:
        captured["subject"] = input_subject
        captured["output"] = output

    monkeypatch.setattr(cli, "build_patient_twin", fake_build)

    assert cli.main([str(subject)]) == 0
    assert captured == {"subject": subject.resolve(), "output": subject.resolve()}
