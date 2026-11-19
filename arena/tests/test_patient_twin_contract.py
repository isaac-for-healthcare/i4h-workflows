# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-repository contract: bundles the real patient-digital-twin writes load here.

The other bundle tests write bundles with this repository's own fixture, which only
proves the reader is self-consistent. This one builds a bundle with the pinned
patient-digital-twin exporter, in the tools/patient_twin environment where it is
installed, and loads it through ``PatientTwin.load`` and ``PatientVolume.load``. It is
skipped when that environment has not been set up (``./third_party/setup.sh
patient-twin`` and ``uv sync --project tools/patient_twin``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_volume import PatientVolume

ROOT = Path(__file__).resolve().parents[2]
TOOL_PYTHON = ROOT / "tools" / "patient_twin" / ".venv" / "bin" / "python"
WRITER = Path(__file__).with_name("patient_twin_contract_writer.py")
SHAPE_KJI = (30, 24, 24)
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _oblique_ras_mm() -> np.ndarray:
    angle = 0.3
    affine = np.diag([0.9, 1.1, 1.25, 1.0])
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    affine[:3, :3] = rotation @ affine[:3, :3]
    affine[:3, 3] = (-40.0, 25.0, 110.0)
    return affine


def _tool_has_the_library() -> bool:
    if not TOOL_PYTHON.is_file():
        return False
    probe = subprocess.run([str(TOOL_PYTHON), "-c", "import patient_digital_twin"], capture_output=True, check=False)
    return probe.returncode == 0


requires_patient_twin_tool = pytest.mark.skipif(
    not _tool_has_the_library(), reason="tools/patient_twin environment with patient-digital-twin is not set up"
)


@pytest.fixture(scope="module")
def library_bundle(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("contract")
    affine = directory / "affine.json"
    affine.write_text(json.dumps(_oblique_ras_mm().tolist()), encoding="utf-8")
    result = subprocess.run(
        [str(TOOL_PYTHON), str(WRITER), str(directory / "out"), str(affine)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return Path(result.stdout.strip().splitlines()[-1])


@requires_patient_twin_tool
def test_a_library_bundle_resolves_to_patient_lps_millimetres(library_bundle) -> None:
    twin = PatientTwin.load(library_bundle)

    np.testing.assert_allclose(twin.voxel_to_patient_mm, _RAS_TO_LPS @ _oblique_ras_mm(), atol=1e-4)
    assert twin.shape_ijk == SHAPE_KJI[::-1]
    mask = twin.vessel_mask_kji()
    assert mask is not None and mask.shape == SHAPE_KJI
    assert mask[3:27, 9:14, 9:14].mean() > 0.5
    assert mask[:2].sum() == 0


@requires_patient_twin_tool
def test_a_library_centerline_runs_inside_its_vessel(library_bundle) -> None:
    twin = PatientTwin.load(library_bundle)
    centerline = twin.centerline()
    assert centerline is not None and len(centerline.points_mm) >= 2

    voxels = np.rint(
        (np.linalg.inv(twin.voxel_to_patient_mm) @ np.c_[centerline.points_mm, np.ones(len(centerline.points_mm))].T)[
            :3
        ]
    ).astype(int)
    # The tube runs along k between slices 3 and 26 and spans voxels 9-13 in i and j.
    assert np.all((voxels[0] >= 8) & (voxels[0] <= 14) & (voxels[1] >= 8) & (voxels[1] <= 14))
    assert np.ptp(voxels[2]) >= 12


@requires_patient_twin_tool
def test_a_library_bundle_renders_on_its_native_grid(library_bundle) -> None:
    volume = PatientVolume.load(PatientTwin.load(library_bundle))

    assert volume.shape_zyx == SHAPE_KJI
    np.testing.assert_allclose(volume.spacing_xyz_mm, (0.9, 1.1, 1.25), atol=1e-4)
    vessel, tissue, air = volume.mu_volume[15, 11, 11], volume.mu_volume[15, 4, 4], volume.mu_volume[0, 0, 0]
    assert vessel > tissue > air >= 0.0
