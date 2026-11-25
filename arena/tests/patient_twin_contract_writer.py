# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a tiny schema-2 bundle with the real patient-digital-twin exporter.

Run with the tools/patient_twin interpreter, not arena's:
``python patient_twin_contract_writer.py <output_dir> <affine.json>``. The CT is stored
in ``(k, j, i)`` order as ``hu_kji``, with a 300 HU aorta tube along ``k`` inside 100 HU
tissue. Prints the bundle's ``patient_twin.yaml`` path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from patient_digital_twin import HumanBody, SegmentationImporter
from patient_digital_twin.scan_volume import from_nifti

SHAPE_KJI = (30, 24, 24)


def main(output: Path, affine_ras_mm: np.ndarray) -> Path:
    hu_kji = np.full(SHAPE_KJI, -1000.0, dtype=np.float32)
    hu_kji[2:-2, 2:-2, 2:-2] = 100.0
    mask_kji = np.zeros(SHAPE_KJI, dtype=np.uint8)
    mask_kji[3:27, 9:14, 9:14] = 1
    hu_kji[mask_kji == 1] = 300.0
    output.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(hu_kji.transpose(2, 1, 0), affine_ras_mm)
    image.header.set_xyzt_units("mm")
    ct_path = output / "ct.nii.gz"
    nib.save(image, ct_path)
    affine_m = nib.load(ct_path).affine.copy()
    affine_m[:3] *= 0.001
    body = HumanBody(
        SegmentationImporter(mask_kji, {1: "aorta"}, affine_xyz_to_imaging_m=affine_m).to_anatomy_collection()
    )
    body.attach_scan(from_nifti(ct_path), source_path=ct_path)
    return body.export_patient_twin(output / "bundle", vessel_names=["aorta"])


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), np.asarray(json.loads(Path(sys.argv[2]).read_text()), dtype=float)))
