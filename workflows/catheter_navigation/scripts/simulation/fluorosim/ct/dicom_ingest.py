# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CtVolume:
    """A CT volume in a consistent in-memory convention.

    - `hu_zyx`: CT in Hounsfield Units, shape (Z,Y,X)
    - `spacing_zyx_mm`: voxel spacing in millimeters (Z,Y,X)
    - `origin_xyz_mm`: physical origin in millimeters (X,Y,Z)
    - `direction`: 3x3 direction cosine matrix (row-major)
    """

    hu_zyx: np.ndarray
    spacing_zyx_mm: tuple[float, float, float] | None = None
    origin_xyz_mm: tuple[float, float, float] | None = None
    direction: tuple[float, ...] | None = None

    def to_json_dict(self) -> dict:
        d: dict = {
            "shape_zyx": list(self.hu_zyx.shape),
            "dtype": str(self.hu_zyx.dtype),
        }
        if self.spacing_zyx_mm is not None:
            d["spacing_zyx_mm"] = list(self.spacing_zyx_mm)
        if self.origin_xyz_mm is not None:
            d["origin_xyz_mm"] = list(self.origin_xyz_mm)
        if self.direction is not None:
            d["direction_row_major_3x3"] = list(self.direction)
        return d


def load_dicom_series_hu(dicom_dir: str | Path) -> CtVolume:
    """Load a DICOM series directory into a HU volume.

    This is the first missing piece of the "CT ingestion pipeline" described in the
    catheter PDF: read clinical CT/CTA volumes and normalize them into a consistent
    `(Z,Y,X)` numpy array for downstream HU→μ and rendering.

    Implementation choice:
    - We use SimpleITK for DICOM series reading because it is lightweight and stable.
    - MONAI can also be used (as in HoloHub apps), but MONAI typically pulls in PyTorch,
      which is a heavy dependency for this workflow. If you already have MONAI installed,
      you can wrap this function or add a MONAI reader path later.
    """
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SimpleITK is required to load DICOM series. Install it with:\n"
            "  pip install SimpleITK\n"
            "If you prefer MONAI-based IO, install MONAI + its deps and we can wire that in."
        ) from e

    ddir = Path(dicom_dir)
    if not ddir.exists() or not ddir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {ddir}")

    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(str(ddir)))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found under: {ddir}")

    # If multiple series exist, pick the first. You can extend this to choose by SeriesDescription.
    series_uid = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(str(ddir), series_uid)
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    img = reader.Execute()

    # SimpleITK returns arrays in (Z,Y,X) already.
    arr_zyx = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)

    # Convert to HU if rescale tags exist (common for CT).
    # Tags:
    # - (0028,1052) RescaleIntercept
    # - (0028,1053) RescaleSlope
    intercept = None
    slope = None
    try:
        if reader.HasMetaDataKey(0, "0028|1052"):
            intercept = float(reader.GetMetaData(0, "0028|1052"))
        if reader.HasMetaDataKey(0, "0028|1053"):
            slope = float(reader.GetMetaData(0, "0028|1053"))
    except Exception:
        intercept = None
        slope = None

    if slope is not None and intercept is not None:
        arr_zyx = arr_zyx * float(slope) + float(intercept)

    spacing_xyz = tuple(float(x) for x in img.GetSpacing())  # (X,Y,Z)
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    origin_xyz = tuple(float(x) for x in img.GetOrigin())
    direction = tuple(float(x) for x in img.GetDirection())  # 9 floats, row-major

    return CtVolume(
        hu_zyx=arr_zyx,
        spacing_zyx_mm=spacing_zyx,
        origin_xyz_mm=origin_xyz,
        direction=direction,
    )


def load_nifti_hu(nifti_path: str | Path) -> CtVolume:
    """Load a NIfTI file into a HU volume.

    This supports loading CT volumes stored in NIfTI format (.nii or .nii.gz),
    which is common for research datasets like ImageCAS, ASOCA, etc.

    The NIfTI file is assumed to contain HU values directly (no rescaling needed).
    """
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("nibabel is required to load NIfTI files. Install it with:\n" "  pip install nibabel") from e

    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    img = nib.load(nifti_path)
    arr = img.get_fdata().astype(np.float32)

    # NIfTI convention is typically (X, Y, Z), we need (Z, Y, X)
    arr_zyx = np.transpose(arr, (2, 1, 0))

    # Get spacing from affine or header
    spacing_xyz = tuple(float(x) for x in img.header.get_zooms()[:3])
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    # Get origin from affine
    affine = img.affine
    origin_xyz = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))

    # Direction cosines from affine: normalize each column by its voxel spacing
    # (affine[:3, :3] == R @ diag(spacing), so dividing column j by spacing_xyz[j]
    # recovers unit-length cosines, matching the SimpleITK/DICOM convention above).
    rotation = affine[:3, :3]
    direction = tuple(float(x) for x in (rotation / np.array(spacing_xyz)).flatten())

    return CtVolume(
        hu_zyx=arr_zyx,
        spacing_zyx_mm=spacing_zyx,
        origin_xyz_mm=origin_xyz,
        direction=direction,
    )
