from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CtVolume:
    """A CT volume in a consistent in-memory convention."""

    hu_zyx: np.ndarray
    spacing_zyx_mm: tuple[float, float, float] | None = None
    origin_xyz_mm: tuple[float, float, float] | None = None
    direction: tuple[float, ...] | None = None

    def to_json_dict(self) -> dict:
        data: dict = {
            "shape_zyx": list(self.hu_zyx.shape),
            "dtype": str(self.hu_zyx.dtype),
        }
        if self.spacing_zyx_mm is not None:
            data["spacing_zyx_mm"] = list(self.spacing_zyx_mm)
        if self.origin_xyz_mm is not None:
            data["origin_xyz_mm"] = list(self.origin_xyz_mm)
        if self.direction is not None:
            data["direction_row_major_3x3"] = list(self.direction)
        return data


def load_dicom_series_hu(dicom_dir: str | Path) -> CtVolume:
    """Load a DICOM series directory into a HU volume."""
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SimpleITK is required to load DICOM series. Install it with:\n"
            "  pip install SimpleITK"
        ) from exc

    ddir = Path(dicom_dir)
    if not ddir.exists() or not ddir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {ddir}")

    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(str(ddir)))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found under: {ddir}")

    series_uid = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(str(ddir), series_uid)
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    arr_zyx = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)

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

    spacing_xyz = tuple(float(x) for x in image.GetSpacing())
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    origin_xyz = tuple(float(x) for x in image.GetOrigin())
    direction = tuple(float(x) for x in image.GetDirection())

    return CtVolume(
        hu_zyx=arr_zyx,
        spacing_zyx_mm=spacing_zyx,
        origin_xyz_mm=origin_xyz,
        direction=direction,
    )


def load_nifti_hu(nifti_path: str | Path) -> CtVolume:
    """Load a NIfTI file into a HU volume."""
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required to load NIfTI files. Install with:\n  pip install nibabel") from exc

    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    image = nib.load(nifti_path)
    arr = image.get_fdata().astype(np.float32)
    arr_zyx = np.transpose(arr, (2, 1, 0))

    spacing_xyz = tuple(float(x) for x in image.header.get_zooms()[:3])
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    affine = image.affine
    origin_xyz = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
    rotation = affine[:3, :3]
    direction = tuple(float(x) for x in (rotation / np.array(spacing_xyz)).flatten())

    return CtVolume(
        hu_zyx=arr_zyx,
        spacing_zyx_mm=spacing_zyx,
        origin_xyz_mm=origin_xyz,
        direction=direction,
    )
