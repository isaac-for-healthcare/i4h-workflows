from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import HuToMuMapping, PreprocessingSettings
from .ct.dicom_ingest import load_dicom_series_hu, load_nifti_hu
from .volume import PreprocessedVolume, VolumeMetadata


class VolumePreprocessor:
    """Convert CT HU volumes into mu volumes plus metadata."""

    def __init__(
        self,
        hu_volume: np.ndarray,
        spacing_zyx_mm: tuple[float, float, float],
        origin_xyz_mm: tuple[float, float, float] | None = None,
        source: str | None = None,
        settings: PreprocessingSettings | None = None,
    ):
        if hu_volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {hu_volume.shape}")

        self._hu_volume = hu_volume.astype(np.float32)
        self._spacing_zyx_mm = spacing_zyx_mm
        self._origin_xyz_mm = origin_xyz_mm
        self._source = source
        self._settings = settings or PreprocessingSettings()

    @classmethod
    def from_dicom(cls, dicom_dir: str | Path, settings: PreprocessingSettings | None = None) -> "VolumePreprocessor":
        dicom_dir = Path(dicom_dir)
        if not dicom_dir.exists():
            raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
        ct = load_dicom_series_hu(dicom_dir)
        return cls(
            hu_volume=ct.hu_zyx,
            spacing_zyx_mm=ct.spacing_zyx_mm or (1.0, 1.0, 1.0),
            origin_xyz_mm=ct.origin_xyz_mm,
            source=str(dicom_dir),
            settings=settings,
        )

    @classmethod
    def from_nifti(cls, nifti_path: str | Path, settings: PreprocessingSettings | None = None) -> "VolumePreprocessor":
        nifti_path = Path(nifti_path)
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
        ct = load_nifti_hu(nifti_path)
        return cls(
            hu_volume=ct.hu_zyx,
            spacing_zyx_mm=ct.spacing_zyx_mm or (1.0, 1.0, 1.0),
            origin_xyz_mm=ct.origin_xyz_mm,
            source=str(nifti_path),
            settings=settings,
        )

    @classmethod
    def from_numpy(
        cls,
        hu_volume: np.ndarray,
        spacing_zyx_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        settings: PreprocessingSettings | None = None,
    ) -> "VolumePreprocessor":
        return cls(hu_volume=hu_volume, spacing_zyx_mm=spacing_zyx_mm, settings=settings)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._hu_volume.shape

    @property
    def hu_volume_zyx(self) -> np.ndarray:
        return self._hu_volume

    @property
    def hu_range(self) -> tuple[float, float]:
        return (float(self._hu_volume.min()), float(self._hu_volume.max()))

    def preprocess(self, output_dir: str | Path | None = None) -> PreprocessedVolume:
        settings = self._settings
        hu = self._hu_volume.copy()
        hu_range = (float(hu.min()), float(hu.max()))

        if settings.clip_hu:
            hu = np.clip(hu, settings.hu_clip_min, settings.hu_clip_max)

        mu = self._hu_to_mu(hu, settings.hu_to_mu)
        mu_range = (float(mu.min()), float(mu.max()))

        metadata = VolumeMetadata(
            shape_zyx=mu.shape,
            spacing_zyx_mm=self._spacing_zyx_mm,
            origin_xyz_mm=self._origin_xyz_mm,
            hu_range=hu_range,
            mu_range=mu_range,
            source=self._source,
        )
        volume = PreprocessedVolume(mu, metadata)

        if output_dir is not None:
            volume.save(output_dir)
            print(f"[VolumePreprocessor] Saved to: {output_dir}")
        return volume

    def _hu_to_mu(self, hu: np.ndarray, cfg: HuToMuMapping) -> np.ndarray:
        hu_clipped = np.clip(hu, cfg.hu_min, cfg.hu_max)
        t = (hu_clipped - cfg.hu_min) / (cfg.hu_max - cfg.hu_min + 1e-12)
        mu = cfg.mu_min + t * (cfg.mu_max - cfg.mu_min)
        return mu.astype(np.float32)
