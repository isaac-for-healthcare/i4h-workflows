from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VolumeMetadata:
    """Metadata for a preprocessed CT volume."""

    shape_zyx: tuple[int, int, int]
    spacing_zyx_mm: tuple[float, float, float]
    origin_xyz_mm: tuple[float, float, float] | None = None
    hu_range: tuple[float, float] | None = None
    mu_range: tuple[float, float] | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_zyx": list(self.shape_zyx),
            "spacing_zyx_mm": list(self.spacing_zyx_mm),
            "origin_xyz_mm": list(self.origin_xyz_mm) if self.origin_xyz_mm else None,
            "hu_range": list(self.hu_range) if self.hu_range else None,
            "mu_range": list(self.mu_range) if self.mu_range else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VolumeMetadata":
        return cls(
            shape_zyx=tuple(data["shape_zyx"]),
            spacing_zyx_mm=tuple(data["spacing_zyx_mm"]),
            origin_xyz_mm=tuple(data["origin_xyz_mm"]) if data.get("origin_xyz_mm") else None,
            hu_range=tuple(data["hu_range"]) if data.get("hu_range") else None,
            mu_range=tuple(data["mu_range"]) if data.get("mu_range") else None,
            source=data.get("source"),
        )


class PreprocessedVolume:
    """A preprocessed CT volume ready for rendering or downstream processing."""

    def __init__(self, mu_volume: np.ndarray, metadata: VolumeMetadata):
        if mu_volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {mu_volume.shape}")
        self._mu_volume = np.ascontiguousarray(mu_volume.astype(np.float32))
        self._metadata = metadata

    @property
    def mu_volume(self) -> np.ndarray:
        return self._mu_volume

    @property
    def metadata(self) -> VolumeMetadata:
        return self._metadata

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._mu_volume.shape

    @property
    def spacing_zyx_mm(self) -> tuple[float, float, float]:
        return self._metadata.spacing_zyx_mm

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "mu_volume.npy", self._mu_volume)
        (output_dir / "metadata.json").write_text(
            json.dumps(self._metadata.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return output_dir

    @classmethod
    def load(cls, input_dir: str | Path) -> "PreprocessedVolume":
        input_dir = Path(input_dir)
        mu_path = input_dir / "mu_volume.npy"
        meta_path = input_dir / "metadata.json"
        if not mu_path.exists():
            raise FileNotFoundError(f"Volume file not found: {mu_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        mu_volume = np.load(mu_path)
        metadata = VolumeMetadata.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        return cls(mu_volume, metadata)
