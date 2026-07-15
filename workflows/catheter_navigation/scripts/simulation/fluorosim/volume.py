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

"""Preprocessed volume representation for the Fluoroscopy Simulator.

This module defines the PreprocessedVolume class which encapsulates a
CT volume that has been converted to linear attenuation coefficients (μ)
and is ready for rendering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VolumeMetadata:
    """Metadata for a preprocessed CT volume.

    Attributes:
        shape_zyx: Volume shape in (Z, Y, X) order.
        spacing_zyx_mm: Voxel spacing in mm, (Z, Y, X) order.
        origin_xyz_mm: Volume origin in mm, (X, Y, Z) order.
        hu_range: Original HU range before conversion [min, max].
        mu_range: μ range after conversion [min, max].
        source: Original source path (DICOM dir or NIfTI file).
    """

    shape_zyx: tuple[int, int, int]
    spacing_zyx_mm: tuple[float, float, float]
    origin_xyz_mm: tuple[float, float, float] | None = None
    hu_range: tuple[float, float] | None = None
    mu_range: tuple[float, float] | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "shape_zyx": list(self.shape_zyx),
            "spacing_zyx_mm": list(self.spacing_zyx_mm),
            "origin_xyz_mm": list(self.origin_xyz_mm) if self.origin_xyz_mm else None,
            "hu_range": list(self.hu_range) if self.hu_range else None,
            "mu_range": list(self.mu_range) if self.mu_range else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VolumeMetadata":
        """Create from dictionary."""
        return cls(
            shape_zyx=tuple(d["shape_zyx"]),
            spacing_zyx_mm=tuple(d["spacing_zyx_mm"]),
            origin_xyz_mm=tuple(d["origin_xyz_mm"]) if d.get("origin_xyz_mm") else None,
            hu_range=tuple(d["hu_range"]) if d.get("hu_range") else None,
            mu_range=tuple(d["mu_range"]) if d.get("mu_range") else None,
            source=d.get("source"),
        )


class PreprocessedVolume:
    """A preprocessed CT volume ready for fluoroscopy rendering.

    This class encapsulates a 3D volume of linear attenuation coefficients (μ)
    along with metadata needed for rendering. Volumes can be saved to and loaded
    from disk for caching.

    Attributes:
        mu_volume: 3D numpy array of μ values in (Z, Y, X) order.
        metadata: Volume metadata including spacing and origin.

    Example:
        >>> # Load a cached volume
        >>> volume = PreprocessedVolume.load("/tmp/fluoro_cache")
        >>>
        >>> # Access properties
        >>> print(volume.shape)  # (256, 512, 512)
        >>> print(volume.spacing_zyx_mm)  # (1.0, 0.5, 0.5)
    """

    def __init__(
        self,
        mu_volume: np.ndarray,
        metadata: VolumeMetadata,
    ):
        """Initialize a preprocessed volume.

        Args:
            mu_volume: 3D numpy array of μ values (Z, Y, X), dtype float32.
            metadata: Volume metadata.

        Raises:
            ValueError: If mu_volume is not 3D or has wrong dtype.
        """
        if mu_volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {mu_volume.shape}")

        self._mu_volume = np.ascontiguousarray(mu_volume.astype(np.float32))
        self._metadata = metadata

    @property
    def mu_volume(self) -> np.ndarray:
        """Return the μ volume array (Z, Y, X)."""
        return self._mu_volume

    @property
    def metadata(self) -> VolumeMetadata:
        """Return volume metadata."""
        return self._metadata

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return volume shape (Z, Y, X)."""
        return self._mu_volume.shape

    @property
    def spacing_zyx_mm(self) -> tuple[float, float, float]:
        """Return voxel spacing in mm (Z, Y, X)."""
        return self._metadata.spacing_zyx_mm

    @property
    def spacing_xyz_mm(self) -> tuple[float, float, float]:
        """Return voxel spacing in mm (X, Y, Z)."""
        sz, sy, sx = self._metadata.spacing_zyx_mm
        return (sx, sy, sz)

    def save(self, output_dir: str | Path) -> Path:
        """Save the preprocessed volume to disk.

        Creates two files in output_dir:
        - mu_volume.npy: The μ volume array
        - metadata.json: Volume metadata

        Args:
            output_dir: Directory to save files to.

        Returns:
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save volume
        mu_path = output_dir / "mu_volume.npy"
        np.save(mu_path, self._mu_volume)

        # Save metadata
        meta_path = output_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(self._metadata.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

        return output_dir

    @classmethod
    def load(cls, input_dir: str | Path) -> "PreprocessedVolume":
        """Load a preprocessed volume from disk.

        Args:
            input_dir: Directory containing mu_volume.npy and metadata.json.

        Returns:
            Loaded PreprocessedVolume.

        Raises:
            FileNotFoundError: If required files are not found.
        """
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

    def __repr__(self) -> str:
        z, y, x = self.shape
        return (
            f"PreprocessedVolume(\n"
            f"  shape=({z}, {y}, {x}),\n"
            f"  spacing_mm={self.spacing_zyx_mm},\n"
            f"  mu_range=[{self._mu_volume.min():.4f}, {self._mu_volume.max():.4f}]\n"
            f")"
        )
