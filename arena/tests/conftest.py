# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for patient-twin consumers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

_FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])
_METERS_PER_UNIT = {"mm": 0.001, "m": 1.0}


def to_scan(voxel_to_lps_mm: np.ndarray, *, frame: str, unit: str) -> np.ndarray:
    """Express an LPS-millimetre affine in another scan frame and unit."""
    scale = np.diag([1000.0 * _METERS_PER_UNIT[unit]] * 3 + [1.0])
    return np.linalg.inv(scale) @ (_FLIP if frame == "RAS" else np.eye(4)) @ voxel_to_lps_mm


@pytest.fixture
def write_patient_bundle(tmp_path):
    """Write a native patient-digital-twin bundle and return its manifest path.

    Arrays and centerline points are given in ``(i, j, k)`` order and scan coordinates;
    the bundle stores them in ``array_axes`` order, exactly as the exporter does.
    """
    from xray_simulator.scan_volume import from_array

    def write(
        hu_ijk: np.ndarray,
        voxel_to_scan: np.ndarray,
        *,
        frame: str = "LPS",
        unit: str = "mm",
        array_axes: str = "ijk",
        name: str = "bundle",
        vessel_mask_ijk: np.ndarray | None = None,
        centerline_scan: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        world_from_scan_m: np.ndarray | None = None,
    ) -> Path:
        order = ["ijk".index(axis) for axis in array_axes]
        permutation = np.eye(4)
        permutation[:3, :3] = np.eye(3)[:, [array_axes.index(axis) for axis in "ijk"]]
        array_to_scan = np.asarray(voxel_to_scan, dtype=float) @ np.linalg.inv(permutation)
        scan = from_array(
            np.asarray(hu_ijk, dtype=np.float32).transpose(order),
            array_to_scan,
            array_axes=array_axes,
            world_frame=frame,
            world_unit=unit,
        )
        folder = tmp_path / name
        scan.save(folder)
        artifacts = {"hu_volume": "volume.npy", "volume_metadata": "volume.yaml"}
        if vessel_mask_ijk is not None:
            np.save(folder / "vessel_mask.npy", np.asarray(vessel_mask_ijk, dtype=np.uint8).transpose(order))
            artifacts["vessel_mask"] = "vessel_mask.npy"
        if centerline_scan is not None:
            for key, values in zip(("centerline_points", "centerline_edges", "centerline_radii"), centerline_scan):
                np.save(folder / f"{key}.npy", values)
                artifacts[key] = f"{key}.npy"
        transforms = {"voxel_to_scan": np.asarray(voxel_to_scan, dtype=float).tolist()}
        if world_from_scan_m is not None:
            transforms["world_from_patient_m"] = np.asarray(world_from_scan_m, dtype=float).tolist()
        manifest = folder / "patient_twin.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    "patient_id": name,
                    "coordinate_frame": frame,
                    "spatial_unit": unit,
                    "meters_per_unit": _METERS_PER_UNIT[unit],
                    "transforms": transforms,
                    "artifacts": artifacts,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return manifest

    return write
