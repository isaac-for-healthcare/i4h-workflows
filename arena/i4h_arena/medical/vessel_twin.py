# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patient-twin inputs for xray_simulator's vessel contrast.

The carve, the contrast transport and the chord integrator live in :mod:`xray_simulator.vessel`
and take explicit arrays. This module reads them out of a twin, in the renderer's volume
millimetres: the twin's patient LPS frame, reached from a voxel index through the scan's own
affine, which is the frame :func:`~i4h_arena.medical.slang_fluoroscopy.solve_projection_geometry`
poses the C-arm in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from xray_simulator.vessel import HuToMu, VesselTree

if TYPE_CHECKING:
    from xray_simulator.vessel import VesselContrast

    from .patient_volume import PatientVolume


def volume_scale(patient: PatientVolume) -> float:
    """Uniform length scale taking patient millimetres to volume millimetres.

    Measured by sending unit separations through the same two calls a point takes, rather than
    composing the affines here: ``patient_mm_to_world`` folds the millimetre-to-metre step into
    itself, so a hand-composed product with ``world_from_patient_m`` -- which places patient
    *metres* -- comes out a thousand times wrong, silently and in the direction that still looks
    plausible.

    Raises:
        ValueError: If the chain scales axes unequally, since a scalar lumen radius then has no
            single length to scale by.
    """
    probe_patient_mm = np.concatenate((np.zeros((1, 3)), np.eye(3)), axis=0)
    probe_volume_mm = patient.world_to_volume_mm(patient.twin.patient_mm_to_world(probe_patient_mm))
    scales = np.linalg.norm(probe_volume_mm[1:] - probe_volume_mm[0], axis=-1)
    if not np.allclose(scales, scales[0], rtol=1.0e-5, atol=1.0e-8):
        raise ValueError(f"patient-to-volume transform scales axes unequally ({scales.tolist()})")
    return float(scales[0])


def ijk_to_volume_mm(patient: PatientVolume) -> np.ndarray:
    """The renderer's voxel-centre index to volume millimetre transform."""
    return np.asarray(patient.voxel_to_volume_mm, dtype=np.float64)


def vessel_tree(patient: PatientVolume, *, radius_scale: float = 1.0) -> VesselTree:
    """The twin's centerline in volume millimetres, smoothed off its voxel staircase.

    Points go through the twin's placement and then the volume transform, while radii are lengths
    and take only the chain's uniform scale.

    Raises:
        ValueError: If the twin declares no centerline points, radii or edges, or they disagree.
    """
    artifacts = patient.twin.artifacts
    missing = [
        name for name in ("centerline_points", "centerline_radii", "centerline_edges") if artifacts.get(name) is None
    ]
    if missing:
        raise ValueError(f"vessel contrast needs twin artifacts {', '.join(missing)}")
    points_patient_mm = np.asarray(np.load(artifacts["centerline_points"]), dtype=np.float64)
    radii_patient_mm = np.asarray(np.load(artifacts["centerline_radii"]), dtype=np.float64).reshape(-1)
    tree = VesselTree(
        patient.world_to_volume_mm(patient.twin.patient_mm_to_world(points_patient_mm)),
        np.load(artifacts["centerline_edges"]),
        radii_patient_mm * volume_scale(patient) * float(radius_scale),
    )
    return tree.smoothed()


def hu_to_mu(patient: PatientVolume) -> HuToMu:
    """The HU-to-mu curve the patient's attenuation volume was built with."""
    from xray_simulator import HuToMuMapping

    return HuToMu.from_mapping(HuToMuMapping.preset(patient.hu_to_mu_preset))


def vessel_contrast(patient: PatientVolume, *, supersample: int = 1, device: str | None = None) -> VesselContrast:
    """Carve the twin's lumen and hold it ready to composite back, in volume millimetres."""
    from xray_simulator.vessel import VesselContrast

    mask_path = patient.twin.artifacts.get("vessel_mask")
    return VesselContrast(
        patient.mu_volume,
        ijk_to_volume_mm(patient),
        vessel_tree(patient),
        hu_to_mu(patient),
        vessel_mask_kji=None if mask_path is None else np.load(mask_path),
        supersample=supersample,
        device=device,
    )


__all__ = ["hu_to_mu", "ijk_to_volume_mm", "vessel_contrast", "vessel_tree", "volume_scale"]
