# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TotalSegmentator-subject to Arena patient-twin pipeline.

CT ingest, canonical reorientation, the HU-to-mu transfer function, and the attenuation
volume all come from the ``vasculature_digital_twin`` package, so a twin built here is
expressed in that package's canonical LPS patient frame regardless of the slice order the
source study was stored in. What stays here is the glue Arena needs and that package does
not produce: the anatomy USD and the ``patient_twin.yaml`` manifest carrying the transforms
that keep the renderer, the solver, and the USD stage in one coordinate system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage
from skimage.morphology import skeletonize
from vasculature_digital_twin import (
    CenterlineGraph,
    HuToMuMapping,
    PreprocessingSettings,
    VolumePreprocessor,
    get_vessel_mask,
)
from vasculature_digital_twin.ct.dicom_ingest import CtVolume, load_nifti_hu

from .anatomy import build_anatomy_usd

DEFAULT_LABELS = ("aorta", "iliac_artery_left", "iliac_artery_right")

# The curve itself comes from the digital-twin package; these are the two settings this
# workflow ships. `interventional` suppresses soft tissue, which a fluoroscopic beam barely
# sees, and gives contrast, bone and implant density their own slopes so an implant does not
# saturate at the same value as dense cortical bone. Both share the 3000 HU -> 0.02 /mm
# ceiling, so the overall attenuation scale, and the DSA vessel boost tuned against it, is
# unchanged between them.
INTERVENTIONAL = HuToMuMapping(
    control_points=(
        (-1000.0, 0.0000),
        (-300.0, 0.0000),
        (100.0, 0.0008),
        (300.0, 0.0028),
        (500.0, 0.0060),
        (900.0, 0.0090),
        (1500.0, 0.0120),
        (3000.0, 0.0200),
        (8000.0, 0.0440),
    )
)

# The ramp patient twins used before named curves existed, and the package default.
LINEAR = HuToMuMapping()

PRESETS: dict[str, HuToMuMapping] = {"interventional": INTERVENTIONAL, "linear": LINEAR}
DEFAULT_PRESET = "interventional"

# Patient LPS to Isaac world: +X toward the head, +Y toward the patient's right, +Z
# anterior, which lays a supine patient along the table facing up.
_WORLD_FROM_PATIENT_ROTATION = np.asarray(
    ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    dtype=np.float64,
)

# A working-height isocenter keeps the patient, catheter, table, and C-arm aligned in a
# recognizable clinical layout.
_ISOCENTER_WORLD_M = np.asarray((0.0, 0.0, 0.85), dtype=np.float64)


def preset(name: str) -> HuToMuMapping:
    """Look up a named curve.

    Raises:
        KeyError: If the name is not a known preset.
    """
    if name not in PRESETS:
        raise KeyError(f"unknown HU to mu preset {name!r}; choose from {', '.join(sorted(PRESETS))}")
    return PRESETS[name]


def _ingest(nifti: Path) -> CtVolume:
    """Read a NIfTI volume into the digital twin's canonical LPS frame.

    The package resolves the study's own direction cosines, so an axial series stored feet
    first, or a label file saved with a different slice order, arrives on the same axes as
    everything else instead of silently mirroring the anatomy.

    Raises:
        ValueError: If the acquisition is oblique enough that permuting and flipping cannot
            align it.
    """
    volume = load_nifti_hu(nifti)
    direction = np.asarray(volume.direction, dtype=np.float64).reshape(3, 3)
    if not np.allclose(direction, np.eye(3), atol=1e-4):
        raise ValueError(
            f"{nifti} is an oblique acquisition that permuting and flipping cannot align. The DRR "
            "renderer samples an axis-aligned voxel grid, so resample the CT and its labels onto "
            "the patient axes first."
        )
    return volume


def _write_attenuation_volume(
    ct: CtVolume,
    output: Path,
    *,
    source: Path,
    mapping: HuToMuMapping,
    hu_to_mu_name: str,
) -> None:
    preprocessor = VolumePreprocessor(
        hu_volume=ct.hu_zyx,
        spacing_zyx_mm=ct.spacing_zyx_mm,
        origin_xyz_mm=ct.origin_xyz_mm,
        source=str(source),
        # The interventional curve puts its top knot at 8000 HU so implant density keeps a
        # slope of its own, well past the 3071 HU end of the CT storage range the package
        # clips to by default. Clipping there would collapse that band into cortical bone.
        settings=PreprocessingSettings(hu_to_mu=mapping, clip_hu=False),
        anatomical_frame=ct.anatomical_frame,
        source_orientation=ct.source_orientation,
        direction=ct.direction,
    )
    volume = preprocessor.preprocess()
    # Record the named curve alongside its knots so a twin can be traced back to the
    # invocation that built it, not just to a shape.
    volume.metadata.hu_to_mu = {"preset": hu_to_mu_name, **mapping.to_dict()}
    volume.save(output)
    np.save(output / "hu_volume.npy", ct.hu_zyx.astype(np.float32, copy=False))


def _require_same_grid(volume: CtVolume, ct: CtVolume, source: Path) -> None:
    if volume.hu_zyx.shape != ct.hu_zyx.shape:
        raise ValueError(f"segmentation {source} shape {volume.hu_zyx.shape} does not match the CT {ct.hu_zyx.shape}")
    if not np.allclose(volume.spacing_zyx_mm, ct.spacing_zyx_mm, atol=1e-5):
        raise ValueError(f"segmentation {source} voxel spacing does not match the CT")
    if not np.allclose(volume.origin_xyz_mm, ct.origin_xyz_mm, atol=1e-3):
        raise ValueError(f"segmentation {source} patient origin does not match the CT")


def _load_vessel_mask(segmentations: Path, labels: tuple[str, ...], ct: CtVolume) -> np.ndarray:
    union = np.zeros(ct.hu_zyx.shape, dtype=bool)
    found: list[str] = []
    for label in labels:
        path = segmentations / f"{label}.nii.gz"
        if not path.is_file():
            continue
        volume = _ingest(path)
        _require_same_grid(volume, ct, path)
        union |= volume.hu_zyx > 0.5
        found.append(label)
    if not found:
        raise FileNotFoundError(f"none of the requested labels exist under {segmentations}: {', '.join(labels)}")
    return union


def _segment_vessel_mask(ct: CtVolume) -> np.ndarray:
    """Derive a vessel mask with the digital twin's own segmenter.

    Runs TotalSegmentator where it is installed and falls back to the package's HU
    threshold otherwise, so a subject that ships a CT and no label files still builds.
    """
    result = get_vessel_mask(ct.hu_zyx, ct.spacing_zyx_mm)
    mask = np.asarray(result.combined_mask) > 0
    if not mask.any():
        raise ValueError("vessel segmentation produced an empty mask")
    return mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0:
        raise ValueError("vessel labels produced an empty mask")
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _centerline(
    mask_zyx: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
) -> CenterlineGraph:
    """Skeletonize the vessel mask into a centerline graph in patient millimetres.

    The package also offers a VMTK centerline extractor, which resolves branches far better
    than a voxel skeleton but pulls in a conda-only toolchain. This keeps the dependency
    surface to scikit-image while still returning the package's graph type.
    """
    skeleton = skeletonize(mask_zyx)
    coordinates = np.argwhere(skeleton)
    if len(coordinates) < 4:
        raise RuntimeError(f"vessel skeleton has too few nodes ({len(coordinates)})")

    distance = ndimage.distance_transform_edt(mask_zyx, sampling=spacing_zyx_mm)
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    origin = np.asarray(origin_xyz_mm, dtype=np.float64)
    points_xyz_mm = origin + coordinates[:, ::-1] * spacing[::-1]
    radii_mm = distance[tuple(coordinates.T)].astype(np.float32)
    index = {tuple(int(value) for value in coordinate): i for i, coordinate in enumerate(coordinates)}
    edges: list[tuple[int, int]] = []
    for i, coordinate in enumerate(coordinates):
        z, y, x = (int(value) for value in coordinate)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    neighbor = index.get((z + dz, y + dy, x + dx))
                    if neighbor is not None and neighbor > i:
                        edges.append((i, neighbor))
    if not edges:
        raise RuntimeError("vessel skeleton produced no edges")
    return CenterlineGraph(
        points=points_xyz_mm.astype(np.float32),
        radii=radii_mm,
        edges=np.asarray(edges, dtype=np.int64),
    )


def _voxel_to_patient_mm(ct: CtVolume) -> np.ndarray:
    """Affine from voxel indices in xyz order to canonical LPS millimetres.

    Diagonal because :func:`_ingest` has already rejected anything the canonical
    reorientation could not bring onto the patient axes.
    """
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag(np.asarray(ct.spacing_zyx_mm, dtype=np.float64)[::-1])
    affine[:3, 3] = np.asarray(ct.origin_xyz_mm, dtype=np.float64)
    return affine


def _write_manifest(
    output: Path,
    *,
    patient_id: str,
    voxel_to_patient_mm: np.ndarray,
    shape_xyz: tuple[int, int, int],
) -> Path:
    center_voxel_xyz = 0.5 * (np.asarray(shape_xyz, dtype=np.float64) - 1.0)
    center_patient_mm = (voxel_to_patient_mm @ np.append(center_voxel_xyz, 1.0))[:3]
    world_from_patient_m = np.eye(4, dtype=np.float64)
    world_from_patient_m[:3, :3] = _WORLD_FROM_PATIENT_ROTATION
    world_from_patient_m[:3, 3] = _ISOCENTER_WORLD_M - (_WORLD_FROM_PATIENT_ROTATION @ center_patient_mm * 0.001)
    manifest = {
        "schema_version": 1,
        "patient_id": patient_id,
        "coordinate_frame": "DICOM_LPS",
        "transforms": {
            "voxel_to_patient_mm": voxel_to_patient_mm.tolist(),
            "world_from_patient_m": world_from_patient_m.tolist(),
        },
        "artifacts": {
            "attenuation_volume": "mu_volume.npy",
            "volume_metadata": "metadata.json",
            "hu_volume": "hu_volume.npy",
            "vessel_mask": "vessel_mask.npy",
            "centerline_points": "centerline_points_mm.npy",
            "centerline_edges": "centerline_edges.npy",
            "centerline_radii": "centerline_radii_mm.npy",
            "anatomy_usd": "patient_anatomy.usdc",
        },
    }
    path = output / "patient_twin.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def build_patient_twin(
    subject: Path,
    output: Path,
    *,
    patient_id: str | None = None,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    close_iterations: int = 2,
    surface_step: int = 3,
    hu_to_mu_preset: str = DEFAULT_PRESET,
    segment_vessels: bool = False,
) -> Path:
    subject = subject.expanduser().resolve()
    output = output.expanduser().resolve()
    ct_path = subject / "ct.nii.gz"
    segmentations = subject / "segmentations"
    if not ct_path.is_file():
        raise FileNotFoundError(f"subject CT does not exist: {ct_path}")
    if not segment_vessels:
        if not segmentations.is_dir():
            raise FileNotFoundError(f"subject segmentations do not exist: {segmentations}")
        if not labels:
            raise ValueError("at least one vessel label is required")
    if close_iterations < 0 or surface_step < 1:
        raise ValueError("close_iterations must be non-negative and surface_step must be positive")

    mapping = preset(hu_to_mu_preset)

    output.mkdir(parents=True, exist_ok=True)
    print(f"==> patient twin output {output}")
    print(f"==> preprocess CT (HU to mu preset {hu_to_mu_preset})")
    ct = _ingest(ct_path)
    print(f"==> CT stored as {ct.source_orientation}, reoriented to {ct.anatomical_frame}")
    _write_attenuation_volume(
        ct,
        output,
        source=ct_path,
        mapping=mapping,
        hu_to_mu_name=hu_to_mu_preset,
    )

    if segment_vessels:
        print("==> segment vessels (digital twin segmenter)")
        vessel_mask = _segment_vessel_mask(ct)
    else:
        print(f"==> segment vessels ({', '.join(labels)})")
        vessel_mask = _load_vessel_mask(segmentations, labels, ct)
    if close_iterations:
        vessel_mask = ndimage.binary_closing(
            vessel_mask,
            structure=np.ones((3, 3, 3), dtype=np.uint8),
            iterations=close_iterations,
        )
    vessel_mask = _largest_component(vessel_mask)
    np.save(output / "vessel_mask.npy", vessel_mask.astype(np.uint8))
    centerline = _centerline(vessel_mask, ct.spacing_zyx_mm, ct.origin_xyz_mm)
    np.save(output / "centerline_points_mm.npy", centerline.points)
    np.save(output / "centerline_edges.npy", centerline.edges)
    np.save(output / "centerline_radii_mm.npy", centerline.radii)

    print("==> build anatomy USD")
    build_anatomy_usd(
        ct.hu_zyx,
        vessel_mask,
        ct.spacing_zyx_mm,
        ct.origin_xyz_mm,
        output / "patient_anatomy.usdc",
        surface_step=surface_step,
    )
    print("==> write Arena manifest")
    manifest = _write_manifest(
        output,
        patient_id=patient_id or subject.name,
        voxel_to_patient_mm=_voxel_to_patient_mm(ct),
        shape_xyz=ct.hu_zyx.shape[::-1],
    )
    print(f"PATIENT_TWIN={manifest}")
    return manifest
