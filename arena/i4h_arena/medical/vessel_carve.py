# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Removing the contrast lumen from the attenuation volume so it can be injected instead.

A contrast CT already carries the lumen baked in at the concentration the scan was taken
with. That is fine while the vessel is static and the contrast never changes, and wrong as
soon as either moves: a bolus that washes in and out cannot be represented by a constant, and
a wall the catheter pushes cannot be represented by voxels fixed to the original pose. So the
lumen is carved out at build time and composited back every frame as attenuating geometry,
the way :mod:`~i4h_arena.medical.catheter_attenuation` already composites the instrument.

Three things make the carve harder than ``volume[mask] = soft_tissue``:

**There is no single value to fill with.** The aorta's neighbourhood is soft tissue *and*
lung *and* vertebral body *and* fat. Filling with one scalar leaves a vessel-shaped column of
uniform attenuation where lung should sit behind the descending aorta on one side and bone on
the other, and since it is static it never clears. The fill here is instead a nearest-donor
inpaint: every carved voxel takes the value of the closest voxel that is *eligible*, and
eligibility is an attenuation band measured from the shell around the vessel, so air and bone
are excluded without hard-coding Hounsfield thresholds for a particular scanner.

**What is carved and what is added back have to be the same shape.** The carve could use the
segmentation; the add-back can only use the physics model, which is a centerline with per-node
radii, and on ``s0011`` only about 83% of the segmented voxels fall inside that tube. Carving
the tube alone is what keeps the accounting exact -- removed and restored are then one region,
so no ray can double-count and nothing is lost. Carving the *union* instead removes 10,988
voxels of real attenuation that nothing can put back, as a permanent static error; it is
available through ``include_segmentation`` for the case where leaving unmodelled contrast in
the volume is the worse problem, but it is not the default, because the measurement says the
union is both lossier and no more consistent.

**The boundary matters at the scale of the effect.** A binary mask on a 1.5 mm grid
stair-steps by half a voxel, against wall motion of 0.5-2 mm. ``supersample`` evaluates the
tube at sub-voxel offsets and carves fractionally, which is the same reason the chord
integrals are solved analytically instead of marched.

The static residual this leaves is why the DSA view subtracts a contrast-free mask frame
rather than boosting voxels: a static error appears identically in the mask and the live
frame and cancels, so carve accuracy stops being a correctness requirement for the view the
contrast actually lives in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .patient_volume import PatientVolume

#: Donor band, as quantiles of the attenuation in the shell around the vessel. The shell is
#: whatever the vessel actually borders, so its interquartile range is the local soft tissue
#: without needing to name a Hounsfield window; the tails are the lung and the spine.
DONOR_QUANTILES = (0.25, 0.75)
#: Shell thickness used to measure that band, in voxels.
SHELL_VOXELS = 3
#: Margin around the carve region for the inpaint search, in voxels. Donors are looked for
#: inside this halo, so the fill stays local rather than reaching across the body.
INPAINT_MARGIN_VOXELS = 12


@dataclass(frozen=True, slots=True)
class CarveResult:
    """A carved attenuation volume and the measurements that describe what was taken out."""

    #: Carved volume, ZYX, in the same units and layout as the original.
    mu_volume: np.ndarray
    #: Fraction of each voxel that was carved, ZYX in ``[0, 1]``. Binary unless supersampled.
    occupancy: np.ndarray
    #: Mean attenuation inside the carved region before the carve.
    lumen_mu_per_mm: float
    #: Mean attenuation of the material filled in.
    fill_mu_per_mm: float
    #: Donor band the inpaint drew from.
    donor_band_mu_per_mm: tuple[float, float]
    #: Carved voxel count and volume.
    carved_voxels: int
    carved_volume_ml: float
    #: Share of the segmentation mask the centerline tube covers, when both were available.
    tube_covers_mask: float | None

    @property
    def contrast_mu_per_mm(self) -> float:
        """Attenuation the lumen contributed over the tissue it displaced.

        This is the coefficient the composited vessel carries at the scan's own contrast
        concentration, and the reference the bolus model scales. Deriving it from the volume
        rather than from a physical table is what keeps the dynamic geometry on the same
        calibrated transfer function as the anatomy behind it, which the catheter's
        hard-coded physical coefficients notably do not.
        """
        return float(self.lumen_mu_per_mm - self.fill_mu_per_mm)


def volume_scale(patient: PatientVolume) -> float:
    """Uniform length scale taking patient millimetres to volume millimetres.

    Measured by sending unit separations through the same two calls a point takes, rather
    than composing the affines here: ``patient_mm_to_world`` folds the millimetre-to-metre
    step into itself, so a hand-composed product with ``world_from_patient_m`` -- which places
    patient *metres* -- comes out a thousand times wrong, silently and in the direction that
    still looks plausible.

    Raises:
        ValueError: If the chain scales axes unequally, since a scalar lumen radius then has
            no single length to scale by.
    """
    probe_patient_mm = np.concatenate((np.zeros((1, 3)), np.eye(3)), axis=0)
    probe_volume_mm = patient.world_to_volume_mm(patient.twin.patient_mm_to_world(probe_patient_mm))
    scales = np.linalg.norm(probe_volume_mm[1:] - probe_volume_mm[0], axis=-1)
    if not np.allclose(scales, scales[0], rtol=1.0e-5, atol=1.0e-8):
        raise ValueError(f"patient-to-volume transform scales axes unequally ({scales.tolist()})")
    return float(scales[0])


#: Laplacian passes applied to the centerline before anything geometric is built on it.
#: The stored nodes sit at voxel centres, so the path is a staircase that turns a mean 32 degrees
#: per node on ``s0011`` -- discretisation, not anatomy, and it wrecks any chain of convex
#: primitives laid along it because the joint planes then lean further than a segment is long.
#: Two passes bring the mean turn to 7 degrees while moving nodes 0.37 mm on average and 1.12 mm
#: at worst, under one voxel, so the staircase goes and the vessel stays where it was. More
#: passes keep straightening but start cutting the corners off genuine bends.
_SMOOTHING_PASSES = 2
_SMOOTHING_RATE = 0.5


def smooth_centerline(points_mm: np.ndarray, edges: np.ndarray, *, passes: int = _SMOOTHING_PASSES) -> np.ndarray:
    """Relax a voxel-stepped centerline towards the smooth curve it samples.

    Each pass moves every node a fraction of the way to the mean of its graph neighbours. Degree
    is taken from the edge list, so bifurcations and endpoints need no special case: an endpoint
    has one neighbour and simply drifts towards it.
    """
    points = np.asarray(points_mm, dtype=np.float64).copy()
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not passes or not edges.size:
        return points
    degree = np.bincount(edges.reshape(-1), minlength=len(points)).astype(np.float64)
    degree[degree == 0.0] = 1.0
    for _ in range(int(passes)):
        neighbour_sum = np.zeros_like(points)
        np.add.at(neighbour_sum, edges[:, 0], points[edges[:, 1]])
        np.add.at(neighbour_sum, edges[:, 1], points[edges[:, 0]])
        points += _SMOOTHING_RATE * (neighbour_sum / degree[:, None] - points)
    return points


def centerline_in_volume_mm(patient: PatientVolume, *, radius_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Centerline nodes and radii in the renderer's volume millimetres.

    The twin stores them in patient millimetres, so points go through the twin's placement and
    then the volume transform, while radii are lengths and take only the chain's uniform
    scale. Keeping that distinction here is the whole reason the manifest carries explicit
    transforms.

    Nodes are smoothed on the way through, and this is the single place it happens so that the
    carve and the per-frame compositing cannot end up describing different curves.

    Raises:
        ValueError: If the twin declares no centerline points or radii, the two disagree in
            length, or any radius is not positive.
    """
    artifacts = patient.twin.artifacts
    missing = [name for name in ("centerline_points", "centerline_radii") if artifacts.get(name) is None]
    if missing:
        raise ValueError(f"the vessel carve needs twin artifacts {', '.join(missing)}")

    points_patient_mm = np.asarray(np.load(artifacts["centerline_points"]), dtype=np.float64)
    radii_patient_mm = np.asarray(np.load(artifacts["centerline_radii"]), dtype=np.float64).reshape(-1)
    if len(radii_patient_mm) != len(points_patient_mm):
        raise ValueError(
            f"centerline_radii has {len(radii_patient_mm)} entries but centerline_points has "
            f"{len(points_patient_mm)}; one radius per node is required"
        )
    if not np.all(radii_patient_mm > 0.0):
        raise ValueError("centerline_radii must be strictly positive")

    points_volume_mm = patient.world_to_volume_mm(patient.twin.patient_mm_to_world(points_patient_mm))
    edges = artifacts.get("centerline_edges")
    if edges is not None:
        points_volume_mm = smooth_centerline(points_volume_mm, np.load(edges))
    radii_volume_mm = radii_patient_mm * volume_scale(patient) * float(radius_scale)
    return points_volume_mm, radii_volume_mm


def tube_occupancy(
    patient: PatientVolume,
    *,
    supersample: int = 1,
    radius_scale: float = 1.0,
) -> np.ndarray:
    """Fraction of each voxel inside the centerline tube, ZYX in ``[0, 1]``.

    The tube is the union of spheres swept along the centerline graph with each node's lumen
    radius, which is the same primitive the per-frame compositing integrates analytically, so
    what is carved and what is added back describe one geometry.

    Args:
        patient: Loaded volume, whose twin supplies the centerline artifacts.
        supersample: Sub-voxel samples per axis. ``1`` carves whole voxels; ``2`` or more
            resolves the boundary, at a cost cubic in this value.
        radius_scale: Scale on every node radius. Above ``1.0`` it widens the carve, which is
            how the 83% coverage of a segmentation is traded for a wider static loss.

    Raises:
        ValueError: If the twin carries no centerline, or ``supersample`` is below one.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415 - build-time only

    if int(supersample) < 1:
        raise ValueError(f"supersample must be at least 1, got {supersample}")
    points_mm, radii_mm = centerline_in_volume_mm(patient, radius_scale=radius_scale)

    spacing_xyz = np.asarray(patient.spacing_xyz_mm, dtype=np.float64)
    shape_zyx = patient.shape_zyx
    occupancy = np.zeros(shape_zyx, dtype=np.float32)

    # Only the tube's bounding box can be occupied, and it is a small part of a torso scan.
    lower_mm = (points_mm - radii_mm[:, None]).min(axis=0)
    upper_mm = (points_mm + radii_mm[:, None]).max(axis=0)
    lower = np.maximum(np.floor(lower_mm / spacing_xyz).astype(np.int64) - 1, 0)
    upper = np.minimum(
        np.ceil(upper_mm / spacing_xyz).astype(np.int64) + 2,
        np.asarray(shape_zyx[::-1], dtype=np.int64),
    )
    if np.any(upper <= lower):
        return occupancy

    grid_x = np.arange(lower[0], upper[0], dtype=np.float64)
    grid_y = np.arange(lower[1], upper[1], dtype=np.float64)
    tree = cKDTree(points_mm)
    steps = (np.arange(supersample, dtype=np.float64) + 0.5) / float(supersample) - 0.5
    weight = 1.0 / float(supersample**3)

    # A z-slab at a time: the whole box supersampled is several hundred million query points,
    # which is a needless allocation when the reduction is per voxel anyway.
    for z_index in range(int(lower[2]), int(upper[2])):
        slab = np.zeros((len(grid_y), len(grid_x)), dtype=np.float32)
        for dz in steps:
            z_mm = (float(z_index) + dz) * spacing_xyz[2]
            for dy in steps:
                for dx in steps:
                    sample_x, sample_y = np.meshgrid(
                        (grid_x + dx) * spacing_xyz[0],
                        (grid_y + dy) * spacing_xyz[1],
                        indexing="xy",
                    )
                    samples = np.stack(
                        (sample_x.ravel(), sample_y.ravel(), np.full(sample_x.size, z_mm)),
                        axis=-1,
                    )
                    distance, nearest = tree.query(samples, k=1, workers=-1)
                    inside = distance <= radii_mm[nearest]
                    slab += weight * inside.reshape(sample_x.shape).astype(np.float32)
        occupancy[z_index, lower[1] : upper[1], lower[0] : upper[0]] = slab
    return occupancy


def carve_lumen(
    patient: PatientVolume,
    *,
    supersample: int = 1,
    radius_scale: float = 1.0,
    include_segmentation: bool = False,
    donor_quantiles: tuple[float, float] = DONOR_QUANTILES,
) -> CarveResult:
    """Carve the contrast lumen out of ``patient.mu_volume`` and report what was taken.

    Args:
        patient: Loaded volume. Its own ``mu_volume`` is not modified.
        supersample: Sub-voxel samples per axis for the tube boundary.
        radius_scale: Scale on the node radii, widening or narrowing the tube.
        include_segmentation: Carve the union of the tube and the twin's ``vessel_mask`` rather
            than the tube alone. Off by default: the add-back can only cover the tube, so the
            union's extra voxels are removed and never restored. With no mask declared this
            changes nothing but the reported coverage.
        donor_quantiles: Quantiles of the shell attenuation bounding the inpaint's donors.

    Raises:
        ValueError: If the twin carries no centerline, the quantiles are not ordered inside
            ``[0, 1]``, a declared mask does not match the volume, or the carve is empty.
    """
    low, high = (float(value) for value in donor_quantiles)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"donor_quantiles must satisfy 0 <= low < high <= 1, got {donor_quantiles}")

    occupancy = tube_occupancy(patient, supersample=supersample, radius_scale=radius_scale)
    tube_covers_mask = None
    mask_path = patient.twin.artifacts.get("vessel_mask")
    if mask_path is not None:
        segmentation = np.asarray(np.load(mask_path)).astype(bool)
        if segmentation.shape != occupancy.shape:
            raise ValueError(f"vessel_mask shape {segmentation.shape} does not match volume {occupancy.shape}")
        if segmentation.any():
            # Reported whether or not the union is carved, because it is the number that says how
            # much contrast the centerline model cannot represent at all.
            tube_covers_mask = float((occupancy > 0.0)[segmentation].mean())
        if include_segmentation:
            occupancy = np.maximum(occupancy, segmentation.astype(np.float32))

    carved_region = occupancy > 0.0
    if not carved_region.any():
        raise ValueError("the carve region is empty; check the centerline placement against the volume")

    mu = np.asarray(patient.mu_volume, dtype=np.float32)
    band = _donor_band(mu, carved_region, quantiles=(low, high))
    fill = _inpaint_from_donors(mu, carved_region, band)

    # Fractional at the boundary: a voxel half inside keeps half its original attenuation.
    carved = mu * (1.0 - occupancy) + fill * occupancy

    voxel_ml = float(np.prod(patient.spacing_zyx_mm)) * 1.0e-3
    return CarveResult(
        mu_volume=carved,
        occupancy=occupancy,
        lumen_mu_per_mm=float(mu[carved_region].mean()),
        fill_mu_per_mm=float(fill[carved_region].mean()),
        donor_band_mu_per_mm=band,
        carved_voxels=int(carved_region.sum()),
        carved_volume_ml=float(occupancy.sum()) * voxel_ml,
        tube_covers_mask=tube_covers_mask,
    )


def node_contrast_mu_per_mm(
    patient: PatientVolume,
    result: CarveResult,
    *,
    radius_scale: float = 1.0,
) -> np.ndarray:
    """Attenuation to composite back at each centerline node, measured per node.

    :attr:`CarveResult.contrast_mu_per_mm` is one number for the whole tree, and the lumen is not
    one number: on ``s0011`` its attenuation has a coefficient of variation of 0.37, from beam
    hardening, bolus timing during the scan and partial volume at the wall. Putting a uniform
    value back therefore leaves a residual that is dominated not by the carve's geometry but by
    that spread -- which is worth knowing, because it means refining the carve boundary would not
    have helped.

    Each carved voxel is attributed to its nearest centerline node, and the node's coefficient is
    what the carve actually took there: the mean original attenuation minus the mean fill. So the
    rest state reproduces the scan node by node rather than on average, and the bolus model scales
    a profile that already carries the anatomy's own variation.

    Returns:
        ``(nodes,)`` attenuation per millimetre, zero at nodes no carved voxel was nearest to.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415 - build-time only

    points_mm, _ = centerline_in_volume_mm(patient, radius_scale=radius_scale)
    carved = result.occupancy > 0.5
    indices = np.argwhere(carved)
    if not len(indices):
        return np.zeros(len(points_mm))

    # argwhere gives ZYX; the centerline is in XYZ volume millimetres.
    spacing_xyz = np.asarray(patient.spacing_xyz_mm, dtype=np.float64)
    voxel_centres_mm = indices[:, ::-1].astype(np.float64) * spacing_xyz
    _, nearest = cKDTree(points_mm).query(voxel_centres_mm, k=1, workers=-1)

    original = np.asarray(patient.mu_volume, dtype=np.float64)[carved]
    fill = np.asarray(result.mu_volume, dtype=np.float64)[carved]
    counts = np.bincount(nearest, minlength=len(points_mm))
    removed = np.bincount(nearest, weights=original - fill, minlength=len(points_mm))
    return np.divide(removed, counts, out=np.zeros(len(points_mm)), where=counts > 0)


def _donor_band(mu: np.ndarray, region: np.ndarray, *, quantiles: tuple[float, float]) -> tuple[float, float]:
    """Attenuation range the inpaint may draw from, measured on the shell around ``region``.

    Taken from the vessel's own surroundings rather than a fixed Hounsfield window, because
    what the aorta borders is a property of the anatomy and the scanner, not of this code. The
    interquartile range of that shell is the local soft tissue; the tails are the lung on one
    side and the vertebral body on the other, which are exactly the donors to refuse.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure  # noqa: PLC0415

    shell = binary_dilation(region, structure=generate_binary_structure(3, 1), iterations=SHELL_VOXELS) & ~region
    values = mu[shell] if shell.any() else mu[~region]
    return (float(np.quantile(values, quantiles[0])), float(np.quantile(values, quantiles[1])))


def _inpaint_from_donors(mu: np.ndarray, region: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Fill ``region`` from the nearest voxel whose attenuation sits inside ``band``.

    A Euclidean distance transform over the eligible donors gives, for every voxel at once,
    the index of its closest acceptable neighbour, so the fill is local by construction and
    costs one pass rather than a per-voxel search. It runs on a halo around the carve instead
    of the whole torso, which is two orders of magnitude fewer voxels.
    """
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415

    box = _halo(region, INPAINT_MARGIN_VOXELS, mu.shape)
    local_mu = mu[box]
    local_region = region[box]
    eligible = (~local_region) & (local_mu >= band[0]) & (local_mu <= band[1])

    fill = np.array(mu, dtype=np.float32)
    if not eligible.any():
        # Nothing in the halo passes the band, so fall back to its median rather than leaving
        # the lumen in place, which would double-count against the composited tube.
        fill[region] = float(np.median(local_mu[~local_region])) if (~local_region).any() else 0.0
        return fill

    _, indices = distance_transform_edt(~eligible, return_indices=True)
    local_fill = local_mu[tuple(indices)]
    fill[box] = np.where(local_region, local_fill, local_mu)
    return fill


def _halo(region: np.ndarray, margin: int, shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Bounding box of ``region`` grown by ``margin`` voxels and clipped to ``shape``."""
    bounds = []
    for axis, extent in enumerate(shape):
        along = np.any(region, axis=tuple(index for index in range(region.ndim) if index != axis))
        occupied = np.nonzero(along)[0]
        start = max(int(occupied[0]) - margin, 0)
        stop = min(int(occupied[-1]) + margin + 1, extent)
        bounds.append(slice(start, stop))
    return tuple(bounds)


def carve_residual(
    patient: PatientVolume,
    result: CarveResult,
    *,
    axis: int = 0,
) -> dict[str, float]:
    """How much line integral the carve moved, before any deformation is composited.

    The validation the carve has to pass is that putting the vessel back where it already was
    reproduces the original image. This is that test without a renderer: parallel line
    integrals along one volume axis, comparing the original volume against the carved volume
    with the tube's own attenuation restored at ``contrast_mu_per_mm``. The residual it
    reports is in the same units as the effect being added, where one millimetre of wall
    motion is about 0.002, so the two can be compared directly.

    Args:
        patient: The volume the carve was built from.
        result: What :func:`carve_lumen` produced.
        axis: Volume axis to integrate along, ZYX.

    Returns:
        ``max``, ``mean`` and ``rms`` of the absolute residual, plus the ``signal`` a
        millimetre of wall motion would produce, for scale.
    """
    step_mm = float(patient.spacing_zyx_mm[axis])
    original = np.asarray(patient.mu_volume, dtype=np.float64).sum(axis=axis) * step_mm
    restored = (result.mu_volume + result.occupancy * result.contrast_mu_per_mm).sum(axis=axis) * step_mm
    residual = np.abs(restored - original)
    return {
        "max": float(residual.max()),
        "mean": float(residual.mean()),
        "rms": float(np.sqrt(np.mean(np.square(residual)))),
        # One millimetre of wall displacing tissue at the lumen's own contrast.
        "signal": float(result.contrast_mu_per_mm * 1.0),
    }


__all__ = [
    "DONOR_QUANTILES",
    "CarveResult",
    "carve_lumen",
    "carve_residual",
    "centerline_in_volume_mm",
    "tube_occupancy",
    "volume_scale",
]
