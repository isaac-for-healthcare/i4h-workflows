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

"""Vasculature utilities: vessel boosting, centerline extraction, and bolus dynamics.

This module provides:
- ``apply_vessel_boost``: Increase vessel μ to make them visible in standard DRR
- ``vessel_mask_from_hu``: Threshold-based vessel segmentation from HU volume
- ``extract_centerlines``: VMTK-based vessel centerline extraction
- ``extract_vessel_mesh``: Convert vessel mask → closed triangle mesh → wp.Mesh
- ``compute_arrival_map``: Dijkstra shortest-path arrival times on the centerline graph
- ``ct_coords_to_voxel``: Convert physical mm coordinates → voxel indices
- ``gamma_variate``: Gamma-variate bolus concentration model C(t)
- ``build_contrast_volume``: Per-frame μ update: μ(v,t) = μ_tissue + Δμ·C(t − T(v))
- ``vessel_mask_from_totalsegmentator``: Named territory vessel masks via TotalSegmentator nnU-Net
- ``get_vessel_mask``: Unified entry point — TotalSegmentator with HU threshold fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import warp as wp


def apply_vessel_boost(
    mu_volume: np.ndarray,
    vessel_mask: np.ndarray,
    boost_factor: float = 8.0,
) -> np.ndarray:
    """Boost vessel attenuation to make them visible in standard DRR.

    Multiplies μ inside the vessel mask by ``boost_factor``, simulating the
    increased X-ray absorption from iodinated contrast or simply making CTA
    vessels more prominent in a non-subtracted projection.

    Args:
        mu_volume: 3D float32 array (Z, Y, X) of linear attenuation coefficients.
        vessel_mask: 3D boolean or binary array of the same shape. Non-zero voxels
            are treated as vessel interior.
        boost_factor: Multiplicative factor applied inside the mask. A value of 8
            approximates the contrast enhancement from full iodine concentration.

    Returns:
        New 3D float32 array with boosted vessel regions.
    """
    if mu_volume.shape != vessel_mask.shape:
        raise ValueError(f"Shape mismatch: mu_volume {mu_volume.shape} vs vessel_mask {vessel_mask.shape}")

    out = mu_volume.astype(np.float32, copy=True)
    mask = vessel_mask.astype(bool)
    out[mask] *= boost_factor
    return out


# ---------------------------------------------------------------------------
# VMTK centerline extraction
# ---------------------------------------------------------------------------


@dataclass
class CenterlineGraph:
    """Vessel centerline representation.

    Attributes:
        points: (N, 3) array of centerline point coordinates in mm (X, Y, Z).
        radii: (N,) array of estimated vessel radii at each point.
        edges: (M, 2) array of edge indices connecting centerline points.
        branch_ids: (N,) optional array labelling which branch each point belongs to.
    """

    points: np.ndarray
    radii: np.ndarray
    edges: np.ndarray
    branch_ids: np.ndarray | None = None


def vessel_mask_from_hu(
    hu_zyx: np.ndarray,
    hu_threshold: float = 200.0,
    min_component_voxels: int = 500,
) -> np.ndarray:
    """Segment vessels from a contrast-enhanced CT volume using HU thresholding.

    For contrast-enhanced CTA, iodinated vessels typically appear at HU > 200.
    After thresholding, small isolated components are removed to eliminate noise
    and bone fragments (bone HU > 700 — use ``hu_threshold=200`` with
    ``hu_zyx < 700`` masking to exclude cortical bone if needed).

    Args:
        hu_zyx: 3D HU volume, shape (Z, Y, X), float32 or int16.
        hu_threshold: Minimum HU value to be classified as vessel (default 200).
            Use 150–250 for CTA; lower for non-contrast acquisitions.
        min_component_voxels: Connected components smaller than this are discarded.
            Removes small isolated high-HU voxels (e.g. calcifications, noise).

    Returns:
        Binary mask ``(Z, Y, X)`` uint8, 1 = vessel, 0 = background.
    """
    from scipy import ndimage as ndi

    # Threshold: vessel voxels are bright in CTA
    mask = (hu_zyx >= hu_threshold).astype(np.uint8)

    # Remove connected components smaller than min_component_voxels.
    # This retains the main vessel tree and discards noise specks.
    labeled, n_labels = ndi.label(mask)
    if n_labels > 0:
        sizes = ndi.sum(mask, labeled, range(1, n_labels + 1))
        keep = np.array(sizes) >= min_component_voxels
        keep_labels = np.where(keep)[0] + 1  # 1-indexed
        mask = np.isin(labeled, keep_labels).astype(np.uint8)

    return mask


# ---------------------------------------------------------------------------
# TotalSegmentator-based vessel segmentation
# ---------------------------------------------------------------------------

#: Maps TotalSegmentator ``total`` task label IDs to I4H vascular territories.
#: Arteries only — veins (brachiocephalic_vein, iliac_vena, pulmonary_vein) excluded.
TOTALSEG_VESSEL_TERRITORY_MAP: dict[int, str] = {
    52: "aortic",  # aorta
    54: "aortic",  # brachiocephalic_trunk (arch branch origin)
    55: "cerebral",  # subclavian_artery_right
    56: "cerebral",  # subclavian_artery_left
    57: "cerebral",  # common_carotid_artery_right
    58: "cerebral",  # common_carotid_artery_left  ← LCCA is label 58
    65: "peripheral",  # iliac_artery_left
    66: "peripheral",  # iliac_artery_right
}

#: Coronary arteries come from a separate TotalSegmentator task.
TOTALSEG_CORONARY_LABEL: int = 1


@dataclass
class VesselSegmentationResult:
    """Per-territory binary vessel masks produced by TotalSegmentator.

    Attributes:
        territory_masks: Dict mapping territory name → binary (Z,Y,X) uint8 mask.
            Keys are a subset of ``{"aortic", "cerebral", "peripheral", "coronary"}``.
            When produced by the HU fallback the single key is ``"unknown"``.
        combined_mask: Union of all territory masks — drop-in replacement for the
            old ``vessel_mask_from_hu()`` output.
        label_map: Raw integer label volume (Z,Y,X) from TotalSegmentator.
        spacing_zyx_mm: Voxel spacing of the masks (same as input CT).
    """

    territory_masks: dict[str, np.ndarray]
    combined_mask: np.ndarray
    label_map: np.ndarray
    spacing_zyx_mm: tuple[float, float, float]

    def get_territory(self, territory: str) -> np.ndarray:
        """Return binary mask for a named territory, or zeros if absent."""
        return self.territory_masks.get(territory, np.zeros_like(self.combined_mask))


def vessel_mask_from_totalsegmentator(
    hu_zyx: "np.ndarray | None" = None,
    spacing_zyx_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    nifti_input: Any = None,
    include_coronary: bool = False,
    device: str = "gpu",
    fast: bool = False,
    cache_dir: "str | None" = None,
) -> "VesselSegmentationResult":
    """Segment vessels from a CT volume using TotalSegmentator (nnU-Net).

    Runs TotalSegmentator's ``total`` task to produce named, territory-labelled
    vessel masks. Optionally runs the ``coronary_arteries`` task as a second pass.

    Label → territory mapping (``TOTALSEG_VESSEL_TERRITORY_MAP``):

    +-----------+-----------------------------------------------------------+
    | Territory | TotalSegmentator labels (``total`` task)                   |
    +===========+===========================================================+
    | aortic    | aorta (52), brachiocephalic_trunk (54)                     |
    +-----------+-----------------------------------------------------------+
    | cerebral  | common_carotid_artery_{left,right} (57,58),               |
    |           | subclavian_artery_{left,right} (55,56)                     |
    +-----------+-----------------------------------------------------------+
    | peripheral| iliac_artery_{left,right} (65,66)                         |
    +-----------+-----------------------------------------------------------+
    | coronary  | coronary_arteries (1) — separate task, opt-in             |
    +-----------+-----------------------------------------------------------+

    Args:
        hu_zyx: 3D HU volume (Z,Y,X) float32/int16. Either this or
            ``nifti_input`` must be provided.
        spacing_zyx_mm: Voxel spacing in mm (Z,Y,X). Required when ``hu_zyx``
            is provided so the NIfTI header is built correctly.
        nifti_input: Pre-loaded ``nibabel.Nifti1Image``. Takes priority over
            ``hu_zyx`` when both are supplied.
        include_coronary: Run a second pass with the ``coronary_arteries`` task
            and add a ``"coronary"`` territory mask. Adds ~30 s on GPU.
            Quality is best on ECG-gated CTA; degrades on non-gated scans.
        device: ``"gpu"`` (default) or ``"cpu"``. GPU strongly recommended —
            ``total`` task inference takes ~20 s on GPU, ~5 min on CPU.
        fast: Use TotalSegmentator fast mode (~2x faster, slightly lower
            accuracy for small vessels).
        cache_dir: Directory to cache the raw TotalSegmentator NIfTI output.
            If a cached result exists it is loaded instead of re-running
            inference. Pass ``None`` to disable caching.

    Returns:
        :class:`VesselSegmentationResult` with per-territory binary masks,
        a combined mask, the raw integer label volume, and voxel spacing.

    Raises:
        ImportError: If ``totalsegmentator`` or ``nibabel`` is not installed.
        ValueError: If neither ``hu_zyx`` nor ``nifti_input`` is provided.

    Example::

        from fluorosim.vasculature import vessel_mask_from_totalsegmentator
        from fluorosim.ct.dicom_ingest import load_dicom_series_hu

        ct = load_dicom_series_hu("/path/to/dicom/")
        result = vessel_mask_from_totalsegmentator(
            hu_zyx=ct.hu_zyx,
            spacing_zyx_mm=ct.spacing_zyx_mm,
        )

        aortic_mask   = result.get_territory("aortic")
        cerebral_mask = result.get_territory("cerebral")  # includes LCCA

        # Drop-in replacement for the old combined mask
        vessel_mesh = extract_vessel_mesh(result.combined_mask, ct.spacing_zyx_mm)
    """
    try:
        import nibabel as nib
        from totalsegmentator.python_api import totalsegmentator as _ts_run
    except ImportError as exc:
        raise ImportError(
            "totalsegmentator and nibabel are required. " "Install with: pip install totalsegmentator nibabel"
        ) from exc

    from pathlib import Path as _Path

    if hu_zyx is None and nifti_input is None:
        raise ValueError("Provide either hu_zyx or nifti_input.")

    # ---- Build nibabel NIfTI from numpy if needed --------------------------
    if nifti_input is None:
        sz, sy, sx = spacing_zyx_mm
        affine = np.diag([sx, sy, sz, 1.0]).astype(np.float64)
        # NIfTI data is stored (X,Y,Z); transpose from (Z,Y,X)
        data_xyz = hu_zyx.astype(np.int16).transpose(2, 1, 0)
        nifti_input = nib.Nifti1Image(data_xyz, affine)

    # ---- Cache paths -------------------------------------------------------
    seg_nifti = None
    coronary_nifti = None
    cache_path = None
    coronary_cache_path = None

    if cache_dir is not None:
        _cache = _Path(cache_dir)
        _cache.mkdir(parents=True, exist_ok=True)
        cache_path = _cache / "totalseg_total.nii.gz"
        coronary_cache_path = _cache / "totalseg_coronary.nii.gz"
        if cache_path.exists():
            print(f"[vessel_mask_from_totalsegmentator] Loading cached segmentation: {cache_path}")
            seg_nifti = nib.load(str(cache_path))
        if include_coronary and coronary_cache_path.exists():
            print(f"[vessel_mask_from_totalsegmentator] Loading cached coronary: {coronary_cache_path}")
            coronary_nifti = nib.load(str(coronary_cache_path))

    # ---- Run TotalSegmentator (total task) ---------------------------------
    if seg_nifti is None:
        print("[vessel_mask_from_totalsegmentator] Running TotalSegmentator (total task) ...")
        seg_nifti = _ts_run(
            input=nifti_input,
            task="total",
            device=device,
            fast=fast,
            quiet=True,
            skip_saving=True,
        )
        if cache_path is not None:
            nib.save(seg_nifti, str(cache_path))
            print(f"[vessel_mask_from_totalsegmentator] Cached to: {cache_path}")

    # ---- Run coronary task (optional) --------------------------------------
    if include_coronary and coronary_nifti is None:
        print("[vessel_mask_from_totalsegmentator] Running TotalSegmentator (coronary_arteries task) ...")
        coronary_nifti = _ts_run(
            input=nifti_input,
            task="coronary_arteries",
            device=device,
            fast=fast,
            quiet=True,
            skip_saving=True,
        )
        if coronary_cache_path is not None:
            nib.save(coronary_nifti, str(coronary_cache_path))

    # ---- Extract label volume: NIfTI (X,Y,Z) → pipeline (Z,Y,X) -----------
    seg_xyz: np.ndarray = np.asarray(seg_nifti.dataobj, dtype=np.int16)
    label_zyx: np.ndarray = seg_xyz.transpose(2, 1, 0).copy()

    # Recover spacing from NIfTI header (handles both input paths)
    hdr_zooms = seg_nifti.header.get_zooms()  # (sx, sy, sz)
    recovered_spacing_zyx = (float(hdr_zooms[2]), float(hdr_zooms[1]), float(hdr_zooms[0]))

    # ---- Build per-territory masks -----------------------------------------
    territory_masks: dict[str, np.ndarray] = {}
    for label_id, territory in TOTALSEG_VESSEL_TERRITORY_MAP.items():
        component = (label_zyx == label_id).astype(np.uint8)
        if component.any():
            if territory not in territory_masks:
                territory_masks[territory] = np.zeros(label_zyx.shape, dtype=np.uint8)
            np.maximum(territory_masks[territory], component, out=territory_masks[territory])

    # ---- Coronary mask (optional) ------------------------------------------
    if include_coronary and coronary_nifti is not None:
        cor_xyz = np.asarray(coronary_nifti.dataobj, dtype=np.int16)
        cor_zyx = cor_xyz.transpose(2, 1, 0).copy()
        coronary_mask = (cor_zyx == TOTALSEG_CORONARY_LABEL).astype(np.uint8)
        if coronary_mask.any():
            territory_masks["coronary"] = coronary_mask

    # ---- Combined mask (union of all territories) --------------------------
    combined = np.zeros(label_zyx.shape, dtype=np.uint8)
    for m in territory_masks.values():
        np.maximum(combined, m, out=combined)

    print(
        f"[vessel_mask_from_totalsegmentator] Found territories: {list(territory_masks.keys())} "
        f"— {int(combined.sum()):,} vessel voxels total"
    )

    return VesselSegmentationResult(
        territory_masks=territory_masks,
        combined_mask=combined,
        label_map=label_zyx,
        spacing_zyx_mm=recovered_spacing_zyx,
    )


def get_vessel_mask(
    hu_zyx: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    use_totalsegmentator: bool = True,
    include_coronary: bool = False,
    device: str = "gpu",
    fast: bool = False,
    cache_dir: "str | None" = None,
    hu_threshold: float = 200.0,
    min_component_voxels: int = 500,
) -> "VesselSegmentationResult":
    """Unified vessel segmentation — TotalSegmentator with HU threshold fallback.

    Attempts TotalSegmentator first (preferred: named territories, no bone
    bleed-through). Falls back to ``vessel_mask_from_hu()`` if TotalSegmentator
    is not installed or fails, wrapping the result in a
    :class:`VesselSegmentationResult` with a single ``"unknown"`` territory key.

    Args:
        hu_zyx: 3D HU volume (Z,Y,X).
        spacing_zyx_mm: Voxel spacing in mm (Z,Y,X).
        use_totalsegmentator: Set ``False`` to force HU threshold fallback.
        include_coronary: Run the coronary_arteries task too (TotalSegmentator only).
        device: ``"gpu"`` or ``"cpu"`` for TotalSegmentator inference.
        fast: Use TotalSegmentator fast mode.
        cache_dir: Cache directory for TotalSegmentator outputs.
        hu_threshold: HU threshold for fallback segmentation.
        min_component_voxels: Minimum component size for fallback.

    Returns:
        :class:`VesselSegmentationResult` — same interface regardless of method used.
    """
    if use_totalsegmentator:
        try:
            return vessel_mask_from_totalsegmentator(
                hu_zyx=hu_zyx,
                spacing_zyx_mm=spacing_zyx_mm,
                include_coronary=include_coronary,
                device=device,
                fast=fast,
                cache_dir=cache_dir,
            )
        except ImportError:
            print("[get_vessel_mask] TotalSegmentator not available — " "falling back to HU threshold segmentation.")
        except Exception as exc:
            print(f"[get_vessel_mask] TotalSegmentator failed ({exc}) — " "falling back to HU threshold segmentation.")

    mask = vessel_mask_from_hu(
        hu_zyx=hu_zyx,
        hu_threshold=hu_threshold,
        min_component_voxels=min_component_voxels,
    )
    return VesselSegmentationResult(
        territory_masks={"unknown": mask},
        combined_mask=mask,
        label_map=mask.astype(np.int16),
        spacing_zyx_mm=spacing_zyx_mm,
    )


def extract_vessel_mesh(
    vessel_mask: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    smooth_iterations: int = 20,
    smooth_pass_band: float = 0.1,
    device: str = "cuda",
) -> "wp.Mesh":
    """Convert a binary vessel mask to a smoothed ``wp.Mesh`` in physical metres.

    Pipeline:
        1. VTK marching cubes on the binary mask → raw triangle surface in mm
        2. VTK Windowed Sinc smoothing (removes staircasing artefacts)
        3. Vertex normals computed and outward-pointing (required for
           ``XCathRodSolver`` signed-distance convention with ``sign_scale=1``)
        4. Vertices converted from mm → metres, physical origin applied
        5. Warp ``wp.Mesh`` constructed with auto-built BVH

    Args:
        vessel_mask: Binary (Z, Y, X) uint8 array — 1 = vessel lumen.
        spacing_zyx_mm: Voxel spacing (Z, Y, X) in mm.
        origin_xyz_mm: Physical origin of the CT volume (X, Y, Z) in mm.
            Usually ``CtVolume.origin_xyz_mm``.
        smooth_iterations: VTK Windowed Sinc iteration count (default 20).
            Higher = smoother but slower. 15–30 is the practical range.
        smooth_pass_band: VTK Windowed Sinc pass-band frequency (default 0.1).
            Lower = more smoothing of high-frequency staircasing.
        device: Warp device string (``"cuda"`` or ``"cpu"``).

    Returns:
        ``wp.Mesh`` with BVH built and ready for ``wp.mesh_query_point_sign_normal``.
        Vertex positions are in **metres**, sharing the same coordinate frame as
        ``XCathRodSolver`` physics (which also operates in metres).

    Raises:
        ImportError: If ``vtk`` is not installed.
        ValueError: If the mask is empty (no vessel voxels).
    """
    try:
        import warp as wp
    except ImportError as e:
        raise ImportError("NVIDIA Warp is required. Install with: pip install warp-lang") from e

    if vessel_mask.sum() == 0:
        raise ValueError("vessel_mask contains no vessel voxels. " "Check HU threshold or segmentation input.")

    sz, sy, sx = spacing_zyx_mm
    ox, oy, oz = origin_xyz_mm

    verts_mm: np.ndarray
    faces: np.ndarray

    # ---- Path A: VTK (preferred — provides Windowed Sinc smoothing) -----------
    _have_vtk = False
    try:
        import vtk  # noqa: F401

        _have_vtk = True
    except ImportError:
        pass

    if _have_vtk:
        from vtk.util.numpy_support import vtk_to_numpy

        mask_uint8 = (vessel_mask > 0).astype(np.uint8)
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(mask_uint8.shape[2], mask_uint8.shape[1], mask_uint8.shape[0])
        vtk_image.SetSpacing(sx, sy, sz)
        vtk_image.SetOrigin(ox, oy, oz)
        flat = mask_uint8.ravel()
        arr = vtk.util.numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        arr.SetName("mask")
        vtk_image.GetPointData().SetScalars(arr)

        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(vtk_image)
        mc.SetValue(0, 0.5)
        mc.ComputeNormalsOn()
        mc.Update()

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(mc.GetOutput())
        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.SetPassBand(smooth_pass_band)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(smoother.GetOutput())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        final = normals.GetOutput()

        verts_mm = vtk_to_numpy(final.GetPoints().GetData()).reshape(-1, 3).astype(np.float32)
        polys_flat = vtk_to_numpy(final.GetPolys().GetData())
        n_tris = final.GetNumberOfCells()
        faces = polys_flat.reshape(n_tris, 4)[:, 1:].astype(np.int32)
        print(f"[extract_vessel_mesh] VTK path: {len(verts_mm):,} verts, {n_tris:,} tris")

    else:
        # ---- Path B: scikit-image marching_cubes (no smoothing pass) ----------
        try:
            from skimage.measure import marching_cubes  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Neither VTK nor scikit-image is available. "
                "Install one: pip install vtk  or  pip install scikit-image"
            ) from e

        # skimage.marching_cubes operates in voxel index space.
        # We scale the output into physical mm by multiplying by spacing.
        mask_f32 = (vessel_mask > 0).astype(np.float32)
        verts_vox, faces_raw, _, _ = marching_cubes(
            mask_f32,
            level=0.5,
            spacing=(sz, sy, sx),  # converts voxel indices → mm
            allow_degenerate=False,
            method="lewiner",
        )
        # marching_cubes returns (Z, Y, X) based on spacing order; reorder to (X, Y, Z)
        verts_mm = np.column_stack(
            [
                verts_vox[:, 2] + ox,  # X mm
                verts_vox[:, 1] + oy,  # Y mm
                verts_vox[:, 0] + oz,  # Z mm
            ]
        ).astype(np.float32)
        faces = faces_raw.astype(np.int32)
        print(
            f"[extract_vessel_mesh] skimage path: {len(verts_mm):,} verts, " f"{len(faces):,} tris  (no Sinc smoothing)"
        )

    # ---- mm → metres -----------------------------------------------------------
    verts_m = verts_mm / 1000.0

    # ---- Build wp.Mesh with BVH ------------------------------------------------
    wp.init()
    mesh = wp.Mesh(
        points=wp.array(verts_m, dtype=wp.vec3, device=device),
        indices=wp.array(faces.flatten(), dtype=int, device=device),
    )

    print(f"[extract_vessel_mesh] {len(verts_m):,} vertices, {len(faces):,} triangles — " f"BVH built on {device}")
    return mesh


def ct_coords_to_voxel(
    pos_mm: np.ndarray,
    origin_xyz_mm: tuple[float, float, float],
    spacing_zyx_mm: tuple[float, float, float],
) -> np.ndarray:
    """Convert physical mm coordinates (X, Y, Z) to fractional voxel indices (X, Y, Z).

    Used to map catheter rod particle positions (in mm, CT physical frame) to
    the voxel index space expected by the Slang DRR renderer's VolumeInfo.

    The renderer expects positions in the ``(X, Y, Z)`` voxel coordinate system
    where the CT volume occupies ``[0, vol_shape_X] × [0, vol_shape_Y] × [0, vol_shape_Z]``.

    Args:
        pos_mm: ``(N, 3)`` float32 array of positions in mm, columns = (X, Y, Z).
        origin_xyz_mm: CT volume origin (X, Y, Z) in mm — ``CtVolume.origin_xyz_mm``.
        spacing_zyx_mm: Voxel spacing (Z, Y, X) in mm — ``CtVolume.spacing_zyx_mm``.

    Returns:
        ``(N, 3)`` float32 array of fractional voxel indices (X, Y, Z).
    """
    ox, oy, oz = origin_xyz_mm
    sz, sy, sx = spacing_zyx_mm
    origin = np.array([ox, oy, oz], dtype=np.float32)
    spacing_xyz = np.array([sx, sy, sz], dtype=np.float32)  # reorder to XYZ
    return (np.asarray(pos_mm, dtype=np.float32) - origin) / spacing_xyz


def extract_centerlines(
    vessel_mask: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    seed_point_xyz: tuple[float, float, float] | None = None,
) -> CenterlineGraph:
    """Extract vessel centerlines from a binary mask using VMTK.

    Requires the ``vmtk`` package (``pip install vmtk`` or a dedicated conda env).

    Pipeline: mask → marching cubes → VMTK network extraction → centerline
    points, radii, and connectivity.

    Args:
        vessel_mask: 3D binary array (Z, Y, X), non-zero = vessel.
        spacing_zyx_mm: Voxel spacing in mm (Z, Y, X).
        seed_point_xyz: Optional seed point in mm for the source of the
            centerline tree. If None, VMTK auto-detects.

    Returns:
        CenterlineGraph with points, radii, and edges.

    Raises:
        ImportError: If vmtk is not installed.
    """
    try:
        import vtk
        from vmtk import vmtkscripts  # type: ignore
    except ImportError as e:
        raise ImportError(
            "VMTK is required for centerline extraction. "
            "Install via: conda create -n vmtk -c conda-forge python=3.10 vmtk"
        ) from e

    sz, sy, sx = spacing_zyx_mm

    # Convert binary mask to VTK image
    mask_uint8 = (vessel_mask > 0).astype(np.uint8)
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(mask_uint8.shape[2], mask_uint8.shape[1], mask_uint8.shape[0])
    vtk_image.SetSpacing(sx, sy, sz)
    vtk_image.SetOrigin(0.0, 0.0, 0.0)

    flat = mask_uint8.ravel()
    arr = vtk.util.numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    arr.SetName("mask")
    vtk_image.GetPointData().SetScalars(arr)

    # Marching cubes to get surface
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_image)
    mc.SetValue(0, 0.5)
    mc.Update()
    surface = mc.GetOutput()

    # VMTK network extraction
    net_extraction = vmtkscripts.vmtkNetworkExtraction()
    net_extraction.Surface = surface
    net_extraction.Execute()

    # VMTK centerline computation
    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = surface
    if seed_point_xyz is not None:
        centerlines.SeedSelectorName = "pointlist"
        centerlines.SourcePoints = list(seed_point_xyz)
    else:
        centerlines.SeedSelectorName = "openprofiles"
    centerlines.Execute()

    cl_polydata = centerlines.Centerlines

    # Extract points
    n_pts = cl_polydata.GetNumberOfPoints()
    points = np.array([cl_polydata.GetPoint(i) for i in range(n_pts)], dtype=np.float32)

    # Extract radii
    radius_array = cl_polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if radius_array is not None:
        radii = np.array([radius_array.GetValue(i) for i in range(n_pts)], dtype=np.float32)
    else:
        radii = np.ones(n_pts, dtype=np.float32)

    # Extract edges from cell connectivity
    edges = []
    for i in range(cl_polydata.GetNumberOfCells()):
        cell = cl_polydata.GetCell(i)
        n_cell_pts = cell.GetNumberOfPoints()
        for j in range(n_cell_pts - 1):
            edges.append([cell.GetPointId(j), cell.GetPointId(j + 1)])
    edges = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)

    return CenterlineGraph(points=points, radii=radii, edges=edges)


# ---------------------------------------------------------------------------
# Dijkstra arrival map
# ---------------------------------------------------------------------------


def compute_arrival_map(
    centerline: CenterlineGraph,
    injection_point_xyz: np.ndarray,
    flow_speed_mm_per_s: float = 200.0,
) -> np.ndarray:
    """Compute contrast arrival time at each centerline point via Dijkstra.

    Builds a weighted graph from the centerline edges (weight = Euclidean
    distance / flow_speed), finds the node closest to ``injection_point_xyz``,
    and computes shortest-path arrival times from that source.

    Args:
        centerline: CenterlineGraph with points and edges.
        injection_point_xyz: 3D coordinate of the contrast injection site (mm).
        flow_speed_mm_per_s: Assumed constant flow speed along vessels (mm/s).

    Returns:
        (N,) float32 array of arrival times in seconds, one per centerline point.
    """
    import heapq

    pts = centerline.points
    edges = centerline.edges
    n = len(pts)

    # Build adjacency list
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for e in edges:
        a, b = int(e[0]), int(e[1])
        dist = float(np.linalg.norm(pts[a] - pts[b]))
        travel_time = dist / flow_speed_mm_per_s
        adj[a].append((b, travel_time))
        adj[b].append((a, travel_time))

    # Find source node (closest to injection point)
    injection = np.asarray(injection_point_xyz, dtype=np.float32)
    dists_to_injection = np.linalg.norm(pts - injection, axis=1)
    source = int(np.argmin(dists_to_injection))

    # Dijkstra
    arrival = np.full(n, np.inf, dtype=np.float32)
    arrival[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        t, u = heapq.heappop(heap)
        if t > arrival[u]:
            continue
        for v, w in adj[u]:
            t_new = t + w
            if t_new < arrival[v]:
                arrival[v] = t_new
                heapq.heappush(heap, (t_new, v))

    return arrival


# ---------------------------------------------------------------------------
# Gamma-variate bolus model
# ---------------------------------------------------------------------------


def gamma_variate(
    t: np.ndarray,
    alpha: float = 3.0,
    beta: float = 1.5,
    c_peak: float = 1.0,
) -> np.ndarray:
    """Gamma-variate concentration–time curve.

    C(t) = c_peak · (t / t_peak)^α · exp(α · (1 − t/t_peak))

    where t_peak = α · β.

    This is the standard model for indicator dilution in vascular imaging
    (Stewart–Hamilton theory). For t < 0, C = 0.

    Args:
        t: Time array in seconds (can be negative for pre-arrival).
        alpha: Shape parameter (controls rise steepness).
        beta: Scale parameter in seconds.
        c_peak: Peak concentration (dimensionless, typically 1.0).

    Returns:
        Concentration array, same shape as t.
    """
    t = np.asarray(t, dtype=np.float64)
    t_peak = alpha * beta
    c = np.zeros_like(t)
    pos = t > 0
    ratio = t[pos] / t_peak
    c[pos] = c_peak * np.power(ratio, alpha) * np.exp(alpha * (1.0 - ratio))
    return c.astype(np.float32)


def build_contrast_volume(
    mu_tissue: np.ndarray,
    vessel_mask: np.ndarray,
    arrival_times: np.ndarray,
    centerline: CenterlineGraph,
    t: float,
    delta_mu: float = 0.015,
    alpha: float = 3.0,
    beta: float = 1.5,
    spacing_zyx_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Build a contrast-enhanced μ volume for a specific time point.

    For each vessel voxel, finds the nearest centerline point, looks up its
    arrival time T(v), and computes:

        μ(v, t) = μ_tissue(v) + Δμ · C(t − T(v))

    where C is the gamma-variate bolus model.

    Args:
        mu_tissue: Baseline 3D μ volume (Z, Y, X), float32.
        vessel_mask: 3D binary mask (Z, Y, X), same shape.
        arrival_times: Per-centerline-point arrival times from ``compute_arrival_map``.
        centerline: CenterlineGraph used to map voxels to arrival times.
        t: Current time in seconds.
        delta_mu: Maximum μ increase at peak contrast concentration (mm^-1).
        alpha: Gamma-variate shape parameter.
        beta: Gamma-variate scale parameter (seconds).
        spacing_zyx_mm: Voxel spacing in (Z, Y, X) order.

    Returns:
        Updated 3D μ volume with time-dependent contrast enhancement.
    """
    out = mu_tissue.astype(np.float32, copy=True)
    mask = vessel_mask.astype(bool)
    vessel_indices = np.argwhere(mask)  # (K, 3) in ZYX

    if len(vessel_indices) == 0 or len(centerline.points) == 0:
        return out

    sz, sy, sx = spacing_zyx_mm

    # Convert voxel indices to mm (XYZ to match centerline points)
    vessel_mm = np.zeros((len(vessel_indices), 3), dtype=np.float32)
    vessel_mm[:, 0] = vessel_indices[:, 2] * sx  # X
    vessel_mm[:, 1] = vessel_indices[:, 1] * sy  # Y
    vessel_mm[:, 2] = vessel_indices[:, 0] * sz  # Z

    # For each vessel voxel, find nearest centerline point (batched for speed)
    from scipy.spatial import cKDTree  # type: ignore

    tree = cKDTree(centerline.points)
    _, nearest_idx = tree.query(vessel_mm)

    # Look up arrival time for each voxel
    voxel_arrival = arrival_times[nearest_idx]

    # Compute concentration at current time
    c = gamma_variate(t - voxel_arrival, alpha=alpha, beta=beta)

    # Apply μ boost
    zz = vessel_indices[:, 0]
    yy = vessel_indices[:, 1]
    xx = vessel_indices[:, 2]
    out[zz, yy, xx] += delta_mu * c

    return out
