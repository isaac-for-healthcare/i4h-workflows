from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import warp as wp


def apply_vessel_boost(mu_volume: np.ndarray, vessel_mask: np.ndarray, boost_factor: float = 8.0) -> np.ndarray:
    if mu_volume.shape != vessel_mask.shape:
        raise ValueError(f"Shape mismatch: mu_volume {mu_volume.shape} vs vessel_mask {vessel_mask.shape}")
    out = mu_volume.astype(np.float32, copy=True)
    out[vessel_mask.astype(bool)] *= boost_factor
    return out


@dataclass
class CenterlineGraph:
    points: np.ndarray
    radii: np.ndarray
    edges: np.ndarray
    branch_ids: np.ndarray | None = None


def vessel_mask_from_hu(
    hu_zyx: np.ndarray,
    hu_threshold: float = 200.0,
    min_component_voxels: int = 500,
) -> np.ndarray:
    from scipy import ndimage as ndi

    mask = (hu_zyx >= hu_threshold).astype(np.uint8)
    labeled, n_labels = ndi.label(mask)
    if n_labels > 0:
        sizes = ndi.sum(mask, labeled, range(1, n_labels + 1))
        keep = np.array(sizes) >= min_component_voxels
        keep_labels = np.where(keep)[0] + 1
        mask = np.isin(labeled, keep_labels).astype(np.uint8)
    return mask


TOTALSEG_VESSEL_TERRITORY_MAP: dict[int, str] = {
    52: "aortic",
    54: "aortic",
    55: "cerebral",
    56: "cerebral",
    57: "cerebral",
    58: "cerebral",
    65: "peripheral",
    66: "peripheral",
}

TOTALSEG_CORONARY_LABEL: int = 1


@dataclass
class VesselSegmentationResult:
    territory_masks: dict[str, np.ndarray]
    combined_mask: np.ndarray
    label_map: np.ndarray
    spacing_zyx_mm: tuple[float, float, float]

    def get_territory(self, territory: str) -> np.ndarray:
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
    try:
        import nibabel as nib
        from totalsegmentator.python_api import totalsegmentator as _ts_run
    except ImportError as exc:
        raise ImportError("totalsegmentator and nibabel are required. Install with: pip install totalsegmentator nibabel") from exc

    from pathlib import Path as _Path

    if hu_zyx is None and nifti_input is None:
        raise ValueError("Provide either hu_zyx or nifti_input.")

    if nifti_input is None:
        sz, sy, sx = spacing_zyx_mm
        affine = np.diag([sx, sy, sz, 1.0]).astype(np.float64)
        data_xyz = hu_zyx.astype(np.int16).transpose(2, 1, 0)
        nifti_input = nib.Nifti1Image(data_xyz, affine)

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
            seg_nifti = nib.load(str(cache_path))
        if include_coronary and coronary_cache_path.exists():
            coronary_nifti = nib.load(str(coronary_cache_path))

    if seg_nifti is None:
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

    if include_coronary and coronary_nifti is None:
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

    seg_xyz: np.ndarray = np.asarray(seg_nifti.dataobj, dtype=np.int16)
    label_zyx: np.ndarray = seg_xyz.transpose(2, 1, 0).copy()

    hdr_zooms = seg_nifti.header.get_zooms()
    recovered_spacing_zyx = (float(hdr_zooms[2]), float(hdr_zooms[1]), float(hdr_zooms[0]))

    territory_masks: dict[str, np.ndarray] = {}
    for label_id, territory in TOTALSEG_VESSEL_TERRITORY_MAP.items():
        component = (label_zyx == label_id).astype(np.uint8)
        if component.any():
            if territory not in territory_masks:
                territory_masks[territory] = np.zeros(label_zyx.shape, dtype=np.uint8)
            np.maximum(territory_masks[territory], component, out=territory_masks[territory])

    if include_coronary and coronary_nifti is not None:
        cor_xyz = np.asarray(coronary_nifti.dataobj, dtype=np.int16)
        cor_zyx = cor_xyz.transpose(2, 1, 0).copy()
        coronary_mask = (cor_zyx == TOTALSEG_CORONARY_LABEL).astype(np.uint8)
        if coronary_mask.any():
            territory_masks["coronary"] = coronary_mask

    combined = np.zeros(label_zyx.shape, dtype=np.uint8)
    for mask in territory_masks.values():
        np.maximum(combined, mask, out=combined)

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
            pass
        except Exception:
            pass

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
    try:
        import warp as wp
    except ImportError as exc:
        raise ImportError("NVIDIA Warp is required. Install with: pip install warp-lang") from exc

    if vessel_mask.sum() == 0:
        raise ValueError("vessel_mask contains no vessel voxels.")

    sz, sy, sx = spacing_zyx_mm
    ox, oy, oz = origin_xyz_mm

    verts_mm: np.ndarray
    faces: np.ndarray

    _have_vtk = False
    try:
        import vtk  # noqa: F401

        _have_vtk = True
    except ImportError:
        pass

    if _have_vtk:
        import vtk
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
    else:
        try:
            from skimage.measure import marching_cubes  # type: ignore
        except ImportError as exc:
            raise ImportError("Need vtk or scikit-image for mesh extraction.") from exc

        mask_f32 = (vessel_mask > 0).astype(np.float32)
        verts_vox, faces_raw, _, _ = marching_cubes(
            mask_f32,
            level=0.5,
            spacing=(sz, sy, sx),
            allow_degenerate=False,
            method="lewiner",
        )
        verts_mm = np.column_stack(
            [
                verts_vox[:, 2] + ox,
                verts_vox[:, 1] + oy,
                verts_vox[:, 0] + oz,
            ]
        ).astype(np.float32)
        faces = faces_raw.astype(np.int32)

    verts_m = verts_mm / 1000.0
    wp.init()
    return wp.Mesh(
        points=wp.array(verts_m, dtype=wp.vec3, device=device),
        indices=wp.array(faces.flatten(), dtype=int, device=device),
    )


def ct_coords_to_voxel(
    pos_mm: np.ndarray,
    origin_xyz_mm: tuple[float, float, float],
    spacing_zyx_mm: tuple[float, float, float],
) -> np.ndarray:
    ox, oy, oz = origin_xyz_mm
    sz, sy, sx = spacing_zyx_mm
    origin = np.array([ox, oy, oz], dtype=np.float32)
    spacing_xyz = np.array([sx, sy, sz], dtype=np.float32)
    return (np.asarray(pos_mm, dtype=np.float32) - origin) / spacing_xyz


def extract_centerlines(
    vessel_mask: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    seed_point_xyz: tuple[float, float, float] | None = None,
) -> CenterlineGraph:
    try:
        import vtk
        from vmtk import vmtkscripts  # type: ignore
    except ImportError as exc:
        raise ImportError("VMTK is required for centerline extraction.") from exc

    sz, sy, sx = spacing_zyx_mm
    mask_uint8 = (vessel_mask > 0).astype(np.uint8)
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(mask_uint8.shape[2], mask_uint8.shape[1], mask_uint8.shape[0])
    vtk_image.SetSpacing(sx, sy, sz)
    vtk_image.SetOrigin(0.0, 0.0, 0.0)
    flat = mask_uint8.ravel()
    arr = vtk.util.numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    arr.SetName("mask")
    vtk_image.GetPointData().SetScalars(arr)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_image)
    mc.SetValue(0, 0.5)
    mc.Update()
    surface = mc.GetOutput()

    net_extraction = vmtkscripts.vmtkNetworkExtraction()
    net_extraction.Surface = surface
    net_extraction.Execute()

    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = surface
    if seed_point_xyz is not None:
        centerlines.SeedSelectorName = "pointlist"
        centerlines.SourcePoints = list(seed_point_xyz)
    else:
        centerlines.SeedSelectorName = "openprofiles"
    centerlines.Execute()

    cl_polydata = centerlines.Centerlines
    n_pts = cl_polydata.GetNumberOfPoints()
    points = np.array([cl_polydata.GetPoint(i) for i in range(n_pts)], dtype=np.float32)
    radius_array = cl_polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if radius_array is not None:
        radii = np.array([radius_array.GetValue(i) for i in range(n_pts)], dtype=np.float32)
    else:
        radii = np.ones(n_pts, dtype=np.float32)

    edges = []
    for i in range(cl_polydata.GetNumberOfCells()):
        cell = cl_polydata.GetCell(i)
        n_cell_pts = cell.GetNumberOfPoints()
        for j in range(n_cell_pts - 1):
            edges.append([cell.GetPointId(j), cell.GetPointId(j + 1)])
    edges = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
    return CenterlineGraph(points=points, radii=radii, edges=edges)


def compute_arrival_map(
    centerline: CenterlineGraph,
    injection_point_xyz: np.ndarray,
    flow_speed_mm_per_s: float = 200.0,
) -> np.ndarray:
    import heapq

    pts = centerline.points
    edges = centerline.edges
    n = len(pts)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for edge in edges:
        a, b = int(edge[0]), int(edge[1])
        dist = float(np.linalg.norm(pts[a] - pts[b]))
        travel_time = dist / flow_speed_mm_per_s
        adj[a].append((b, travel_time))
        adj[b].append((a, travel_time))

    injection = np.asarray(injection_point_xyz, dtype=np.float32)
    dists_to_injection = np.linalg.norm(pts - injection, axis=1)
    source = int(np.argmin(dists_to_injection))

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


def gamma_variate(
    t: np.ndarray,
    alpha: float = 3.0,
    beta: float = 1.5,
    c_peak: float = 1.0,
) -> np.ndarray:
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
    out = mu_tissue.astype(np.float32, copy=True)
    mask = vessel_mask.astype(bool)
    vessel_indices = np.argwhere(mask)
    if len(vessel_indices) == 0 or len(centerline.points) == 0:
        return out

    sz, sy, sx = spacing_zyx_mm
    vessel_mm = np.zeros((len(vessel_indices), 3), dtype=np.float32)
    vessel_mm[:, 0] = vessel_indices[:, 2] * sx
    vessel_mm[:, 1] = vessel_indices[:, 1] * sy
    vessel_mm[:, 2] = vessel_indices[:, 0] * sz

    from scipy.spatial import cKDTree  # type: ignore

    tree = cKDTree(centerline.points)
    _, nearest_idx = tree.query(vessel_mm)
    voxel_arrival = arrival_times[nearest_idx]
    c = gamma_variate(t - voxel_arrival, alpha=alpha, beta=beta)
    zz = vessel_indices[:, 0]
    yy = vessel_indices[:, 1]
    xx = vessel_indices[:, 2]
    out[zz, yy, xx] += delta_mu * c
    return out
