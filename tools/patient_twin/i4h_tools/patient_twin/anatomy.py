# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a lightweight CT-derived patient anatomy USD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from scipy import ndimage
from skimage import measure


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("body threshold produced an empty CT mask")
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _surface_mesh(mask_zyx: np.ndarray, spacing_zyx_mm: np.ndarray, origin_xyz_mm: np.ndarray) -> trimesh.Trimesh:
    vertices_zyx, faces, _normals, _values = measure.marching_cubes(
        np.pad(mask_zyx.astype(np.uint8), 1),
        level=0.5,
        spacing=tuple(float(value) for value in spacing_zyx_mm),
        allow_degenerate=False,
    )
    vertices_zyx -= spacing_zyx_mm
    vertices_xyz_m = (vertices_zyx[:, ::-1] + origin_xyz_mm) * 0.001
    mesh = trimesh.Trimesh(vertices=vertices_xyz_m, faces=faces, process=True)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.45, nu=0.5, iterations=12)
    return mesh


def _material(stage: Usd.Stage, path: str, color: tuple[float, float, float], opacity: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _add_mesh(
    stage: Usd.Stage,
    path: str,
    mesh: trimesh.Trimesh,
    material: UsdShade.Material,
    *,
    double_sided: bool,
) -> None:
    usd_mesh = UsdGeom.Mesh.Define(stage, path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    usd_mesh.CreatePointsAttr([Gf.Vec3f(*(float(value) for value in point)) for point in vertices])
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    usd_mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    usd_mesh.CreateNormalsAttr([Gf.Vec3f(*(float(value) for value in normal)) for normal in normals])
    usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    usd_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    usd_mesh.CreateDoubleSidedAttr().Set(double_sided)
    extent = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    usd_mesh.CreateExtentAttr([Gf.Vec3f(*(float(value) for value in point)) for point in extent])
    UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)


def _make_anterior_cutaway(mesh: trimesh.Trimesh) -> None:
    """Open a window over the chest and abdomen so the vasculature stays visible.

    Mesh coordinates are patient LPS, so anterior is the negative y side.
    """
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    centroids = np.asarray(mesh.triangles_center)
    remove = (
        (np.abs(centroids[:, 0] - center[0]) < 0.34 * extent[0])
        & (centroids[:, 1] < center[1] - 0.02 * extent[1])
        & (np.abs(centroids[:, 2] - center[2]) < 0.42 * extent[2])
    )
    mesh.update_faces(~remove)
    mesh.remove_unreferenced_vertices()


def build_anatomy_usd(
    hu_zyx: np.ndarray,
    vessel_mask_zyx: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    output: Path,
    *,
    surface_step: int = 3,
    threshold_hu: float = -300.0,
    cutaway: bool = True,
) -> None:
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    origin = np.asarray(origin_xyz_mm, dtype=np.float64)
    body_sample = np.asarray(hu_zyx[::surface_step, ::surface_step, ::surface_step]) > threshold_hu
    body_sample = _largest_component(ndimage.binary_closing(body_sample, iterations=2))
    body_sample = ndimage.binary_fill_holes(body_sample)
    body_mesh = _surface_mesh(body_sample, spacing * surface_step, origin)
    if cutaway:
        _make_anterior_cutaway(body_mesh)

    vessel_indices = np.argwhere(vessel_mask_zyx)
    if vessel_indices.size == 0:
        raise ValueError("vessel mask is empty")
    lower = np.maximum(vessel_indices.min(axis=0) - 2, 0)
    upper = np.minimum(vessel_indices.max(axis=0) + 3, vessel_mask_zyx.shape)
    vessel_crop = np.asarray(
        vessel_mask_zyx[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))], dtype=bool
    )
    vessel_origin = origin + lower[::-1] * spacing[::-1]
    vessel_mesh = _surface_mesh(vessel_crop, spacing, vessel_origin)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/PatientAnatomy")
    stage.SetDefaultPrim(root.GetPrim())
    skin = _material(stage, "/PatientAnatomy/Materials/Skin", (0.62, 0.36, 0.28), 0.78)
    artery = _material(stage, "/PatientAnatomy/Materials/Vasculature", (0.75, 0.025, 0.02), 1.0)
    _add_mesh(stage, "/PatientAnatomy/BodySurface", body_mesh, skin, double_sided=False)
    _add_mesh(stage, "/PatientAnatomy/Vasculature", vessel_mesh, artery, double_sided=True)
    stage.GetRootLayer().Save()
