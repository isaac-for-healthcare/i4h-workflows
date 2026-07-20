#!/usr/bin/env python3
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

"""Native Slang-powered catheter viewport with cursor control.

This replaces web widgets with a direct viewport loop:
  - Left pane: Slang DRR fluoroscopy with catheter composited by Beer-Lambert.
  - Right pane: control/telemetry panel with a top-down catheter map.

Controls
--------
Mouse (left-drag):
    Horizontal drag -> insertion/retraction velocity
    Vertical drag   -> clockwise/counterclockwise torque
Keyboard:
    W / S : advance / retract
    A / D : rotate CCW / CW
    SHIFT : boost command magnitude
    SPACE : pause/resume
    R     : reset catheter to initial state
    1/2/3/4 : AP / LAO-45 / Lateral / RAO-30
    Q or ESC: quit
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# On Linux X11 sessions, some OpenCV Qt builds auto-select a platform plugin that
# does not deliver keyboard events (notably on remote/virtual displays), so W/S,
# A/D and 1-4 silently do nothing. Force the xcb (X11) plugin before importing
# cv2. We override when QT_QPA_PLATFORM is unset *or* empty (some sessions export
# it blank). Wayland sessions and any explicit non-empty value are left untouched.
if sys.platform.startswith("linux") and not os.environ.get("WAYLAND_DISPLAY"):
    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("OpenCV is required. Install with: pip install opencv-python") from exc


def _ensure_importable(package: str, *candidate_dirs: str) -> None:
    """Add first valid candidate directory to sys.path for package lookup."""
    import sys

    if importlib.util.find_spec(package) is not None:
        return
    for directory in candidate_dirs:
        if Path(directory).is_dir():
            sys.path.insert(0, directory)
            if importlib.util.find_spec(package) is not None:
                return
    raise ImportError(
        f"Cannot import '{package}'. Install it with pip install -e <repo_root>. " f"Searched: {list(candidate_dirs)}"
    )


_SCRIPT_DIR = Path(__file__).resolve().parent
_FLUORO_ROOT = str(_SCRIPT_DIR.parent)

_ensure_importable("fluorosim", _FLUORO_ROOT)
_ensure_importable("catheter_vasculature_solver")
_ensure_importable("vasculature_digital_twin")

import torch
import warp as wp
from catheter_vasculature_solver import RodConfig, XCathRodSolver, compute_signed_distances  # noqa: E402
from vasculature_digital_twin import extract_vessel_mesh, vessel_mask_from_hu  # noqa: E402

try:
    from fluorosim.rendering.diffdrr_slang_renderer import (  # noqa: E402
        CatheterSegmentData,
        SlangDiffDRRConfig,
        SlangDiffDRRRenderer,
    )
except ImportError:
    _local_renderer_path = _SCRIPT_DIR.parent / "rendering" / "diffdrr_slang_renderer.py"
    _spec = importlib.util.spec_from_file_location("catheter_local_diffdrr", _local_renderer_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot load local renderer fallback: {_local_renderer_path}")
    _local_renderer = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _local_renderer
    _spec.loader.exec_module(_local_renderer)
    CatheterSegmentData = _local_renderer.CatheterSegmentData
    SlangDiffDRRConfig = _local_renderer.SlangDiffDRRConfig
    SlangDiffDRRRenderer = _local_renderer.SlangDiffDRRRenderer

DET_SIZE = 320
PHYSICS_FPS = 30
DT = 1.0 / PHYSICS_FPS
DEFAULT_NUM_SEGMENTS = 40
OUTSIDE_METRIC_EVERY_N_FRAMES = 15

CATHETER_R = 1.8
CATHETER_MU = 0.50
_TRANS_ZERO = np.zeros((1, 3), dtype=np.float32)

PROJECTIONS = {
    "1: AP (0deg)": np.zeros((1, 3), dtype=np.float32),
    "2: LAO-45": np.array([[0.0, math.radians(45.0), 0.0]], dtype=np.float32),
    "3: Lateral (90deg)": np.array([[0.0, math.radians(90.0), 0.0]], dtype=np.float32),
    "4: RAO-30": np.array([[0.0, math.radians(-30.0), 0.0]], dtype=np.float32),
}
PROJECTION_KEYS = {
    ord("1"): "1: AP (0deg)",
    ord("2"): "2: LAO-45",
    ord("3"): "3: Lateral (90deg)",
    ord("4"): "4: RAO-30",
}


def _build_centerline_voxel_mask(
    cl_pts_xyz_mm: np.ndarray,
    cl_radii_mm: np.ndarray | None,
    vol_shape_zyx: tuple[int, int, int],
    origin_xyz_mm: tuple[float, float, float],
    spacing_zyx_mm: tuple[float, float, float],
    default_radius_mm: float = 1.5,
    max_radius_mm: float = 4.0,
) -> np.ndarray:
    """Voxelize centerline nodes into a thin binary mask (ZYX, uint8).

    Each centerline point is converted to a voxel index and marked.  The mask
    is then dilated isotropically by the average vessel radius so each node
    becomes a small sphere — producing a sparse 'tube' network.

    Using this thin mask for DSA boosting (instead of the full vessel mask)
    avoids the 'solid blob' problem: a dense vessel mask causes almost every
    ray through the brain to hit a vessel, filling the entire 2-D projection.
    The thin centerline mask projects as fine bright lines, matching clinical DSA.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure

    nz, ny, nx = vol_shape_zyx
    ox, oy, oz = origin_xyz_mm
    sz, sy, sx = spacing_zyx_mm

    ix = np.round((cl_pts_xyz_mm[:, 0] - ox) / sx).astype(np.int32)
    iy = np.round((cl_pts_xyz_mm[:, 1] - oy) / sy).astype(np.int32)
    iz = np.round((cl_pts_xyz_mm[:, 2] - oz) / sz).astype(np.int32)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    mask = np.zeros((nz, ny, nx), dtype=np.uint8)
    mask[iz[valid], iy[valid], ix[valid]] = 1

    # Dilation radius ≈ average vessel radius, capped at max_radius_mm
    if cl_radii_mm is not None and len(cl_radii_mm) > 0:
        usable = cl_radii_mm[cl_radii_mm < max_radius_mm]
        avg_r = float(np.mean(usable)) if len(usable) > 0 else default_radius_mm
    else:
        avg_r = default_radius_mm
    min_spacing = min(sx, sy, sz)
    n_iter = max(1, int(round(avg_r / min_spacing)))
    print(
        f"[Viewport]   Centerline mask: {int(valid.sum())} nodes, avg_r={avg_r:.2f}mm, dilation={n_iter} vox",
        flush=True,
    )

    struct = generate_binary_structure(3, 1)  # 6-connected ball
    mask = binary_dilation(mask, structure=struct, iterations=n_iter).astype(np.uint8)
    return mask


def _translate_wp_mesh(mesh: "wp.Mesh", offset_xyz_m: np.ndarray, device: str = "cuda") -> "wp.Mesh":
    """Rebase collision mesh vertices into solver-local coordinates."""
    off = np.asarray(offset_xyz_m, dtype=np.float32).reshape(3)
    pts = wp.to_torch(mesh.points).cpu().numpy().astype(np.float32) - off[None, :]
    idx = wp.to_torch(mesh.indices).cpu().numpy().astype(np.int32)
    return wp.Mesh(
        points=wp.array(pts, dtype=wp.vec3, device=device),
        indices=wp.array(idx, dtype=wp.int32, device=device),
    )


def _build_vessel_mask_downsampled(mu_zyx: np.ndarray, meta: dict) -> tuple[np.ndarray, tuple, tuple, tuple]:
    """Build downsampled vessel mask for collision geometry extraction."""
    sz_mm, sy_mm, sx_mm = meta["spacing_zyx_mm"]
    ox, oy, oz = meta["origin_xyz_mm"]
    nz, ny, nx = mu_zyx.shape
    cx_mm = ox + (nx / 2) * sx_mm
    cy_mm = oy + (ny / 2) * sy_mm
    cz_mm = oz + (nz * 0.45) * sz_mm

    ds = 4
    nzd, nyd, nxd = nz // ds, ny // ds, nx // ds
    mask = np.zeros((nzd, nyd, nxd), dtype=np.uint8)

    vcy = nyd // 2
    vcz = int(round((cz_mm - oz) / (sz_mm * ds)))
    r_y = max(1, int(round(8.0 / (sy_mm * ds))))
    r_z = max(1, int(round(8.0 / (sz_mm * ds))))

    for xi in range(nxd):
        frac = xi / max(nxd, 1)
        dz_off = int(round(6 * math.sin(math.pi * frac)))
        y0 = max(0, vcy - r_y - 2)
        y1 = min(nyd, vcy + r_y + 2)
        z0 = max(0, vcz + dz_off - r_z - 2)
        z1 = min(nzd, vcz + dz_off + r_z + 2)
        for yi in range(y0, y1):
            for zi in range(z0, z1):
                dy = (yi - vcy) / r_y
                dz = (zi - (vcz + dz_off)) / r_z
                if dy * dy + dz * dz <= 1.0:
                    mask[zi, yi, xi] = 1

    spacing_ds = (sz_mm * ds, sy_mm * ds, sx_mm * ds)
    return mask, spacing_ds, (ox, oy, oz), (cx_mm, cy_mm, cz_mm)


def _downsample_binary_mask(
    mask_zyx: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    factor: int,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Downsample binary mask for faster collision-mesh extraction."""
    f = max(1, int(factor))
    if f == 1:
        return (mask_zyx > 0).astype(np.uint8), spacing_zyx_mm
    ds_mask = (mask_zyx[::f, ::f, ::f] > 0).astype(np.uint8)
    sz, sy, sx = spacing_zyx_mm
    return ds_mask, (sz * f, sy * f, sx * f)


def _mask_centroid_xyz_mm(
    mask_zyx: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compute physical-space centroid (x,y,z) of a binary mask."""
    idx = np.argwhere(mask_zyx > 0)
    if idx.size == 0:
        raise ValueError("Vessel mask is empty after thresholding/downsampling.")
    sz, sy, sx = spacing_zyx_mm
    ox, oy, oz = origin_xyz_mm
    z_mean = float(idx[:, 0].mean())
    y_mean = float(idx[:, 1].mean())
    x_mean = float(idx[:, 2].mean())
    cx = ox + x_mean * sx
    cy = oy + y_mean * sy
    cz = oz + z_mean * sz
    return cx, cy, cz


def _catheter_segment_data(pos_vol_mm: np.ndarray) -> CatheterSegmentData:
    """Pack catheter polyline into Slang renderer segment format."""
    return CatheterSegmentData(positions=pos_vol_mm, radii=CATHETER_R, mu_values=CATHETER_MU)


def _moving_average_polyline(points: np.ndarray, passes: int = 2) -> np.ndarray:
    """Light smoothing along curve index while preserving endpoints."""
    if points.shape[0] < 3:
        return points
    out = points.copy()
    for _ in range(max(1, passes)):
        nxt = out.copy()
        nxt[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = nxt
    return out


def _clip_polyline_jumps(points: np.ndarray, jump_cap_mm: float) -> np.ndarray:
    """Clamp per-segment jump size to suppress outlier spikes."""
    if points.shape[0] < 2:
        return points
    cleaned = [points[0]]
    for i in range(1, points.shape[0]):
        prev = cleaned[-1]
        cur = points[i]
        vec = cur - prev
        dist = float(np.linalg.norm(vec))
        if dist > jump_cap_mm and dist > 1e-8:
            cur = prev + vec / dist * jump_cap_mm
        cleaned.append(cur)
    return np.asarray(cleaned, dtype=np.float32)


def _catmull_rom_resample(points: np.ndarray, samples_per_segment: int = 3) -> np.ndarray:
    """Resample polyline with a smooth Catmull-Rom spline."""
    n = points.shape[0]
    if n < 4 or samples_per_segment <= 1:
        return points
    out = []
    for i in range(n - 1):
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, n - 1)]
        for j in range(samples_per_segment):
            t = j / float(samples_per_segment)
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(pt)
    out.append(points[-1])
    return np.asarray(out, dtype=np.float32)


def _prepare_render_polyline(pos_vol_mm: np.ndarray, expected_seg_len_mm: float) -> np.ndarray:
    """Make centerline render-safe and cable-like for fluoroscopy display."""
    if pos_vol_mm.shape[0] < 2:
        return pos_vol_mm

    # Keep only finite coordinates.
    finite_mask = np.isfinite(pos_vol_mm).all(axis=1)
    pts = pos_vol_mm[finite_mask]
    if pts.shape[0] < 2:
        return pos_vol_mm

    # Suppress pathological jumps that create long "spikes" in the rendered wire.
    jump_cap = max(3.0 * expected_seg_len_mm, 2.0)
    cleaned_pts = _clip_polyline_jumps(pts, jump_cap_mm=jump_cap)

    # Smooth and upsample for cable-like continuity.
    smoothed = _moving_average_polyline(cleaned_pts, passes=2)
    samples_per_segment = 2 if smoothed.shape[0] >= 180 else 3
    return _catmull_rom_resample(smoothed, samples_per_segment=samples_per_segment)


def _auto_tune_solver_and_render(num_segments: int, seg_len_m: float) -> dict[str, float | int]:
    """Scale key parameters so high segment counts remain stable and realistic."""
    # Keep collision particles safely smaller than the segment length to avoid self-overlap artifacts.
    particle_radius_m = float(np.clip(seg_len_m * 0.35, 0.00018, 0.00045))

    # Increase integration effort with finer discretization.
    num_substeps = int(np.clip(8 + (num_segments // 40) * 2, 8, 20))
    collision_iterations = int(np.clip(2 + (num_segments // 80), 2, 5))

    # Thinner, more realistic fluoroscopy appearance than the previous "thick bright wire".
    render_radius_mm = float(np.clip(0.7 + 20.0 * seg_len_m, 0.8, 1.1))
    render_mu = 0.28

    return {
        "particle_radius_m": particle_radius_m,
        "num_substeps": num_substeps,
        "collision_iterations": collision_iterations,
        "render_radius_mm": render_radius_mm,
        "render_mu": render_mu,
    }


def _quat_from_z_to_dir(target_dir: np.ndarray) -> np.ndarray:
    """Quaternion (xyzw) rotating local +Z axis onto target_dir."""
    d = np.asarray(target_dir, dtype=np.float64)
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    d = d / d_norm
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = float(np.dot(z, d))
    if c > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if c < -0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = np.cross(z, d)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    half = 0.5 * float(np.arccos(c))
    s = float(np.sin(half))
    co = float(np.cos(half))
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, co], dtype=np.float32)


def _centerline_insertion_axis(
    centerline_points_mm: np.ndarray,
    rod_len_m: float,
    align_hint_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Derive straight insertion axis from centerline points (PCA chord)."""
    pts = np.asarray(centerline_points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 3:
        raise ValueError(f"Centerline must be (N,3) with N>=4, got {pts.shape}")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered / max(1, pts.shape[0] - 1)
    _, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]
    hint = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if align_hint_dir is None
        else np.asarray(align_hint_dir, dtype=np.float64)
    )
    if float(np.dot(axis, hint)) < 0.0:
        axis = -axis
    axis_unit = axis / max(float(np.linalg.norm(axis)), 1e-9)
    proj = centered @ axis_unit
    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))
    proximal_mm = pts[i_min].astype(np.float32)
    distal_mm = pts[i_max].astype(np.float32)
    chord_mm = float(np.linalg.norm(distal_mm - proximal_mm))
    track_start_m = (proximal_mm / 1000.0).astype(np.float32)
    track_dir = axis_unit.astype(np.float32)
    track_length_m = float(max(rod_len_m, chord_mm / 1000.0))
    info = {
        "n_points": int(pts.shape[0]),
        "chord_mm": chord_mm,
        "proximal_mm": proximal_mm.tolist(),
        "distal_mm": distal_mm.tolist(),
    }
    return track_start_m, track_dir, track_length_m, info


def _resample_polyline_uniform(path_m: np.ndarray, spacing_m: float) -> np.ndarray:
    """Resample polyline to approximately uniform arc-length spacing."""
    if path_m.shape[0] < 2:
        return path_m
    seg = np.linalg.norm(np.diff(path_m, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-6))
    path = path_m[keep]
    if path.shape[0] < 2:
        return path_m[:2]
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1])
    if total <= 1e-6:
        return path
    n = max(2, int(total / max(spacing_m, 1e-6)) + 1)
    target = np.linspace(0.0, total, n, dtype=np.float32)
    out = np.zeros((n, 3), dtype=np.float32)
    j = 0
    for i, ti in enumerate(target):
        while j < len(s) - 2 and s[j + 1] < ti:
            j += 1
        s0 = float(s[j])
        s1 = float(s[j + 1])
        a = 0.0 if s1 <= s0 else (float(ti) - s0) / (s1 - s0)
        out[i] = (1.0 - a) * path[j] + a * path[j + 1]
    return out


def _compute_rest_darboux_from_polyline(
    guide_path_m: np.ndarray,
    num_edges: int,
    seg_len: float,
) -> np.ndarray:
    """Compute rest-Darboux vectors encoding the vessel-centerline curvature.

    Samples *guide_path_m* at *seg_len* intervals to produce up to
    ``num_edges + 1`` rod nodes, builds a material-frame quaternion per edge
    (local +Z aligned with the tangent, no twist), then derives

        Ω_i = 2 · Im( q_i⁻¹ ⊗ q_{i+1} ) / seg_len

    for each consecutive edge pair.  Edges beyond the guide path retain a
    zero rest-shape (straight).  Quaternions are in (x, y, z, w) order,
    matching the ``_quat_from_z_to_dir`` convention used everywhere else.
    """
    if guide_path_m is None or guide_path_m.shape[0] < 2 or num_edges < 2 or seg_len < 1e-8:
        return np.zeros((num_edges, 3), dtype=np.float32)

    # Arc length of the guide path.
    arc_total = float(np.sum(np.linalg.norm(np.diff(guide_path_m, axis=0), axis=1)))
    if arc_total < 1e-6:
        return np.zeros((num_edges, 3), dtype=np.float32)

    # Sample exactly (n_cover) evenly spaced points along the guide arc.
    n_cover = min(num_edges + 1, max(2, int(arc_total / seg_len) + 2))
    seg_np = np.linalg.norm(np.diff(guide_path_m, axis=0), axis=1)
    cum_s = np.concatenate(([0.0], np.cumsum(seg_np)))
    target_s = np.linspace(0.0, min(arc_total, seg_len * (n_cover - 1)), n_cover)
    sampled = np.zeros((n_cover, 3), dtype=np.float32)
    j = 0
    for k, ts in enumerate(target_s):
        while j < len(cum_s) - 2 and cum_s[j + 1] < ts:
            j += 1
        s0, s1 = float(cum_s[j]), float(cum_s[j + 1])
        alpha = 0.0 if s1 <= s0 else (ts - s0) / (s1 - s0)
        sampled[k] = (1.0 - alpha) * guide_path_m[j] + alpha * guide_path_m[j + 1]

    # Tangent per guided edge.
    n_guided_edges = sampled.shape[0] - 1
    tangents = np.diff(sampled, axis=0).astype(np.float64)  # (n_guided_edges, 3)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(norms, 1e-8)

    # Material-frame quaternion per edge (no twist).
    quats = np.array([_quat_from_z_to_dir(t) for t in tangents], dtype=np.float64)  # (n_guided_edges, 4)

    def _qconj(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )

    rest_darboux = np.zeros((num_edges, 3), dtype=np.float32)
    n_pairs = min(n_guided_edges - 1, num_edges - 1)
    for i in range(n_pairs):
        q_rel = _qmul(_qconj(quats[i]), quats[i + 1])
        rest_darboux[i] = (2.0 * q_rel[:3] / seg_len).astype(np.float32)

    # Extend the last computed curvature to the final guided edge to avoid an
    # abrupt step back to zero at the tip of the covered region.
    if n_pairs > 0:
        rest_darboux[n_pairs] = rest_darboux[n_pairs - 1]

    return rest_darboux


def _centerline_polyline_from_points(
    centerline_points_mm: np.ndarray,
    target_seg_len_m: float,
    max_nodes: int = 2500,
    k_neighbors: int = 6,
) -> tuple[np.ndarray, dict]:
    """Build an ordered centerline polyline from an unordered centerline point cloud.

    Mirrors XCATH's graph + Dijkstra idea:
      1) Build a sparse nearest-neighbor graph on centerline points.
      2) Start from lowest-Z node (injection-like root heuristic).
      3) Run Dijkstra and backtrack the farthest reachable node path.
      4) Resample path to roughly rod segment spacing.
    """
    pts = np.asarray(centerline_points_mm, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 8 or pts.shape[1] != 3:
        raise ValueError(f"Centerline points must be (N,3) with N>=8, got {pts.shape}")
    if pts.shape[0] > max_nodes:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(pts.shape[0], size=max_nodes, replace=False))
        pts = pts[idx]

    n = pts.shape[0]
    k = max(2, min(int(k_neighbors), n - 1))

    # Pairwise Euclidean distances (n<=2500 by design).
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2).astype(np.float32)
    np.fill_diagonal(dmat, np.inf)
    nn = np.argpartition(dmat, kth=k - 1, axis=1)[:, :k]

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in nn[i]:
            w = float(dmat[i, j])
            if not np.isfinite(w):
                continue
            adj[i].append((int(j), w))
            adj[int(j)].append((i, w))

    start = int(np.argmin(pts[:, 2]))  # lowest-Z root, same heuristic family as XCATH slides

    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int32)
    dist[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > float(dist[u]):
            continue
        for v, w in adj[u]:
            new_dist = d + w
            if new_dist < float(dist[v]):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    reachable = np.isfinite(dist)
    if not np.any(reachable):
        raise RuntimeError("Dijkstra found no reachable nodes in centerline graph")
    end = int(np.argmax(np.where(reachable, dist, -1.0)))

    path_idx: list[int] = []
    cur = end
    while cur >= 0:
        path_idx.append(cur)
        if cur == start:
            break
        cur = int(prev[cur])
    path_idx = path_idx[::-1]
    if len(path_idx) < 2:
        raise RuntimeError("Recovered centerline path is too short")

    path_mm = pts[np.asarray(path_idx, dtype=np.int32)]
    # Light smoothing + uniform resample for stable root guidance.
    path_mm = _moving_average_polyline(path_mm, passes=2)
    path_m = _resample_polyline_uniform(path_mm / 1000.0, spacing_m=max(target_seg_len_m, 1e-4))
    seg = np.linalg.norm(np.diff(path_m, axis=0), axis=1)
    total_m = float(np.sum(seg)) if seg.size > 0 else 0.0
    info = {
        "n_cloud_points": int(centerline_points_mm.shape[0]),
        "n_graph_nodes": int(n),
        "n_path_nodes": int(path_m.shape[0]),
        "path_length_m": total_m,
        "start_idx": int(start),
        "end_idx": int(end),
    }
    return path_m.astype(np.float32), info


def _centerline_polyline_from_graph(
    centerline_points_mm: np.ndarray,
    centerline_edges: np.ndarray,
    target_seg_len_m: float,
    centerline_radii_mm: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Build ordered polyline from true centerline graph (VMTK-style edges)."""
    pts = np.asarray(centerline_points_mm, dtype=np.float32)
    edges = np.asarray(centerline_edges)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
        raise ValueError(f"centerline_points must be (N,3), got {pts.shape}")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] < 1:
        raise ValueError(f"centerline_edges must be (M,2), got {edges.shape}")
    edges = edges.astype(np.int64)
    if np.any(edges < 0) or np.any(edges >= pts.shape[0]):
        raise ValueError("centerline_edges contains out-of-range node indices")

    n = int(pts.shape[0])
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    deg = np.zeros(n, dtype=np.int32)
    radii = None
    if centerline_radii_mm is not None:
        r = np.asarray(centerline_radii_mm, dtype=np.float32).reshape(-1)
        if r.shape[0] == n:
            radii = np.maximum(r, 1e-3)

    for u, v in edges:
        uu = int(u)
        vv = int(v)
        d = float(np.linalg.norm(pts[uu] - pts[vv]))  # mm
        if not np.isfinite(d) or d <= 1e-6:
            continue
        if radii is not None:
            # XCATH slide logic: weight ~ distance / velocity(radius). Larger radius => cheaper.
            ravg = float(0.5 * (radii[uu] + radii[vv]))
            vel = max((ravg**0.5), 1e-3)
            w = d / vel
        else:
            w = d
        adj[uu].append((vv, w))
        adj[vv].append((uu, w))
        deg[uu] += 1
        deg[vv] += 1

    endpoints = np.where(deg == 1)[0]
    if endpoints.size > 0:
        start = int(endpoints[np.argmin(pts[endpoints, 2])])  # lowest-Z endpoint
    else:
        start = int(np.argmin(pts[:, 2]))  # fallback

    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int32)
    dist[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > float(dist[u]):
            continue
        for v, w in adj[u]:
            new_dist = d + w
            if new_dist < float(dist[v]):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    reachable = np.isfinite(dist)
    if not np.any(reachable):
        raise RuntimeError("No reachable nodes in centerline graph")
    end = int(np.argmax(np.where(reachable, dist, -1.0)))

    path_idx: list[int] = []
    cur = end
    while cur >= 0:
        path_idx.append(cur)
        if cur == start:
            break
        cur = int(prev[cur])
    path_idx = path_idx[::-1]
    if len(path_idx) < 2:
        raise RuntimeError("Recovered graph path is too short")

    path_mm = pts[np.asarray(path_idx, dtype=np.int32)]
    path_mm = _moving_average_polyline(path_mm, passes=1)
    path_m = _resample_polyline_uniform(path_mm / 1000.0, spacing_m=max(target_seg_len_m, 1e-4))
    seg = np.linalg.norm(np.diff(path_m, axis=0), axis=1)
    total_m = float(np.sum(seg)) if seg.size > 0 else 0.0
    info = {
        "method": "graph-edges",
        "n_nodes": n,
        "n_edges": int(edges.shape[0]),
        "n_path_nodes": int(path_m.shape[0]),
        "path_length_m": total_m,
        "start_idx": int(start),
        "end_idx": int(end),
        "used_radii": bool(radii is not None),
    }
    return path_m.astype(np.float32), info


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product (xyzw): compose base orientation with local twist."""
    return np.array(
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ],
        dtype=np.float32,
    )


def _polyline_interp(path_m: np.ndarray, s_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate point+tangent on polyline at arc-length s."""
    if path_m.shape[0] < 2:
        p = path_m[0] if path_m.shape[0] == 1 else np.zeros(3, dtype=np.float32)
        return p.astype(np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32)
    seg = np.linalg.norm(np.diff(path_m, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    s = float(np.clip(s_m, 0.0, total))
    j = int(np.searchsorted(cum, s, side="right") - 1)
    j = max(0, min(j, path_m.shape[0] - 2))
    s0 = float(cum[j])
    s1 = float(cum[j + 1])
    a = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
    p = (1.0 - a) * path_m[j] + a * path_m[j + 1]
    t = path_m[j + 1] - path_m[j]
    nrm = float(np.linalg.norm(t))
    if nrm > 1e-8:
        t = t / nrm
    else:
        t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return p.astype(np.float32), t.astype(np.float32)


def _polyline_interp_with_cum(path_m: np.ndarray, cum_m: np.ndarray, s_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate point+tangent on polyline using precomputed cumulative arclength."""
    if path_m.shape[0] < 2:
        p = path_m[0] if path_m.shape[0] == 1 else np.zeros(3, dtype=np.float32)
        return p.astype(np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32)
    total = float(cum_m[-1])
    s = float(np.clip(s_m, 0.0, total))
    j = int(np.searchsorted(cum_m, s, side="right") - 1)
    j = max(0, min(j, path_m.shape[0] - 2))
    s0 = float(cum_m[j])
    s1 = float(cum_m[j + 1])
    a = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
    p = (1.0 - a) * path_m[j] + a * path_m[j + 1]
    t = path_m[j + 1] - path_m[j]
    nrm = float(np.linalg.norm(t))
    if nrm > 1e-8:
        t = t / nrm
    else:
        t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return p.astype(np.float32), t.astype(np.float32)


class MouseDriveController:
    """Maps cursor drag motions to proximal velocity and torque commands."""

    def __init__(self, max_velocity_mm_s: float = 14.0, max_torque: float = 0.03):
        self.dragging = False
        self.last_xy: tuple[int, int] | None = None
        self.velocity_cmd_m_s = 0.0
        self.torque_cmd = 0.0
        self.max_velocity_mm_s = max_velocity_mm_s
        self.max_torque = max_torque
        self._decay = 0.88

    def handle_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:  # noqa: ARG002
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_xy = (x, y)
            return
        if event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.last_xy = None
            self.velocity_cmd_m_s = 0.0
            self.torque_cmd = 0.0
            return
        if event != cv2.EVENT_MOUSEMOVE or not self.dragging or self.last_xy is None:
            return

        last_x, last_y = self.last_xy
        dx = x - last_x
        dy = y - last_y
        self.last_xy = (x, y)

        # Horizontal drag drives insertion/retraction, vertical drag drives twist.
        vel_mm_s = np.clip(dx * 0.7, -self.max_velocity_mm_s, self.max_velocity_mm_s)
        torque = np.clip(-dy * 0.0018, -self.max_torque, self.max_torque)
        self.velocity_cmd_m_s = float(vel_mm_s / 1000.0)
        self.torque_cmd = float(torque)

    def commands(self) -> tuple[float, float]:
        """Return smoothed current command; decays to zero when not dragging."""
        if not self.dragging:
            self.velocity_cmd_m_s *= self._decay
            self.torque_cmd *= self._decay
            if abs(self.velocity_cmd_m_s) < 1e-5:
                self.velocity_cmd_m_s = 0.0
            if abs(self.torque_cmd) < 1e-4:
                self.torque_cmd = 0.0
        return self.velocity_cmd_m_s, self.torque_cmd


class KeyboardDriveController:
    """Maps key holds to proximal velocity/torque commands."""

    def __init__(self, max_velocity_mm_s: float = 16.0, max_torque: float = 1.5, key_hold_ttl_s: float = 0.25):
        self.max_velocity_mm_s = max_velocity_mm_s
        self.max_torque = max_torque
        self.fast_scale = 1.8
        self.slow_scale = 1.0
        # OpenCV waitKey has no key-up events. Keep key presses alive across OS key-repeat delay.
        self._last_seen_time: dict[int, float] = {}
        self._active_ttl_s = max(0.05, float(key_hold_ttl_s))

    def set_key_down(self, key: int) -> None:
        if key != 255:
            key_i = int(key)
            self._last_seen_time[key_i] = time.monotonic()

    def set_key_up(self, key: int) -> None:
        self._last_seen_time.pop(int(key), None)

    def clear(self) -> None:
        self._last_seen_time.clear()

    def _is_active(self, key: int) -> bool:
        t = self._last_seen_time.get(key)
        if t is None:
            return False
        return (time.monotonic() - t) <= self._active_ttl_s

    def _held(self, key: int, upper_key: int | None = None) -> bool:
        if self._is_active(key):
            return True
        if upper_key is not None and self._is_active(upper_key):
            return True
        return False

    def commands(self) -> tuple[float, float]:
        forward = self._held(ord("w"), ord("W"))
        backward = self._held(ord("s"), ord("S"))
        ccw = self._held(ord("a"), ord("A"))
        cw = self._held(ord("d"), ord("D"))
        fast = self._held(225) or self._held(226)  # common OpenCV shift scan codes

        vel_sign = float(forward) - float(backward)
        torque_sign = float(cw) - float(ccw)
        scale = self.fast_scale if fast else self.slow_scale
        vel_mm_s = np.clip(
            vel_sign * self.max_velocity_mm_s * scale, -self.max_velocity_mm_s * scale, self.max_velocity_mm_s * scale
        )
        torque = np.clip(torque_sign * self.max_torque * scale, -self.max_torque * scale, self.max_torque * scale)
        return float(vel_mm_s / 1000.0), float(torque)


class SlangViewportApp:
    def __init__(
        self,
        ct_dir: str,
        det_size: int = DET_SIZE,
        num_segments: int = DEFAULT_NUM_SEGMENTS,
        particle_radius_mm: float | None = None,
        catheter_radius_mm: float | None = None,
        catheter_mu: float | None = None,
        visual_style: str = "default",
        idle_mode: str = "hold",
        centerline_mode: str = "raw",
        pose_guard: str = "safe",
        collision_mode: str = "mesh-edge",
        sign_scale: float = 1.0,
        vessel_source: str = "auto",
        vessel_mask_path: str | None = None,
        hu_volume_path: str | None = None,
        hu_threshold: float = 200.0,
        min_component_voxels: int = 500,
        mask_downsample: int = 4,
        track_stiffness: float = 0.35,
        advance_guide_stiffness: float = 0.65,
        insertion_axis: str = "centerline",
        centerline_file: str | None = None,
        guidewire_young_modulus: float = 1e6,
        guidewire_bend_stiffness: float = 0.1,
        pixel_spacing_mm: float = 0.5,
        auto_advance_mm_s: float = 0.0,
        key_hold_ttl_s: float = 0.25,
        dsa: bool = False,
        dsa_boost: float = 6.0,
        dsa_gamma: float = 1.0,
    ):
        self.ct_dir = ct_dir
        self.det_size = det_size
        self.pixel_spacing_mm = float(pixel_spacing_mm)
        self.num_segments = int(max(3, num_segments))
        self._particle_radius_mm_override = particle_radius_mm
        self._catheter_radius_mm_override = catheter_radius_mm
        self._catheter_mu_override = catheter_mu
        self.visual_style = visual_style
        self.idle_mode = idle_mode
        self.centerline_mode = centerline_mode
        self.pose_guard = pose_guard
        self.collision_mode = collision_mode
        self.sign_scale = float(sign_scale)
        self.vessel_source = vessel_source
        self.vessel_mask_path = vessel_mask_path
        self.hu_volume_path = hu_volume_path
        self.hu_threshold = float(hu_threshold)
        self.min_component_voxels = int(min_component_voxels)
        self.mask_downsample = int(mask_downsample)
        self.track_stiffness = float(np.clip(track_stiffness, 0.0, 1.0))
        self.advance_guide_stiffness = float(np.clip(advance_guide_stiffness, 0.0, 1.0))
        self.insertion_axis = insertion_axis
        self.centerline_file = centerline_file
        self.guidewire_young_modulus = float(guidewire_young_modulus)
        self.guidewire_bend_stiffness = float(guidewire_bend_stiffness)
        self.key_hold_ttl_s = max(0.05, float(key_hold_ttl_s))
        self.sim: dict[str, object] = {}
        self.mouse = MouseDriveController()
        self.keyboard = KeyboardDriveController(key_hold_ttl_s=self.key_hold_ttl_s)
        self.window_name = "Catheter Navigation"
        self.current_projection = "2: LAO-45"
        self.paused = False
        self.frame_idx = 0
        self.last_loop_ms = 0.0
        self.fps_smooth = 0.0
        self._cuda_available = torch.cuda.is_available()
        self._temporal_frames: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(7)
        self._last_key_code = -1
        self._invalid_pose_count = 0
        self._last_valid_positions_m: np.ndarray | None = None
        self._last_valid_orientations_xyzw: np.ndarray | None = None
        self._outside_max_phi_mm = 0.0
        self._outside_percent = 0.0
        self._outside_metric_status = "n/a"
        self._outside_plus_percent = 0.0
        self._outside_minus_percent = 0.0
        self._suggested_sign_scale = 1
        self._mesh_aabb_min_m: np.ndarray | None = None
        self._mesh_aabb_max_m: np.ndarray | None = None
        self._aabb_overlap = False
        self._guide_s_m = 0.0
        self._guide_ds_frame_mm = 0.0
        self._guide_free_active = False
        self._guide_boundary_state = "none"
        self._guide_blend_t = 1.0
        # Guided-only runtime.
        self.navigation_mode = "guided"
        self.vessel_overlay_on = True  # F key toggles vessel highlight
        self.centerline_on = True  # C key toggles projected centerline on right panel
        # DSA contrast bake: render from a vessel-μ-boosted renderer (X key toggles).
        self.dsa_contrast_boost = float(max(1.5, dsa_boost))
        self._dsa_on = bool(dsa)
        self._dsa_renderer = None  # built in _init_simulation when a vessel mask exists
        # DSA brightness: gamma applied to the normalized angiogram (<1 brighter, >1 darker).
        # Adjusted live with '-' (darker) / '=' (brighter).
        self.dsa_gamma = float(np.clip(dsa_gamma, 0.3, 3.0))
        self._vessel_overlays: dict[str, np.ndarray] = {}
        self._vessel_diff_overlays: dict[str, np.ndarray] = {}  # cyan guide-path tint
        self._last_raw_frame: np.ndarray | None = None  # pre-overlay BGR frame (left panel)
        self._init_simulation()

    def _init_simulation(self) -> None:
        mu_zyx = np.load(os.path.join(self.ct_dir, "mu_volume.npy"))
        with open(os.path.join(self.ct_dir, "metadata.json"), encoding="utf-8") as stream:
            meta = json.load(stream)
        sz_mm, sy_mm, sx_mm = meta["spacing_zyx_mm"]
        ox, oy, oz = meta["origin_xyz_mm"]
        nz, ny, nx = mu_zyx.shape

        mask_full = None
        selected_source = self.vessel_source

        # Resolve real vessel mask source (explicit mask, HU volume, or auto-detect files).
        candidate_mask_paths = []
        if self.vessel_mask_path:
            candidate_mask_paths.append(self.vessel_mask_path)
        candidate_mask_paths.extend(
            [
                os.path.join(self.ct_dir, "vessel_mask.npy"),
                os.path.join(self.ct_dir, "vessel_segmentation.npy"),
                os.path.join(self.ct_dir, "segmentation.npy"),
                os.path.join(self.ct_dir, "mask.npy"),
            ]
        )
        real_mask_path = next((p for p in candidate_mask_paths if p and os.path.isfile(p)), None)

        hu_path = self.hu_volume_path
        if hu_path is None:
            for p in [
                os.path.join(self.ct_dir, "hu_volume.npy"),
                os.path.join(self.ct_dir, "ct_hu_zyx.npy"),
                os.path.join(self.ct_dir, "ct_hu.npy"),
            ]:
                if os.path.isfile(p):
                    hu_path = p
                    break

        overlay_mask_zyx: np.ndarray | None = None
        if self.vessel_source in ("auto", "real") and (real_mask_path is not None or hu_path is not None):
            if real_mask_path is not None:
                mask_full = (np.load(real_mask_path) > 0).astype(np.uint8)
                selected_source = "real-mask"
            else:
                hu_zyx = np.load(hu_path).astype(np.float32)
                mask_full = vessel_mask_from_hu(
                    hu_zyx,
                    hu_threshold=self.hu_threshold,
                    min_component_voxels=self.min_component_voxels,
                )
                selected_source = "real-hu-threshold"

            if mask_full.shape != mu_zyx.shape:
                raise ValueError(f"Vessel mask shape {mask_full.shape} does not match mu_volume shape {mu_zyx.shape}.")
            # Collision mesh uses a coarser mask than the fluoro overlay.
            collision_ds = max(1, min(2, int(self.mask_downsample)))
            mask, spacing_ds = _downsample_binary_mask(
                mask_full,
                (sz_mm, sy_mm, sx_mm),
                factor=collision_ds,
            )
            origin_xyz = (ox, oy, oz)
            cx_mm, cy_mm, cz_mm = _mask_centroid_xyz_mm(mask, spacing_ds, origin_xyz)
            overlay_mask_zyx = mask_full
        elif self.vessel_source == "real":
            raise FileNotFoundError(
                "Requested --vessel-source real but no real vessel input found. "
                "Provide --vessel-mask /path/to/vessel_mask.npy or --hu-volume /path/to/hu_volume.npy."
            )
        else:
            mask, spacing_ds, origin_xyz, (cx_mm, cy_mm, cz_mm) = _build_vessel_mask_downsampled(mu_zyx, meta)
            selected_source = "synthetic"
            overlay_mask_zyx = mask

        # Stash the vessel mask so the vessel-boost renderer can be built after the main renderer.
        # Using Method 2 (vessel μ×8 rendered directly) avoids the DSA blob problem: Method 3
        # (subtraction) causes almost every brain ray to accumulate vessel signal, filling the
        # entire 2-D image with a solid blob even for thin centerline masks.
        _boost_mask_zyx: np.ndarray | None = (
            overlay_mask_zyx if overlay_mask_zyx is not None and overlay_mask_zyx.shape == mu_zyx.shape else None
        )

        mesh = extract_vessel_mesh(mask, spacing_zyx_mm=spacing_ds, origin_xyz_mm=origin_xyz, device="cuda")

        x_start_mm = ox + nx * sx_mm * 0.15
        x_end_mm = ox + nx * sx_mm * 0.80
        rod_len_m = (x_end_mm - x_start_mm) / 1000.0
        seg_len = rod_len_m / float(self.num_segments)
        track_start_m = np.array([x_start_mm / 1000.0, cy_mm / 1000.0, cz_mm / 1000.0], dtype=np.float32)
        track_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        track_length_m = rod_len_m
        axis_status = "straight +X"
        guide_path_local_m: np.ndarray | None = None
        guide_path_length_m = 0.0
        ct_offset_m = track_start_m.copy()
        local_z0_m = np.array([0.0, 0.0, float(track_start_m[2])], dtype=np.float32)
        if self.insertion_axis == "centerline":
            cl_candidates = []
            if self.centerline_file:
                cl_candidates.append(self.centerline_file)
            cl_candidates.append(os.path.join(self.ct_dir, "centerline_points_mm.npy"))
            cl_path = next((p for p in cl_candidates if p and os.path.isfile(p)), None)
            if cl_path is None:
                axis_status = "centerline missing -> straight +X"
            else:
                try:
                    cl_pts_mm = np.load(cl_path).astype(np.float32)
                    cl_dir = os.path.dirname(cl_path)
                    edge_candidates = [
                        os.path.join(cl_dir, "centerline_edges.npy"),
                        os.path.join(self.ct_dir, "centerline_edges.npy"),
                    ]
                    edges_path = next((p for p in edge_candidates if os.path.isfile(p)), None)
                    radii_candidates = [
                        os.path.join(cl_dir, "centerline_radii_mm.npy"),
                        os.path.join(cl_dir, "centerline_radii.npy"),
                        os.path.join(self.ct_dir, "centerline_radii_mm.npy"),
                        os.path.join(self.ct_dir, "centerline_radii.npy"),
                    ]
                    radii_path = next((p for p in radii_candidates if os.path.isfile(p)), None)
                    if edges_path is not None:
                        cl_edges = np.load(edges_path)
                        cl_radii = np.load(radii_path) if radii_path is not None else None
                        path_world_m, path_info = _centerline_polyline_from_graph(
                            cl_pts_mm,
                            cl_edges,
                            target_seg_len_m=seg_len,
                            centerline_radii_mm=cl_radii,
                        )
                        axis_status = (
                            f"centerline graph+edges "
                            f"({path_info['n_path_nodes']} nodes, {path_info['path_length_m']*1000.0:.0f} mm"
                            f"{', radii' if path_info['used_radii'] else ''})"
                        )
                    else:
                        path_world_m, path_info = _centerline_polyline_from_points(
                            cl_pts_mm,
                            target_seg_len_m=seg_len,
                            max_nodes=2500,
                            k_neighbors=6,
                        )
                        axis_status = (
                            f"centerline graph+dijkstra "
                            f"({path_info['n_path_nodes']} nodes, {path_info['path_length_m']*1000.0:.0f} mm)"
                        )
                    ct_offset_m = path_world_m[0].astype(np.float32)
                    local_z0_m = np.zeros(3, dtype=np.float32)
                    guide_path_local_m = (path_world_m - ct_offset_m).astype(np.float32)
                    guide_path_length_m = float(np.sum(np.linalg.norm(np.diff(guide_path_local_m, axis=0), axis=1)))

                    # Boost mask: ONLY the Dijkstra guide-path nodes (~260 nodes).
                    # Adding r>=1.5mm large-vessel nodes produces thousands of scattered
                    # points throughout the brain → blob artifacts in 2D projection.
                    # The guide path is a CONNECTED corridor → projects as a coherent line.
                    _guide_pts_mm = path_world_m.astype(np.float32) * 1000.0  # m → mm (XYZ)
                    print(
                        f"[Viewport]   Boost mask: {len(_guide_pts_mm)} guide-path nodes only "
                        f"(r=1.2mm fixed dilation — avoids scatter from full artery tree)",
                        flush=True,
                    )
                    _boost_mask_zyx = _build_centerline_voxel_mask(
                        _guide_pts_mm,
                        None,  # use fixed dilation for a tight visual guide tube
                        mu_zyx.shape,
                        (ox, oy, oz),
                        (sz_mm, sy_mm, sx_mm),
                        default_radius_mm=1.2,
                    )

                    # Initial straight-track approximation follows local tangent at guide root.
                    p0, t0 = _polyline_interp(guide_path_local_m, 0.0)
                    track_start_m = ct_offset_m.copy()  # rendering offset / reporting
                    track_dir = t0.astype(np.float32)
                    track_length_m = max(rod_len_m, guide_path_length_m)
                except Exception as exc:
                    axis_status = f"centerline failed ({type(exc).__name__}) -> straight +X"
                    print(f"[viewport] failed centerline axis from {cl_path}: {exc}")

        # Collision mesh from extract_vessel_mesh is in absolute CT metres, but the rod
        # solver runs in local metres (origin at catheter entry / guide root). Without
        # this shift, containment never engages and the catheter appears to
        # move through empty space while overlays look correctly registered.
        mesh_frame_offset_m = ct_offset_m.astype(np.float32)
        mesh = _translate_wp_mesh(mesh, mesh_frame_offset_m, device="cuda")
        mesh_points_m = wp.to_torch(mesh.points).cpu().numpy()
        self._mesh_aabb_min_m = np.min(mesh_points_m, axis=0).astype(np.float32)
        self._mesh_aabb_max_m = np.max(mesh_points_m, axis=0).astype(np.float32)
        print(
            "[Viewport] Collision mesh rebased to solver-local frame " f"(offset mm: {mesh_frame_offset_m * 1000.0})",
            flush=True,
        )

        tuned = _auto_tune_solver_and_render(self.num_segments, seg_len)
        particle_radius_m = (
            float(self._particle_radius_mm_override) / 1000.0
            if self._particle_radius_mm_override is not None
            else float(tuned["particle_radius_m"])
        )
        render_radius_mm = (
            float(self._catheter_radius_mm_override)
            if self._catheter_radius_mm_override is not None
            else float(tuned["render_radius_mm"])
        )
        render_mu = (
            float(self._catheter_mu_override) if self._catheter_mu_override is not None else float(tuned["render_mu"])
        )

        rod_cfg = RodConfig()
        rod_cfg.device = "cuda"
        rod_cfg.geometry.num_segments = self.num_segments
        rod_cfg.geometry.rest_length = rod_len_m
        rod_cfg.geometry.segment_length = seg_len
        rod_cfg.solver.num_substeps = int(tuned["num_substeps"])
        # Guidewire-appropriate bending compliance.  Default 1e9 (steel) is 5+
        # orders of magnitude too stiff for cerebral navigation in this viewport.
        rod_cfg.material.young_modulus = float(self.guidewire_young_modulus)
        rod_cfg.material.shear_modulus = float(self.guidewire_young_modulus) / (2.0 * 1.3)
        rod_cfg.material.bend_stiffness = float(self.guidewire_bend_stiffness)

        # Track guidance is evaluated in solver-space; render uses ct_offset/local_z0 for translation.
        track_start_solver = np.array([0.0, 0.0, float(local_z0_m[2])], dtype=np.float32)
        if guide_path_local_m is not None and guide_path_local_m.shape[0] >= 1:
            track_start_solver = guide_path_local_m[0].astype(np.float32)

        solver = XCathRodSolver(
            rod_cfg,
            collision_mesh=mesh,
            track_start=track_start_solver,
            track_dir=track_dir,
            track_length=track_length_m,
            tip_num_edges=8,
            particle_radius=particle_radius_m,
            segment_length=seg_len,
            collision_iterations=int(tuned["collision_iterations"]),
            mesh_edge_collision_enabled=(self.collision_mode == "mesh-edge"),
            mesh_edge_collision_max_triangles=192,
            sign_scale=float(self.sign_scale),
            target_phi=-0.001,
            max_dist=0.05,
            initial_height=float(track_start_solver[2]),
            track_stiffness=float(self.track_stiffness),
        )

        # Align initial rod polyline and orientations to chosen track direction in solver-space.
        if float(np.linalg.norm(track_dir - np.array([1.0, 0.0, 0.0], dtype=np.float32))) > 1e-6:
            local_start = track_start_solver
            new_pos_np = np.zeros((self.num_segments + 1, 3), dtype=np.float32)
            for i in range(self.num_segments + 1):
                new_pos_np[i] = local_start + float(i) * float(seg_len) * track_dir
            q_align = _quat_from_z_to_dir(track_dir)
            new_ori_np = np.tile(q_align, (self.num_segments + 1, 1)).astype(np.float32)
            zero_vec3_np = np.zeros((self.num_segments + 1, 3), dtype=np.float32)
            ws = solver._ws
            ws.positions.assign(wp.array(new_pos_np, dtype=wp.vec3, device="cuda"))
            ws.predicted_positions.assign(wp.array(new_pos_np, dtype=wp.vec3, device="cuda"))
            ws.orientations.assign(wp.array(new_ori_np, dtype=wp.quat, device="cuda"))
            ws.predicted_orientations.assign(wp.array(new_ori_np, dtype=wp.quat, device="cuda"))
            ws.prev_orientations.assign(wp.array(new_ori_np, dtype=wp.quat, device="cuda"))
            ws.velocities.assign(wp.array(zero_vec3_np, dtype=wp.vec3, device="cuda"))
            ws.angular_velocities.assign(wp.array(zero_vec3_np, dtype=wp.vec3, device="cuda"))

        # If we have a curved guide, initialize all particles on that guide (not just root).
        if guide_path_local_m is not None and guide_path_local_m.shape[0] >= 2:
            seg_guide = np.linalg.norm(np.diff(guide_path_local_m, axis=0), axis=1)
            cum_guide = np.concatenate(([0.0], np.cumsum(seg_guide))).astype(np.float32)
            total_guide = float(cum_guide[-1])
            num_points = self.num_segments + 1
            init_pos = np.zeros((num_points, 3), dtype=np.float32)
            init_ori = np.zeros((num_points, 4), dtype=np.float32)
            for i in range(num_points):
                s_i = min(float(i) * float(seg_len), total_guide)
                p_i, t_i = _polyline_interp_with_cum(guide_path_local_m, cum_guide, s_i)
                init_pos[i] = p_i
                init_ori[i] = _quat_from_z_to_dir(t_i)
            ws = solver._ws
            ws.positions.assign(wp.array(init_pos, dtype=wp.vec3, device="cuda"))
            ws.predicted_positions.assign(wp.array(init_pos, dtype=wp.vec3, device="cuda"))
            ws.orientations.assign(wp.array(init_ori, dtype=wp.quat, device="cuda"))
            ws.predicted_orientations.assign(wp.array(init_ori, dtype=wp.quat, device="cuda"))
            ws.prev_orientations.assign(wp.array(init_ori, dtype=wp.quat, device="cuda"))
            ws.velocities.assign(wp.zeros(num_points, dtype=wp.vec3, device="cuda"))
            ws.angular_velocities.assign(wp.zeros(num_points, dtype=wp.vec3, device="cuda"))

        _drr_cfg = SlangDiffDRRConfig(
            det_width_px=self.det_size,
            det_height_px=self.det_size,
            pixel_spacing_mm=self.pixel_spacing_mm,
            step_mm=1.0,
            i0=1.0,
        )
        renderer = SlangDiffDRRRenderer(
            mu_zyx,
            spacing_zyx_mm=(sz_mm, sy_mm, sx_mm),
            cfg=_drr_cfg,
            num_envs=1,
        )

        # DSA contrast renderer: a second DRR whose μ is amplified inside the full
        # vessel lumen and rendered with native-angiogram polarity (invert=False →
        # dense contrast + catheter read DARK on a brighter anatomy background, like
        # a real digital-subtraction angiogram). Toggled live with the X key; built
        # once here so toggling is free (no per-frame volume re-upload).
        self._dsa_renderer = None
        if overlay_mask_zyx is not None and overlay_mask_zyx.shape == mu_zyx.shape:
            print(
                f"[Viewport] Building DSA contrast renderer "
                f"(vessel μ×{self.dsa_contrast_boost:.1f}, angiogram polarity) ...",
                flush=True,
            )
            _dsa_runtime_cfg = SlangDiffDRRConfig(
                det_width_px=self.det_size,
                det_height_px=self.det_size,
                pixel_spacing_mm=self.pixel_spacing_mm,
                step_mm=1.0,
                i0=1.0,
                normalize=True,
                invert=False,
            )
            _mu_contrast = mu_zyx.astype(np.float32).copy()
            _mu_contrast[overlay_mask_zyx > 0] *= self.dsa_contrast_boost
            self._dsa_renderer = SlangDiffDRRRenderer(
                _mu_contrast,
                spacing_zyx_mm=(sz_mm, sy_mm, sx_mm),
                cfg=_dsa_runtime_cfg,
                num_envs=1,
            )
            del _mu_contrast
        elif self._dsa_on:
            print("[Viewport] --dsa requested but no vessel mask available; DSA disabled.", flush=True)
            self._dsa_on = False

        # Precompute DSA vessel diff overlays for the 4 C-arm projections.
        # Uses two UNNORMALIZED renderers (normalize=False) so the subtraction is on raw
        # transmitted intensities — independent normalization is the cause of scattered
        # artifacts and must be avoided here.  Both temp renderers are deleted after use.
        if _boost_mask_zyx is not None:
            _dsa_cfg = SlangDiffDRRConfig(
                det_width_px=self.det_size,
                det_height_px=self.det_size,
                pixel_spacing_mm=self.pixel_spacing_mm,
                step_mm=1.0,
                i0=1.0,
                normalize=False,  # raw transmitted intensity — no per-image scale difference
                invert=False,
            )
            print("[Viewport] Building DSA renderers for vessel overlay (unnormalized) ...", flush=True)
            _rnd_normal = SlangDiffDRRRenderer(
                mu_zyx.astype(np.float32),
                spacing_zyx_mm=(sz_mm, sy_mm, sx_mm),
                cfg=_dsa_cfg,
                num_envs=1,
            )
            _mu_v = mu_zyx.astype(np.float32).copy()
            _mu_v[_boost_mask_zyx > 0] *= 8.0
            _rnd_vessel = SlangDiffDRRRenderer(
                _mu_v,
                spacing_zyx_mm=(sz_mm, sy_mm, sx_mm),
                cfg=_dsa_cfg,
                num_envs=1,
            )
            del _mu_v
            print("[Viewport] Precomputing vessel diff overlays (4 projections) ...", flush=True)
            for _proj_name, _rot in PROJECTIONS.items():
                _I_n = _rnd_normal.render_batch(_rot, _TRANS_ZERO)[0]  # raw transmitted, normal
                _I_v = _rnd_vessel.render_batch(_rot, _TRANS_ZERO)[0]  # raw transmitted, vessel
                # I_normal > I_vessel (vessels absorb extra → less transmitted).
                # Raw transmission per 3mm vessel with μ×8 is ~0.008, so k=80 maps it
                # to ~0.65 before gamma, giving ~0.7 alpha — clearly visible cyan stripe.
                _diff = np.clip((_I_n - _I_v) * 80.0, 0.0, 1.0)  # k=80 amplification
                _diff = np.power(_diff + 1e-8, 0.7).astype(np.float32)  # γ=0.7 gamma correction
                self._vessel_diff_overlays[_proj_name] = _diff
                print(f"[Viewport]   {_proj_name}: vessel diff max={_diff.max():.3f}", flush=True)
            del _rnd_normal, _rnd_vessel
            self.vessel_overlay_on = True  # guarantee ON whenever overlays are built
            print("[Viewport] Vessel overlays ready. Press F to toggle (cyan=vessels, white=catheter).", flush=True)

        self.sim["solver"] = solver
        self.sim["renderer"] = renderer
        self.sim["mesh_frame_offset_m"] = mesh_frame_offset_m
        self.sim["base_collision_iterations"] = int(tuned["collision_iterations"])
        self.sim["ct_offset_m"] = ct_offset_m
        self.sim["local_z0_m"] = local_z0_m
        self.sim["ct_origin_mm"] = np.array([ox, oy, oz], dtype=np.float32)
        self.sim["vol_shape_xyz"] = (int(nx), int(ny), int(nz))
        self.sim["spacing_xyz_mm"] = (float(sx_mm), float(sy_mm), float(sz_mm))
        self.sim["initial_pos"] = solver.positions.cpu().numpy().copy()
        self.sim["initial_ori"] = solver.orientations.cpu().numpy().copy()
        self.sim["tip_ct_mm"] = np.zeros(3, dtype=np.float32)
        self.sim["particle_radius_mm"] = particle_radius_m * 1000.0
        self.sim["render_radius_mm"] = render_radius_mm
        self.sim["render_mu"] = render_mu
        self.sim["num_substeps"] = int(tuned["num_substeps"])
        self.sim["collision_iterations"] = int(tuned["collision_iterations"])
        self.sim["expected_seg_len_mm"] = seg_len * 1000.0
        self.sim["expected_seg_len_m"] = seg_len
        self.sim["vessel_source"] = selected_source
        self.sim["track_axis_status"] = axis_status
        self.sim["guided_track_axis_status"] = axis_status
        self.sim["track_dir"] = track_dir.copy()
        self.sim["guide_path_local_m"] = guide_path_local_m
        self.sim["guide_path_length_m"] = float(guide_path_length_m)
        if guide_path_local_m is not None and guide_path_local_m.shape[0] >= 2:
            seg_guide = np.linalg.norm(np.diff(guide_path_local_m, axis=0), axis=1)
            cum_guide = np.concatenate(([0.0], np.cumsum(seg_guide))).astype(np.float32)
            self.sim["guide_cumlen_m"] = cum_guide
            self.sim["guide_max_root_s_m"] = float(
                max(0.0, float(cum_guide[-1]) - float(seg_len) * float(self.num_segments))
            )
        else:
            self.sim["guide_cumlen_m"] = None
            self.sim["guide_max_root_s_m"] = 0.0
        self._guide_s_m = 0.0

        # Warm up renderer once to avoid a long first interaction frame.
        _ = self._render_fluoro(self.current_projection)
        print("[Viewport] Warming up solver kernels (one-time compile on cold cache) ...", flush=True)
        _t_warm = time.perf_counter()
        try:
            solver.apply_proximal_control(0.0, 0.0, DT)
            solver.step(DT)
            self._reset_solver()
            print(
                f"[Viewport] Solver warmup complete in {(time.perf_counter() - _t_warm):.1f}s",
                flush=True,
            )
        except Exception as exc:
            print(f"[Viewport] Solver warmup skipped ({type(exc).__name__}: {exc})", flush=True)

        # Seed valid-state rollback buffers.
        self._last_valid_positions_m = solver.positions.cpu().numpy().copy()
        self._last_valid_orientations_xyzw = solver.orientations.cpu().numpy().copy()
        self._apply_sign_scale_autofix()

    def _apply_sign_scale_autofix(self, force: bool = False) -> None:
        """Apply the sign convention that minimizes outside-mesh particles."""
        solver = self.sim.get("solver")
        if solver is None:
            return
        self._update_outside_metric()
        suggested = float(self._suggested_sign_scale)
        mismatch = abs(float(self.sign_scale) - suggested) > 0.5
        if force or mismatch or self._outside_percent > 25.0:
            if mismatch or force:
                self.sign_scale = suggested
                solver.sign_scale = suggested
                print(
                    f"[Viewport] Auto-selected collision sign_scale={int(suggested):+d} "
                    f"(outside +1/-1 = {self._outside_plus_percent:.1f}% / "
                    f"{self._outside_minus_percent:.1f}%, "
                    f"containment={100.0 - self._outside_percent:.1f}%)",
                    flush=True,
                )
                self._update_outside_metric()

    def _pos_to_vol_mm(self, pos_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos_ct_mm = (pos_m - self.sim["local_z0_m"] + self.sim["ct_offset_m"]) * 1000.0
        pos_vol_mm = pos_ct_mm - self.sim["ct_origin_mm"]
        return pos_vol_mm, pos_ct_mm

    def _render_fluoro(self, projection: str) -> np.ndarray:
        solver = self.sim["solver"]
        # DSA mode swaps in the vessel-μ-boosted renderer so the lumen + catheter
        # bake into the DRR as dark contrast (native angiogram); else the normal DRR.
        use_dsa = self._dsa_on and self._dsa_renderer is not None
        renderer = self._dsa_renderer if use_dsa else self.sim["renderer"]
        pos_vol_mm, pos_ct_mm = self._pos_to_vol_mm(solver.positions.cpu().numpy())
        if self.centerline_mode == "smooth":
            render_positions = _prepare_render_polyline(pos_vol_mm, float(self.sim["expected_seg_len_mm"]))
        else:
            # Even in raw mode, clamp pathological segment jumps to avoid one-frame spikes.
            render_positions = _clip_polyline_jumps(
                pos_vol_mm.astype(np.float32),
                jump_cap_mm=max(3.0 * float(self.sim["expected_seg_len_mm"]), 2.0),
            )
        cat = CatheterSegmentData(
            positions=render_positions,
            radii=float(self.sim["render_radius_mm"]),
            mu_values=float(self.sim["render_mu"]),
        )
        img = renderer.render_batch_with_catheter(PROJECTIONS[projection], _TRANS_ZERO, [cat])[0]
        self.sim["tip_ct_mm"] = pos_ct_mm[-1]
        if use_dsa and abs(self.dsa_gamma - 1.0) > 1e-3:
            # Brightness control for the angiogram (γ<1 lifts mids → brighter).
            img = np.power(np.clip(img, 0.0, 1.0), self.dsa_gamma)
        img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        frame = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        # Cache the clean frame (catheter wire only, no vessel overlay) for the left panel.
        self._last_raw_frame = frame.copy()
        # Overlay vessel cues for guided navigation.
        # In DSA mode the contrast bolus already shows the lumen, so the overlay
        # is suppressed to keep the angiogram clean.
        if self.vessel_overlay_on and not use_dsa:
            diff = self._vessel_diff_overlays.get(projection)
            if diff is not None:
                alpha = np.clip(diff, 0.0, 0.95).astype(np.float32)
                out = frame.astype(np.float32)
                out[:, :, 0] = out[:, :, 0] * (1.0 - alpha) + 255.0 * alpha  # B  (cyan)
                out[:, :, 1] = out[:, :, 1] * (1.0 - alpha) + 255.0 * alpha  # G  (cyan)
                out[:, :, 2] = out[:, :, 2] * (1.0 - alpha) + 0.0 * alpha  # R  (no red)
                frame = np.clip(out, 0, 255).astype(np.uint8)
        if use_dsa:
            cv2.putText(
                frame,
                f"DSA ANGIO  g{self.dsa_gamma:.1f}",
                (12, 26),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (60, 200, 255),
                1,
                cv2.LINE_AA,
            )
        return frame

    def _restore_last_valid_pose(self) -> None:
        """Restore last known physically-valid solver state."""
        if self._last_valid_positions_m is None or self._last_valid_orientations_xyzw is None:
            return
        solver = self.sim["solver"]
        ws = solver._ws
        pos = torch.from_numpy(self._last_valid_positions_m)
        ori = torch.from_numpy(self._last_valid_orientations_xyzw)
        wp.to_torch(ws.positions).copy_(pos)
        wp.to_torch(ws.predicted_positions).copy_(pos)
        if hasattr(ws, "prev_positions"):
            wp.to_torch(ws.prev_positions).copy_(pos)
        wp.to_torch(ws.velocities).zero_()
        if hasattr(ws, "forces"):
            wp.to_torch(ws.forces).zero_()
        wp.to_torch(ws.orientations).copy_(ori)
        if hasattr(ws, "predicted_orientations"):
            wp.to_torch(ws.predicted_orientations).copy_(ori)
        if hasattr(ws, "prev_orientations"):
            wp.to_torch(ws.prev_orientations).copy_(ori)

    def _validate_and_guard_pose(self) -> bool:
        """Validate pose; rollback on unstable outliers. Returns True when pose is valid."""
        if self.pose_guard == "off":
            solver = self.sim["solver"]
            self._last_valid_positions_m = solver.positions.cpu().numpy().copy()
            self._last_valid_orientations_xyzw = solver.orientations.cpu().numpy().copy()
            return True

        solver = self.sim["solver"]
        positions_m = solver.positions.cpu().numpy()
        # "safe" mode is intentionally permissive for interactive control:
        # only rollback on clearly corrupted states (NaN/Inf or huge coordinate explosion).
        if not np.isfinite(positions_m).all():
            self._invalid_pose_count += 1
            self._restore_last_valid_pose()
            return False

        if float(np.max(np.abs(positions_m))) > 5.0:
            self._invalid_pose_count += 1
            self._restore_last_valid_pose()
            return False

        # Pose accepted: update rollback buffer.
        self._last_valid_positions_m = positions_m.copy()
        self._last_valid_orientations_xyzw = solver.orientations.cpu().numpy().copy()
        return True

    def _update_outside_metric(self) -> None:
        """Compute containment diagnostics from signed distance to collision mesh."""
        solver = self.sim["solver"]
        if not hasattr(solver, "collision_mesh") or solver.collision_mesh is None:
            self._outside_metric_status = "no-mesh"
            self._outside_max_phi_mm = 0.0
            self._outside_percent = 0.0
            return
        try:
            pos_m = solver.positions.cpu().numpy()
            phi = compute_signed_distances(
                points=pos_m.astype(np.float32),
                mesh=solver.collision_mesh,
                max_dist=float(getattr(solver, "max_dist", 0.05)),
                sign_scale=float(getattr(solver, "sign_scale", 1.0)),
                device="cuda",
            )
            outside = phi > 0.0
            self._outside_max_phi_mm = float(max(0.0, np.max(phi)) * 1000.0)
            self._outside_percent = float(100.0 * np.mean(outside))

            # Sign-convention diagnostic: compare outside% at +1 vs -1.
            phi_plus = compute_signed_distances(
                points=pos_m.astype(np.float32),
                mesh=solver.collision_mesh,
                max_dist=float(getattr(solver, "max_dist", 0.05)),
                sign_scale=1.0,
                device="cuda",
            )
            phi_minus = compute_signed_distances(
                points=pos_m.astype(np.float32),
                mesh=solver.collision_mesh,
                max_dist=float(getattr(solver, "max_dist", 0.05)),
                sign_scale=-1.0,
                device="cuda",
            )
            self._outside_plus_percent = float(100.0 * np.mean(phi_plus > 0.0))
            self._outside_minus_percent = float(100.0 * np.mean(phi_minus > 0.0))
            self._suggested_sign_scale = 1.0 if self._outside_plus_percent <= self._outside_minus_percent else -1.0
            # Live correction: keep sign convention aligned with measured containment.
            if abs(float(self.sign_scale) - float(self._suggested_sign_scale)) > 0.5:
                self.sign_scale = float(self._suggested_sign_scale)
                solver.sign_scale = float(self._suggested_sign_scale)

            # AABB overlap diagnostic between catheter points and collision mesh.
            if self._mesh_aabb_min_m is not None and self._mesh_aabb_max_m is not None:
                cat_min = np.min(pos_m, axis=0)
                cat_max = np.max(pos_m, axis=0)
                overlap_axes = np.logical_and(cat_max >= self._mesh_aabb_min_m, cat_min <= self._mesh_aabb_max_m)
                self._aabb_overlap = bool(np.all(overlap_axes))
            else:
                self._aabb_overlap = False

            self._outside_metric_status = "ok"
        except Exception:
            self._outside_metric_status = "error"
            self._outside_max_phi_mm = 0.0
            self._outside_percent = 0.0
            self._outside_plus_percent = 0.0
            self._outside_minus_percent = 0.0
            self._suggested_sign_scale = 1
            self._aabb_overlap = False

    def _apply_visual_style(self, frame_bgr: np.ndarray, panel_key: str = "default") -> np.ndarray:
        """Apply stylistic detector post-processing for presentation-quality output."""
        if self.visual_style == "default":
            return frame_bgr

        # Convert once to float for the post-processing stack.
        frame = frame_bgr.astype(np.float32) / 255.0

        # Slight detector blur + bloom softens sharp synthetic edges.
        base = cv2.GaussianBlur(frame, (0, 0), sigmaX=0.9)
        bloom = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.2)
        frame = np.clip(0.85 * base + 0.15 * bloom, 0.0, 1.0)

        # Temporal persistence creates cine-fluoro continuity.
        # Each panel uses its own buffer so overlays on the right panel
        # cannot bleed into the left panel through temporal blending.
        alpha = 0.24
        if panel_key not in self._temporal_frames:
            self._temporal_frames[panel_key] = frame.copy()
        else:
            self._temporal_frames[panel_key] = (1.0 - alpha) * self._temporal_frames[panel_key] + alpha * frame
        frame = self._temporal_frames[panel_key]

        # Mild detector noise and vignette for realistic look.
        noise = self._rng.normal(0.0, 0.012, frame.shape).astype(np.float32)
        frame = np.clip(frame + noise, 0.0, 1.0)
        h, w = frame.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cx, cy = w * 0.5, h * 0.5
        rr = np.sqrt(((xx - cx) / max(w, 1)) ** 2 + ((yy - cy) / max(h, 1)) ** 2)
        vignette = np.clip(1.0 - 0.9 * rr, 0.78, 1.0).astype(np.float32)
        frame *= vignette[:, :, None]

        # Filmic contrast curve.
        frame = np.clip(np.power(frame, 1.06), 0.0, 1.0)
        return (frame * 255.0).astype(np.uint8)

    def _reset_solver(self) -> None:
        solver = self.sim["solver"]
        ws = solver._ws

        init_pos = torch.from_numpy(self.sim["initial_pos"])
        init_ori = torch.from_numpy(self.sim["initial_ori"])

        wp.to_torch(ws.positions).copy_(init_pos)
        wp.to_torch(ws.predicted_positions).copy_(init_pos)
        if hasattr(ws, "prev_positions"):
            wp.to_torch(ws.prev_positions).copy_(init_pos)
        wp.to_torch(ws.velocities).zero_()
        if hasattr(ws, "forces"):
            wp.to_torch(ws.forces).zero_()

        wp.to_torch(ws.orientations).copy_(init_ori)
        if hasattr(ws, "predicted_orientations"):
            wp.to_torch(ws.predicted_orientations).copy_(init_ori)
        if hasattr(ws, "prev_orientations"):
            wp.to_torch(ws.prev_orientations).copy_(init_ori)

        solver.reset_cuda_graph()
        if self._cuda_available:
            torch.cuda.synchronize()
        self._guide_s_m = 0.0
        self._guide_ds_frame_mm = 0.0
        self._guide_free_active = False
        self._guide_boundary_state = "none"
        self._guide_blend_t = 1.0
        self.frame_idx = 0

    def _apply_centerline_guide(self, velocity_cmd_m_s: float, dt: float) -> float:
        """Project full catheter onto curved centerline path; return residual velocity.

        This runs in two modes:
          - pre-step: dt>0, velocity command updates the root arclength state.
          - post-step: dt==0, re-project at the same arclength to remove physics drift.
        """
        guide = self.sim.get("guide_path_local_m")
        if guide is None:
            return velocity_cmd_m_s
        path = np.asarray(guide, dtype=np.float32)
        if path.ndim != 2 or path.shape[0] < 2:
            return velocity_cmd_m_s

        total = float(self.sim.get("guide_path_length_m", 0.0))
        if total <= 1e-6:
            return velocity_cmd_m_s
        cum = self.sim.get("guide_cumlen_m")
        if cum is None:
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
            cum = np.concatenate(([0.0], np.cumsum(seg))).astype(np.float32)
        max_root_s = float(
            self.sim.get(
                "guide_max_root_s_m", max(0.0, total - float(self.sim["expected_seg_len_m"]) * float(self.num_segments))
            )
        )
        prev_s = float(self._guide_s_m)
        # Always clamp arclength to valid guide range — no free mode, no shape artifacts.
        if dt > 0.0:
            self._guide_s_m = float(np.clip(prev_s + float(velocity_cmd_m_s) * float(dt), 0.0, max_root_s))
            self._guide_ds_frame_mm = (self._guide_s_m - prev_s) * 1000.0
            # Track boundary state for HUD only (no behavioral change).
            if self._guide_s_m <= 0.0 and velocity_cmd_m_s < 0.0:
                self._guide_boundary_state = "retract_end"
            elif self._guide_s_m >= max_root_s and velocity_cmd_m_s > 0.0:
                self._guide_boundary_state = "advance_end"
            else:
                self._guide_boundary_state = "none"
        self._guide_free_active = False

        solver = self.sim["solver"]
        ws = solver._ws
        pos_t = wp.to_torch(ws.positions)
        pred_t = wp.to_torch(ws.predicted_positions)
        num_points = self.num_segments + 1
        seg_len = float(self.sim["expected_seg_len_m"])
        target_pos = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            s_i = self._guide_s_m + float(i) * seg_len
            p_i, t_i = _polyline_interp_with_cum(path, cum, s_i)
            target_pos[i] = p_i
        target_pos_t = torch.as_tensor(target_pos, dtype=pos_t.dtype, device=pos_t.device)

        # Guidance follow strength.
        # - During active advance, use the configured stiffness (default 0.65).
        # - During retraction/idle pre-step, keep a strong blend.
        # - During post-step stabilization (dt==0), do a full snap to remove residual drift.
        active_advance = dt > 0.0 and velocity_cmd_m_s > 1e-8
        if dt <= 0.0:
            follow = 1.0
        elif active_advance:
            follow = max(float(self.advance_guide_stiffness), 0.92)
        else:
            follow = 0.95

        # Safety clamp: if tip diverges from target path, force near-hard projection.
        # This eliminates the "looks off" frames where the catheter visibly runs
        # outside the cyan corridor during transitions.
        tip_err_mm = float(torch.linalg.norm(pos_t[-1] - target_pos_t[-1]).item() * 1000.0)
        if tip_err_mm > 2.0:
            follow = max(follow, 0.98)

        pos_t.mul_(1.0 - follow).add_(target_pos_t, alpha=follow)
        pred_t.copy_(pos_t)
        if hasattr(ws, "prev_positions"):
            wp.to_torch(ws.prev_positions).copy_(pos_t)
        if hasattr(ws, "velocities"):
            wp.to_torch(ws.velocities).zero_()

        # Keep solver track aligned with current guide tangent.
        _, t0 = _polyline_interp_with_cum(path, cum, self._guide_s_m)
        solver.track_start = wp.vec3(float(target_pos[0, 0]), float(target_pos[0, 1]), float(target_pos[0, 2]))
        solver.track_dir = wp.vec3(float(t0[0]), float(t0[1]), float(t0[2]))
        self.sim["track_dir"] = np.array([float(t0[0]), float(t0[1]), float(t0[2])], dtype=np.float32)
        return 0.0

    def _tip_guide_metrics(self) -> tuple[float, float]:
        """Return (tip_s_m, tip_to_guide_err_mm) in centerline mode."""
        guide = self.sim.get("guide_path_local_m")
        cum = self.sim.get("guide_cumlen_m")
        if guide is None or cum is None:
            return 0.0, 0.0
        path = np.asarray(guide, dtype=np.float32)
        cum_m = np.asarray(cum, dtype=np.float32)
        if path.ndim != 2 or path.shape[0] < 2 or cum_m.ndim != 1 or cum_m.shape[0] != path.shape[0]:
            return 0.0, 0.0
        tip = self.sim["solver"].positions.cpu().numpy()[-1].astype(np.float32)

        best_dist2 = float("inf")
        best_s = 0.0
        for i in range(path.shape[0] - 1):
            a = path[i]
            b = path[i + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 <= 1e-12:
                continue
            t = float(np.clip(np.dot(tip - a, ab) / ab2, 0.0, 1.0))
            p = a + t * ab
            d2 = float(np.dot(tip - p, tip - p))
            if d2 < best_dist2:
                best_dist2 = d2
                best_s = float(cum_m[i] + t * np.sqrt(ab2))
        return best_s, float(np.sqrt(max(best_dist2, 0.0)) * 1000.0)

    # ── panel rendering helpers ───────────────────────────────────────────────

    @staticmethod
    def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple = (130, 148, 128)) -> None:
        """Small category label — FONT_HERSHEY_PLAIN renders crisply at
        small sizes (unlike DUPLEX which blurs below scale 0.5)."""
        cv2.putText(img, text.upper(), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.90, color, 1, cv2.LINE_AA)

    @staticmethod
    def _value(img: np.ndarray, text: str, x: int, y: int, color: tuple = (230, 240, 230), scale: float = 0.68) -> None:
        """Large metric value — FONT_HERSHEY_DUPLEX for crisp strokes."""
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, 1, cv2.LINE_AA)

    @staticmethod
    def _section_bg(img: np.ndarray, y0: int, y1: int, width: int, shade: int = 26) -> None:
        """Subtle alternating background stripe for a section band."""
        cv2.rectangle(img, (0, y0), (width, y1), (shade, shade + 2, shade), -1)

    @staticmethod
    def _divider(img: np.ndarray, y: int, width: int, color: tuple = (55, 68, 55)) -> None:
        cv2.line(img, (8, y), (width - 8, y), color, 1, cv2.LINE_AA)

    def _build_control_panel(self, size: int) -> np.ndarray:
        """Render the HUD at its tuned design resolution, then upscale to the
        requested panel size so text stays large and legible at high --det-size."""
        design = 512
        panel = self._render_control_panel(design)
        if size != design:
            interp = cv2.INTER_AREA if size < design else cv2.INTER_CUBIC
            panel = cv2.resize(panel, (size, size), interpolation=interp)
        return panel

    def _render_control_panel(self, size: int) -> np.ndarray:
        """Fixed-layout info panel — each block occupies a defined pixel band
        so content never overflows regardless of size."""
        panel = np.full((size, size, 3), 16, dtype=np.uint8)

        tip = self.sim["tip_ct_mm"]
        mouse_vel_cmd, mouse_torque_cmd = self.mouse.commands()
        key_vel_cmd, key_torque_cmd = self.keyboard.commands()
        vel_mm_s = (key_vel_cmd + mouse_vel_cmd) * 1000.0
        torque_cmd = key_torque_cmd + mouse_torque_cmd
        fps = 1000.0 / max(self.last_loop_ms, 1.0)
        self.fps_smooth = 0.9 * self.fps_smooth + 0.1 * fps if self.fps_smooth > 0.0 else fps

        containment = 100.0 - self._outside_percent

        # BGR palette
        WHITE = (245, 248, 245)
        CYAN = (200, 230, 60)  # chartreuse-cyan — stands out on dark bg
        GREEN = (70, 215, 95)
        RED = (75, 75, 210)
        AMBER = (50, 180, 240)
        DIM = (180, 198, 175)  # label color — brightened for readability
        HINT = (145, 158, 108)

        P = 12  # left pad

        # ── Fixed pixel bands (designed for size=320, scales with size) ───
        # Bands are expressed as fractions of size so the layout adapts.
        H = size
        b = {
            "title": (0, int(H * 0.085)),  # 0 – 27
            "proj": (int(H * 0.09), int(H * 0.21)),  # 29 – 67
            "guide": (int(H * 0.21), int(H * 0.32)),  # 67 – 102
            "tip": (int(H * 0.32), int(H * 0.57)),  # 102 – 182
            "motion": (int(H * 0.57), int(H * 0.70)),  # 182 – 224
            "status": (int(H * 0.70), int(H * 0.82)),  # 224 – 262
            "map": (int(H * 0.82), H),  # 262 – 320
        }

        # ── Title ──────────────────────────────────────────────────────────
        t0, t1 = b["title"]
        cv2.rectangle(panel, (0, t0), (size, t1), (32, 42, 32), -1)
        cv2.line(panel, (0, t1), (size, t1), (60, 80, 60), 1)

        # ── C-arm projection ───────────────────────────────────────────────
        p0, p1 = b["proj"]
        self._section_bg(panel, p0, p1, size, 20)
        self._label(panel, "C-Arm Projection", P, p0 + 13, DIM)
        self._value(panel, self.current_projection, P, p0 + 36, CYAN, scale=0.76)

        # ── Guidance / vessel ──────────────────────────────────────────────
        g0, g1 = b["guide"]
        gh = g1 - g0
        self._divider(panel, g0, size)
        self._section_bg(panel, g0, g1, size, 24)
        self._label(panel, "Guidance / Vessel", P, g0 + int(gh * 0.25), DIM)
        self._value(panel, self.navigation_mode, P, g0 + int(gh * 0.52), WHITE, scale=0.58)
        v_col = GREEN if self.vessel_overlay_on else RED
        v_txt = "Vessel ON  [F=off]" if self.vessel_overlay_on else "Vessel OFF [F=on]"
        self._value(panel, v_txt, P, g0 + int(gh * 0.74), v_col, scale=0.46)
        cl_col = GREEN if self.centerline_on else RED
        cl_txt = "Centerline ON  [C=off]" if self.centerline_on else "Centerline OFF [C=on]"
        self._value(panel, cl_txt, P, g0 + int(gh * 0.95), cl_col, scale=0.46)

        # ── Tip position ───────────────────────────────────────────────────
        tp0, tp1 = b["tip"]
        self._divider(panel, tp0, size)
        self._section_bg(panel, tp0, tp1, size, 20)
        self._label(panel, "Tip Position  (CT mm)", P, tp0 + 13, DIM)
        row_h = (tp1 - tp0 - 20) // 3
        for i, (axis, val) in enumerate([("X", tip[0]), ("Y", tip[1]), ("Z", tip[2])]):
            ry = tp0 + 20 + row_h * i + int(row_h * 0.72)
            self._value(panel, f"{axis}   {val:+7.1f}", P, ry, CYAN, scale=0.64)

        # ── Velocity / torque ──────────────────────────────────────────────
        m0, m1 = b["motion"]
        self._divider(panel, m0, size)
        self._section_bg(panel, m0, m1, size, 24)
        self._label(panel, "Velocity  (mm/s)", P, m0 + 13, DIM)
        vel_col = CYAN if abs(vel_mm_s) > 0.5 else WHITE
        self._value(panel, f"{vel_mm_s:+.1f}", P, m0 + 32, vel_col, scale=0.74)
        self._label(panel, "Torque", size // 2, m0 + 13, DIM)
        self._value(panel, f"{torque_cmd:+.3f}", size // 2, m0 + 32, WHITE, scale=0.62)

        # ── Status row: containment / FPS / paused ─────────────────────────
        s0, s1 = b["status"]
        self._divider(panel, s0, size)
        self._section_bg(panel, s0, s1, size, 20)
        cont_col = GREEN if containment > 80 else (AMBER if containment > 50 else RED)
        pau_col = RED if self.paused else GREEN
        col2 = size // 2
        lrow = s0 + 13
        self._label(panel, "Contain", P, lrow, DIM)
        self._label(panel, "State", col2, lrow, DIM)
        vrow = s0 + 32
        self._value(panel, f"{containment:.0f}%", P, vrow, cont_col, scale=0.62)
        pau_txt = "PAUSE" if self.paused else "RUN"
        self._value(panel, pau_txt, col2, vrow, pau_col, scale=0.62)
        extra_y = s0 + 50
        if self.insertion_axis == "centerline":
            tip_s_m, tip_err_mm = self._tip_guide_metrics()
            root_max = float(self.sim.get("guide_max_root_s_m", 0.0))
            self._label(
                panel,
                f"Guide {self._guide_s_m * 1000.0:.0f}/{root_max * 1000.0:.0f} mm" f"  err {tip_err_mm:.1f} mm",
                P,
                extra_y,
                AMBER,
            )
        else:
            self._label(panel, f"Loop {self.last_loop_ms:.1f} ms", P, extra_y, DIM)

        # ── Top-down catheter map + key hints ─────────────────────────────
        mp0, mp1 = b["map"]
        self._divider(panel, mp0, size)

        map_x1 = int(size * 0.50)
        hint_x0 = map_x1 + 8

        # Map box with subtle border
        cv2.rectangle(panel, (P - 2, mp0 + 3), (map_x1, mp1 - 3), (20, 24, 20), -1)
        cv2.rectangle(panel, (P - 2, mp0 + 3), (map_x1, mp1 - 3), (55, 70, 55), 1, cv2.LINE_AA)
        cv2.putText(panel, "TOP VIEW", (P, mp0 + 13), cv2.FONT_HERSHEY_PLAIN, 0.80, DIM, 1, cv2.LINE_AA)

        pos_vol_mm, pos_ct_mm = self._pos_to_vol_mm(self.sim["solver"].positions.cpu().numpy())
        _ = pos_vol_mm
        xy = pos_ct_mm[:, :2]
        if xy.shape[0] > 1:
            center = xy.mean(axis=0, keepdims=True)
            span = max(float(np.max(np.linalg.norm(xy - center, axis=1))), 1.0)
            norm_xy = (xy - center) / span
            mc = np.array([(P + map_x1) * 0.5, (mp0 + 16 + mp1) * 0.5], dtype=np.float32)
            ms = min(map_x1 - P, mp1 - mp0 - 18) * 0.40
            pts = np.zeros((xy.shape[0], 2), dtype=np.int32)
            pts[:, 0] = np.clip((mc[0] + norm_xy[:, 0] * ms).astype(np.int32), P, map_x1 - 1)
            pts[:, 1] = np.clip((mc[1] - norm_xy[:, 1] * ms).astype(np.int32), mp0 + 14, mp1 - 2)
            cv2.polylines(panel, [pts.reshape(-1, 1, 2)], isClosed=False, color=(120, 200, 255), thickness=2)
            cv2.circle(panel, tuple(pts[0]), 4, (80, 80, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, tuple(pts[-1]), 5, (40, 255, 255), -1, cv2.LINE_AA)

        # Key hints in the right column
        hints = [
            "W/S  fwd / back",
            "A/D  rotate",
            "1-4  C-arm view",
            "F  vessel  C  centerline",
            "X  DSA   -/=  brightness",
            "Guided mode only  V  style",
            "R reset  Q quit  SPC pause",
        ]
        hy = mp0 + 14
        line_h = max(13, (mp1 - mp0 - 10) // len(hints))
        for h in hints:
            if hy + 10 > mp1:
                break
            cv2.putText(panel, h, (hint_x0, hy), cv2.FONT_HERSHEY_PLAIN, 0.85, HINT, 1, cv2.LINE_AA)
            hy += line_h

        return panel

    def _project_rod_to_pixels(self, proj_name: str) -> np.ndarray:
        """Project rod particle positions (vol_mm) → 2D pixel coords.

        Uses the same ZXY-Euler cone-beam geometry as the Slang DRR shader:
          source_local  = (0, 0, -SID)
          detector_z    = SDD - SID
          pixel (u, v)  = (det_x / ps + W/2,  det_y / ps + H/2)
        """
        euler = PROJECTIONS[proj_name][0]
        rx, ry, rz = float(euler[0]), float(euler[1]), float(euler[2])
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        R = np.array(
            [
                [cz * cy - sz * sx * sy, -sz * cx, cz * sy + sz * sx * cy],
                [sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * sx * cy],
                [-cx * sy, sx, cx * cy],
            ],
            dtype=np.float64,
        )

        SID = 510.0  # source-to-isocenter mm  (SlangDiffDRRConfig default)
        SDD = 1020.0  # source-to-detector mm   (SlangDiffDRRConfig default)
        ps = self.pixel_spacing_mm  # must match the DRR detector spacing (--pixel-spacing-mm)
        W = H = float(self.det_size)

        nx, ny, nz = self.sim["vol_shape_xyz"]
        sx_mm, sy_mm, sz_mm = self.sim["spacing_xyz_mm"]
        iso = np.array([nx * sx_mm * 0.5, ny * sy_mm * 0.5, nz * sz_mm * 0.5], dtype=np.float64)

        pos_vol_mm, _ = self._pos_to_vol_mm(self.sim["solver"].positions.cpu().numpy())
        pts_local = (pos_vol_mm.astype(np.float64) - iso) @ R  # R^T via row-vector convention

        denom = pts_local[:, 2] + SID
        u = SDD * pts_local[:, 0] / denom / ps + W * 0.5
        v = SDD * pts_local[:, 1] / denom / ps + H * 0.5
        return np.stack([u, v], axis=-1).astype(np.float32)

    def _build_centerline_panel(self, fluoro_bgr: np.ndarray, proj_name: str) -> np.ndarray:
        """Return fluoro_bgr with the projected catheter centerline drawn on top."""
        panel = fluoro_bgr.copy()
        uv = self._project_rod_to_pixels(proj_name)
        H, W = panel.shape[:2]
        pts = [(int(round(float(u))), int(round(float(v)))) for u, v in uv if 0 <= float(u) < W and 0 <= float(v) < H]
        # Scale dot size proportionally to detector resolution so it's always visible.
        dot_r = max(4, W // 100)
        line_t = max(3, W // 160)
        if len(pts) >= 2:
            cv2.polylines(
                panel,
                [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)],
                isClosed=False,
                color=(0, 255, 180),
                thickness=line_t,
            )
        for p in pts:
            cv2.circle(panel, p, dot_r, (0, 255, 180), -1, cv2.LINE_AA)
        return panel

    def _handle_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Combined mouse handler: ignore clicks in the right panels, drag on the left."""
        if x >= self.det_size:
            return
        self.mouse.handle_mouse(event, x, y, flags, param)

    def run(self) -> None:
        if os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None:
            raise RuntimeError("No display found. This viewport app requires a GUI session.")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Fit window to screen: leave room for OS title bar + taskbar (~80px).
        try:
            import subprocess  # noqa: PLC0415

            out = subprocess.check_output(["xdpyinfo"], timeout=2).decode()
            for line in out.splitlines():
                if "dimensions:" in line:
                    parts = line.split()
                    idx = parts.index("dimensions:")
                    wh = parts[idx + 1].split("x")
                    screen_w, screen_h = int(wh[0]), int(wh[1])
                    max_h = screen_h - 80
                    max_w = screen_w - 20
                    scale = min(max_h / self.det_size, max_w / (self.det_size * 3))
                    win_w = int(self.det_size * 3 * scale)
                    win_h = int(self.det_size * scale)
                    cv2.resizeWindow(self.window_name, win_w, win_h)
                    cv2.moveWindow(self.window_name, 0, 0)
                    break
            else:
                cv2.resizeWindow(self.window_name, self.det_size * 3, self.det_size)
        except Exception:
            cv2.resizeWindow(self.window_name, self.det_size * 3, self.det_size)
        cv2.setMouseCallback(self.window_name, self._handle_mouse)

        # Attempt to raise and focus the window automatically on Linux/X11.
        try:
            subprocess.Popen(
                ["xdotool", "search", "--name", self.window_name, "windowfocus", "--sync"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError):
            pass  # xdotool not installed — user must click manually

        while True:
            t0 = time.perf_counter()
            solver = self.sim["solver"]

            if not self.paused:
                mouse_vel_cmd, mouse_torque_cmd = self.mouse.commands()
                key_vel_cmd, key_torque_cmd = self.keyboard.commands()
                velocity_cmd = mouse_vel_cmd + key_vel_cmd
                torque_cmd = mouse_torque_cmd + key_torque_cmd

                # Input deadband: suppress tiny residual decay values so hold mode is truly static.
                if abs(velocity_cmd) < 2e-4:  # 0.2 mm/s
                    velocity_cmd = 0.0
                if abs(torque_cmd) < 2e-4:
                    torque_cmd = 0.0

                command_mag = abs(velocity_cmd) + abs(torque_cmd)
                should_step = True
                # Match Gradio behavior by default: no autonomous dynamics at zero input.
                if self.idle_mode == "hold" and command_mag < 1e-8:
                    should_step = False
                if should_step:
                    velocity_for_solver = self._apply_centerline_guide(velocity_cmd, DT)
                    solver.track_stiffness = 0.0
                    solver.apply_proximal_control(velocity_for_solver, torque_cmd, DT)
                    solver.step(DT)
                    self._apply_centerline_guide(0.0, 0.0)
                    self._validate_and_guard_pose()
                    self.frame_idx += 1
                else:
                    self._guide_ds_frame_mm = 0.0
            else:
                self._guide_ds_frame_mm = 0.0

            # Update containment diagnostics at a lower rate to reduce overhead.
            if self.frame_idx % OUTSIDE_METRIC_EVERY_N_FRAMES == 0:
                self._update_outside_metric()

            if self._cuda_available:
                torch.cuda.synchronize()

            # Single GPU render — vessel overlay goes to right panel only.
            fluoro_right = self._render_fluoro(self.current_projection)
            fluoro_right = self._apply_visual_style(fluoro_right, panel_key="right")

            # Left panel: Beer-Lambert catheter wire only — no vessel overlay, no projected dots.
            # This is the "raw fluoroscope" view: what the clinician sees without AI assistance.
            raw = self._last_raw_frame if self._last_raw_frame is not None else fluoro_right
            fluoro_left = self._apply_visual_style(raw.copy(), panel_key="left")

            # Right panel: vessel overlay (DSA) + projected catheter dots — the AI-assisted view.
            if self.centerline_on:
                fluoro_right = self._build_centerline_panel(fluoro_right, self.current_projection)

            panel = self._build_control_panel(self.det_size)
            canvas = np.hstack([fluoro_left, fluoro_right, panel])

            cv2.imshow(self.window_name, canvas)

            key_full = cv2.waitKeyEx(1)
            self._last_key_code = int(key_full)
            key = 255 if key_full < 0 else (key_full & 0xFF)

            if key in (27, ord("q")):
                break
            if key == ord(" "):
                self.paused = not self.paused
                if self.paused:
                    self.keyboard.clear()
            elif key in PROJECTION_KEYS:
                self.current_projection = PROJECTION_KEYS[key]
            elif key in (ord("r"), ord("R")):
                self._reset_solver()
            elif key in (ord("v"), ord("V")):
                self.visual_style = "cinematic" if self.visual_style == "default" else "default"
                self._temporal_frames.clear()
            elif key in (ord("f"), ord("F")):
                self.vessel_overlay_on = not self.vessel_overlay_on
            elif key in (ord("x"), ord("X")):
                if self._dsa_renderer is not None:
                    self._dsa_on = not self._dsa_on
                    self._temporal_frames.clear()  # avoid cross-polarity ghosting
            elif key in (ord("-"), ord("_")):
                self.dsa_gamma = float(min(3.0, self.dsa_gamma + 0.1))  # darker
                self._temporal_frames.clear()
            elif key in (ord("="), ord("+")):
                self.dsa_gamma = float(max(0.3, self.dsa_gamma - 0.1))  # brighter
                self._temporal_frames.clear()
            elif key in (ord("c"), ord("C")):
                self.centerline_on = not self.centerline_on
            elif key in (
                ord("w"),
                ord("W"),
                ord("s"),
                ord("S"),
                ord("a"),
                ord("A"),
                ord("d"),
                ord("D"),
                225,
                226,
            ):
                self.keyboard.set_key_down(key)
            # Arrow-key support from waitKeyEx across common backends.
            elif key_full in (2490368, 65362, 82):  # Up
                self.keyboard.set_key_down(ord("w"))
            elif key_full in (2621440, 65364, 84):  # Down
                self.keyboard.set_key_down(ord("s"))
            elif key_full in (2424832, 65361, 81):  # Left
                self.keyboard.set_key_down(ord("a"))
            elif key_full in (2555904, 65363, 83):  # Right
                self.keyboard.set_key_down(ord("d"))
            elif key == 255:
                # No key this frame; active commands naturally time out in KeyboardDriveController.
                pass

            self.last_loop_ms = (time.perf_counter() - t0) * 1000.0

        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Slang-powered catheter viewport (native UI, cursor control)")
    parser.add_argument(
        "--ct-dir",
        default="/tmp/patient_001",
        help="Directory with mu_volume.npy and metadata.json",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=DET_SIZE,
        help="Detector resolution in pixels (default: 320)",
    )
    parser.add_argument(
        "--pixel-spacing-mm",
        type=float,
        default=0.5,
        help=(
            "Detector pixel pitch in mm (default: 0.5). "
            "FOV at isocenter = det_size × pixel_spacing × (SID/SDD). "
            "Use 1.25 for a full brain (~220 mm FOV at 320px), "
            "0.5 for a tight vessel view."
        ),
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        default=DEFAULT_NUM_SEGMENTS,
        help=f"Catheter segment count (default: {DEFAULT_NUM_SEGMENTS})",
    )
    parser.add_argument(
        "--particle-radius-mm",
        type=float,
        default=None,
        help="Override solver particle collision radius in mm (default: auto-tuned)",
    )
    parser.add_argument(
        "--catheter-radius-mm",
        type=float,
        default=None,
        help="Override rendered catheter radius in mm (default: auto-tuned)",
    )
    parser.add_argument(
        "--catheter-mu",
        type=float,
        default=None,
        help="Override rendered catheter attenuation coefficient (default: auto-tuned)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="cinematic",
        choices=["default", "cinematic"],
        help="Visual post-processing style (default: cinematic)",
    )
    parser.add_argument(
        "--idle-mode",
        type=str,
        default="hold",
        choices=["hold", "simulate"],
        help="hold: freeze when no input (Gradio-like), simulate: always step physics",
    )
    parser.add_argument(
        "--centerline-mode",
        type=str,
        default="raw",
        choices=["raw", "smooth"],
        help="raw: direct solver polyline, smooth: filtered/spline render polyline",
    )
    parser.add_argument(
        "--pose-guard",
        type=str,
        default="safe",
        choices=["off", "safe"],
        help="safe: rollback only on broken poses (default), off: disable rollback guard",
    )
    parser.add_argument(
        "--collision-mode",
        type=str,
        default="mesh-edge",
        choices=["mesh-edge", "sdf"],
        help="Vessel containment mode (default: mesh-edge for demo)",
    )
    parser.add_argument(
        "--sign-scale",
        type=float,
        default=1.0,
        choices=[-1.0, 1.0],
        help="Collision SDF normal sign convention; +1 matches the outward-normal meshes "
        "built by vasculature.py (auto-corrected live in guided mode)",
    )
    parser.add_argument(
        "--vessel-source",
        type=str,
        default="auto",
        choices=["auto", "real", "synthetic"],
        help="Collision vessel source: auto (prefer real), real, or synthetic fallback",
    )
    parser.add_argument(
        "--vessel-mask",
        type=str,
        default=None,
        help="Path to binary vessel mask .npy (ZYX) aligned to mu_volume",
    )
    parser.add_argument(
        "--hu-volume",
        type=str,
        default=None,
        help="Path to HU CT .npy (ZYX); used to derive vessel mask via thresholding",
    )
    parser.add_argument(
        "--hu-threshold",
        type=float,
        default=200.0,
        help="HU threshold for vessel segmentation when --hu-volume is used",
    )
    parser.add_argument(
        "--mask-min-component-voxels",
        type=int,
        default=500,
        help="Connected-component min size when thresholding HU to vessel mask",
    )
    parser.add_argument(
        "--mask-downsample",
        type=int,
        default=4,
        help="Downsample factor for vessel mask before mesh extraction (default: 4)",
    )
    parser.add_argument(
        "--track-stiffness",
        type=float,
        default=0.35,
        help=(
            "Strength of track guidance per substep [0..1]. "
            "1.0 = rigid rail. "
            "0.3 = soft guide that lets mesh collision deform the rod into curved vessels."
        ),
    )
    parser.add_argument(
        "--advance-guide-stiffness",
        type=float,
        default=0.15,
        help=(
            "Track/guide stiffness used only while actively advancing (W / positive velocity). "
            "Lower values reduce over-constraining and help progression."
        ),
    )
    parser.add_argument(
        "--insertion-axis",
        type=str,
        default="centerline",
        choices=["straight", "centerline"],
        help="Track axis source: straight +X or centerline PCA from centerline_points_mm.npy",
    )
    parser.add_argument(
        "--centerline-file",
        type=str,
        default=None,
        help="Optional path to centerline_points_mm.npy. Defaults to <ct_dir>/centerline_points_mm.npy",
    )
    parser.add_argument(
        "--guidewire-young-modulus",
        type=float,
        default=1e6,
        help=(
            "Young's modulus [Pa] for the rod material (default: 1e4). "
            "1e9 = steel-stiff, 1e4 = flexible cerebral guidewire. "
            "Reduce further for ultra-flexible wires."
        ),
    )
    parser.add_argument(
        "--guidewire-bend-stiffness",
        type=float,
        default=0.1,
        help="Normalised bend-stiffness multiplier 0-1 (default: 0.1). Lower = more flexible.",
    )
    parser.add_argument(
        "--key-hold-ttl",
        type=float,
        default=0.25,
        help=(
            "Seconds a key stays active after last repeat event (default: 0.25). "
            "Increase to 0.4-0.8 only if remote/X11 drops key repeats."
        ),
    )
    parser.add_argument(
        "--dsa",
        action="store_true",
        help=(
            "Start in DSA contrast mode: bake a contrast bolus into the live DRR so the "
            "vessel lumen + catheter render dark like a native angiogram. Toggle with X."
        ),
    )
    parser.add_argument(
        "--dsa-boost",
        type=float,
        default=6.0,
        help="Attenuation multiplier applied inside the vessel lumen for DSA contrast (default: 6.0).",
    )
    parser.add_argument(
        "--dsa-gamma",
        type=float,
        default=1.0,
        help=(
            "Brightness gamma for the DSA angiogram (<1 brighter, >1 darker; default: 1.0). "
            "Adjust live with '-' (darker) and '=' (brighter)."
        ),
    )
    args = parser.parse_args()

    app = SlangViewportApp(
        ct_dir=args.ct_dir,
        det_size=int(args.det_size),
        pixel_spacing_mm=float(args.pixel_spacing_mm),
        num_segments=int(args.num_segments),
        particle_radius_mm=args.particle_radius_mm,
        catheter_radius_mm=args.catheter_radius_mm,
        catheter_mu=args.catheter_mu,
        visual_style=args.style,
        idle_mode=args.idle_mode,
        centerline_mode=args.centerline_mode,
        pose_guard=args.pose_guard,
        collision_mode=args.collision_mode,
        sign_scale=args.sign_scale,
        vessel_source=args.vessel_source,
        vessel_mask_path=args.vessel_mask,
        hu_volume_path=args.hu_volume,
        hu_threshold=args.hu_threshold,
        min_component_voxels=args.mask_min_component_voxels,
        mask_downsample=args.mask_downsample,
        track_stiffness=args.track_stiffness,
        advance_guide_stiffness=args.advance_guide_stiffness,
        insertion_axis=args.insertion_axis,
        centerline_file=args.centerline_file,
        guidewire_young_modulus=float(args.guidewire_young_modulus),
        guidewire_bend_stiffness=float(args.guidewire_bend_stiffness),
        key_hold_ttl_s=float(args.key_hold_ttl),
        dsa=bool(args.dsa),
        dsa_boost=float(args.dsa_boost),
        dsa_gamma=float(args.dsa_gamma),
    )
    app.run()


if __name__ == "__main__":
    main()
