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

"""Produce a real vessel mask + centerline graph for the interactive viewport.

This is the data-prep step for the *real-vessel* navigation path of
``interactive_catheter_slang_viewport``. It writes, into ``--ct-dir``:

* ``vessel_mask.npy``          — binary (Z,Y,X) mask aligned to ``mu_volume.npy``
* ``centerline_points_mm.npy`` — (N,3) world-mm XYZ centerline nodes
* ``centerline_edges.npy``     — (M,2) int connectivity (skeleton 26-graph)
* ``centerline_radii_mm.npy``  — (N,) inscribed-sphere radius per node (mm)

The viewport then picks these up automatically with::

    python -m fluorosim.examples.interactive_catheter_slang_viewport \\
        --ct-dir <ct-dir> --vessel-source real --insertion-axis centerline

Mask source: TotalSegmentator (named arteries, no bone bleed-through) with a
HU-threshold fallback. **Use a contrast CT angiography (CTA)** — a non-contrast
scan (e.g. a plain head CT) has no vessel signal and HU thresholding will pick
up bone instead.

Centerline: a 3-D skeleton of the mask (``skimage.morphology.skeletonize``)
converted to a node/edge graph. This avoids the VMTK dependency; the viewport's
graph loader extracts the dominant (longest) path on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make the `fluorosim` package importable regardless of launch style.
_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from fluorosim import PreprocessingSettings, VolumePreprocessor  # noqa: E402
from fluorosim.vasculature import get_vessel_mask  # noqa: E402


def _expand(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment vessels and extract a centerline graph for the real-vessel "
            "interactive viewport path. Requires a contrast CTA."
        ),
    )
    parser.add_argument(
        "--ct-dir",
        required=True,
        help="Cache dir with mu_volume.npy + metadata.json. Outputs are written here.",
    )
    hu_src = parser.add_mutually_exclusive_group()
    hu_src.add_argument(
        "--hu-volume",
        default=None,
        help="Explicit HU volume .npy (ZYX). Defaults to <ct-dir>/hu_volume.npy.",
    )
    hu_src.add_argument("--dicom", default=None, help="Load HU directly from a DICOM series dir.")
    hu_src.add_argument("--nifti", default=None, help="Load HU directly from a NIfTI file.")
    parser.add_argument(
        "--ts-gt-dir",
        default=None,
        help=(
            "Use ground-truth TotalSegmentator label masks from this 'segmentations/' "
            "directory instead of running inference. Masks are unioned per --ts-labels "
            "and loaded with the same axis convention as the CT, so they co-register. "
            "No HU volume is needed in this mode."
        ),
    )
    parser.add_argument(
        "--ts-labels",
        default="aorta,iliac_artery_left,iliac_artery_right",
        help=(
            "Comma-separated label names (no extension) to union when --ts-gt-dir is "
            "set. Default is the aorta->iliac arterial tree (classic catheter path)."
        ),
    )
    parser.add_argument(
        "--close-iterations",
        type=int,
        default=2,
        help=(
            "Binary-closing iterations applied to the mask before skeletonizing, to "
            "bridge small gaps (e.g. the aortic bifurcation). 0 disables."
        ),
    )
    parser.add_argument(
        "--territory",
        default="combined",
        choices=["combined", "aortic", "cerebral", "peripheral", "coronary"],
        help="Which TotalSegmentator territory to keep (default: combined).",
    )
    parser.add_argument(
        "--no-totalsegmentator",
        action="store_true",
        help="Force HU-threshold segmentation instead of TotalSegmentator.",
    )
    parser.add_argument(
        "--include-coronary",
        action="store_true",
        help="Also run the TotalSegmentator coronary_arteries task.",
    )
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"], help="TotalSegmentator device.")
    parser.add_argument("--fast", action="store_true", help="TotalSegmentator fast mode.")
    parser.add_argument("--hu-threshold", type=float, default=200.0, help="HU fallback threshold.")
    parser.add_argument(
        "--min-component-voxels",
        type=int,
        default=500,
        help="HU fallback: minimum connected-component size.",
    )
    parser.add_argument(
        "--largest-component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only the largest connected component of the mask before "
            "skeletonizing (gives one clean navigable vessel; default on)."
        ),
    )
    parser.add_argument(
        "--max-centerline-nodes",
        type=int,
        default=0,
        help="If >0, uniformly subsample skeleton nodes to at most this many.",
    )
    return parser


def _load_hu(args, expected_shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Load the HU volume from an explicit .npy, the cache, or DICOM/NIfTI."""
    if args.dicom or args.nifti:
        settings = PreprocessingSettings()
        if args.dicom:
            print(f"[segment_vessels] Loading HU from DICOM: {_expand(args.dicom)}")
            pre = VolumePreprocessor.from_dicom(_expand(args.dicom), settings=settings)
        else:
            print(f"[segment_vessels] Loading HU from NIfTI: {_expand(args.nifti)}")
            pre = VolumePreprocessor.from_nifti(_expand(args.nifti), settings=settings)
        hu = pre.hu_volume_zyx.astype(np.float32)
    else:
        hu_path = _expand(args.hu_volume) or str(Path(args.ct_dir).expanduser() / "hu_volume.npy")
        if not Path(hu_path).is_file():
            raise FileNotFoundError(
                f"HU volume not found at {hu_path}. Re-run preprocess with --save-hu, "
                f"or pass --hu-volume / --dicom / --nifti."
            )
        print(f"[segment_vessels] Loading HU volume: {hu_path}")
        hu = np.load(hu_path).astype(np.float32)

    # Validate against the cached mu_volume shape for every source (DICOM/NIfTI
    # reprocessing or an explicit .npy can mismatch and mis-register the masks).
    if hu.shape != expected_shape_zyx:
        raise ValueError(f"HU volume shape {hu.shape} does not match mu_volume shape {expected_shape_zyx}.")
    return hu


def _load_ts_gt_masks(
    gt_dir: str,
    labels: list[str],
    expected_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Union named TotalSegmentator ground-truth label masks into a (Z,Y,X) mask.

    The label NIfTIs share the CT's grid; we apply the same ``transpose(2, 1, 0)``
    as ``load_nifti_hu`` so the result co-registers with ``mu_volume.npy``.
    """
    import nibabel as nib

    gt = Path(gt_dir).expanduser()
    union: np.ndarray | None = None
    found: list[tuple[str, int]] = []
    for name in labels:
        path = gt / f"{name}.nii.gz"
        if not path.is_file():
            print(f"[segment_vessels]   skip missing label: {name}")
            continue
        arr = np.transpose(np.asarray(nib.load(str(path)).dataobj), (2, 1, 0)) > 0
        if arr.shape != expected_shape_zyx:
            raise ValueError(f"GT mask '{name}' shape {arr.shape} != volume shape {expected_shape_zyx}.")
        union = arr if union is None else (union | arr)
        found.append((name, int(arr.sum())))
    if union is None:
        raise FileNotFoundError(f"No label masks found in {gt} among {labels}.")
    for name, count in found:
        print(f"[segment_vessels]   {name}: {count:,} vox")
    return union.astype(np.uint8)


def _keep_largest_component(mask_zyx: np.ndarray) -> np.ndarray:
    """Return only the largest 26-connected component of a binary mask."""
    from scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, n = ndimage.label(mask_zyx > 0, structure=structure)
    if n <= 1:
        return (mask_zyx > 0).astype(np.uint8)
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background
    keep = int(np.argmax(counts))
    return (labels == keep).astype(np.uint8)


def _centerline_from_mask(
    mask_zyx: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    max_nodes: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Skeletonize a binary mask into a (points_mm, edges, radii_mm) graph.

    Points are world-mm XYZ (origin + voxel_index * spacing), matching the
    convention the viewport uses for ``mu_volume`` / ``vessel_mask``.
    """
    from scipy import ndimage

    try:
        from skimage.morphology import skeletonize

        skel = skeletonize(mask_zyx.astype(bool))
    except TypeError:
        from skimage.morphology import skeletonize_3d

        skel = skeletonize_3d(mask_zyx.astype(bool)) > 0

    sz, sy, sx = (float(v) for v in spacing_zyx_mm)
    ox, oy, oz = (float(v) for v in origin_xyz_mm)

    # Inscribed-sphere radius (mm) from the full mask, sampled on the skeleton.
    edt = ndimage.distance_transform_edt(mask_zyx.astype(bool), sampling=(sz, sy, sx))

    coords = np.argwhere(skel)  # (N, 3) in (z, y, x)
    if coords.shape[0] < 4:
        raise RuntimeError(f"Skeleton has too few nodes ({coords.shape[0]}). Mask may be empty or too small.")

    if max_nodes and coords.shape[0] > max_nodes:
        sel = np.linspace(0, coords.shape[0] - 1, max_nodes).round().astype(np.int64)
        sel = np.unique(sel)
        coords = coords[sel]

    index_of = {(int(z), int(y), int(x)): i for i, (z, y, x) in enumerate(coords)}

    pts = np.empty((coords.shape[0], 3), dtype=np.float32)
    pts[:, 0] = ox + coords[:, 2] * sx
    pts[:, 1] = oy + coords[:, 1] * sy
    pts[:, 2] = oz + coords[:, 0] * sz
    radii = edt[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)

    neighbors = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]
    edges: list[tuple[int, int]] = []
    for i, (z, y, x) in enumerate(coords):
        for dz, dy, dx in neighbors:
            j = index_of.get((int(z + dz), int(y + dy), int(x + dx)))
            if j is not None and j > i:
                edges.append((i, j))
    edges_arr = np.asarray(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
    if edges_arr.shape[0] < 1:
        raise RuntimeError("Skeleton produced no edges (disconnected single voxels).")
    return pts, edges_arr, radii


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ct_dir = Path(args.ct_dir).expanduser()

    meta_path = ct_dir / "metadata.json"
    mu_path = ct_dir / "mu_volume.npy"
    if not meta_path.is_file() or not mu_path.is_file():
        raise FileNotFoundError(f"{ct_dir} must contain mu_volume.npy and metadata.json (run preprocess_ct first).")
    with open(meta_path, encoding="utf-8") as stream:
        meta = json.load(stream)
    spacing_zyx_mm = tuple(float(v) for v in meta["spacing_zyx_mm"])
    # origin_xyz_mm is optional in metadata.json (VolumeMetadata defaults it to null);
    # a missing origin means no world offset, so fall back to the volume-local frame.
    origin_raw = meta.get("origin_xyz_mm")
    origin_xyz_mm = tuple(float(v) for v in origin_raw) if origin_raw else (0.0, 0.0, 0.0)
    shape_zyx = tuple(int(v) for v in meta["shape_zyx"])

    if args.ts_gt_dir:
        labels = [s.strip() for s in args.ts_labels.split(",") if s.strip()]
        print(f"[segment_vessels] Using ground-truth masks from {args.ts_gt_dir}: {labels}")
        mask = _load_ts_gt_masks(args.ts_gt_dir, labels, shape_zyx)
    else:
        hu_zyx = _load_hu(args, shape_zyx)
        print(
            f"[segment_vessels] Segmenting vessels "
            f"(method={'HU-threshold' if args.no_totalsegmentator else 'TotalSegmentator'}, "
            f"territory={args.territory}) ...",
            flush=True,
        )
        result = get_vessel_mask(
            hu_zyx=hu_zyx,
            spacing_zyx_mm=spacing_zyx_mm,
            use_totalsegmentator=not args.no_totalsegmentator,
            include_coronary=args.include_coronary,
            device=args.device,
            fast=args.fast,
            cache_dir=str(ct_dir / "totalseg_cache"),
            hu_threshold=args.hu_threshold,
            min_component_voxels=args.min_component_voxels,
        )
        if args.territory == "combined":
            mask = result.combined_mask
        else:
            mask = result.get_territory(args.territory)
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != shape_zyx:
        raise ValueError(f"Vessel mask shape {mask.shape} != volume shape {shape_zyx}.")
    n_vessel = int(mask.sum())
    if n_vessel == 0:
        raise RuntimeError(
            "Vessel mask is empty. For a contrast CTA try a different --territory; "
            "for the HU fallback lower --hu-threshold. A non-contrast scan has no vessel signal."
        )
    print(f"[segment_vessels] Vessel voxels: {n_vessel:,} ({100.0 * n_vessel / mask.size:.3f}% of volume)")

    if args.close_iterations > 0:
        from scipy import ndimage

        mask = ndimage.binary_closing(
            mask > 0, structure=np.ones((3, 3, 3), np.uint8), iterations=args.close_iterations
        ).astype(np.uint8)
        print(f"[segment_vessels] After closing (i={args.close_iterations}): {int(mask.sum()):,} voxels")

    if args.largest_component:
        mask = _keep_largest_component(mask)
        print(f"[segment_vessels] Largest component: {int(mask.sum()):,} voxels")

    mask_path = ct_dir / "vessel_mask.npy"
    np.save(mask_path, mask)
    print(f"[segment_vessels] Wrote {mask_path}")

    print("[segment_vessels] Extracting centerline (skeleton graph) ...", flush=True)
    pts_mm, edges, radii_mm = _centerline_from_mask(
        mask, spacing_zyx_mm, origin_xyz_mm, max_nodes=args.max_centerline_nodes
    )
    np.save(ct_dir / "centerline_points_mm.npy", pts_mm)
    np.save(ct_dir / "centerline_edges.npy", edges)
    np.save(ct_dir / "centerline_radii_mm.npy", radii_mm)
    print(
        f"[segment_vessels] Centerline: {pts_mm.shape[0]:,} nodes, {edges.shape[0]:,} edges, "
        f"radius {radii_mm.min():.2f}-{radii_mm.max():.2f} mm"
    )

    print(
        "[segment_vessels] Done. Launch the viewport with:\n"
        f"  python -m fluorosim.examples.interactive_catheter_slang_viewport \\\n"
        f"      --ct-dir {ct_dir} --vessel-source real --insertion-axis centerline"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
