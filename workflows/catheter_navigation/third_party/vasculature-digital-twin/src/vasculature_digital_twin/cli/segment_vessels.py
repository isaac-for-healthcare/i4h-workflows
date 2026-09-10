from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vasculature_digital_twin import PreprocessingSettings, VolumePreprocessor
from vasculature_digital_twin.vasculature import get_vessel_mask


def _expand(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment vessels and extract centerline graph from CT cache.")
    parser.add_argument("--ct-dir", required=True, help="Directory containing mu_volume.npy + metadata.json.")
    hu_src = parser.add_mutually_exclusive_group()
    hu_src.add_argument("--hu-volume", default=None, help="Explicit HU volume .npy (ZYX).")
    hu_src.add_argument("--dicom", default=None, help="Load HU from DICOM series.")
    hu_src.add_argument("--nifti", default=None, help="Load HU from NIfTI.")
    parser.add_argument("--ts-gt-dir", default=None, help="Optional TotalSegmentator segmentations directory.")
    parser.add_argument(
        "--ts-labels",
        default="aorta,iliac_artery_left,iliac_artery_right",
        help="Comma-separated TotalSegmentator label names to union when using --ts-gt-dir.",
    )
    parser.add_argument("--close-iterations", type=int, default=2)
    parser.add_argument(
        "--territory",
        default="combined",
        choices=["combined", "aortic", "cerebral", "peripheral", "coronary"],
    )
    parser.add_argument("--no-totalsegmentator", action="store_true")
    parser.add_argument("--include-coronary", action="store_true")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--hu-threshold", type=float, default=200.0)
    parser.add_argument("--min-component-voxels", type=int, default=500)
    parser.add_argument("--largest-component", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-centerline-nodes", type=int, default=0)
    return parser


def _load_hu(args, expected_shape_zyx: tuple[int, int, int]) -> np.ndarray:
    if args.dicom or args.nifti:
        settings = PreprocessingSettings()
        if args.dicom:
            pre = VolumePreprocessor.from_dicom(_expand(args.dicom), settings=settings)
        else:
            pre = VolumePreprocessor.from_nifti(_expand(args.nifti), settings=settings)
        hu = pre.hu_volume_zyx.astype(np.float32)
    else:
        hu_path = _expand(args.hu_volume) or str(Path(args.ct_dir).expanduser() / "hu_volume.npy")
        if not Path(hu_path).is_file():
            raise FileNotFoundError(
                f"HU volume not found at {hu_path}. Re-run preprocess with --save-hu, or pass --hu-volume/--dicom/--nifti."
            )
        hu = np.load(hu_path).astype(np.float32)

    if hu.shape != expected_shape_zyx:
        raise ValueError(f"HU shape {hu.shape} does not match mu_volume shape {expected_shape_zyx}.")
    return hu


def _load_ts_gt_masks(
    gt_dir: str,
    labels: list[str],
    expected_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    import nibabel as nib

    gt = Path(gt_dir).expanduser()
    union: np.ndarray | None = None
    for name in labels:
        path = gt / f"{name}.nii.gz"
        if not path.is_file():
            continue
        arr = np.transpose(np.asarray(nib.load(str(path)).dataobj), (2, 1, 0)) > 0
        if arr.shape != expected_shape_zyx:
            raise ValueError(f"GT mask '{name}' shape {arr.shape} != volume shape {expected_shape_zyx}.")
        union = arr if union is None else (union | arr)
    if union is None:
        raise FileNotFoundError(f"No label masks found in {gt} among {labels}.")
    return union.astype(np.uint8)


def _keep_largest_component(mask_zyx: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, n = ndimage.label(mask_zyx > 0, structure=structure)
    if n <= 1:
        return (mask_zyx > 0).astype(np.uint8)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return (labels == keep).astype(np.uint8)


def _centerline_from_mask(
    mask_zyx: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    max_nodes: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy import ndimage

    try:
        from skimage.morphology import skeletonize

        skel = skeletonize(mask_zyx.astype(bool))
    except TypeError:
        from skimage.morphology import skeletonize_3d

        skel = skeletonize_3d(mask_zyx.astype(bool)) > 0

    sz, sy, sx = (float(v) for v in spacing_zyx_mm)
    ox, oy, oz = (float(v) for v in origin_xyz_mm)

    edt = ndimage.distance_transform_edt(mask_zyx.astype(bool), sampling=(sz, sy, sx))
    coords = np.argwhere(skel)
    if coords.shape[0] < 4:
        raise RuntimeError(f"Skeleton has too few nodes ({coords.shape[0]}).")

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
        raise RuntimeError("Skeleton produced no edges.")
    return pts, edges_arr, radii


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ct_dir = Path(args.ct_dir).expanduser()
    meta_path = ct_dir / "metadata.json"
    mu_path = ct_dir / "mu_volume.npy"
    if not meta_path.is_file() or not mu_path.is_file():
        raise FileNotFoundError(f"{ct_dir} must contain mu_volume.npy and metadata.json.")

    with open(meta_path, encoding="utf-8") as stream:
        meta = json.load(stream)
    spacing_zyx_mm = tuple(float(v) for v in meta["spacing_zyx_mm"])
    origin_raw = meta.get("origin_xyz_mm")
    origin_xyz_mm = tuple(float(v) for v in origin_raw) if origin_raw else (0.0, 0.0, 0.0)
    shape_zyx = tuple(int(v) for v in meta["shape_zyx"])

    if args.ts_gt_dir:
        labels = [s.strip() for s in args.ts_labels.split(",") if s.strip()]
        mask = _load_ts_gt_masks(args.ts_gt_dir, labels, shape_zyx)
    else:
        hu_zyx = _load_hu(args, shape_zyx)
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
        mask = result.combined_mask if args.territory == "combined" else result.get_territory(args.territory)

    mask = (mask > 0).astype(np.uint8)
    if args.close_iterations > 0:
        from scipy import ndimage

        mask = ndimage.binary_closing(
            mask > 0,
            structure=np.ones((3, 3, 3), np.uint8),
            iterations=args.close_iterations,
        ).astype(np.uint8)

    if args.largest_component:
        mask = _keep_largest_component(mask)

    np.save(ct_dir / "vessel_mask.npy", mask)
    pts_mm, edges, radii_mm = _centerline_from_mask(
        mask, spacing_zyx_mm, origin_xyz_mm, max_nodes=args.max_centerline_nodes
    )
    np.save(ct_dir / "centerline_points_mm.npy", pts_mm)
    np.save(ct_dir / "centerline_edges.npy", edges)
    np.save(ct_dir / "centerline_radii_mm.npy", radii_mm)
    print(f"[vdt-segment-vessels] Saved outputs in {ct_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
