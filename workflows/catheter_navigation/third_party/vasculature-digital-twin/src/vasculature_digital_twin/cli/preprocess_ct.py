from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vasculature_digital_twin import PreprocessingSettings, VolumePreprocessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a CT volume into mu_volume + metadata cache artifacts.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dicom", type=str, default=None, help="Path to a DICOM series directory.")
    src.add_argument("--nifti", type=str, default=None, help="Path to a NIfTI file (.nii / .nii.gz).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for cache outputs.")
    parser.add_argument("--hu-clip-min", type=float, default=PreprocessingSettings.hu_clip_min)
    parser.add_argument("--hu-clip-max", type=float, default=PreprocessingSettings.hu_clip_max)
    parser.add_argument(
        "--save-hu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write raw HU as hu_volume.npy.",
    )
    return parser


def _expand(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = PreprocessingSettings(hu_clip_min=args.hu_clip_min, hu_clip_max=args.hu_clip_max)
    dicom = _expand(args.dicom)
    nifti = _expand(args.nifti)
    output_dir = _expand(args.output_dir)

    if dicom is not None:
        preprocessor = VolumePreprocessor.from_dicom(dicom, settings=settings)
    else:
        preprocessor = VolumePreprocessor.from_nifti(nifti, settings=settings)

    volume = preprocessor.preprocess(output_dir=output_dir)
    print(volume)

    if args.save_hu:
        hu_path = Path(output_dir) / "hu_volume.npy"
        np.save(hu_path, preprocessor.hu_volume_zyx.astype(np.float32))
        print(f"[vdt-preprocess-ct] Saved HU volume: {hu_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
