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

"""Preprocess a DICOM/NIfTI CT volume into a fluorosim-ready mu volume.

Usage::

    python -m fluorosim.examples.preprocess_ct --dicom /path/to/dicom --output-dir /tmp/fluoro_cache
    python -m fluorosim.examples.preprocess_ct --nifti /path/to/ct.nii.gz --output-dir /tmp/fluoro_cache

The resulting directory (``mu_volume.npy`` + ``metadata.json``) can be fed
straight into ``python -m fluorosim.examples.render_drr --cache <output-dir>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the `fluorosim` package importable as a top-level package regardless of
# how this entry point is launched (the package uses absolute `fluorosim.*`
# imports internally). `parents[2]` is the `simulation/` directory.
_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from fluorosim import PreprocessingSettings, VolumePreprocessor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess a CT volume (DICOM or NIfTI) into a fluorosim mu volume.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dicom", type=str, default=None, help="Path to a DICOM series directory.")
    src.add_argument("--nifti", type=str, default=None, help="Path to a NIfTI file (.nii / .nii.gz).")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the preprocessed volume (mu_volume.npy + metadata.json).",
    )
    parser.add_argument(
        "--hu-clip-min",
        type=float,
        default=PreprocessingSettings.hu_clip_min,
        help="Minimum HU value used for clipping.",
    )
    parser.add_argument(
        "--hu-clip-max",
        type=float,
        default=PreprocessingSettings.hu_clip_max,
        help="Maximum HU value used for clipping.",
    )
    parser.add_argument(
        "--save-hu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also write the raw HU volume as hu_volume.npy alongside mu_volume.npy. "
            "This lets the interactive viewport derive a real vessel segmentation "
            "(via --vessel-source real / auto-detect). Use --no-save-hu to skip."
        ),
    )
    return parser


def _expand(path: str | None) -> str | None:
    """Expand a leading ``~`` and environment variables in a user-supplied path."""
    if path is None:
        return None
    return str(Path(path).expanduser())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = PreprocessingSettings(
        hu_clip_min=args.hu_clip_min,
        hu_clip_max=args.hu_clip_max,
    )

    dicom = _expand(args.dicom)
    nifti = _expand(args.nifti)
    output_dir = _expand(args.output_dir)

    if dicom is not None:
        print(f"[preprocess_ct] Loading DICOM series: {dicom}")
        preprocessor = VolumePreprocessor.from_dicom(dicom, settings=settings)
    else:
        print(f"[preprocess_ct] Loading NIfTI volume: {nifti}")
        preprocessor = VolumePreprocessor.from_nifti(nifti, settings=settings)

    print(preprocessor)

    volume = preprocessor.preprocess(output_dir=output_dir)
    print(volume)

    if args.save_hu:
        hu_path = Path(output_dir) / "hu_volume.npy"
        np.save(hu_path, preprocessor.hu_volume_zyx.astype(np.float32))
        print(f"[preprocess_ct] Saved raw HU volume for vessel segmentation: {hu_path}")

    print(f"[preprocess_ct] Done. Render it with:\n" f"  python -m fluorosim.examples.render_drr --cache {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
