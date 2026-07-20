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

"""Render a single DRR (fluoroscopy) frame from a preprocessed CT volume.

This entry script supports three input modes:
  - Synthetic phantom (default, no input data required)
  - Preprocessed cache directory (`--cache`)
  - Raw DICOM/NIfTI source (`--dicom` / `--nifti`)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Ensure top-level package imports (`fluorosim.*`) resolve regardless of launch mode.
_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from fluorosim import (  # noqa: E402
    CarmGeometry,
    FluoroSimulator,
    PreprocessedVolume,
    PreprocessingSettings,
    RealismSettings,
    SimulatorConfig,
    VolumePreprocessor,
)
from fluorosim.rendering import diffdrr_slang_renderer as _slang_renderer  # noqa: E402

_LOCAL_SHADER_PATH = Path(__file__).resolve().parents[1] / "rendering" / "diffdrr_slang.slang"
# Some pip builds of fluorosim may miss packaging the .slang shader asset.
# Prefer the workflow-local shader file when present.
if _LOCAL_SHADER_PATH.is_file():
    _slang_renderer._SLANG_SHADER_PATH = _LOCAL_SHADER_PATH


def make_synthetic_phantom(size: int = 128) -> PreprocessedVolume:
    """Build a compact CT-like phantom for self-contained DRR smoke runs."""
    z = y = x = int(size)
    hu = np.full((z, y, x), -1000.0, dtype=np.float32)

    cy, cx = y / 2.0, x / 2.0
    _zz, yy, xx = np.mgrid[0:z, 0:y, 0:x]
    radial = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    body_r = 0.42 * x
    bone_r = 0.40 * x

    hu[radial <= body_r] = 40.0
    hu[(radial <= body_r) & (radial >= bone_r)] = 1200.0

    vessel = np.sqrt((yy - (cy + 0.12 * y)) ** 2 + (xx - (cx - 0.10 * x)) ** 2)
    hu[vessel <= 0.04 * x] = 3000.0

    preprocessor = VolumePreprocessor.from_numpy(
        hu_volume=hu,
        spacing_zyx_mm=(1.0, 0.8, 0.8),
        settings=PreprocessingSettings(),
    )
    return preprocessor.preprocess()


def _expand(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser())


def resolve_volume(args: argparse.Namespace) -> PreprocessedVolume:
    if args.cache:
        cache = _expand(args.cache)
        print(f"[render_drr] Loading preprocessed volume: {cache}")
        return PreprocessedVolume.load(cache)
    if args.dicom:
        dicom = _expand(args.dicom)
        print(f"[render_drr] Preprocessing DICOM series: {dicom}")
        return VolumePreprocessor.from_dicom(dicom).preprocess()
    if args.nifti:
        nifti = _expand(args.nifti)
        print(f"[render_drr] Preprocessing NIfTI volume: {nifti}")
        return VolumePreprocessor.from_nifti(nifti).preprocess()
    print(f"[render_drr] No input data provided - using synthetic phantom (size={args.phantom_size}).")
    return make_synthetic_phantom(args.phantom_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a single DRR frame from a CT/preprocessed volume.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--cache", type=str, default=None, help="Preprocessed cache dir (mu_volume.npy + metadata.json).")
    src.add_argument("--dicom", type=str, default=None, help="Path to a DICOM series directory.")
    src.add_argument("--nifti", type=str, default=None, help="Path to a NIfTI file (.nii / .nii.gz).")

    parser.add_argument("--output", type=str, default="drr.png", help="Output image path (.png or .npy).")
    parser.add_argument("--phantom-size", type=int, default=128, help="Synthetic phantom cube size.")
    parser.add_argument(
        "--view",
        choices=["axial", "ap", "lat"],
        default="axial",
        help="Projection preset applied before user rotations.",
    )
    parser.add_argument("--rx", type=float, default=0.0, help="Extra rotation X in degrees.")
    parser.add_argument("--ry", type=float, default=0.0, help="Extra rotation Y in degrees.")
    parser.add_argument("--rz", type=float, default=0.0, help="Extra rotation Z in degrees.")
    parser.add_argument("--tx", type=float, default=0.0, help="Translation X (mm).")
    parser.add_argument("--ty", type=float, default=0.0, help="Translation Y (mm).")
    parser.add_argument("--tz", type=float, default=0.0, help="Translation Z (mm).")
    parser.add_argument("--width", type=int, default=None, help="Detector width in pixels.")
    parser.add_argument("--height", type=int, default=None, help="Detector height in pixels.")

    parser.add_argument("--realism", action="store_true", help="Enable realism post-processing.")
    parser.add_argument("--gamma", type=float, default=None, help="Display gamma (implies --realism).")
    parser.add_argument("--scatter", type=float, default=None, help="Compton scatter sigma (implies --realism).")
    parser.add_argument(
        "--poisson", type=float, default=None, help="Photon count for Poisson noise (implies --realism)."
    )
    parser.add_argument("--gaussian", type=float, default=None, help="Gaussian noise sigma (implies --realism).")
    parser.add_argument("--blur", type=float, default=None, help="Detector blur sigma (implies --realism).")
    return parser


_REALISM_DEFAULTS = {
    "gamma": 0.8,
    "scatter_sigma_px": 16.0,
    "poisson_photons": 6000.0,
    "gaussian_sigma": 0.008,
    "blur_sigma_px": 0.6,
}

_VIEW_PRESETS = {
    "axial": (0.0, 0.0, 0.0),
    "ap": (90.0, 0.0, 0.0),
    "lat": (90.0, 90.0, 0.0),
}


def build_realism(args: argparse.Namespace) -> RealismSettings:
    knobs = {
        "gamma": args.gamma,
        "scatter_sigma_px": args.scatter,
        "poisson_photons": args.poisson,
        "gaussian_sigma": args.gaussian,
        "blur_sigma_px": args.blur,
    }
    any_knob = any(value is not None for value in knobs.values())
    if not args.realism and not any_knob:
        return RealismSettings(enabled=False)

    resolved = dict(_REALISM_DEFAULTS)
    for key, value in knobs.items():
        if value is not None:
            resolved[key] = value
    return RealismSettings(enabled=True, **resolved)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    volume = resolve_volume(args)

    geometry_kwargs = {}
    if args.width is not None:
        geometry_kwargs["detector_width_px"] = args.width
    if args.height is not None:
        geometry_kwargs["detector_height_px"] = args.height

    config = SimulatorConfig(
        geometry=CarmGeometry(**geometry_kwargs) if geometry_kwargs else CarmGeometry(),
        realism=build_realism(args),
    )

    sim = FluoroSimulator(volume, config)

    base = _VIEW_PRESETS[args.view]
    rotation = (
        math.radians(base[0] + args.rx),
        math.radians(base[1] + args.ry),
        math.radians(base[2] + args.rz),
    )
    translation = (args.tx, args.ty, args.tz)

    output = _expand(args.output)
    frame = sim.render_frame(rotation=rotation, translation=translation)
    frame.save(output)

    metrics = sim.get_metrics()
    print(f"[render_drr] Saved frame to: {output}")
    print(f"[render_drr] Image shape: {frame.image.shape}, range: [{frame.image.min():.4f}, {frame.image.max():.4f}]")
    print(f"[render_drr] Render time: {frame.timestamp_ms:.2f} ms ({metrics.fps:.1f} FPS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
