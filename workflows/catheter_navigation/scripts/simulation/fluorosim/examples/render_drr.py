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

This is the default workflow run. With no input data it falls back to a
built-in synthetic phantom so the command is self-contained::

    # Self-contained (synthetic phantom)
    python -m fluorosim.examples.render_drr --output drr.png

    # From a cache produced by preprocess_ct
    python -m fluorosim.examples.render_drr --cache /tmp/fluoro_cache --output drr.png

    # Directly from a CT source
    python -m fluorosim.examples.render_drr --dicom /path/to/dicom --output drr.png
    python -m fluorosim.examples.render_drr --nifti /path/to/ct.nii.gz --output drr.png

Rendering requires ``slangpy`` and a CUDA-capable GPU (same class of host
requirement as the other i4h workflows).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Make the `fluorosim` package importable as a top-level package regardless of
# how this entry point is launched (the package uses absolute `fluorosim.*`
# imports internally). `parents[2]` is the `simulation/` directory.
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


def make_synthetic_phantom(size: int = 128) -> PreprocessedVolume:
    """Build a small CT-like phantom so the demo runs without input data.

    The phantom is a soft-tissue cylinder enclosed in a denser bone shell,
    with a high-attenuation "contrast" vessel running through it. Values are
    expressed in Hounsfield Units and passed through the normal HU->mu path.
    """
    z = y = x = int(size)
    hu = np.full((z, y, x), -1000.0, dtype=np.float32)  # air background

    cy, cx = y / 2.0, x / 2.0
    zz, yy, xx = np.mgrid[0:z, 0:y, 0:x]
    radial = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    body_r = 0.42 * x
    bone_r = 0.40 * x

    # Soft tissue body
    hu[radial <= body_r] = 40.0
    # Bone shell
    hu[(radial <= body_r) & (radial >= bone_r)] = 1200.0

    # Contrast-filled vessel: an off-center vertical cylinder
    vessel = np.sqrt((yy - (cy + 0.12 * y)) ** 2 + (xx - (cx - 0.10 * x)) ** 2)
    hu[vessel <= 0.04 * x] = 3000.0

    preprocessor = VolumePreprocessor.from_numpy(
        hu_volume=hu,
        spacing_zyx_mm=(1.0, 0.8, 0.8),
        settings=PreprocessingSettings(),
    )
    return preprocessor.preprocess()


def _expand(path: str | None) -> str | None:
    """Expand a leading ``~`` and environment variables in a user-supplied path."""
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
    parser = argparse.ArgumentParser(
        description="Render a single DRR frame from a preprocessed CT volume.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--cache", type=str, default=None, help="Directory with a preprocessed volume (mu_volume.npy + metadata.json)."
    )
    src.add_argument("--dicom", type=str, default=None, help="Path to a DICOM series directory.")
    src.add_argument("--nifti", type=str, default=None, help="Path to a NIfTI file (.nii / .nii.gz).")

    parser.add_argument("--output", type=str, default="drr.png", help="Output image path (.png or .npy).")
    parser.add_argument(
        "--phantom-size", type=int, default=128, help="Synthetic phantom cube size (when no input data)."
    )

    parser.add_argument(
        "--view",
        choices=["axial", "ap", "lat"],
        default="axial",
        help="Projection preset: axial (default beam axis), ap (anterior-posterior), lat (lateral). "
        "Adds on top of any --rx/--ry/--rz.",
    )
    parser.add_argument("--rx", type=float, default=0.0, help="Extra C-arm rotation about X (degrees).")
    parser.add_argument("--ry", type=float, default=0.0, help="Extra C-arm rotation about Y (degrees).")
    parser.add_argument("--rz", type=float, default=0.0, help="Extra C-arm rotation about Z (degrees).")
    parser.add_argument("--tx", type=float, default=0.0, help="Translation X (mm).")
    parser.add_argument("--ty", type=float, default=0.0, help="Translation Y (mm).")
    parser.add_argument("--tz", type=float, default=0.0, help="Translation Z (mm).")

    parser.add_argument("--width", type=int, default=None, help="Detector width in pixels (overrides default).")
    parser.add_argument("--height", type=int, default=None, help="Detector height in pixels (overrides default).")

    # Realism post-processing. --realism turns on a physically plausible default
    # fluoro look; the individual knobs below override those defaults.
    parser.add_argument(
        "--realism", action="store_true", help="Enable realism post-processing (scatter/noise/blur/gamma)."
    )
    parser.add_argument("--gamma", type=float, default=None, help="Display gamma (clinical ~0.8). Implies --realism.")
    parser.add_argument("--scatter", type=float, default=None, help="Compton scatter sigma in px. Implies --realism.")
    parser.add_argument(
        "--poisson", type=float, default=None, help="Poisson photon count (quantum noise). Implies --realism."
    )
    parser.add_argument(
        "--gaussian", type=float, default=None, help="Additive Gaussian noise sigma. Implies --realism."
    )
    parser.add_argument("--blur", type=float, default=None, help="Detector PSF blur sigma in px. Implies --realism.")
    return parser


# Sensible defaults for a plausible interventional-fluoroscopy look.
_REALISM_DEFAULTS = dict(
    gamma=0.8,
    scatter_sigma_px=16.0,
    poisson_photons=6000.0,
    gaussian_sigma=0.008,
    blur_sigma_px=0.6,
)

# Projection presets (degrees) applied before the user's extra --rx/--ry/--rz.
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
    any_knob = any(v is not None for v in knobs.values())
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
    print(volume)

    geometry_kwargs = {}
    if args.width is not None:
        geometry_kwargs["detector_width_px"] = args.width
    if args.height is not None:
        geometry_kwargs["detector_height_px"] = args.height

    realism = build_realism(args)
    config = SimulatorConfig(
        geometry=CarmGeometry(**geometry_kwargs) if geometry_kwargs else CarmGeometry(),
        realism=realism,
    )

    sim = FluoroSimulator(volume, config)
    print(sim)
    print(f"[render_drr] view={args.view}, realism={'on' if realism.enabled else 'off'}")

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
    print(
        f"[render_drr] Image shape: {frame.image.shape}, " f"range: [{frame.image.min():.4f}, {frame.image.max():.4f}]"
    )
    print(f"[render_drr] Render time: {frame.timestamp_ms:.2f} ms ({metrics.fps:.1f} FPS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
