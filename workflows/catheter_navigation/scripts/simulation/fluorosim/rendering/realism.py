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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RealismConfig:
    """Realism knobs for fluoroscopy intensity images.

    CPU-side (NumPy) post-processing applied to rendered DRR frames.

    Conventions:
    - Input is expected to be a non-negative float image (intensity-like),
      typically I = I0 * exp(-∫μ ds).
    - Output is float32. If normalize_output=True, output is scaled to [0,1].
    """

    gain: float = 1.0
    bias: float = 0.0

    poisson_photons: float = 0.0
    gaussian_sigma: float = 0.0

    blur_sigma_px: float = 0.0
    scatter_sigma_px: float = 0.0

    gamma: float = 1.0

    normalize_output: bool = True
    eps: float = 1e-8

    seed: int | None = 0


def apply_scatter(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Physical scatter model: S = G(sigma_s) * I, added to primary signal.

    Models the broad, low-frequency haze caused by Compton scatter in the
    patient and detector housing. The scatter-to-primary ratio is fixed at
    the Gaussian amplitude (no tuneable SPR parameter for now).

    Args:
        img: 2D float32 image (non-negative).
        sigma_px: Gaussian kernel sigma in pixels.

    Returns:
        Image with scatter contribution added.
    """
    import scipy.ndimage  # type: ignore

    scatter = scipy.ndimage.gaussian_filter(img, sigma=sigma_px).astype(np.float32, copy=False)
    return img + scatter


def apply_misregistration(img: np.ndarray, shift_px: tuple[float, float]) -> np.ndarray:
    """Sub-pixel shift to simulate patient motion between mask and contrast acquisitions.

    Uses scipy affine_transform for sub-pixel accuracy.

    Args:
        img: 2D float32 image.
        shift_px: (dy, dx) shift in pixels.

    Returns:
        Shifted image.
    """
    dy, dx = shift_px
    if abs(dy) < 1e-6 and abs(dx) < 1e-6:
        return img

    import scipy.ndimage  # type: ignore

    return scipy.ndimage.shift(img, (dy, dx), order=1, mode="constant", cval=0.0).astype(np.float32, copy=False)


def apply_realism(img: np.ndarray, cfg: RealismConfig = RealismConfig()) -> np.ndarray:
    """Apply realism post-processing to a single 2D frame.

    Pipeline order: gain/bias → scatter → Poisson → Gaussian → blur → gamma → normalize.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image; got shape={img.shape}")

    out = img.astype(np.float32, copy=True)
    out = out * float(cfg.gain) + float(cfg.bias)
    out = np.clip(out, 0.0, None)

    if float(cfg.scatter_sigma_px) > 0.0:
        try:
            out = apply_scatter(out, float(cfg.scatter_sigma_px))
        except ImportError:
            pass

    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()

    if float(cfg.poisson_photons) > 0.0:
        lam = np.clip(out, 0.0, None) * float(cfg.poisson_photons)
        out = rng.poisson(lam=lam).astype(np.float32) / float(cfg.poisson_photons)

    if float(cfg.gaussian_sigma) > 0.0:
        out = out + rng.normal(loc=0.0, scale=float(cfg.gaussian_sigma), size=out.shape).astype(np.float32)
        out = np.clip(out, 0.0, None)

    if float(cfg.blur_sigma_px) > 0.0:
        try:
            import scipy.ndimage  # type: ignore

            out = scipy.ndimage.gaussian_filter(out, sigma=float(cfg.blur_sigma_px)).astype(np.float32, copy=False)
        except ImportError:
            pass

    if float(cfg.gamma) != 1.0 and float(cfg.gamma) > 0.0:
        out = np.clip(out, 0.0, None)
        vmax = float(np.max(out))
        if vmax > 0:
            out = out / vmax
            out = np.power(out, 1.0 / float(cfg.gamma))
            out = out * vmax

    if cfg.normalize_output:
        vmin = float(np.min(out))
        vmax = float(np.max(out))
        out = (out - vmin) / (vmax - vmin + float(cfg.eps))

    return out.astype(np.float32, copy=False)
