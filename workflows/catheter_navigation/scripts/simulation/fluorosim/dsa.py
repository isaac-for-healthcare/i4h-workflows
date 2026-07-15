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

"""Digital Subtraction Angiography (DSA) pipeline.

DSA is the primary rendering mode clinicians use during catheter procedures.
The pipeline renders two DRRs from the same pose — a *mask* (no contrast) and
a *contrast* (with iodine in vessels) — subtracts them, and post-processes the
difference to reveal vascular anatomy.

Pipeline:
    1. Render **mask DRR** from baseline μ volume (no contrast)
    2. Render **contrast DRR** from contrast-enhanced μ volume
    3. Optionally apply scatter convolution to both DRRs
    4. Optionally apply misregistration jitter to the mask
    5. Subtract: diff = contrast_drr − mask_drr
    6. Post-process: contrast boost (k), gamma correction (γ), noise
    7. Normalize to [0, 1]

Usage:
    >>> from fluorosim.dsa import DSAPipeline
    >>> from fluorosim.vasculature import apply_vessel_boost
    >>>
    >>> # Prepare volumes
    >>> mu_mask = volume.mu_volume
    >>> mu_contrast = apply_vessel_boost(mu_mask, vessel_mask, boost_factor=8)
    >>>
    >>> # Create pipeline
    >>> pipeline = DSAPipeline(renderer, dsa_settings)
    >>>
    >>> # Render a single DSA frame
    >>> dsa_frame = pipeline.render_dsa_frame(mu_mask, mu_contrast, rotation, translation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DSASettings


@dataclass
class DSAFrame:
    """A single DSA frame with intermediate products.

    Attributes:
        dsa_image: Final post-processed DSA image, float32 in [0, 1].
        mask_drr: Raw mask DRR (before subtraction).
        contrast_drr: Raw contrast DRR (before subtraction).
        subtraction: Raw subtraction image (contrast - mask), before post-processing.
    """

    dsa_image: np.ndarray
    mask_drr: np.ndarray
    contrast_drr: np.ndarray
    subtraction: np.ndarray


class DSAPipeline:
    """Digital Subtraction Angiography rendering pipeline.

    Wraps an existing DRR renderer and applies the full DSA workflow:
    mask DRR → contrast DRR → optional scatter → optional jitter →
    subtraction → boost → gamma → noise → normalize.
    """

    def __init__(
        self,
        renderer,
        settings: DSASettings = DSASettings(),
    ):
        """Initialize the DSA pipeline.

        Args:
            renderer: A DRR renderer instance that supports both
                ``render(rotation, translation)`` and ``update_volume(mu_volume)``.
            settings: DSA pipeline settings.
        """
        self._renderer = renderer
        self._settings = settings

    @property
    def settings(self) -> DSASettings:
        return self._settings

    def render_dsa_frame(
        self,
        mu_mask: np.ndarray,
        mu_contrast: np.ndarray,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        seed: int | None = None,
    ) -> DSAFrame:
        """Render a single DSA frame.

        Both the mask and contrast volumes are rendered at the same pose. The
        renderer's internal μ texture is swapped for each render call using
        ``update_volume``.

        Args:
            mu_mask: Baseline μ volume (Z, Y, X), float32.
            mu_contrast: Contrast-enhanced μ volume, same shape.
            rotation: C-arm rotation (Euler angles in radians).
            translation: C-arm translation in mm.
            seed: Random seed for noise (None = random).

        Returns:
            DSAFrame with the final image and intermediates.
        """
        cfg = self._settings

        # 1. Render mask DRR
        self._renderer.update_volume(mu_mask)
        mask_drr = self._renderer.render(rotation, translation)

        # 2. Render contrast DRR
        self._renderer.update_volume(mu_contrast)
        contrast_drr = self._renderer.render(rotation, translation)

        # 3. Apply scatter convolution
        if cfg.scatter_sigma_px > 0.0:
            from .rendering.realism import apply_scatter

            mask_drr = apply_scatter(mask_drr, cfg.scatter_sigma_px)
            contrast_drr = apply_scatter(contrast_drr, cfg.scatter_sigma_px)

        # 4. Apply noise independently to each DRR
        rng = np.random.default_rng(seed)
        mask_drr = self._apply_dsa_noise(mask_drr, rng)
        contrast_drr = self._apply_dsa_noise(contrast_drr, rng)

        # 5. Apply misregistration jitter to mask
        dx, dy = cfg.misregistration_px
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            from .rendering.realism import apply_misregistration

            mask_drr = apply_misregistration(mask_drr, (dy, dx))

        # 6. Subtraction
        subtraction = contrast_drr - mask_drr

        # 7. Post-process: contrast boost + gamma + normalize
        dsa_image = self._postprocess(subtraction)

        return DSAFrame(
            dsa_image=dsa_image,
            mask_drr=mask_drr,
            contrast_drr=contrast_drr,
            subtraction=subtraction,
        )

    def render_dsa_sequence(
        self,
        mu_mask: np.ndarray,
        contrast_volumes: list[np.ndarray],
        rotations: list[tuple[float, float, float]],
        translations: list[tuple[float, float, float]],
        progress: bool = True,
    ) -> list[DSAFrame]:
        """Render a temporal DSA sequence with per-frame contrast volumes.

        Args:
            mu_mask: Baseline μ volume (shared across all frames).
            contrast_volumes: List of contrast-enhanced μ volumes, one per frame.
            rotations: Per-frame C-arm rotations.
            translations: Per-frame C-arm translations.
            progress: Print progress.

        Returns:
            List of DSAFrame objects.
        """
        n = len(contrast_volumes)
        frames = []

        for i in range(n):
            frame = self.render_dsa_frame(
                mu_mask=mu_mask,
                mu_contrast=contrast_volumes[i],
                rotation=rotations[i] if i < len(rotations) else rotations[-1],
                translation=translations[i] if i < len(translations) else translations[-1],
                seed=i,
            )
            frames.append(frame)

            if progress and (i + 1) % 10 == 0:
                print(f"[DSA] Frame {i + 1}/{n}")

        return frames

    def _apply_dsa_noise(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply Poisson + Gaussian noise to a single DRR (pre-subtraction)."""
        cfg = self._settings
        out = img.copy()

        if cfg.poisson_photons > 0.0:
            lam = np.clip(out, 0.0, None) * cfg.poisson_photons
            out = rng.poisson(lam=lam).astype(np.float32) / cfg.poisson_photons

        if cfg.gaussian_sigma > 0.0:
            out = out + rng.normal(0.0, cfg.gaussian_sigma, size=out.shape).astype(np.float32)

        return out

    def _postprocess(self, subtraction: np.ndarray) -> np.ndarray:
        """Post-process the subtraction image: boost, gamma, normalize."""
        cfg = self._settings

        out = subtraction.astype(np.float32, copy=True)

        # Contrast boost
        out = out * cfg.contrast_boost

        # Clip to non-negative (vessels should be brighter in contrast DRR)
        out = np.clip(out, 0.0, None)

        # Gamma correction
        if cfg.gamma != 1.0 and cfg.gamma > 0.0:
            vmax = float(np.max(out))
            if vmax > 0:
                out = out / vmax
                out = np.power(out, 1.0 / cfg.gamma)
                out = out * vmax

        # Normalize to [0, 1]
        vmin = float(np.min(out))
        vmax = float(np.max(out))
        eps = 1e-8
        if vmax - vmin > eps:
            out = (out - vmin) / (vmax - vmin)
        else:
            out = np.zeros_like(out)

        return out.astype(np.float32)
