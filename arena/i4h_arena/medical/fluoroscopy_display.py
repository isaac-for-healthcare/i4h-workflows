# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fluoroscopy display mapping and cine look as batched torch operations.

Every frame of every environment goes through these on the simulation's device, so a rendered
detector image never has to visit the host before a policy reads it. Each function reproduces a
CPU reference that tests hold it to: ``display_frames`` is ``xray_simulator.display.apply_display``,
``gaussian_blur`` is Pillow's ``GaussianBlur``, and ``CinematicStyle`` is the reference viewport's
blur, persistence, noise and vignette.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from xray_simulator.config import DisplaySettings

_EPS = 1e-8
# Pillow approximates a Gaussian with this many box passes along each axis.
_BOX_PASSES = 3


def display_frames(intensity: torch.Tensor, settings: DisplaySettings, i0: float = 1.0) -> torch.Tensor:
    """Map rendered intensity ``(N, H, W)`` to display values in ``[0, 1]``, frame by frame."""
    if intensity.ndim != 3:
        raise ValueError(f"expected (N, H, W) frames, got shape {tuple(intensity.shape)}")
    if i0 <= 0.0:
        raise ValueError(f"i0 must be positive, got {i0}")
    values = (intensity.to(torch.float32) / float(i0)).clamp(0.0, 1.0)
    if settings.scaling == "log":
        low, high = settings.log_window
        line_integral = -torch.log(values.clamp(_EPS, 1.0))
        scaled = 1.0 - ((line_integral - low) / (high - low)).clamp(0.0, 1.0)
    elif settings.scaling == "transmission":
        scaled = values
    elif settings.scaling == "window":
        low, high = settings.window
        scaled = ((values - low) / (high - low)).clamp(0.0, 1.0)
    else:
        flat = values.flatten(1)
        minimum = flat.amin(dim=1)[:, None, None]
        maximum = flat.amax(dim=1)[:, None, None]
        scaled = (values - minimum) / (maximum - minimum + _EPS)
    if settings.polarity == "diagnostic":
        scaled = 1.0 - scaled
    if settings.gamma != 1.0:
        scaled = scaled.clamp(0.0, 1.0).pow(1.0 / settings.gamma)
    return scaled.clamp(0.0, 1.0)


def percentile(frames: torch.Tensor, q: float) -> torch.Tensor:
    """Per-frame ``numpy.percentile`` with linear interpolation, shape ``(N,)``.

    ``torch.quantile`` refuses inputs past 16M elements, which a batch of megapixel frames passes.
    """
    flat = frames.flatten(1)
    count = flat.shape[1]
    position = (float(q) / 100.0) * (count - 1)
    lower = int(math.floor(position))
    fraction = position - lower
    low = flat.kthvalue(lower + 1, dim=1).values
    if fraction == 0.0 or lower + 1 >= count:
        return low
    high = flat.kthvalue(lower + 2, dim=1).values
    return low + fraction * (high - low)


def _box_radius(radius: float) -> float:
    """Pillow's box radius for one of ``_BOX_PASSES`` passes approximating a Gaussian of ``radius``."""
    sigma2 = radius * radius / _BOX_PASSES
    extent = math.sqrt(12.0 * sigma2 + 1.0)
    whole = math.floor((extent - 1.0) / 2.0)
    fraction = (2 * whole + 1) * (whole * (whole + 1) - 3 * sigma2)
    fraction /= 6 * (sigma2 - (whole + 1) * (whole + 1))
    return whole + fraction


def _box_kernel(radius: float, *, device: torch.device) -> torch.Tensor:
    """A box of ``2 * radius + 1`` pixels whose fractional ends get partial weight."""
    whole = int(radius)
    kernel = torch.ones(2 * whole + 3, dtype=torch.float32, device=device)
    kernel[0] = kernel[-1] = radius - whole
    return kernel / (2.0 * radius + 1.0)


def gaussian_blur(grayscale: torch.Tensor, radius: float) -> torch.Tensor:
    """Pillow's ``GaussianBlur`` over ``(N, H, W)`` grey levels, rounded like its 8-bit passes.

    Pillow runs three box passes along x, then three along y, replicating edge pixels and storing
    each pass as 8 bits; the rounding is kept so the result lands on the same grey levels.
    """
    kernel = _box_kernel(_box_radius(float(radius)), device=grayscale.device)
    pad = (kernel.numel() - 1) // 2
    images = grayscale.to(torch.float32)[:, None]
    for weights, padding in (
        (kernel.view(1, 1, 1, -1), (pad, pad, 0, 0)),
        (kernel.view(1, 1, -1, 1), (0, 0, pad, pad)),
    ):
        for _ in range(_BOX_PASSES):
            images = torch.floor(F.conv2d(F.pad(images, padding, mode="replicate"), weights) + 0.5)
    return images[:, 0]


class CinematicStyle:
    """The reference viewport's cine blur, persistence, noise and vignette, per environment.

    Persistence blends each frame into the previous one, so the state is kept per stream (plain and
    contrast run) and per environment, which is the leading axis of the stored frame.
    """

    def __init__(self, *, height: int, width: int, device: torch.device, seed: int = 42) -> None:
        rows = torch.arange(height, dtype=torch.float32, device=device)[:, None]
        columns = torch.arange(width, dtype=torch.float32, device=device)[None, :]
        radius = torch.sqrt(((columns - 0.5 * width) / width) ** 2 + ((rows - 0.5 * height) / height) ** 2)
        self._vignette = (1.0 - 0.9 * radius).clamp(0.78, 1.0)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self._persistence: dict[str, torch.Tensor] = {}

    def __call__(self, grayscale: torch.Tensor, *, key: str) -> torch.Tensor:
        """Style ``(N, H, W)`` grey levels and return grey levels on the same 0-255 scale."""
        base = gaussian_blur(grayscale, 0.9) / 255.0
        bloom = gaussian_blur(grayscale, 2.2) / 255.0
        frame = (0.85 * base + 0.15 * bloom).clamp(0.0, 1.0)
        previous = self._persistence.get(key)
        if previous is None or previous.shape != frame.shape:
            persisted = frame
        else:
            persisted = 0.76 * previous + 0.24 * frame
        self._persistence[key] = persisted
        noise = torch.randn(frame.shape, generator=self._generator, device=frame.device, dtype=frame.dtype)
        styled = ((persisted + 0.012 * noise).clamp(0.0, 1.0) * self._vignette).pow(1.06).clamp(0.0, 1.0)
        return torch.round(255.0 * styled)


def vessel_overlay(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Tint ``(N, H, W, 3)`` uint8 frames cyan where the contrast run shows vessel."""
    result = rgb.to(torch.float32)
    keep = 1.0 - alpha
    tinted = torch.stack(
        (result[..., 0] * keep, result[..., 1] * keep + 255.0 * alpha, result[..., 2] * keep + 255.0 * alpha),
        dim=-1,
    )
    return tinted.clamp(0.0, 255.0).to(torch.uint8)
