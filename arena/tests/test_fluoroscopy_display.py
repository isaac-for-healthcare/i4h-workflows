# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The batched torch display chain against the CPU references it replaced."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter
from xray_simulator.config import DisplaySettings
from xray_simulator.display import apply_display

from i4h_arena.medical.fluoroscopy_display import (
    CinematicStyle,
    display_frames,
    gaussian_blur,
    percentile,
    vessel_overlay,
)


def _intensities(count: int = 3, size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(7)
    ramp = np.linspace(0.01, 0.95, size * size, dtype=np.float32).reshape(size, size)
    return np.stack(
        [np.clip(ramp * (0.6 + 0.2 * index) + rng.normal(0.0, 0.02, ramp.shape), 0.0, 1.0) for index in range(count)]
    ).astype(np.float32)


def _grayscale(count: int = 2, size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (count, size, size), dtype=np.uint8)


@pytest.mark.parametrize(
    "settings",
    [
        DisplaySettings(scaling="log", log_window=(0.1, 2.5)),
        DisplaySettings(polarity="diagnostic", scaling="log", log_window=(0.0, 3.0)),
        DisplaySettings(scaling="log", log_window=(0.2, 1.5), gamma=2.2),
        DisplaySettings(scaling="transmission"),
        DisplaySettings(scaling="window", window=(0.2, 0.7)),
        DisplaySettings(scaling="per_frame"),
    ],
    ids=["log", "diagnostic", "gamma", "transmission", "window", "per_frame"],
)
def test_display_frames_matches_apply_display_frame_by_frame(settings) -> None:
    intensity = _intensities()

    batched = display_frames(torch.from_numpy(intensity), settings).numpy()

    for index, frame in enumerate(intensity):
        np.testing.assert_allclose(batched[index], apply_display(frame, settings), atol=2e-6)


@pytest.mark.parametrize("q", [0.0, 1.0, 50.0, 99.0, 99.5, 100.0])
def test_percentile_matches_numpy_per_frame(q) -> None:
    frames = _intensities()

    np.testing.assert_allclose(
        percentile(torch.from_numpy(frames), q).numpy(), np.percentile(frames.reshape(3, -1), q, axis=1), rtol=1e-6
    )


@pytest.mark.parametrize("radius", [0.9, 2.2])
def test_gaussian_blur_lands_on_pillows_grey_levels(radius) -> None:
    images = _grayscale()
    images[1] = np.linspace(0, 255, images[1].size).reshape(images[1].shape).astype(np.uint8)

    blurred = gaussian_blur(torch.from_numpy(images), radius).numpy()

    for index, image in enumerate(images):
        expected = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
        assert np.abs(blurred[index] - expected).max() <= 1.0


def _reference_cinematic(frames: list[np.ndarray]) -> list[np.ndarray]:
    """The removed per-image numpy implementation, without its noise."""
    persisted = None
    styled_frames = []
    for grayscale in frames:
        source = Image.fromarray(grayscale)
        base = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=0.9)), dtype=np.float32) / 255.0
        bloom = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=2.2)), dtype=np.float32) / 255.0
        frame = np.clip(0.85 * base + 0.15 * bloom, 0.0, 1.0)
        persisted = frame.copy() if persisted is None else 0.76 * persisted + 0.24 * frame
        height, width = frame.shape
        yy, xx = np.ogrid[:height, :width]
        radius = np.sqrt(((xx - 0.5 * width) / width) ** 2 + ((yy - 0.5 * height) / height) ** 2)
        vignette = np.clip(1.0 - 0.9 * radius, 0.78, 1.0).astype(np.float32)
        styled = np.clip(np.power(np.clip(persisted, 0.0, 1.0) * vignette, 1.06), 0.0, 1.0)
        styled_frames.append(np.rint(255.0 * styled))
    return styled_frames


def test_the_cinematic_look_matches_the_numpy_reference_across_frames(monkeypatch) -> None:
    monkeypatch.setattr(torch, "randn", lambda shape, **kwargs: torch.zeros(shape, dtype=kwargs.get("dtype")))
    first, second = _grayscale(count=2)
    style = CinematicStyle(height=48, width=48, device=torch.device("cpu"))

    styled = [
        style(torch.from_numpy(frame[None].astype(np.float32)), key="drr")[0].numpy() for frame in (first, second)
    ]

    for actual, expected in zip(styled, _reference_cinematic([first, second]), strict=True):
        assert np.abs(actual - expected).max() <= 1.0


def test_cinematic_persistence_is_kept_per_environment_and_per_stream() -> None:
    style = CinematicStyle(height=48, width=48, device=torch.device("cpu"))
    dark = torch.zeros((2, 48, 48))
    bright = torch.full((2, 48, 48), 255.0)
    style(dark, key="drr")
    style(bright, key="dsa")

    lit = torch.stack((bright[0], dark[1]))
    after = style(lit, key="drr")

    # Persistence drags the newly lit environment toward its own dark past and nothing else.
    assert 40.0 < float(after[0].mean()) < 100.0
    assert float(after[1].mean()) < 5.0


def test_cinematic_noise_is_seeded() -> None:
    frames = torch.full((1, 48, 48), 128.0)

    first = CinematicStyle(height=48, width=48, device=torch.device("cpu"))(frames, key="drr")
    again = CinematicStyle(height=48, width=48, device=torch.device("cpu"))(frames, key="drr")

    torch.testing.assert_close(first, again)
    assert float(first.std()) > 0.0


def test_vessel_overlay_matches_the_numpy_tint() -> None:
    rgb = np.repeat(_grayscale(count=2)[..., None], 3, axis=-1)
    alpha = np.random.default_rng(5).uniform(0.0, 0.82, rgb.shape[:-1]).astype(np.float32)

    tinted = vessel_overlay(torch.from_numpy(rgb), torch.from_numpy(alpha)).numpy()

    expected = rgb.astype(np.float32)
    expected[..., 0] *= 1.0 - alpha
    expected[..., 1] = expected[..., 1] * (1.0 - alpha) + 255.0 * alpha
    expected[..., 2] = expected[..., 2] * (1.0 - alpha) + 255.0 * alpha
    np.testing.assert_array_equal(tinted, np.clip(expected, 0.0, 255.0).astype(np.uint8))
