# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from i4h_common.centerline import (
    _smooth_along_arc,
    ordered_centerline_lumen,
    ordered_centerline_path,
    sample_polyline,
    sample_polyline_scalar,
)


def test_ordered_centerline_uses_lowest_endpoint_and_farthest_branch() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0],
            [2.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [2, 4]], dtype=np.int64)

    path = ordered_centerline_path(points, edges, target_spacing_mm=0.5)

    np.testing.assert_allclose(path[0], points[0])
    assert np.linalg.norm(path[-1] - points[3]) < 1e-5


# --------------------------------------------------------------------------- #
# Smoothing the skeleton's spurious curvature
# --------------------------------------------------------------------------- #


def _zigzag(amplitude_mm: float, count: int = 61, step_mm: float = 1.0):
    """A straight run carrying a sawtooth far tighter than any vessel bends."""
    z = np.arange(count, dtype=np.float64) * step_mm
    x = amplitude_mm * (-1.0) ** np.arange(count)
    points = np.stack([x, np.zeros_like(z), z], axis=1)
    edges = np.stack([np.arange(count - 1), np.arange(1, count)], axis=1)
    return points, edges


def _min_radius_of_curvature(path: np.ndarray) -> float:
    first, middle, last = path[:-2], path[1:-1], path[2:]
    a = np.linalg.norm(middle - first, axis=1)
    b = np.linalg.norm(last - middle, axis=1)
    c = np.linalg.norm(last - first, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(middle - first, last - first), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = (a * b * c) / (4.0 * area)
    return float(np.nanmin(np.where(np.isfinite(radius), radius, np.inf)))


def test_smoothing_removes_curvature_the_anatomy_does_not_have() -> None:
    """A rod of 40 segments over 660 mm spans 16.5 mm a segment, so a 10 mm
    radius bend would need 104 degrees at one joint. It cannot, so it deviates
    instead -- worst at the tip, whose last joint is constrained on one side."""
    points, edges = _zigzag(1.5)

    raw = ordered_centerline_path(points, edges, target_spacing_mm=1.0, smoothing_mm=0.0)
    smoothed = ordered_centerline_path(points, edges, target_spacing_mm=1.0)

    assert _min_radius_of_curvature(smoothed) > 5.0 * _min_radius_of_curvature(raw)


def test_smoothing_holds_both_ends_exactly_where_they_were() -> None:
    """The first sample is the access point the rod is seeded from and aims its
    track down; the last is the target success is measured against. Tidying the
    path in between must move neither."""
    points, edges = _zigzag(1.5)

    raw = ordered_centerline_path(points, edges, target_spacing_mm=1.0, smoothing_mm=0.0)
    smoothed = ordered_centerline_path(points, edges, target_spacing_mm=1.0)

    # Resampling by arc length lands the final sample a float32 hair short of
    # the vertex, so the two paths are compared to each other, not to zero.
    np.testing.assert_allclose(smoothed[0], raw[0], atol=1e-4)
    np.testing.assert_allclose(smoothed[-1], raw[-1], atol=1e-4)


def test_smoothing_moves_the_path_by_less_than_the_lumen_radius() -> None:
    """Straightening a path is only safe while it stays inside the vessel. On
    ``s0011`` an 8.8 mm width moves the path at most 3.4 mm against a lumen
    radius of 9 to 14 mm."""
    points, edges = _zigzag(1.5)

    raw = ordered_centerline_path(points, edges, target_spacing_mm=1.0, smoothing_mm=0.0)
    smoothed = ordered_centerline_path(points, edges, target_spacing_mm=1.0)

    # Straightening shortens the arc, so the two paths hold different sample
    # counts and each smoothed point is measured to the nearest raw one.
    shift = np.linalg.norm(smoothed[:, None, :] - raw[None, :, :], axis=2).min(axis=1).max()
    # The sawtooth is +/-1.5 mm, so no correction can exceed its amplitude.
    assert shift <= 1.5 + 1e-6


def _uniform_zigzag(amplitude_mm: float, count: int = 121) -> np.ndarray:
    """A uniformly sampled sawtooth, so a millimetre of width is one sample."""
    z = np.arange(count, dtype=np.float64)
    return np.stack([amplitude_mm * (-1.0) ** np.arange(count), np.zeros_like(z), z], axis=1)


def test_smoothing_never_moves_a_sample_out_of_a_narrow_vessel() -> None:
    """The rod is seeded along this path, so a tidied path that leaves the lumen
    starts the catheter outside the vessel it is meant to be threaded through.
    One ``s0011`` sample moved 3.46 mm where the iliac radius is 3.31 mm before
    the correction was capped."""

    path = _uniform_zigzag(1.5)
    radii = np.full(path.shape[0], 0.4)

    smoothed = _smooth_along_arc(path, spacing_mm=1.0, width_mm=8.8, radii_mm=radii)

    shift = np.linalg.norm(smoothed - path, axis=1)
    assert np.all(shift <= 0.5 * radii + 1e-9)


def test_a_wide_lumen_is_smoothed_harder_than_a_narrow_one() -> None:
    """The cap has to bind on width, not merely exist: the coarse rod struggles
    in the wide aorta, which is exactly where there is room to help it."""
    path = _uniform_zigzag(1.5)

    def correction(radius: float) -> float:
        radii = np.full(path.shape[0], radius)
        smoothed = _smooth_along_arc(path, spacing_mm=1.0, width_mm=8.8, radii_mm=radii)
        return float(np.linalg.norm(smoothed - path, axis=1).max())

    assert correction(12.0) > 4.0 * correction(0.4)


def test_smoothing_without_radii_is_unconstrained() -> None:
    """A caller that supplies no widths has asked for no lumen check, and must
    not silently get a capped result that reads as a smoothing bug."""
    path = _uniform_zigzag(1.5)

    smoothed = _smooth_along_arc(path, spacing_mm=1.0, width_mm=8.8)

    # Peak-to-peak, not absolute: this sawtooth starts and ends on its own
    # extreme, so pinning the ends leaves a constant offset across the path.
    # That offset is linear in arc length and so carries no curvature, which is
    # the whole point of correcting the ends that way.
    assert float(np.ptp(smoothed[20:-20, 0])) < 0.1
    assert float(np.ptp(path[20:-20, 0])) > 2.9


def test_smoothing_leaves_an_already_straight_path_alone() -> None:
    """Otherwise the smoother would be introducing error of its own."""
    count = 41
    z = np.arange(count, dtype=np.float64)
    points = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    edges = np.stack([np.arange(count - 1), np.arange(1, count)], axis=1)

    smoothed = ordered_centerline_path(points, edges, target_spacing_mm=1.0)

    np.testing.assert_allclose(smoothed[:, :2], 0.0, atol=1e-9)


def test_the_raw_skeleton_is_still_reachable() -> None:
    """Smoothing is a default, not a policy: an ablation has to be able to ask
    what the extracted path actually said."""
    points, edges = _zigzag(1.5)

    raw = ordered_centerline_path(points, edges, target_spacing_mm=1.0, smoothing_mm=0.0)

    assert np.abs(raw[:, 0]).max() > 1.0


def test_smoothing_keeps_a_radius_for_every_sample() -> None:
    """The radii are resampled along the smoothed path, so the two must stay
    the same length or a containment lookup reads the wrong width."""
    points, edges = _zigzag(1.5)
    radii = np.full(points.shape[0], 4.0)

    path, path_radii = ordered_centerline_lumen(points, edges, target_spacing_mm=1.0, radii_mm=radii)

    assert path_radii is not None
    assert path_radii.shape == (path.shape[0],)


def test_sample_polyline_clamps_and_interpolates_arc_length() -> None:
    path = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float32)

    sampled = sample_polyline(path, np.asarray([-1.0, 0.5, 2.0, 5.0]))

    np.testing.assert_allclose(sampled, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]])


def test_sample_polyline_scalar_tracks_the_same_arc_length_as_the_points() -> None:
    """Radii have to land on the stations the path landed on.

    A radius sampled on a different parameterization than the point it
    describes gives a vessel width belonging somewhere else along the vessel.
    """
    path = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float32)

    sampled = sample_polyline_scalar(path, np.asarray([2.0, 4.0, 8.0]), np.asarray([-1.0, 0.5, 2.0, 5.0]))

    np.testing.assert_allclose(sampled, [2.0, 3.0, 6.0, 8.0])


def test_sample_polyline_scalar_rejects_a_mismatched_count() -> None:
    path = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="one entry per point"):
        sample_polyline_scalar(path, np.asarray([1.0, 2.0, 3.0]), np.asarray([0.0]))


def _tapering_vessel() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        dtype=np.float32,
    )
    edges = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    return points, edges, np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)


def test_the_lumen_reports_a_radius_for_every_path_sample() -> None:
    points, edges, radii = _tapering_vessel()

    path, sampled = ordered_centerline_lumen(points, edges, target_spacing_mm=0.5, radii_mm=radii)

    assert sampled is not None
    assert sampled.shape[0] == path.shape[0]


def test_the_lumen_radius_narrows_where_the_vessel_narrows() -> None:
    points, edges, radii = _tapering_vessel()

    _, sampled = ordered_centerline_lumen(points, edges, target_spacing_mm=0.5, radii_mm=radii)

    assert sampled[0] > sampled[-1]
    assert np.all(np.diff(sampled) <= 1e-6)


def test_the_lumen_has_no_radii_when_none_were_supplied() -> None:
    """Better than a default width, which would license a made-up vessel."""
    points, edges, _ = _tapering_vessel()

    _, sampled = ordered_centerline_lumen(points, edges, target_spacing_mm=0.5)

    assert sampled is None


def test_the_path_is_unchanged_by_asking_for_radii() -> None:
    """``ordered_centerline_path`` delegates here, so the two must not diverge."""
    points, edges, radii = _tapering_vessel()

    path, _ = ordered_centerline_lumen(points, edges, target_spacing_mm=0.5, radii_mm=radii)

    np.testing.assert_allclose(path, ordered_centerline_path(points, edges, target_spacing_mm=0.5, radii_mm=radii))
