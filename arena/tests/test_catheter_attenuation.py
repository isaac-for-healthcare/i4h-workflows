# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from i4h_arena.medical.catheter_attenuation import CatheterAttenuation, CatheterMaterial
from i4h_arena.medical.slang_fluoroscopy import ProjectionGeometry

DETECTOR_PX = 128
PIXEL_MM = 1.0
SID_MM = 500.0
SDD_MM = 1000.0
MAGNIFICATION = SDD_MM / SID_MM

# The capsule axis sits at the isocenter, so a point at volume x lands on this column.
CENTER_COLUMN = 0.5 * DETECTOR_PX - 0.5
CENTER_ROW = int(0.5 * DETECTOR_PX)


def _projection() -> ProjectionGeometry:
    """A beam down +z with the isocenter at the origin, so pixels map to volume mm by hand."""
    return ProjectionGeometry(
        rotation_zxy_rad=(0.0, 0.0, 0.0),
        translation_xyz_mm=(0.0, 0.0, 0.0),
        source_to_detector_mm=SDD_MM,
        source_to_isocenter_mm=SID_MM,
        pixel_spacing_mm=PIXEL_MM,
        local_to_volume=np.eye(3),
        isocenter_volume_mm=np.zeros(3),
    )


def _column_of(volume_x_mm: float) -> int:
    return int(round(MAGNIFICATION * volume_x_mm + CENTER_COLUMN))


def _shaft(mu_per_mm: float = 1.0) -> CatheterAttenuation:
    """A catheter of one uniform material, so chords can be read straight off the image."""
    return CatheterAttenuation(CatheterMaterial(shaft_mu_per_mm=mu_per_mm, tip_mu_per_mm=mu_per_mm, tip_length_mm=0.0))


def _straight_catheter(x_from: float = -20.0, x_to: float = 20.0, nodes: int = 41) -> np.ndarray:
    points = np.zeros((nodes, 3), dtype=np.float32)
    points[:, 0] = np.linspace(x_from, x_to, nodes)
    return points


def _line_integral(
    compositor: CatheterAttenuation,
    points: np.ndarray,
    radius_mm: float = 2.0,
) -> np.ndarray:
    return compositor.line_integral(points, radius_mm, _projection(), width=DETECTOR_PX, height=DETECTOR_PX)


def _sampled_chord(points: np.ndarray, radius_mm: float, column: int, row: int, samples: int = 200_000) -> float:
    """Chord through the polyline for one pixel's ray, by dense sampling.

    Deliberately independent of the kernel: the analytic solve is the thing under test, and a
    sampled reference is how the fused shader accumulates the same quantity. Spans are flat-ended
    here too, matching the solid the module models.
    """
    source = np.array([0.0, 0.0, -SID_MM])
    target = np.array(
        [
            (column + 0.5 - 0.5 * DETECTOR_PX) * PIXEL_MM,
            (row + 0.5 - 0.5 * DETECTOR_PX) * PIXEL_MM,
            SDD_MM - SID_MM,
        ]
    )
    beam = target - source
    span = float(np.linalg.norm(beam))
    positions = source + np.linspace(0.0, span, samples)[:, None] * (beam / span)
    inside = np.zeros(samples, dtype=bool)
    for start, end in zip(points[:-1], points[1:], strict=True):
        axis = np.asarray(end - start, dtype=np.float64)
        along = ((positions - start) @ axis) / (axis @ axis)
        radial = np.linalg.norm(positions - (start + along[:, None] * axis), axis=1)
        inside |= (radial <= radius_mm) & (along >= 0.0) & (along <= 1.0)
    return float(np.count_nonzero(inside) * span / samples)


@pytest.mark.parametrize("column_offset", [0, 1, -3, 20])
def test_analytic_chords_match_a_sampled_ray_march(column_offset: int) -> None:
    points = _straight_catheter()
    radius_mm = 2.0

    line_integral = _line_integral(_shaft(), points, radius_mm)

    column = _column_of(0.0) + column_offset
    assert line_integral[CENTER_ROW, column] == pytest.approx(
        _sampled_chord(points, radius_mm, column, CENTER_ROW), abs=0.02
    )


def test_a_bent_catheter_matches_a_sampled_ray_march() -> None:
    """Joints are where an analytic solve could disagree: two capsules overlap around each node."""
    points = np.array([[-20.0, 0.0, 0.0], [0.0, 6.0, 4.0], [15.0, -3.0, -6.0], [22.0, 5.0, 0.0]], dtype=np.float32)
    radius_mm = 2.5

    line_integral = _line_integral(_shaft(), points, radius_mm)

    hits = [(row, column) for row, column in zip(*np.nonzero(line_integral), strict=True)][::37]
    assert len(hits) > 3
    for row, column in hits:
        assert line_integral[row, column] == pytest.approx(_sampled_chord(points, radius_mm, column, row), abs=0.06)


def test_a_catheter_crossing_the_beam_darkens_a_band_two_pixels_wide() -> None:
    """A 1 mm shaft at the isocenter covers its own diameter times the magnification."""
    line_integral = _line_integral(_shaft(), _straight_catheter(), radius_mm=0.5)

    rows = np.flatnonzero(line_integral.any(axis=1))
    assert len(rows) == pytest.approx(2.0 * 0.5 * MAGNIFICATION, abs=1)


def test_geometry_closer_to_the_source_projects_wider() -> None:
    """Magnification comes out of the projection, which a drawn overlay cannot reproduce."""
    at_isocenter = _straight_catheter()
    toward_source = at_isocenter.copy()
    toward_source[:, 2] = -0.5 * SID_MM

    isocenter_rows = np.flatnonzero(_line_integral(_shaft(), at_isocenter).any(axis=1))
    source_rows = np.flatnonzero(_line_integral(_shaft(), toward_source).any(axis=1))

    assert len(source_rows) > 1.5 * len(isocenter_rows)


def test_the_catheter_lands_where_the_projection_puts_it() -> None:
    """The kernel and the renderer must agree on the detector grid, half-pixel offset included.

    The offset is a quarter millimetre off a round number so the shaft projects onto the centre of
    one column rather than straddling two, which would make the expected answer a coin flip.
    """
    offset_mm = 12.25
    points = _straight_catheter()
    points[:, 0] = offset_mm
    points[:, 1] = np.linspace(-15.0, 15.0, points.shape[0])

    line_integral = _line_integral(_shaft(), points)

    assert int(np.argmax(line_integral.sum(axis=0))) == _column_of(offset_mm)


def test_the_tip_marker_attenuates_more_than_the_shaft() -> None:
    material = CatheterMaterial(shaft_mu_per_mm=0.8, tip_mu_per_mm=3.0, tip_length_mm=2.0)

    line_integral = _line_integral(CatheterAttenuation(material), _straight_catheter())

    shaft = line_integral[CENTER_ROW, _column_of(0.0)]
    marker = line_integral[CENTER_ROW, _column_of(19.0)]
    assert marker > 3.0 * shaft


def test_the_marker_covers_the_requested_length_of_the_distal_end() -> None:
    lengths_mm = np.full(20, 1.0)

    mu_per_mm = CatheterMaterial(shaft_mu_per_mm=0.8, tip_mu_per_mm=3.0, tip_length_mm=3.0).segment_mu_per_mm(
        lengths_mm
    )

    assert np.count_nonzero(mu_per_mm == 3.0) == 3
    np.testing.assert_allclose(mu_per_mm[:-3], 0.8)


def test_a_shaft_without_a_marker_is_uniform() -> None:
    points = _straight_catheter()

    line_integral = _line_integral(_shaft(), points)

    assert line_integral[CENTER_ROW, _column_of(19.0)] == pytest.approx(
        line_integral[CENTER_ROW, _column_of(0.0)], rel=0.05
    )


def test_attenuation_scales_with_the_coefficient() -> None:
    points = _straight_catheter()

    weak = _line_integral(_shaft(0.4), points)
    strong = _line_integral(_shaft(0.8), points)

    np.testing.assert_allclose(strong, 2.0 * weak, atol=1e-5)


def test_a_thicker_catheter_blocks_more_of_the_beam() -> None:
    """Total attenuation follows the volume of material in the beam, so it goes as the radius squared."""
    points = _straight_catheter()

    thin = _line_integral(_shaft(), points, radius_mm=0.5)
    thick = _line_integral(_shaft(), points, radius_mm=1.0)

    assert thick.sum() == pytest.approx(4.0 * thin.sum(), rel=0.1)
    assert thick[CENTER_ROW, _column_of(0.0)] > thin[CENTER_ROW, _column_of(0.0)]


def test_geometry_behind_the_source_does_not_attenuate() -> None:
    points = _straight_catheter()
    points[:, 2] = -2.0 * SID_MM

    assert not _line_integral(_shaft(), points).any()


def test_a_catheter_outside_the_beam_does_not_attenuate() -> None:
    points = _straight_catheter()
    points[:, 1] = 400.0

    assert not _line_integral(_shaft(), points).any()


def test_coincident_nodes_are_ignored_rather_than_dividing_by_a_zero_axis() -> None:
    points = np.zeros((4, 3), dtype=np.float32)

    line_integral = _line_integral(_shaft(), points)

    assert line_integral.shape == (DETECTOR_PX, DETECTOR_PX)
    assert not line_integral.any()


def test_a_single_node_carries_no_geometry() -> None:
    assert not _line_integral(_shaft(), np.zeros((1, 3), dtype=np.float32)).any()


def test_the_buffer_is_reused_without_leaking_the_previous_frame() -> None:
    compositor = _shaft()
    moving = _straight_catheter()

    first = _line_integral(compositor, moving).copy()
    moving[:, 1] += 10.0
    second = _line_integral(compositor, moving)

    assert first.any() and second.any()
    assert not np.array_equal(first, second)
    assert np.count_nonzero(first) == pytest.approx(np.count_nonzero(second), rel=0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shaft_mu_per_mm": -0.1},
        {"tip_mu_per_mm": float("nan")},
        {"tip_length_mm": -1.0},
    ],
)
def test_invalid_materials_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CatheterMaterial(**kwargs)


@pytest.mark.parametrize("radius_mm", [0.0, -1.0, float("inf")])
def test_a_catheter_must_have_a_positive_radius(radius_mm: float) -> None:
    with pytest.raises(ValueError, match="radius_mm"):
        _line_integral(_shaft(), _straight_catheter(), radius_mm)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((3, 2), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.full((3, 3), np.nan, dtype=np.float32),
    ],
)
def test_malformed_node_arrays_are_rejected(points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _line_integral(_shaft(), points)
