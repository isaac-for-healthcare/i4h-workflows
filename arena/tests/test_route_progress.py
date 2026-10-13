# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remaining vessel has to fall as the tip advances, including where the route doubles back."""

import numpy as np
import pytest

from i4h_arena.medical.route_progress import (
    RouteProgress,
    project_to_route,
    route_arc_m,
    route_progress,
    unambiguous_radius_m,
)


#: A straight run, a 180-degree turn, and a straight run back, which is the
#: aortic arch in miniature: the two limbs are 0.04 m apart and the endpoint is
#: nearer the middle of the route than the middle is to the start.
def _hairpin(limb_m=0.2, gap_m=0.04, turn_points=9):
    up = [(x, 0.0, 0.0) for x in np.linspace(0.0, limb_m, 21)]
    angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, turn_points)
    turn = [(limb_m + 0.5 * gap_m * np.cos(a), 0.5 * gap_m * (1.0 + np.sin(a)), 0.0) for a in angles]
    down = [(x, gap_m, 0.0) for x in np.linspace(limb_m, 0.0, 21)]
    return np.array(up + turn[1:-1] + down, dtype=np.float64)


def _straight(length_m=0.3, count=31):
    return np.stack([np.linspace(0.0, length_m, count), np.zeros(count), np.zeros(count)], axis=1)


# --------------------------------------------------------------------------- #
# Arc length
# --------------------------------------------------------------------------- #
def test_arc_starts_at_zero_and_ends_at_the_route_length():
    arc = route_arc_m(_straight(0.3))

    assert arc[0] == pytest.approx(0.0)
    assert arc[-1] == pytest.approx(0.3)


def test_arc_measures_the_polyline_not_the_straight_line():
    """The whole point: on a hairpin the two differ by nearly the full route."""
    path = _hairpin()

    assert route_arc_m(path)[-1] > 0.4
    assert np.linalg.norm(path[-1] - path[0]) == pytest.approx(0.04)


@pytest.mark.parametrize("path", [np.zeros((1, 3)), np.zeros((0, 3)), np.zeros((4, 2))])
def test_a_route_that_is_not_a_polyline_is_rejected(path):
    with pytest.raises(ValueError, match="two 3-D points"):
        route_arc_m(path)


def test_a_route_with_a_non_finite_point_is_rejected():
    """Silently projecting onto a nan would report the tip as nowhere."""
    path = _straight()
    path[5, 1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        route_arc_m(path)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def test_a_point_on_the_route_projects_onto_itself():
    arc_m, lateral_m = project_to_route(_straight(0.3), (0.12, 0.0, 0.0))

    assert arc_m == pytest.approx(0.12)
    assert lateral_m == pytest.approx(0.0)


def test_an_offset_point_keeps_its_arc_and_reports_the_offset():
    """A catheter is never exactly on the centerline; being 2 mm off the wall
    must not move where along the vessel it is judged to be."""
    arc_m, lateral_m = project_to_route(_straight(0.3), (0.12, 0.002, 0.0))

    assert arc_m == pytest.approx(0.12)
    assert lateral_m == pytest.approx(0.002)


def test_a_point_before_the_route_clamps_to_the_start():
    arc_m, _ = project_to_route(_straight(0.3), (-0.05, 0.0, 0.0))

    assert arc_m == pytest.approx(0.0)


def test_a_point_past_the_route_clamps_to_the_end():
    arc_m, _ = project_to_route(_straight(0.3), (0.5, 0.0, 0.0))

    assert arc_m == pytest.approx(0.3)


def test_a_repeated_route_point_does_not_produce_a_nan_projection():
    """A zero-length segment divides by zero, and a nan would win the argmin."""
    path = np.array([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)])

    arc_m, lateral_m = project_to_route(path, (0.15, 0.0, 0.0))

    assert arc_m == pytest.approx(0.15)
    assert lateral_m == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Monotonicity
#
# The property the whole module exists for. Straight-line distance to the far
# end is not monotonic on a route that doubles back, which is what makes the
# shipped readout unusable for steering.
# --------------------------------------------------------------------------- #
def test_remaining_vessel_falls_all_the_way_along_a_hairpin():
    path = _hairpin()
    remaining = [route_progress(path, point).remaining_m for point in path]

    assert np.all(np.diff(remaining) < 1e-9)
    assert remaining[0] == pytest.approx(route_arc_m(path)[-1])
    assert remaining[-1] == pytest.approx(0.0)


def test_the_straight_line_is_not_monotonic_on_the_same_route():
    """Establishes that the previous test is measuring something real: the
    quantity arrival is judged on genuinely grows while the tip advances."""
    path = _hairpin()
    direct = np.linalg.norm(path - path[-1], axis=1)

    assert np.any(np.diff(direct) > 0.0)


def test_remaining_never_goes_negative_past_the_end():
    path = _straight(0.3)

    assert route_progress(path, (0.4, 0.0, 0.0)).remaining_m == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Ambiguity
#
# Nearest-point projection identifies the tip only while it is closer to its
# own stretch of route than to any other. On a hairpin that fails once the tip
# strays past half the gap, and the arc figures then name the wrong limb.
# --------------------------------------------------------------------------- #
def test_the_fold_between_the_limbs_is_what_limits_the_radius():
    """The two limbs are 40 mm apart, and at the midline between them the tip is
    equidistant from places 400 mm apart along the route."""
    assert unambiguous_radius_m(_hairpin(gap_m=0.04)) == pytest.approx(0.02, abs=1e-3)


def test_a_wider_vessel_tolerates_a_further_stray():
    assert unambiguous_radius_m(_hairpin(gap_m=0.08)) > unambiguous_radius_m(_hairpin(gap_m=0.04))


def test_a_straight_route_is_limited_only_by_the_arc_gap_itself():
    """Projection onto a straight route is exact at any offset, but this
    measurement cannot witness that, so it reports the conservative half-gap
    rather than claiming a guarantee it has not established."""
    assert unambiguous_radius_m(_straight(0.3), min_arc_gap_m=0.1) == pytest.approx(0.05, abs=1e-3)


def test_the_measurement_ignores_the_route_being_near_itself_along_its_length():
    """Every stretch is close to the next; that is continuity, not two places
    to confuse, and counting it would hide the fold behind it."""
    folded = _hairpin(gap_m=0.04)

    # The limbs at 40 mm are found rather than the 100 mm of route excluded.
    assert unambiguous_radius_m(folded, min_arc_gap_m=0.1) < unambiguous_radius_m(_straight(0.3), min_arc_gap_m=0.1)


def test_a_route_too_short_to_double_back_has_no_ambiguity():
    assert unambiguous_radius_m(_hairpin(), min_arc_gap_m=10.0) == np.inf


def test_a_tip_inside_the_lumen_is_on_route():
    path = _hairpin(gap_m=0.04)

    assert route_progress(path, (0.1, 0.005, 0.0)).on_route


def test_a_tip_between_the_limbs_is_flagged_rather_than_mislocated():
    """Equidistant from both limbs, so which one it reports is a coin toss
    between places 400 mm apart along the route. Unflagged it would read as
    progress the tip may not have made."""
    path = _hairpin(gap_m=0.04)

    progress = route_progress(path, (0.1, 0.02, 0.01))

    assert not progress.on_route


def test_a_tip_that_crossed_to_the_other_limb_is_reported_there():
    """Nearest-point is not wrong here, it is the only answer available: the
    tip really is in the return limb, and the flag stays on because the arc
    figure is not a guess."""
    path = _hairpin(gap_m=0.04)

    progress = route_progress(path, (0.1, 0.035, 0.0))

    assert progress.on_route
    assert progress.arc_m > 0.3


def test_progress_is_a_value_not_a_view_of_the_route():
    path = _straight(0.3)
    progress = route_progress(path, (0.12, 0.0, 0.0))
    path[:] = 0.0

    assert isinstance(progress, RouteProgress)
    assert progress.arc_m == pytest.approx(0.12)
