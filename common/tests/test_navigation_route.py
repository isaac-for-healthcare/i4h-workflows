# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The derived navigation goal columns, and that they agree with the Scene.

These columns are labels a policy is trained against, so the failure worth
testing for is not an exception but a plausible wrong number. Each test below
pins one way the derivation could drift from what the Scene publishes while
still looking reasonable.
"""

from __future__ import annotations

import numpy as np
import pytest

from i4h_common.navigation_route import (
    GOAL_COLUMN_NAMES,
    ROUTE_SPACING_MM,
    NavigationRoute,
    goal_columns,
    project_to_route,
)


def straight_route(length_m: float = 1.0, samples: int = 11) -> NavigationRoute:
    """A route along +x, where every quantity is checkable by hand."""
    path = np.zeros((samples, 3))
    path[:, 0] = np.linspace(0.0, length_m, samples)
    return NavigationRoute(path_world_m=path, lumen_radii_m=None, target_world_m=path[-1].copy())


def test_the_route_spacing_is_owned_here() -> None:
    """The Scene imports this; a literal in either place is how the two drift apart."""
    assert pytest.approx(7.5) == ROUTE_SPACING_MM


def test_arc_and_lateral_are_independent() -> None:
    """The pair the reward is built on has to separate along-route from across-route.

    A tip 300 mm along and 40 mm off-axis is the recorded failure this exists
    for: nearly all the length, none of the alignment. Collapsing them into one
    distance scores that episode as nearly arrived.
    """
    route = straight_route()
    arc, lateral = project_to_route(route.path_world_m, np.array([[0.3, 0.04, 0.0]]))

    assert arc[0] == pytest.approx(0.3)
    assert lateral[0] == pytest.approx(0.04)


def test_a_tip_past_the_end_does_not_report_negative_remaining() -> None:
    """Projection clamps to the final sample, and overshoot is not a thing the route can say."""
    route = straight_route()
    columns = goal_columns(np.array([[1.5, 0.0, 0.0]]), route)

    assert columns[0, 3] == pytest.approx(0.0)


def test_the_target_offset_is_in_world_axes() -> None:
    """Matching the ``target_offset`` observation, which is what a policy sees at rollout.

    Tip-local would be the friendlier encoding, and is wrong here precisely
    because it is friendlier: the recorded column would then disagree with the
    observation the checkpoint was trained to expect.
    """
    route = straight_route()
    columns = goal_columns(np.array([[0.25, 0.1, -0.2]]), route)

    assert columns[0, 0:3] == pytest.approx([0.75, -0.1, 0.2])


def test_remaining_arc_falls_monotonically_along_the_route() -> None:
    """What makes it drivable, and what straight-line distance does not do on a route that doubles back."""
    route = straight_route()
    tip = np.zeros((11, 3))
    tip[:, 0] = np.linspace(0.0, 1.0, 11)
    remaining = goal_columns(tip, route)[:, 3]

    assert np.all(np.diff(remaining) < 0.0)
    assert remaining[0] == pytest.approx(1.0)
    assert remaining[-1] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "route_x",
    [
        [0.0, 0.0, 0.5, 1.0],
        [0.0, 0.5, 0.5, 1.0],
        [0.0, 0.5, 1.0, 1.0],
        [0.0, 0.5, 0.5, 0.5, 1.0],
    ],
    ids=["repeated-start", "repeated-middle", "repeated-end", "consecutive-repeats"],
)
def test_repeated_vertices_preserve_route_distances(route_x: list[float]) -> None:
    """Repeated vertices add no arc, including for tips beyond the repeat."""
    path = np.zeros((len(route_x), 3))
    path[:, 0] = route_x
    route = NavigationRoute(path_world_m=path, lumen_radii_m=None, target_world_m=path[-1].copy())
    tips = np.zeros((5, 3))
    tips[:, 0] = [0.0, 0.25, 0.5, 0.75, 1.0]
    arc, lateral = project_to_route(path, tips)
    columns = goal_columns(tips, route)

    assert np.isfinite(columns).all()
    assert arc == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert lateral == pytest.approx(np.zeros(5))
    assert columns[:, 3] == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])


def test_a_non_finite_tip_sample_yields_zeros_rather_than_poisoning_the_dataset() -> None:
    """The solver emits transient non-finite positions, and one nan ruins normalization for every episode."""
    route = straight_route()
    tip = np.array([[0.25, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.5, 0.0, 0.0]])
    columns = goal_columns(tip, route)

    assert np.isfinite(columns).all()
    assert columns[1].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert columns[0, 3] == pytest.approx(0.75)
    assert columns[2, 3] == pytest.approx(0.5)


def test_the_column_order_is_the_one_the_descriptor_declares() -> None:
    """Offsets first, then the remaining/lateral pair, matching ``route_state``."""
    assert GOAL_COLUMN_NAMES == (
        "target_offset_x_m",
        "target_offset_y_m",
        "target_offset_z_m",
        "route_remaining_m",
        "route_lateral_m",
    )


def test_a_mis_shaped_tip_trajectory_is_refused() -> None:
    """Silently broadcasting a wrong shape would label an episode against the wrong frames."""
    route = straight_route()
    with pytest.raises(ValueError, match="shape"):
        goal_columns(np.zeros((5, 2)), route)
