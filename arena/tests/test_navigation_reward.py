# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dense catheter navigation reward.

The valuable cases are the ones that motivated each term over its obvious
alternative: that progress cannot be farmed by a projection jump, that a reset
does not pay out a whole route, that the lateral term still fires on a tip
which has travelled the full arc, and that a fold is graded rather than
flagged. Each runs against a fake env rather than a live Newton model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from i4h_arena.medical.navigation_reward import (
    MAX_STEP_ADVANCE_M,
    approach_reward,
    arrival_reward,
    bend_radius_m,
    fold_penalty,
    lateral_offset_penalty,
    project_to_route,
    reset_route_progress,
    route_length_m,
    route_progress_reward,
    wall_penetration_penalty,
)

#: A metre of straight route along +x, sampled every 100 mm.
ROUTE = tuple((index / 10.0, 0.0, 0.0) for index in range(11))
#: A 5 mm-radius lumen the whole way along.
RADII = tuple(0.005 for _ in ROUTE)


class _FakeEnv:
    """Minimum a reward term reads: a catheter polyline and env sizing."""

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = {"catheter": SimpleNamespace(data=SimpleNamespace(positions_world_m=None))}

    def place(self, *polylines: tuple[tuple[float, float, float], ...]) -> None:
        self.scene["catheter"].data.positions_world_m = torch.tensor(polylines, dtype=torch.float32)

    def place_tip(self, *tips: tuple[float, float, float]) -> None:
        """Give each environment a short polyline ending at ``tips[i]``."""
        self.place(*(((0.0, 0.0, 0.0), (tip[0] / 2.0, 0.0, 0.0), tip) for tip in tips))


def _env_at(*tips: tuple[float, float, float]) -> _FakeEnv:
    env = _FakeEnv(num_envs=len(tips))
    env.place_tip(*tips)
    return env


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def _route_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = torch.tensor(ROUTE, dtype=torch.float32)
    starts, spans = path[:-1], path[1:] - path[:-1]
    lengths = torch.linalg.norm(spans, dim=-1)
    start_arc = torch.cat((torch.zeros(1), torch.cumsum(lengths, dim=0)[:-1]))
    return starts, spans, start_arc


def test_projection_reports_arc_along_the_route_and_offset_from_it():
    starts, spans, start_arc = _route_tensors()
    points = torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.003, 0.0]])

    arc_m, lateral_m, _ = project_to_route(points, starts, spans, start_arc)

    assert arc_m.tolist() == pytest.approx([0.25, 0.25], abs=1e-5)
    assert lateral_m.tolist() == pytest.approx([0.0, 0.003], abs=1e-5)


def test_projection_preserves_the_leading_shape_for_a_batch_of_polylines():
    starts, spans, start_arc = _route_tensors()
    points = torch.zeros((4, 7, 3))

    arc_m, lateral_m, segment = project_to_route(points, starts, spans, start_arc)

    assert arc_m.shape == lateral_m.shape == segment.shape == (4, 7)


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #
def test_the_first_step_of_an_episode_pays_nothing():
    """There is no previous remaining arc to difference against yet."""
    env = _env_at((0.2, 0.0, 0.0))

    assert route_progress_reward(env, ROUTE).tolist() == pytest.approx([0.0])


def test_advancing_along_the_route_pays_the_arc_it_closed():
    env = _env_at((0.2, 0.0, 0.0))
    route_progress_reward(env, ROUTE)
    env.place_tip((0.2015, 0.0, 0.0))

    assert route_progress_reward(env, ROUTE).item() == pytest.approx(0.0015, abs=1e-5)


def test_backing_out_costs_what_advancing_the_same_distance_paid():
    """Potential-based shaping: a round trip has to net to zero."""
    env = _env_at((0.2, 0.0, 0.0))
    route_progress_reward(env, ROUTE)
    env.place_tip((0.2015, 0.0, 0.0))
    forward = route_progress_reward(env, ROUTE).item()
    env.place_tip((0.2, 0.0, 0.0))
    back = route_progress_reward(env, ROUTE).item()

    assert forward + back == pytest.approx(0.0, abs=1e-6)


def test_a_projection_jump_cannot_pay_more_than_insertion_can_deliver():
    """The measured exploit: arc jumped 31 mm in five steps on a real episode.

    Nearest-point projection is discontinuous where the route doubles back, so
    a tip crossing the arch can appear to gain far more vessel than insertion
    could have fed. The clamp is what makes that unprofitable.
    """
    env = _env_at((0.1, 0.0, 0.0))
    route_progress_reward(env, ROUTE)
    env.place_tip((0.9, 0.0, 0.0))

    assert route_progress_reward(env, ROUTE).item() == pytest.approx(MAX_STEP_ADVANCE_M)


def test_the_clamp_is_symmetric_so_a_round_trip_is_never_profitable():
    env = _env_at((0.1, 0.0, 0.0))
    route_progress_reward(env, ROUTE)
    env.place_tip((0.9, 0.0, 0.0))
    out = route_progress_reward(env, ROUTE).item()
    env.place_tip((0.1, 0.0, 0.0))
    home = route_progress_reward(env, ROUTE).item()

    assert out + home == pytest.approx(0.0, abs=1e-6)


def test_a_reset_environment_does_not_earn_the_route_it_was_moved_off():
    """Differencing across a reset would pay the whole task for teleporting."""
    env = _env_at((0.9, 0.0, 0.0))
    route_progress_reward(env, ROUTE)
    reset_route_progress(env, torch.tensor([0]))
    env.place_tip((0.0, 0.0, 0.0))

    assert route_progress_reward(env, ROUTE).tolist() == pytest.approx([0.0])


def test_progress_is_independent_per_environment():
    env = _env_at((0.2, 0.0, 0.0), (0.5, 0.0, 0.0))
    route_progress_reward(env, ROUTE)
    env.place_tip((0.2010, 0.0, 0.0), (0.5, 0.0, 0.0))

    assert route_progress_reward(env, ROUTE).tolist() == pytest.approx([0.0010, 0.0], abs=1e-5)


def test_progress_pays_nothing_before_newton_has_particles():
    env = _FakeEnv()

    assert route_progress_reward(env, ROUTE).tolist() == pytest.approx([0.0])


# --------------------------------------------------------------------------- #
# Lateral offset
# --------------------------------------------------------------------------- #
def test_the_inner_half_of_the_lumen_is_free():
    env = _env_at((0.5, 0.002, 0.0))

    assert lateral_offset_penalty(env, ROUTE, RADII).item() == pytest.approx(0.0)


def test_offset_past_the_free_fraction_is_charged_relative_to_the_lumen():
    """4 mm off-axis in a 5 mm lumen: 1.5 mm past the free 2.5 mm."""
    env = _env_at((0.5, 0.004, 0.0))

    assert lateral_offset_penalty(env, ROUTE, RADII).item() == pytest.approx(0.3, abs=1e-4)


def test_a_tip_at_the_end_of_the_route_is_still_charged_for_being_off_axis():
    """The recorded failure: all of the arc, none of the alignment.

    Arc progress alone scores this as a near-complete episode, which is the
    reason a separate lateral term exists at all.
    """
    env = _env_at((1.0, 0.0072, 0.0))

    assert lateral_offset_penalty(env, ROUTE, RADII).item() > 0.9


def test_lateral_offset_is_not_charged_without_lumen_widths():
    env = _env_at((0.5, 0.05, 0.0))

    assert lateral_offset_penalty(env, ROUTE, None).item() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Wall penetration
# --------------------------------------------------------------------------- #
def test_a_rod_inside_the_lumen_is_not_penalized():
    env = _FakeEnv()
    env.place(((0.1, 0.0, 0.0), (0.2, 0.001, 0.0), (0.3, 0.0, 0.0)))

    assert wall_penetration_penalty(env, ROUTE, RADII).item() == pytest.approx(0.0)


def test_penetration_is_measured_over_every_particle_not_just_the_tip():
    """One of three particles 3 mm through a 5 mm wall averages to 1 mm."""
    env = _FakeEnv()
    env.place(((0.1, 0.0, 0.0), (0.2, 0.008, 0.0), (0.3, 0.0, 0.0)))

    assert wall_penetration_penalty(env, ROUTE, RADII).item() == pytest.approx(0.001, abs=1e-5)


def test_penetration_grades_depth_rather_than_counting_contacts():
    shallow, deep = _FakeEnv(), _FakeEnv()
    shallow.place(((0.2, 0.006, 0.0),) * 2)
    deep.place(((0.2, 0.011, 0.0),) * 2)

    assert wall_penetration_penalty(deep, ROUTE, RADII) > wall_penetration_penalty(shallow, ROUTE, RADII)


# --------------------------------------------------------------------------- #
# Folding
# --------------------------------------------------------------------------- #
def test_a_straight_rod_has_infinite_bend_radius():
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]])

    assert torch.isinf(bend_radius_m(positions)).all()


def test_bend_radius_matches_the_circle_through_three_points():
    """A right-angle triple on a 10 mm grid circumscribes a circle of r = 5*sqrt(2)."""
    positions = torch.tensor([[[0.0, 0.01, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]])

    assert bend_radius_m(positions).item() == pytest.approx(0.01 / (2.0**0.5), abs=1e-5)


def test_a_gentle_anatomical_bend_is_not_called_a_fold():
    env = _FakeEnv()
    env.place(((0.0, 0.05, 0.0), (0.0, 0.0, 0.0), (0.05, 0.0, 0.0)))

    assert fold_penalty(env).item() == pytest.approx(0.0)


def test_folding_is_graded_rather_than_flagged():
    """A boolean fired on nine frames in ten of a real episode, so it carried
    no gradient. A tighter crease has to cost strictly more than a looser one.
    """
    loose, tight = _FakeEnv(), _FakeEnv()
    loose.place(((0.0, 0.004, 0.0), (0.0, 0.0, 0.0), (0.004, 0.0, 0.0)))
    tight.place(((0.0, 0.001, 0.0), (0.0, 0.0, 0.0), (0.001, 0.0, 0.0)))

    assert 0.0 < fold_penalty(loose).item() < fold_penalty(tight).item()


def test_a_rod_too_short_to_have_an_interior_node_is_not_folded():
    env = _FakeEnv()
    env.place(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)))

    assert fold_penalty(env).item() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Arrival and approach
# --------------------------------------------------------------------------- #
def test_arrival_pays_every_step_the_tip_holds_inside_the_tolerance():
    """The hold the success criterion requires has to be worth something."""
    env = _env_at((1.0, 0.0, 0.0), (0.9, 0.0, 0.0))

    assert arrival_reward(env, (1.0, 0.0, 0.0)).tolist() == pytest.approx([1.0, 0.0])


def test_approach_still_has_a_gradient_inside_the_last_centimetre():
    """Remaining arc goes flat within one route sample of the end, so the term
    that decides the 5 mm tolerance needs its own finer scale."""
    near = _env_at((0.998, 0.0, 0.0))
    far = _env_at((0.99, 0.0, 0.0))

    assert approach_reward(near, (1.0, 0.0, 0.0), 0.025) > approach_reward(far, (1.0, 0.0, 0.0), 0.025)


def test_approach_pays_nothing_before_newton_has_particles():
    env = _FakeEnv()

    assert approach_reward(env, (1.0, 0.0, 0.0), 0.025).tolist() == pytest.approx([0.0])


# --------------------------------------------------------------------------- #
# Route length
# --------------------------------------------------------------------------- #
def test_route_length_sums_the_polyline():
    assert route_length_m(ROUTE) == pytest.approx(1.0)
