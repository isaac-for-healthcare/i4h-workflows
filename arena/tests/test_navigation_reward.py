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

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from i4h_arena.medical.navigation_reward import (
    MAX_STEP_ADVANCE_M,
)
from i4h_arena.medical.navigation_reward import _route_tensors as _build_route_tensors
from i4h_arena.medical.navigation_reward import (
    _segment_radii,
    lateral_offset_penalty,
    project_to_route,
    reset_route_progress,
    reset_tip_route_state,
    route_length_m,
    route_progress_reward,
    tip_route_state,
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
        # The terms share one projection per value of this counter, so moving
        # the rod without advancing it would be a second read of the same step
        # and is answered from the cache -- correctly, and not what a test that
        # means "one step later" is asking.
        self.common_step_counter = 0
        self.scene = {"catheter": SimpleNamespace(data=SimpleNamespace(positions_world_m=None))}

    def place(self, *polylines: tuple[tuple[float, float, float], ...]) -> None:
        self.scene["catheter"].data.positions_world_m = torch.tensor(polylines, dtype=torch.float32)
        self.common_step_counter += 1

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
    """A mid-rod particle 3 mm through a 5 mm wall is a 3 mm penetration.

    The tip and the trailing node are both inside the lumen here, so a
    tip-only term would read zero. The shaft cutting a corner while the tip
    threads it is the case this term exists for.
    """
    env = _FakeEnv()
    env.place(((0.1, 0.0, 0.0), (0.2, 0.008, 0.0), (0.3, 0.0, 0.0)))

    assert wall_penetration_penalty(env, ROUTE, RADII).item() == pytest.approx(0.003, abs=1e-5)


def test_a_perforation_costs_the_same_however_much_rod_is_inside_the_vessel():
    """The defect this replaced: averaging over particles diluted the one event
    the term exists to catch, and diluted it *more* the further the catheter
    was inserted, because particles still parked at the entry contribute no
    depth but did count toward the mean. Across the real rod's 121 particles a
    1 mm perforation averaged to 8.3 um and cost about one per cent of a
    traverse for a whole episode of driving the tip through tissue.
    """
    few, many = _FakeEnv(), _FakeEnv()
    few.place(((0.1, 0.0, 0.0), (0.2, 0.008, 0.0)))
    many.place(((0.1, 0.0, 0.0),) * 20 + ((0.2, 0.008, 0.0),))

    assert wall_penetration_penalty(few, ROUTE, RADII).item() == pytest.approx(0.003, abs=1e-5)
    assert wall_penetration_penalty(many, ROUTE, RADII).item() == pytest.approx(0.003, abs=1e-5)


def test_penetration_grades_depth_rather_than_counting_contacts():
    shallow, deep = _FakeEnv(), _FakeEnv()
    shallow.place(((0.2, 0.006, 0.0),) * 2)
    deep.place(((0.2, 0.011, 0.0),) * 2)

    assert wall_penetration_penalty(deep, ROUTE, RADII) > wall_penetration_penalty(shallow, ROUTE, RADII)


# --------------------------------------------------------------------------- #
# The caches distinguish one route from another
# --------------------------------------------------------------------------- #
#: Half as long as ``ROUTE`` and along a different axis, so a term handed this
#: after ``ROUTE`` cannot agree with one handed ``ROUTE`` by coincidence.
OTHER_ROUTE = tuple((0.0, index / 20.0, 0.0) for index in range(11))


def _cached_arc(env: _FakeEnv, route: tuple[tuple[float, float, float], ...]) -> float:
    """Total arc of the route as the cache hands it back."""
    _starts, spans, start_arc = _build_route_tensors(env, route)
    return float(start_arc[-1] + torch.linalg.norm(spans[-1]))


def test_a_second_route_is_not_served_the_first_one_s_tensors():
    """The caches used to key on the device alone, so one env could only ever
    hold one route. Correct while every config is built from the same
    ``rod_spec``, wrong the moment a twin is randomized per episode."""
    env = _env_at((0.4, 0.0, 0.0))

    assert _cached_arc(env, ROUTE) == pytest.approx(1.0)
    assert _cached_arc(env, OTHER_ROUTE) == pytest.approx(0.5)
    # Back to the first, which must not have been evicted by the second.
    assert _cached_arc(env, ROUTE) == pytest.approx(1.0)


def test_a_second_set_of_radii_is_not_served_the_first_one_s():
    env = _env_at((0.4, 0.004, 0.0))
    narrow = tuple(0.001 for _ in ROUTE)

    wide_penalty = lateral_offset_penalty(env, ROUTE, RADII)
    narrow_penalty = lateral_offset_penalty(env, ROUTE, narrow)

    assert narrow_penalty.item() > wide_penalty.item()


def test_a_shorter_route_does_not_get_a_longer_one_s_radius_slice():
    """What the radii cache stores is already truncated to the segment count,
    so the count has to be part of the key and not only the widths."""
    env = _env_at((0.4, 0.004, 0.0))

    lateral_offset_penalty(env, ROUTE, RADII)
    radii = _segment_radii(env, RADII, 4)

    assert radii is not None
    assert int(radii.shape[0]) == 4


def test_a_route_the_configs_did_not_normalize_is_still_cacheable():
    """The signature takes any iterable, and a list of lists is unhashable."""
    env = _env_at((0.4, 0.0, 0.0))
    listed = [list(point) for point in ROUTE]

    assert _cached_arc(env, listed) == pytest.approx(_cached_arc(env, ROUTE))


# --------------------------------------------------------------------------- #
# One diverged rod does not speak for the batch
# --------------------------------------------------------------------------- #
def _with_one_exploded_rod() -> _FakeEnv:
    """Two environments, the second one's tip blown to infinity."""
    env = _FakeEnv(num_envs=2)
    env.place_tip((0.2, 0.004, 0.0), (0.5, 0.004, 0.0))
    points = env.scene["catheter"].data.positions_world_m.clone()
    points[1, -1, :] = float("inf")
    env.scene["catheter"].data.positions_world_m = points
    env.common_step_counter += 1
    return env


def test_one_exploded_rod_does_not_zero_its_neighbours():
    """The defect: a batch-wide finiteness test meant one diverged rod zeroed
    progress, the lateral penalty and the route observation for every
    environment, so seven healthy ones were told nothing they did mattered.
    """
    env = _with_one_exploded_rod()

    lateral = lateral_offset_penalty(env, ROUTE, RADII)
    penetration = wall_penetration_penalty(env, ROUTE, RADII)

    assert lateral[0].item() > 0.0
    assert lateral[1].item() == pytest.approx(0.0)
    assert penetration[1].item() == pytest.approx(0.0)


def test_an_unreadable_tip_is_substituted_before_projecting_not_masked_after():
    """``inf * 0`` is ``nan``, so masking the result of projecting an infinity
    poisons the batch instead of zeroing one row."""
    env = _with_one_exploded_rod()

    state = tip_route_state(env, ROUTE)

    assert torch.isfinite(state.lateral_m).all()
    assert torch.isfinite(lateral_offset_penalty(env, ROUTE, RADII)).all()
    assert state.valid.tolist() == [True, False]


def test_an_unreadable_tip_reads_as_the_whole_route_ahead_not_as_arrival():
    """Zeroing the route observation reports remaining arc zero, which is the
    signature of a perfect arrival. An unreadable rod must not claim the task
    is done."""
    env = _with_one_exploded_rod()

    state = tip_route_state(env, ROUTE)

    assert state.remaining_m[1].item() == pytest.approx(state.total_m.item())


def test_a_tip_that_comes_back_is_not_paid_for_the_route_it_never_travelled():
    """The substituted position projects to the start of the route. Stored as
    an arc, the step it became readable again would difference the whole route
    and pay out the entire task."""
    env = _with_one_exploded_rod()
    route_progress_reward(env, ROUTE)

    env.place_tip((0.2, 0.004, 0.0), (0.5, 0.004, 0.0))

    assert route_progress_reward(env, ROUTE).tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_the_projection_is_shared_by_every_term_within_one_step():
    """Three terms and an observation wanted the same projection of the same
    tip, and the batch-wide test each one ran was a host synchronization."""
    env = _env_at((0.2, 0.004, 0.0))

    first = tip_route_state(env, ROUTE)

    assert tip_route_state(env, ROUTE) is first


def test_a_reset_environment_is_re_projected_rather_than_served_the_stale_step():
    """IsaacLab rewards, then resets, then observes, all under one step
    counter, so the cache has to go with the reset or the new episode's first
    observation describes where the old tip was."""
    env = _env_at((0.9, 0.0, 0.0))
    before = tip_route_state(env, ROUTE).remaining_m.item()

    env.scene["catheter"].data.positions_world_m = torch.tensor(
        [[(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.1, 0.0, 0.0)]], dtype=torch.float32
    )
    reset_tip_route_state(env)

    assert tip_route_state(env, ROUTE).remaining_m.item() > before


# --------------------------------------------------------------------------- #
# Route length
# --------------------------------------------------------------------------- #
def test_route_length_sums_the_polyline():
    assert route_length_m(ROUTE) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The clamp still clears the drive it was sized against
# --------------------------------------------------------------------------- #
#: The embodiments that declare the insertion ceiling. Read statically because
#: both modules import ``isaaclab``.
DRIVE_SOURCES = (
    (Path(__file__).resolve().parents[1] / "i4h_arena/embodiments/catheter.py", "CatheterVelocityActionCfg"),
    (
        Path(__file__).resolve().parents[1] / "i4h_arena/embodiments/franka_catheter.py",
        "ArmDrivenCatheterActionCfg",
    ),
)
#: Where ``catheter.py`` sets the physics step and the control decimation.
SIM_SOURCE = DRIVE_SOURCES[0][0]


def _class_default(path: Path, class_name: str, field: str) -> float:
    """One annotated class-level default, read without importing Isaac Sim."""
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and getattr(statement.target, "id", None) == field:
                return float(ast.literal_eval(statement.value))
    raise AssertionError(f"{class_name}.{field} not found in {path.name}")


def _number(node: ast.AST) -> float:
    """A float from a literal or a product or quotient of literals.

    ``sim.dt`` is written ``1.0 / 120.0`` to keep the rate readable, which
    ``ast.literal_eval`` will not evaluate.
    """
    if isinstance(node, ast.BinOp):
        left, right = _number(node.left), _number(node.right)
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mult):
            return left * right
        raise AssertionError(f"unsupported operator in {ast.unparse(node)}")
    return float(ast.literal_eval(node))


def _attribute_assignment(path: Path, dotted: str) -> float:
    """The value assigned to a dotted attribute path, e.g. ``env_cfg.sim.dt``."""
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            names = []
            while isinstance(target, ast.Attribute):
                names.append(target.attr)
                target = target.value
            if isinstance(target, ast.Name):
                names.append(target.id)
            if ".".join(reversed(names)) == dotted:
                return _number(node.value)
    raise AssertionError(f"no assignment to {dotted} in {path.name}")


#: The per-step ceiling ``MAX_STEP_ADVANCE_M``'s docstring quotes, in metres.
DOCUMENTED_CEILING_M = 0.002


@pytest.mark.parametrize(("source", "class_name"), DRIVE_SOURCES, ids=lambda value: getattr(value, "stem", value))
def test_step_advance_clears_the_drive_ceiling(source: Path, class_name: str):
    """The clamp exceeds what the drive can feed, and the quoted ceiling is real.

    Both halves earn their place. The docstring used to quote a velocity
    neither embodiment declared, so the stated ceiling was wrong while the
    value stayed accidentally safe -- the equality is what catches that. The
    inequality catches an embodiment raising its limit past the clamp, or the
    control rate moving under it.
    """
    velocity_mps = _class_default(source, class_name, "max_insertion_velocity_mps")
    physics_dt_s = _attribute_assignment(SIM_SOURCE, "env_cfg.sim.dt")
    decimation = _attribute_assignment(SIM_SOURCE, "env_cfg.decimation")
    ceiling_m = velocity_mps * physics_dt_s * decimation

    assert ceiling_m == pytest.approx(DOCUMENTED_CEILING_M)
    assert MAX_STEP_ADVANCE_M > ceiling_m
