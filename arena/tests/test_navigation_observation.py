# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the catheter navigation observation terms.

The property worth defending hardest is width stability. IsaacLab concatenates
observation terms into a fixed-width vector once, at startup, and a term that
later returned a different shape -- or a nan -- takes the run down mid-episode
rather than failing loudly at construction. So every term is checked both when
the rod is readable and when it is not, and the two are required to agree on
width.

The rest covers what each term is *for*: that positions are relative so a
policy cannot memorize world coordinates, and that drive state survives a
scene missing the terms it reads from.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from i4h_arena.medical.navigation_observation import (
    DRIVE_STATE_DIM,
    NAVIGATION_STATE_DIM,
    ROUTE_STATE_DIM,
    TARGET_OFFSET_DIM,
    TIP_DIRECTION_DIM,
    TIP_POSITION_DIM,
    drive_state,
    fluoroscopy_image,
    route_state,
    target_offset,
    tip_direction,
    tip_position,
)

#: A metre of straight route along +x, sampled every 100 mm, matching the
#: reward tests so a failure here can be compared against one there.
ROUTE = tuple((index / 10.0, 0.0, 0.0) for index in range(11))


class _Scene(dict):
    """A scene mapping that can also carry cloned-environment origins."""

    env_origins = None
    sensors: dict = {}


class _FakeEnv:
    """Minimum an observation term reads: a polyline, sizing, and maybe a drive."""

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = _Scene({"catheter": SimpleNamespace(data=SimpleNamespace(positions_world_m=None))})

    def place(self, *polylines: tuple[tuple[float, float, float], ...]) -> None:
        self.scene["catheter"].data.positions_world_m = torch.tensor(polylines, dtype=torch.float32)

    def place_tip(self, *tips: tuple[float, float, float]) -> None:
        self.place(*(((0.0, 0.0, 0.0), (tip[0] / 2.0, 0.0, 0.0), tip) for tip in tips))

    def with_fluoroscopy(self, *, renderable: bool, edge: int = 8) -> _FakeEnv:
        """Attach a detector that either renders or is still awaiting its C-arm."""
        frames = torch.full((self.num_envs, edge, edge, 3), 7, dtype=torch.uint8)
        sensor = SimpleNamespace(
            is_renderable=renderable,
            cfg=SimpleNamespace(height=edge, width=edge),
            data=SimpleNamespace(output={"rgb": frames}),
        )
        self.scene.sensors = {"fluoroscopy": sensor}
        return self

    def with_drive(self, **channels: float) -> _FakeEnv:
        """Attach an action manager exposing the four drive channels."""
        catheter = SimpleNamespace(
            insertion_depth_m=torch.full((self.num_envs,), channels.get("depth", 0.0)),
            twist_rad=torch.full((self.num_envs,), channels.get("twist", 0.0)),
            tip_bend_angle=torch.full((self.num_envs,), channels.get("bend", 0.0)),
        )
        carm = SimpleNamespace(angle_rad=torch.full((self.num_envs,), channels.get("orbit", 0.0)))
        terms = {"catheter": catheter, "carm_orbit": carm}
        self.action_manager = SimpleNamespace(get_term=terms.__getitem__)
        return self


def _env_at(*tips: tuple[float, float, float]) -> _FakeEnv:
    env = _FakeEnv(num_envs=len(tips))
    env.place_tip(*tips)
    return env


# --------------------------------------------------------------------------- #
# Width stability
# --------------------------------------------------------------------------- #
def _call(term, env):
    """Invoke a term with whatever binding arguments it needs."""
    if term is target_offset:
        return term(env, (1.0, 0.0, 0.0))
    if term is route_state:
        return term(env, ROUTE)
    return term(env)


TERMS = (
    (tip_position, TIP_POSITION_DIM),
    (tip_direction, TIP_DIRECTION_DIM),
    (target_offset, TARGET_OFFSET_DIM),
    (route_state, ROUTE_STATE_DIM),
    (drive_state, DRIVE_STATE_DIM),
)


@pytest.mark.parametrize(("term", "width"), TERMS)
def test_term_width_matches_declared_dim(term, width):
    assert _call(term, _env_at((0.3, 0.0, 0.0)).with_drive()).shape == (1, width)


@pytest.mark.parametrize(("term", "width"), TERMS)
def test_term_keeps_width_before_the_rod_is_readable(term, width):
    """No polyline yet is the state at startup, and it must not change shape."""
    env = _FakeEnv(num_envs=3)
    assert _call(term, env).shape == (3, width)


@pytest.mark.parametrize(("term", "width"), TERMS)
def test_term_keeps_width_when_the_tip_is_infinite(term, width):
    """The tip reads as infinite before Newton finalizes; terms must absorb it."""
    env = _env_at((float("inf"), float("inf"), float("inf")))
    assert _call(term, env).shape == (1, width)


@pytest.mark.parametrize(("term", "_width"), TERMS)
def test_term_never_emits_non_finite_values(term, _width):
    env = _env_at((float("inf"), 0.0, 0.0), (0.3, 0.0, 0.0))
    assert torch.isfinite(_call(term, env)).all()


def test_declared_total_matches_the_sum_of_the_terms():
    env = _env_at((0.3, 0.0, 0.0)).with_drive()
    total = sum(_call(term, env).shape[-1] for term, _ in TERMS)
    assert total == NAVIGATION_STATE_DIM


# --------------------------------------------------------------------------- #
# Tip pose
# --------------------------------------------------------------------------- #
def test_tip_position_is_the_last_polyline_node():
    assert tip_position(_env_at((0.3, 0.1, -0.2))).squeeze(0).tolist() == pytest.approx([0.3, 0.1, -0.2])


def test_tip_position_is_relative_to_the_cloned_env_origin():
    """Absolute coordinates would let a policy memorize one patient's frame."""
    env = _env_at((0.3, 0.0, 0.0), (0.5, 0.0, 0.0))
    env.scene.env_origins = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    assert tip_position(env)[:, 0].tolist() == pytest.approx([0.3, 0.3])


def test_tip_direction_is_a_unit_vector_along_the_shaft():
    direction = tip_direction(_env_at((0.4, 0.0, 0.0))).squeeze(0)
    assert direction.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert float(torch.linalg.norm(direction)) == pytest.approx(1.0)


def test_tip_direction_is_zero_for_a_degenerate_final_segment():
    """Coincident particles are transient solver output, not a nan direction."""
    env = _FakeEnv()
    env.place(((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.2, 0.0, 0.0)))
    assert tip_direction(env).squeeze(0).tolist() == pytest.approx([0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# Target and route
# --------------------------------------------------------------------------- #
def test_target_offset_points_from_the_tip_to_the_target():
    offset = target_offset(_env_at((0.3, 0.0, 0.0)), (1.0, 0.0, 0.0)).squeeze(0)
    assert offset.tolist() == pytest.approx([0.7, 0.0, 0.0])


def test_target_offset_ignores_env_origins():
    """A difference of two world points is already origin-free; correcting it twice would double-count."""
    env = _env_at((0.3, 0.0, 0.0))
    env.scene.env_origins = torch.tensor([[5.0, 5.0, 5.0]])
    assert target_offset(env, (1.0, 0.0, 0.0)).squeeze(0).tolist() == pytest.approx([0.7, 0.0, 0.0])


def test_route_state_reports_remaining_arc_and_lateral_offset():
    remaining, lateral = route_state(_env_at((0.4, 0.02, 0.0)), ROUTE).squeeze(0).tolist()
    assert remaining == pytest.approx(0.6, abs=1e-4)
    assert lateral == pytest.approx(0.02, abs=1e-4)


def test_route_state_separates_arc_from_alignment():
    """The recorded failure: nearly all the arc closed, badly off the axis."""
    on_axis = route_state(_env_at((0.99, 0.0, 0.0)), ROUTE).squeeze(0)
    off_axis = route_state(_env_at((0.99, 0.03, 0.0)), ROUTE).squeeze(0)
    assert float(on_axis[0]) == pytest.approx(float(off_axis[0]), abs=1e-3)
    assert float(off_axis[1]) > float(on_axis[1])


# --------------------------------------------------------------------------- #
# Drive state
# --------------------------------------------------------------------------- #
def test_drive_state_reads_all_four_channels_in_order():
    env = _env_at((0.3, 0.0, 0.0)).with_drive(depth=0.25, twist=1.5, bend=-0.4, orbit=0.8)
    assert drive_state(env).squeeze(0).tolist() == pytest.approx([0.25, 1.5, -0.4, 0.8])


def test_drive_state_is_zero_without_an_action_manager():
    """Replay and dataset tooling build the env without one."""
    assert drive_state(_env_at((0.3, 0.0, 0.0))).squeeze(0).tolist() == pytest.approx([0.0] * DRIVE_STATE_DIM)


def test_drive_state_tolerates_a_missing_action_term():
    """A route-less scene has no C-arm; the state stays the same width."""
    env = _env_at((0.3, 0.0, 0.0))
    catheter = SimpleNamespace(
        insertion_depth_m=torch.tensor([0.25]),
        twist_rad=torch.tensor([1.5]),
        tip_bend_angle=torch.tensor([-0.4]),
    )
    env.action_manager = SimpleNamespace(get_term={"catheter": catheter}.__getitem__)
    assert drive_state(env).squeeze(0).tolist() == pytest.approx([0.25, 1.5, -0.4, 0.0])


def test_drive_state_scrubs_non_finite_channels():
    env = _env_at((0.3, 0.0, 0.0)).with_drive(depth=float("nan"), twist=float("inf"), bend=0.2, orbit=0.1)
    assert drive_state(env).squeeze(0).tolist() == pytest.approx([0.0, 0.0, 0.2, 0.1])


def test_drive_state_is_per_environment():
    env = _env_at((0.3, 0.0, 0.0), (0.5, 0.0, 0.0)).with_drive(depth=0.25)
    assert drive_state(env).shape == (2, DRIVE_STATE_DIM)
    assert drive_state(env)[:, 0].tolist() == pytest.approx([0.25, 0.25])


#: The sensor name the navigation observation group binds.
_FLUORO = SimpleNamespace(name="fluoroscopy")


def test_fluoroscopy_image_serves_zeros_before_the_carm_is_bound():
    """The slang backend cannot render yet; the shape probe must still succeed.

    This is the regression that took a ``--patient-twin`` run down: the stock
    image term rendered on read and raised before the scene had bound a C-arm.
    """
    env = _env_at((0.3, 0.0, 0.0)).with_fluoroscopy(renderable=False)
    frame = fluoroscopy_image(env, _FLUORO)
    assert frame.shape == (1, 8, 8, 3)
    assert frame.dtype == torch.uint8
    assert not frame.any()


def test_fluoroscopy_image_reads_the_sensor_once_renderable():
    env = _env_at((0.3, 0.0, 0.0)).with_fluoroscopy(renderable=True)
    assert (fluoroscopy_image(env, _FLUORO) == 7).all()


@pytest.mark.parametrize("renderable", [False, True])
def test_fluoroscopy_image_width_is_stable_across_binding(renderable: bool):
    """Both sides of the binding must agree, since IsaacLab fixes width once."""
    env = _env_at((0.3, 0.0, 0.0), (0.5, 0.0, 0.0)).with_fluoroscopy(renderable=renderable)
    assert fluoroscopy_image(env, _FLUORO).shape == (2, 8, 8, 3)


def test_fluoroscopy_image_does_not_alias_the_sensor_buffer():
    """A consumer that writes to the observation must not corrupt the sensor."""
    env = _env_at((0.3, 0.0, 0.0)).with_fluoroscopy(renderable=True)
    fluoroscopy_image(env, _FLUORO)[:] = 0
    assert (env.scene.sensors["fluoroscopy"].data.output["rgb"] == 7).all()


def test_fluoroscopy_image_assumes_a_sensor_without_the_flag_can_render():
    """Synthetic-backend sensors predate ``is_renderable`` and always render."""
    env = _env_at((0.3, 0.0, 0.0)).with_fluoroscopy(renderable=True)
    del env.scene.sensors["fluoroscopy"].is_renderable
    assert (fluoroscopy_image(env, _FLUORO) == 7).all()
